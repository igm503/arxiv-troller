"""Enable the validated backend; restore the legacy flag automatically if HTTP checks fail."""
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import uuid
import httpx
import psycopg2
import archive as a

root=Path('/home/arxiv/arxiv_troller')
for name in ['archive-audit.json','legacy-preservation-verified.json','pilot-preservation-verified.json','application-validation.json']:
    assert json.loads((a.ROOT/name).read_text())['passed'],name
assert (a.ROOT/'general-retrieval-validation.json').exists()
pg=psycopg2.connect(dbname='arxiv');pg.autocommit=True
with pg.cursor() as q:
    q.execute("SELECT count(*) FROM voyage4.pending_papers");assert q.fetchone()[0]==0
    q.execute("SELECT count(*) FROM pg_index WHERE indexrelid IN ('voyage4.v4_bits_hnsw'::regclass,'voyage4.v4_rolling30_hnsw'::regclass) AND indisvalid");assert q.fetchone()[0]==2

def flag(value):
    with pg.cursor() as q:q.execute("UPDATE voyage4.control SET value=%s::jsonb,updated_at=now() WHERE key='ready'",(json.dumps(value),))

def write_env(text):
    target=root/'.env';temp=root/('.env.v4-'+uuid.uuid4().hex)
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temp,target)

def reload_app():
    # Discover the master owned by arxiv; do not signal workers or the other hosted app.
    candidates={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            status=(proc/'status').read_text()
            parent=int(re.search(r'^PPid:\s+(\d+)',status,re.M).group(1))
            if proc.stat().st_uid==os.getuid() and 'arxiv_troller.wsgi:application' in argv and 'gunicorn' in argv:candidates[int(proc.name)]=parent
        except (OSError,UnicodeError,AttributeError):continue
    masters=[pid for pid,parent in candidates.items() if parent not in candidates]
    assert len(masters)==1,masters
    os.kill(masters[0],signal.SIGHUP);return masters[0]

before=(root/'.env').read_text()
after=re.sub(r'^VOYAGE4_ENABLED=.*\n?', '',before,flags=re.M).rstrip()+'\nVOYAGE4_ENABLED=1\n'
try:
    flag(True);write_env(after);master=reload_app()
    os.environ['DJANGO_SETTINGS_MODULE']='arxiv_troller.settings_django_deploy';os.environ['VOYAGE4_ENABLED']='1'
    sys.path.insert(0,str(root/'django'))
    import django;django.setup()
    from django.conf import settings
    from papers.voyage4_search import search_ids,GENERAL_EF,GENERAL_CANDIDATES
    from papers.views import get_date_cutoff
    source=3008578;expected={}
    for cat in ['', 'cs.LG']:
        expected[cat]=search_ids(source,cutoff=get_date_cutoff('1month'),category=cat,excluded={source},limit=20)
        assert expected[cat] is not None and len(expected[cat])==20
    host=next((x.lstrip('.') for x in settings.ALLOWED_HOSTS if x!='*'),'localhost')
    checks=[]
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(root/'run/gunicorn.sock')),timeout=30) as client:
        for category in ['', 'cs.LG','']:
            success=False
            for attempt in range(15):
                start=time.perf_counter()
                response=client.get('http://localhost/',params={'single_paper':source,'date_filter':'1month','category':category},headers={'Host':host,'X-Forwarded-Proto':'https','Connection':'close'})
                ids=list(map(int,re.findall(r'class="paper-card" data-paper-id="(\d+)"',response.text)))
                if response.status_code==200 and ids==expected[category]:
                    checks.append(dict(category=category,status=200,results=len(ids),ms=(time.perf_counter()-start)*1000,matched_v4=True));success=True;break
                time.sleep(2)
            assert success,dict(status=response.status_code,returned=len(ids),category=category)
    stat=os.statvfs(root)
    result=dict(activated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),ready=True,general_ef=GENERAL_EF,general_candidates=GENERAL_CANDIDATES,
                master_pid=master,http_checks=checks,free_bytes=stat.f_bavail*stat.f_frsize)
    a.atomic_json(a.ROOT/'activation.json',result);print(json.dumps(result,indent=2))
except BaseException:
    flag(False);write_env(before);reload_app();raise
finally:pg.close()
