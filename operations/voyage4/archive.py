"""Resumable standard-API archive. Original response vectors are immutable float32 shards."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import uuid

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import httpx
import numpy as np
import psycopg2
from dotenv import dotenv_values
from tokenizers import Tokenizer

MODEL='voyage-4-large'
DIM=2048
ENDPOINT='https://api.voyageai.com/v1/embeddings'
ROOT=Path('/home/arxiv/arxiv_troller/data/voyage4')
PILOT=Path('/home/arxiv/arxiv_troller/experiments/voyage4-pilot-20260919/data')
_progress_cache=None


def event(message, **kw):
    print(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),message=message,**kw)),flush=True)


def atomic_json(path, obj):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def catalog(root):
    root.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(root/'catalog.sqlite3')
    c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=FULL')
    c.executescript('''
      CREATE TABLE IF NOT EXISTS papers(id INTEGER PRIMARY KEY,abstract TEXT NOT NULL,sha TEXT NOT NULL,
        tokens INTEGER NOT NULL,created TEXT NOT NULL,categories TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS paper_identity_idx ON papers(id,sha,tokens);
      CREATE TABLE IF NOT EXISTS versions(id INTEGER NOT NULL,sha TEXT NOT NULL,shard TEXT NOT NULL,row_index INTEGER NOT NULL,
        PRIMARY KEY(id,sha));
      CREATE TABLE IF NOT EXISTS shards(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,papers INTEGER NOT NULL,
        tokens INTEGER NOT NULL,origin TEXT NOT NULL,imported INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY,ids TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending');
      CREATE TABLE IF NOT EXISTS metadata_dirty(id INTEGER PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    ''')
    return c


def state(c,key,value=None):
    if value is not None:
        c.execute('INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,json.dumps(value)));c.commit()
    r=c.execute('SELECT value FROM state WHERE key=?',(key,)).fetchone()
    return json.loads(r[0]) if r else None


def check_space(root, minimum_gib=12):
    free=shutil.disk_usage(root).free
    if free<minimum_gib*2**30:raise RuntimeError(f'Paused to preserve disk headroom: {free/2**30:.1f} GiB available')


def freeze(c, root):
    # Resume the exact initial ID range after a crash; later invocations scan additions/updates.
    pg=psycopg2.connect(dbname='arxiv',application_name='voyage4_archive_snapshot',options='-c default_transaction_read_only=on')
    tokenizer=Tokenizer.from_pretrained(MODEL.replace('voyage-','voyageai/voyage-',1));tokenizer.no_truncation()
    previous=state(c,'snapshot_complete')
    scan=state(c,'scan')
    if not scan or scan.get('complete'):
        with pg.cursor() as q:
            q.execute('SELECT max(id) FROM public.papers_paper');maximum=q.fetchone()[0]
        scan=dict(max_id=maximum,last_id=0,started=dt.datetime.now(dt.timezone.utc).isoformat(),complete=False)
        state(c,'scan',scan)
    q=pg.cursor(name='voyage4_freeze');q.itersize=2048
    if previous:
        q.execute('''WITH todo AS MATERIALIZED (
          SELECT id FROM public.papers_paper WHERE id>%s AND id<=%s
          UNION SELECT paper_id FROM voyage4.pending_papers WHERE queued_at<=%s::timestamptz AND paper_id<=%s)
          SELECT p.id,p.abstract,p.created,p.categories FROM todo t JOIN public.papers_paper p ON p.id=t.id
          WHERE p.id>%s ORDER BY p.id''',
          (previous['max_id'],scan['max_id'],scan['started'],scan['max_id'],scan['last_id']))
    else:
        q.execute('SELECT id,abstract,created,categories FROM public.papers_paper WHERE id>%s AND id<=%s ORDER BY id',
                  (scan['last_id'],scan['max_id']))
    count=0
    while rows:=q.fetchmany(2048):
        check_space(root)
        lengths=[len(x) for x in tokenizer.encode_batch([r[1] for r in rows])]
        records=[]
        for (pid,text,created,categories),tokens in zip(rows,lengths):
            if not text.strip() or tokens>32000:raise ValueError(f'Invalid abstract for paper {pid}: {tokens} tokens')
            records.append((pid,text,hashlib.sha256(text.encode()).hexdigest(),tokens,created.isoformat(),json.dumps(categories)))
        c.executemany('''INSERT INTO papers VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
          abstract=excluded.abstract,sha=excluded.sha,tokens=excluded.tokens,created=excluded.created,categories=excluded.categories''',records)
        c.executemany('INSERT OR IGNORE INTO metadata_dirty VALUES(?)',[(r[0],) for r in records])
        scan['last_id']=rows[-1][0];state(c,'scan',scan);count+=len(rows)
        if count%102400==0:event('snapshot progress',scanned=count,last_id=scan['last_id'])
    q.close();pg.close();scan['complete']=True;state(c,'scan',scan);state(c,'snapshot_complete',scan)
    event('snapshot complete',papers=c.execute('SELECT count(*) FROM papers').fetchone()[0],tokens=c.execute('SELECT sum(tokens) FROM papers').fetchone()[0])
    return count


def register(c,path,origin):
    path=Path(path).resolve()
    old=c.execute('SELECT sha256 FROM shards WHERE path=?',(str(path),)).fetchone()
    if old:return 0
    sha=digest(path)
    with np.load(path,allow_pickle=False) as z:
        ids=z['paper_ids'];hashes=z['text_sha256'];vec=z['vectors'];m=json.loads(str(z['metadata']))
        validate(vec,len(ids));assert m['model']==MODEL and m['dimensions']==DIM and m['input_type'] is None
        assert len(set(map(int,ids)))==len(ids)
        before=c.total_changes
        c.executemany('INSERT OR IGNORE INTO versions VALUES(?,?,?,?)',[(int(pid),str(h),str(path),i) for i,(pid,h) in enumerate(zip(ids,hashes))])
        added=c.total_changes-before
        c.execute('INSERT INTO shards(path,sha256,papers,tokens,origin) VALUES(?,?,?,?,?)',
                  (str(path),sha,len(ids),m['usage']['total_tokens'],origin))
    c.commit();return added


def reuse(c,root):
    saved=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
    if state(c,'pilot_registered') and saved.get('status')=='complete' and saved.get('missing')==0:
        event('completed archive catalog reused',papers=saved['completed']);return
    if not state(c,'pilot_registered'):
        export=json.loads((PILOT/'export.json').read_text())
        for name,info in export['files'].items():
            if name in ['vectors.float32.npy','paper_ids.npy','cohort.sqlite3']:
                assert digest(PILOT/name)==info['sha256'],name
        for p in sorted((PILOT/'shards').glob('*.npz')):register(c,p,'pilot')
        state(c,'pilot_registered',True)
    for p in sorted((root/'shards').glob('*.npz')):register(c,p,'standard-api')
    event('saved outputs registered',versions=c.execute('SELECT count(*) FROM versions').fetchone()[0])


def plan(c):
    # Every job is persisted before API submission; unfinished files are recovered by reuse().
    c.execute("DELETE FROM jobs WHERE state='done'")
    assigned=set()
    for (s,) in c.execute("SELECT ids FROM jobs WHERE state='pending'"):assigned.update(json.loads(s))
    rows=c.execute('''SELECT p.id,p.sha,p.tokens FROM papers p LEFT JOIN versions v ON v.id=p.id AND v.sha=p.sha
      WHERE v.id IS NULL ORDER BY p.created DESC,p.id DESC''')
    batch=[];tokens=0;number=0
    def save(batch):
        key=hashlib.sha256(json.dumps([(r[0],r[1]) for r in batch]).encode()).hexdigest()
        c.execute('INSERT OR IGNORE INTO jobs(name,ids) VALUES(?,?)',(key,json.dumps([r[0] for r in batch])))
    for row in rows:
        if row[0] in assigned:continue
        if batch and (len(batch)>=256 or tokens+row[2]>90000):save(batch);number+=1;batch=[];tokens=0
        batch.append(row);tokens+=row[2]
    if batch:save(batch);number+=1
    c.commit();event('requests planned',new_requests=number)


def validate(vec,n):
    if vec.shape!=(n,DIM) or vec.dtype!=np.dtype('<f4'):raise ValueError('Wrong embedding shape or precision')
    if not np.isfinite(vec).all():raise ValueError('Nonfinite embedding')
    norms=np.linalg.norm(vec,axis=1)
    if not ((norms>.5)&(norms<1.5)).all():raise ValueError('Invalid embedding norm')


def decode(body,n):
    if body.get('model')!=MODEL:raise ValueError('Unexpected response model')
    data=body['data']
    if len(data)!=n or sorted(x['index'] for x in data)!=list(range(n)):raise ValueError('Invalid response indices')
    vec=np.empty((n,DIM),dtype='<f4')
    for x in data:
        raw=base64.b64decode(x['embedding'],validate=True)
        if len(raw)!=DIM*4:raise ValueError('Invalid float32 payload length')
        vec[x['index']]=np.frombuffer(raw,dtype='<f4')
    validate(vec,n);return vec


class Pacer:
    def __init__(self,tpm,rpm):self.tpm=tpm;self.rpm=rpm;self.next=0.;self.lock=threading.Lock()
    def acquire(self,tokens):
        with self.lock:
            time.sleep(max(0,self.next-time.monotonic()))
            self.next=time.monotonic()+max(60/self.rpm,tokens*1.03*60/self.tpm)


def generate_one(root,name,rows,client,pacer):
    target=root/'shards'/f'{name}.npz'
    if target.exists():return target
    tokens=sum(r[3] for r in rows)
    for attempt in range(10):
        check_space(root);pacer.acquire(tokens);start=time.monotonic()
        try:
            response=client.post(ENDPOINT,json=dict(model=MODEL,input=[r[1] for r in rows],input_type=None,
                         output_dimension=DIM,output_dtype='float',encoding_format='base64',truncation=False))
        except httpx.TransportError as e:
            event('transport retry',request=name,attempt=attempt+1,error=type(e).__name__);time.sleep(min(60,2**attempt));continue
        if response.status_code in [429,500,502,503,504]:
            event('API retry',request=name,status=response.status_code,attempt=attempt+1)
            try:delay=float(response.headers.get('retry-after','0'))
            except ValueError:delay=0
            time.sleep(min(120,max(delay,10,2**attempt)));continue
        if response.status_code!=200:raise RuntimeError(f'API HTTP {response.status_code}; request {name}')
        body=response.json();vec=decode(body,len(rows))
        meta=dict(model=body['model'],dimensions=DIM,input_type=None,output_dtype='float32',encoding_format='base64',
                  endpoint=ENDPOINT,usage=body['usage'],completed=dt.datetime.now(dt.timezone.utc).isoformat(),
                  attempts=attempt+1,latency_seconds=time.monotonic()-start,
                  response_headers={k:v for k,v in response.headers.items() if k.lower()=='x-request-id' or 'ratelimit' in k.lower()},
                  vector_payload_sha256=hashlib.sha256(vec.tobytes()).hexdigest())
        # Preserve the exact input snapshot and original decoded float32 bytes, without normalization.
        inputs=gzip.compress(json.dumps([dict(id=r[0],abstract=r[1],sha=r[2],created=r[4],categories=json.loads(r[5])) for r in rows],ensure_ascii=False).encode(),mtime=0)
        tmp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
        with tmp.open('xb') as f:
            np.savez(f,paper_ids=np.asarray([r[0] for r in rows],dtype='<i8'),text_sha256=np.asarray([r[2] for r in rows]),
                     vectors=vec,metadata=np.asarray(json.dumps(meta)),inputs_json_gzip=np.frombuffer(inputs,dtype=np.uint8))
            f.flush();os.fsync(f.fileno())
        # Hard-link publish never overwrites an existing saved response.
        os.link(tmp,target);tmp.unlink();os.chmod(target,0o444)
        fd=os.open(target.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
        return target
    raise RuntimeError(f'Retry budget exhausted for {name}')


def progress(c,root,status):
    global _progress_cache
    if _progress_cache is None:
        total=c.execute('SELECT count(*),sum(tokens) FROM papers').fetchone()
        complete=c.execute('SELECT count(*) FROM papers p JOIN versions v ON v.id=p.id AND v.sha=p.sha').fetchone()[0]
        _progress_cache=dict(total=total,complete=complete)
    total=_progress_cache['total'];complete=_progress_cache['complete']
    usage=c.execute("SELECT coalesce(sum(tokens),0) FROM shards WHERE origin='standard-api'").fetchone()[0]
    result=dict(model=MODEL,dimensions=DIM,dtype='float32',status=status,total=total[0],completed=complete,
                missing=total[0]-complete,new_response_tokens=usage,new_list_price_usd=usage/1e6*.12,
                updated=dt.datetime.now(dt.timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free)
    atomic_json(root/'progress.json',result);event('generation progress',**result);return result


def generate(c,root,tpm,rpm,workers):
    key=dotenv_values('/home/arxiv/arxiv_troller/.env').get('VOYAGE_API_KEY')
    if not key:raise ValueError('Missing VOYAGE_API_KEY')
    pacer=Pacer(tpm,rpm)
    jobs=c.execute("SELECT name,ids FROM jobs WHERE state='pending' ORDER BY rowid").fetchall()
    def records(ids):
        rows={r[0]:r for r in c.execute('SELECT * FROM papers WHERE id IN ('+','.join('?'*len(ids))+')',ids)}
        return [rows[i] for i in ids]
    progress(c,root,'generating');it=iter(jobs);done=0
    with httpx.Client(headers={'Authorization':f'Bearer {key}'},timeout=120,limits=httpx.Limits(max_connections=workers,max_keepalive_connections=workers)) as client, ThreadPoolExecutor(max_workers=workers) as pool:
        pending={}
        def submit():
            try:name,s=next(it)
            except StopIteration:return False
            ids=json.loads(s)
            if all(c.execute('SELECT 1 FROM papers p JOIN versions v ON v.id=p.id AND v.sha=p.sha WHERE p.id=?',(i,)).fetchone() for i in ids):
                c.execute("UPDATE jobs SET state='done' WHERE name=?",(name,));c.commit();return True
            pending[pool.submit(generate_one,root,name,records(ids),client,pacer)]=name;return True
        for _ in range(workers*2):submit()
        while pending:
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in finished:
                name=pending.pop(f);path=f.result();added=register(c,path,'standard-api')
                _progress_cache['complete']+=added
                c.execute("UPDATE jobs SET state='done' WHERE name=?",(name,));c.commit();done+=1
                if done%20==0:progress(c,root,'generating')
                submit()
            while len(pending)<workers*2 and submit():pass
    result=progress(c,root,'complete')
    if result['missing']:raise RuntimeError('Archive incomplete')


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['snapshot','generate','all','status'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--tpm',type=int,default=14_000_000)
    p.add_argument('--rpm',type=int,default=3000);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True);(a.root/'shards').mkdir(exist_ok=True)
    with (a.root/'archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=catalog(a.root)
        if a.phase=='status':progress(c,a.root,'inspection');return
        changed=freeze(c,a.root) if a.phase in ['snapshot','all'] else None
        if a.phase=='all' and changed==0 and (a.root/'progress.json').exists():
            prior=json.loads((a.root/'progress.json').read_text())
            dirty=c.execute('SELECT 1 FROM metadata_dirty LIMIT 1').fetchone()
            pending=c.execute("SELECT 1 FROM jobs WHERE state='pending' LIMIT 1").fetchone()
            if prior.get('status')=='complete' and prior.get('missing')==0 and not dirty and not pending:
                event('no paper changes; saved archive remains complete',papers=prior['completed']);c.close();return
        if a.phase in ['generate','all']:
            reuse(c,a.root);plan(c);generate(c,a.root,a.tpm,a.rpm,a.workers)
        c.close()

if __name__=='__main__':main()
