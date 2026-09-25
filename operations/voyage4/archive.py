"""Resumable standard-API archive. Original response vectors are immutable float32 shards."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import fcntl
import gzip
import hashlib
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
from dotenv import load_dotenv
from tokenizers import Tokenizer

MODEL='voyage-4-large'
DIM=2048
ENDPOINT='https://api.voyageai.com/v1/embeddings'
PROJECT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT / '.env')
ROOT = Path(os.getenv('VOYAGE4_ROOT', PROJECT / 'data/voyage4'))

def database(readonly=True, **kwargs):
    return psycopg2.connect(dbname=os.getenv('PGDATABASE', 'arxiv'),
                           options='-c default_transaction_read_only=' + ('on' if readonly else 'off'), **kwargs)

def require(condition, message):
    if not condition:
        raise ValueError(message)


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
      CREATE TABLE IF NOT EXISTS failures(id INTEGER PRIMARY KEY,reason TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS receipts(id INTEGER PRIMARY KEY,queued_at TEXT NOT NULL);
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
    pg=database(application_name='voyage4_archive_snapshot')
    tokenizer=Tokenizer.from_pretrained(MODEL.replace('voyage-','voyageai/voyage-',1));tokenizer.no_truncation()
    previous=state(c,'snapshot_complete')
    scan=state(c,'scan')
    if not scan or scan.get('complete'):
        with pg.cursor() as q:
            q.execute('SELECT max(id) FROM public.papers_paper');maximum=q.fetchone()[0] or 0
        scan=dict(max_id=maximum,last_id=0,started=dt.datetime.now(dt.timezone.utc).isoformat(),complete=False)
        state(c,'scan',scan)
    q=pg.cursor(name='voyage4_freeze');q.itersize=2048
    if previous:
        q.execute('''WITH todo AS MATERIALIZED (
          SELECT id FROM public.papers_paper WHERE id>%s AND id<=%s
          UNION SELECT paper_id FROM voyage4.pending_papers WHERE queued_at<=%s::timestamptz AND paper_id<=%s
          UNION SELECT id FROM public.papers_paper WHERE id=ANY(%s))
          SELECT p.id,p.abstract,p.created,p.categories,pending.queued_at FROM todo t JOIN public.papers_paper p ON p.id=t.id
          LEFT JOIN voyage4.pending_papers pending ON pending.paper_id=p.id
          WHERE p.id>%s ORDER BY p.id''',
          (previous['max_id'],scan['max_id'],scan['started'],scan['max_id'],[r[0] for r in c.execute('SELECT id FROM failures')],scan['last_id']))
    else:
        q.execute('''SELECT p.id,p.abstract,p.created,p.categories,pending.queued_at FROM public.papers_paper p
            LEFT JOIN voyage4.pending_papers pending ON pending.paper_id=p.id WHERE p.id>%s AND p.id<=%s ORDER BY p.id''',
                  (scan['last_id'],scan['max_id']))
    count=0
    while rows:=q.fetchmany(2048):
        check_space(root)
        lengths=[len(x) for x in tokenizer.encode_batch([r[1] for r in rows])]
        records=[]
        for (pid,text,created,categories,queued_at),tokens in zip(rows,lengths):
            if queued_at is not None:
                c.execute('INSERT OR REPLACE INTO receipts VALUES(?,?)', (pid, queued_at.isoformat()))
            if not text.strip() or tokens > 32000:
                reason = f'Invalid abstract: {tokens} tokens'
                c.execute('INSERT OR REPLACE INTO failures VALUES(?,?)', (pid, reason))
                event('paper quarantined', paper_id=pid, reason=reason)
            else:
                c.execute('DELETE FROM failures WHERE id=?', (pid,))
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
        validate(vec,len(ids))
        require(m['model']==MODEL and m['dimensions']==DIM and m['input_type'] is None, 'Wrong archive model')
        require(len(ids)==len(hashes)==len(set(map(int,ids))), 'Invalid archive IDs or text hashes')
        before=c.total_changes
        c.executemany('INSERT OR IGNORE INTO versions VALUES(?,?,?,?)',[(int(pid),str(h),str(path),i) for i,(pid,h) in enumerate(zip(ids,hashes))])
        added=c.total_changes-before
        c.execute('INSERT INTO shards(path,sha256,papers,tokens,origin) VALUES(?,?,?,?,?)',
                  (str(path),sha,len(ids),m['usage']['total_tokens'],origin))
    c.commit();return added


def reuse(c, root, pilot=None):
    # Recover published files even when the process died before catalog registration.
    if pilot is not None:
        export = json.loads((pilot / 'export.json').read_text())
        for name in ['vectors.float32.npy', 'paper_ids.npy', 'cohort.sqlite3']:
            require(digest(pilot / name) == export['files'][name]['sha256'], f'Pilot checksum: {name}')
        for path in sorted((pilot / 'shards').glob('*.npz')):
            register(c, path, 'pilot')
    for path in sorted((root / 'shards').glob('*.npz')):
        register(c, path, 'standard-api')


def plan(c):
    """Derive deterministic requests from missing versions; no second job queue."""
    rows = c.execute('''SELECT * FROM papers WHERE id IN (
        SELECT p.id FROM papers p LEFT JOIN versions v ON v.id=p.id AND v.sha=p.sha
        LEFT JOIN failures f ON f.id=p.id WHERE v.id IS NULL AND f.id IS NULL)
        ORDER BY created DESC,id DESC''')
    batch, tokens = [], 0
    for row in rows:
        if batch and (len(batch) >= 256 or tokens + row[3] > 90000):
            yield batch
            batch, tokens = [], 0
        batch.append(row)
        tokens += row[3]
    if batch:
        yield batch


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


def progress(c, root):
    total = c.execute('SELECT count(*) FROM papers').fetchone()[0]
    complete = c.execute('SELECT count(*) FROM papers p JOIN versions v ON v.id=p.id AND v.sha=p.sha').fetchone()[0]
    failed = c.execute('SELECT count(*) FROM failures').fetchone()[0]
    failed_missing = c.execute('''SELECT count(*) FROM failures f JOIN papers p ON p.id=f.id
        LEFT JOIN versions v ON v.id=p.id AND v.sha=p.sha WHERE v.id IS NULL''').fetchone()[0]
    missing = total - complete - failed_missing
    usage = c.execute("SELECT coalesce(sum(tokens),0) FROM shards WHERE origin='standard-api'").fetchone()[0]
    return dict(model=MODEL, dimensions=DIM, dtype='float32', total=total, completed=complete,
                missing=missing, failed=failed, status='incomplete' if missing else 'degraded' if failed else 'complete',
                new_response_tokens=usage, new_list_price_usd=usage / 1e6 * .12,
                updated=dt.datetime.now(dt.timezone.utc).isoformat(), free_bytes=shutil.disk_usage(root).free)


def generate(c, root, tpm, rpm, workers):
    jobs = iter(plan(c))
    first = next(jobs, None)
    if first is None:
        return
    key = os.getenv('VOYAGE_API_KEY')
    require(key, 'Missing VOYAGE_API_KEY')
    pacer = Pacer(tpm, rpm)
    with httpx.Client(headers={'Authorization': f'Bearer {key}'}, timeout=120) as client, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        batch = first
        while batch is not None or pending:
            while batch is not None and len(pending) < workers:
                name = hashlib.sha256(json.dumps([(r[0], r[2]) for r in batch]).encode()).hexdigest()
                pending.add(pool.submit(generate_one, root, name, batch, client, pacer))
                batch = next(jobs, None)
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                register(c, future.result(), 'standard-api')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['snapshot', 'generate', 'all', 'status'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--pilot', type=Path, help='Explicitly import a preserved pilot archive once')
    parser.add_argument('--tpm', type=int, default=14_000_000)
    parser.add_argument('--rpm', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    require(min(args.tpm, args.rpm, args.workers) > 0, 'Rate limits and workers must be positive')
    (args.root / 'shards').mkdir(parents=True, exist_ok=True)
    with (args.root / 'archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = catalog(args.root)
        if args.phase in ['snapshot', 'all']:
            changed = freeze(c, args.root)
            saved = args.root / 'progress.json'
            if args.phase == 'all' and changed == 0 and not c.execute('SELECT 1 FROM metadata_dirty LIMIT 1').fetchone():
                if saved.exists() and json.loads(saved.read_text()).get('status') == 'complete':
                    event('no paper changes; saved archive remains complete')
                    c.close()
                    return
        if args.phase in ['generate', 'all']:
            reuse(c, args.root, args.pilot)
            generate(c, args.root, args.tpm, args.rpm, args.workers)
        result = progress(c, args.root)
        if args.phase in ['generate', 'all']:
            require(result['missing'] == 0, 'Archive incomplete')
        if args.phase != 'status':
            atomic_json(args.root / 'progress.json', result)
        event('archive status', **result)
        c.close()


if __name__ == '__main__':
    main()
