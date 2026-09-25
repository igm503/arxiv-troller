"""Import saved originals into derived search tables and maintain the rolling graph."""
import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import archive as a

EPOCH=dt.datetime(2000,1,1,tzinfo=dt.timezone.utc)
COLUMNS='paper_id,abstract_sha,created,categories,vector,bits,archive_path,archive_row,vector_sha'


def connect():
    c=psycopg2.connect(dbname='arxiv',application_name='voyage4_storage')
    with c.cursor() as q:
        q.execute('SET ROLE voyage4_writer')
        q.execute("SET statement_timeout='30min'; SET lock_timeout='10s'; SET timezone='UTC'")
    c.commit();return c


def timestamp(s):return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)


def field(raw):return struct.pack('!i',len(raw))+raw


def varchar_array(items):
    return struct.pack('!iiiii',1,0,1043,len(items),1)+b''.join(field(x.encode()) for x in items)


def copy_rows(rows,vec,path,indices,full=False):
    out=io.BytesIO();out.write(b'PGCOPY\n\xff\r\n\x00'+struct.pack('!ii',0,0))
    for row,v,idx in zip(rows,vec,indices):
        pid,abstract,sha,tokens,created,categories=row
        fields=[struct.pack('!q',pid),sha.encode(),struct.pack('!q',round((timestamp(created)-EPOCH).total_seconds()*1e6)),
                varchar_array(json.loads(categories)),struct.pack('!HH',2048,0)+v.astype('>f2').tobytes()]
        if full:
            fields.append(struct.pack('!HH',2048,0)+v.astype('>f4').tobytes())
        else:
            fields += [struct.pack('!i',2048)+np.packbits(v>0).tobytes(),str(path).encode(),struct.pack('!i',idx),hashlib.sha256(v.astype('<f4').tobytes()).hexdigest().encode()]
        out.write(struct.pack('!h',len(fields)))
        for f in fields:out.write(field(f))
    out.write(struct.pack('!h',-1));out.seek(0);return out


def put_rows(pg,rows,vectors,path,indices):
    with pg.cursor() as q:
        q.execute('CREATE TEMP TABLE IF NOT EXISTS v4_import_stage (LIKE voyage4.embeddings INCLUDING DEFAULTS) ON COMMIT DELETE ROWS')
        q.copy_expert(f'COPY v4_import_stage({COLUMNS}) FROM STDIN WITH(FORMAT BINARY)',copy_rows(rows,vectors,path,indices))
        q.execute(f'''INSERT INTO voyage4.embeddings({COLUMNS}) SELECT {COLUMNS} FROM v4_import_stage
          ON CONFLICT(paper_id) DO UPDATE SET abstract_sha=excluded.abstract_sha,created=excluded.created,
          categories=excluded.categories,vector=excluded.vector,bits=excluded.bits,archive_path=excluded.archive_path,
          archive_row=excluded.archive_row,vector_sha=excluded.vector_sha,saved_at=now()
          WHERE voyage4.embeddings.abstract_sha<>excluded.abstract_sha OR
          (voyage4.embeddings.created,voyage4.embeddings.categories) IS DISTINCT FROM (excluded.created,excluded.categories)''')
    pg.commit()


def import_shards(c,pg,root):
    count=0
    # Snapshot the shard list; a later invocation imports newly generated files.
    paths=c.execute('SELECT path,sha256 FROM shards WHERE imported=0 ORDER BY rowid').fetchall()
    for path,sha in paths:
        a.check_space(root)
        if a.digest(path)!=sha:raise RuntimeError(f'Archive hash mismatch: {path}')
        with np.load(path,allow_pickle=False) as z:
            ids=z['paper_ids'];hashes=z['text_sha256'];vec=z['vectors'];a.validate(vec,len(ids))
            entries={r[0]:r for r in c.execute('SELECT * FROM papers WHERE id IN ('+','.join('?'*len(ids))+')',list(map(int,ids)))}
            keep=[i for i,pid in enumerate(ids) if int(pid) in entries and entries[int(pid)][2]==str(hashes[i])]
            rows=[entries[int(ids[i])] for i in keep]
            if rows:
                put_rows(pg,rows,vec[keep],path,keep)
        c.executemany('DELETE FROM metadata_dirty WHERE id=?',[(r[0],) for r in rows])
        c.execute('UPDATE shards SET imported=1 WHERE path=?',(path,));c.commit();count+=len(rows)
        if count and count%25600==0:a.event('search import progress',papers=count)
    changed_metadata=sync_metadata(c,pg)
    if count or changed_metadata:
        with pg.cursor() as q:q.execute('ANALYZE voyage4.embeddings')
    pg.commit();a.event('search import complete',papers=count,shards=len(paths))


def sync_metadata(c,pg):
    if not c.execute('SELECT 1 FROM metadata_dirty LIMIT 1').fetchone():return 0
    dirty=c.execute('SELECT p.* FROM papers p JOIN metadata_dirty d ON d.id=p.id JOIN versions v ON v.id=p.id AND v.sha=p.sha').fetchall()
    for off in range(0,len(dirty),1024):
        rows=dirty[off:off+1024]
        with pg.cursor() as q:
            q.execute('SELECT paper_id,abstract_sha FROM voyage4.embeddings WHERE paper_id=ANY(%s)',([r[0] for r in rows],))
            existing=dict(q.fetchall())
        groups={}
        for r in rows:
            if existing.get(r[0])!=r[2]:
                path,index=c.execute('SELECT shard,row_index FROM versions WHERE id=? AND sha=?',(r[0],r[2])).fetchone()
                groups.setdefault(path,[]).append((r,index))
        for path,items in groups.items():
            with np.load(path,allow_pickle=False) as z:
                put_rows(pg,[r for r,i in items],z['vectors'][[i for r,i in items]],path,[i for r,i in items])
        with pg.cursor() as q:
            execute_values(q,'''UPDATE voyage4.embeddings e SET created=x.created::timestamptz,categories=x.categories::varchar[]
              FROM (VALUES %s) AS x(id,sha,created,categories) WHERE e.paper_id=x.id AND e.abstract_sha=x.sha
              AND (e.created,e.categories) IS DISTINCT FROM (x.created::timestamptz,x.categories::varchar[])''',
              [(r[0],r[2],r[4],json.loads(r[5])) for r in rows])
            execute_values(q,'''UPDATE voyage4.rolling30 e SET created=x.created::timestamptz,categories=x.categories::varchar[]
              FROM (VALUES %s) AS x(id,sha,created,categories) WHERE e.paper_id=x.id AND e.abstract_sha=x.sha
              AND (e.created,e.categories) IS DISTINCT FROM (x.created::timestamptz,x.categories::varchar[])''',
              [(r[0],r[2],r[4],json.loads(r[5])) for r in rows])
        pg.commit();c.executemany('DELETE FROM metadata_dirty WHERE id=?',[(r[0],) for r in rows]);c.commit()
    return len(dirty)


def acknowledge(c,pg):
    scan=a.state(c,'snapshot_complete')
    with pg.cursor() as q:q.execute('DELETE FROM voyage4.pending_papers WHERE queued_at<=%s',(scan['started'],))
    pg.commit()


def rolling(c,pg,root,now=None):
    now=now or dt.datetime.now(dt.timezone.utc)
    cutoff=now-dt.timedelta(days=30)
    # One extra day provides a safe routing boundary during maintenance. Query predicates enforce the exact date.
    floor=cutoff-dt.timedelta(days=1)
    with pg.cursor() as q:
        q.execute('''SELECT e.paper_id,e.archive_path,e.archive_row,e.abstract_sha FROM voyage4.embeddings e
          LEFT JOIN voyage4.rolling30 r USING(paper_id)
          WHERE e.created >= %s AND (r.paper_id IS NULL OR r.abstract_sha<>e.abstract_sha) ORDER BY e.archive_path''',(floor,))
        pending=q.fetchall()
    groups={}
    for pid,path,index,sha in pending:groups.setdefault(path,[]).append((pid,index,sha))
    for path,items in groups.items():
        a.check_space(root)
        ids=[x[0] for x in items]
        entries={r[0]:r for r in c.execute('SELECT * FROM papers WHERE id IN ('+','.join('?'*len(ids))+')',ids)}
        with np.load(path,allow_pickle=False) as z:
            vec=z['vectors'];rows=[entries[pid] for pid,_,_ in items];indices=[i for _,i,_ in items]
            assert all(r[2]==sha for r,(_,_,sha) in zip(rows,items))
            with pg.cursor() as q:
                q.execute('CREATE TEMP TABLE IF NOT EXISTS v4_rolling_stage (LIKE voyage4.rolling30 INCLUDING DEFAULTS) ON COMMIT DELETE ROWS')
                q.copy_expert('COPY v4_rolling_stage FROM STDIN WITH(FORMAT BINARY)',copy_rows(rows,vec[indices],path,indices,full=True))
                q.execute('''INSERT INTO voyage4.rolling30 SELECT * FROM v4_rolling_stage ON CONFLICT(paper_id)
                  DO UPDATE SET abstract_sha=excluded.abstract_sha,created=excluded.created,categories=excluded.categories,
                  vector=excluded.vector,full_vector=excluded.full_vector''')
            pg.commit()
    with pg.cursor() as q:
        # Expiry touches only this derived cache, never the archive or permanent embeddings.
        q.execute('DELETE FROM voyage4.rolling30 WHERE created < %s',(floor,));expired=q.rowcount
        q.execute("INSERT INTO voyage4.control(key,value) VALUES('rolling',%s::jsonb) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=now()",
                  (json.dumps(dict(floor=floor.isoformat(),maintained_at=now.isoformat())),))
        q.execute('ANALYZE voyage4.rolling30')
    pg.commit();pg.autocommit=True
    with pg.cursor() as q:q.execute('VACUUM voyage4.rolling30')
    pg.autocommit=False
    a.event('rolling membership maintained',added_or_revised=len(pending),expired=expired,floor=floor.isoformat())


def indexes(pg,include_general=True):
    pg.commit();pg.autocommit=True
    with pg.cursor() as q:
        q.execute("SET maintenance_work_mem='4GB'; SET max_parallel_maintenance_workers=2; SET statement_timeout='4h'")
        for name,table,column,ops,m,ef in [
            ('v4_rolling30_hnsw','rolling30','vector','halfvec_l2_ops',16,64),
            ('v4_bits_hnsw','embeddings','bits','bit_hamming_ops',32,256)]:
            if not include_general and table=='embeddings':continue
            q.execute('SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(%s)',('voyage4.'+name,));r=q.fetchone()
            if r and r[0]:continue
            if r:raise RuntimeError(f'Incomplete index {name}; inspect before retry')
            a.event('index build started',index=name)
            q.execute(f'CREATE INDEX {name} ON voyage4.{table} USING hnsw ({column} {ops}) WITH(m={m},ef_construction={ef})')
            a.event('index build complete',index=name)
    pg.autocommit=False


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['import','rolling','rolling-index','indexes','acknowledge','all']);p.add_argument('--root',type=Path,default=a.ROOT)
    args=p.parse_args()
    with (args.root/'storage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=a.catalog(args.root);pg=connect()
        if args.phase in ['import','all']:import_shards(c,pg,args.root)
        if args.phase in ['rolling','all']:rolling(c,pg,args.root)
        if args.phase in ['indexes','all','rolling-index']:indexes(pg,include_general=args.phase!='rolling-index')
        if args.phase=='acknowledge':acknowledge(c,pg)
        pg.close();c.close()

if __name__=='__main__':main()
