"""Audit immutable full-precision archives and their permanent database coverage."""
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import psycopg2
import archive as a


def run(root):
    c=a.catalog(root);pg=psycopg2.connect(dbname='arxiv',options='-c default_transaction_read_only=on')
    shards=c.execute('SELECT path,sha256,papers,tokens,origin FROM shards ORDER BY path').fetchall()
    vectors=0;response_tokens=0;new_tokens=0;bytes_saved=0;samples=[]
    rng=random.Random(20260921)
    for number,(path,sha,n,tokens,origin) in enumerate(shards,1):
        assert a.digest(path)==sha,f'File checksum mismatch: {path}'
        with np.load(path,allow_pickle=False) as z:
            v=z['vectors'];ids=z['paper_ids'];hashes=z['text_sha256'];m=json.loads(str(z['metadata']))
            a.validate(v,n)
            assert len(ids)==n and len(hashes)==n
            assert m['model']==a.MODEL and m['dimensions']==a.DIM and m['input_type'] is None
            if 'vector_payload_sha256' in m:assert hashlib.sha256(v.tobytes()).hexdigest()==m['vector_payload_sha256']
            if 'inputs_json_gzip' in z:
                inputs=json.loads(gzip.decompress(z['inputs_json_gzip'].tobytes()))
                assert len(inputs)==n
                for pid,h,r in zip(ids,hashes,inputs):
                    assert int(pid)==r['id'] and str(h)==r['sha']==hashlib.sha256(r['abstract'].encode()).hexdigest()
            # Check an independently selected original against the actual database search copy.
            if number%100==0 or number==1:
                index=rng.randrange(n);pid=int(ids[index])
                with pg.cursor() as q:
                    q.execute('SELECT abstract_sha,vector_sha,archive_path,archive_row,vector::text FROM voyage4.embeddings WHERE paper_id=%s',(pid,))
                    row=q.fetchone()
                if row and row[0]==str(hashes[index]) and row[2]==path:
                    assert row[1]==hashlib.sha256(v[index].tobytes()).hexdigest() and row[3]==index
                    np.testing.assert_array_equal(np.fromstring(row[4].strip('[]'),sep=',',dtype='<f4'),v[index].astype('f2').astype('f4'))
                    samples.append(pid)
        vectors+=n;response_tokens+=tokens;new_tokens+=tokens if origin=='standard-api' else 0;bytes_saved+=Path(path).stat().st_size
        if number%500==0:a.event('original archive audit progress',shards=number,total=len(shards))
    total=c.execute('SELECT count(*) FROM papers').fetchone()[0]
    missing=c.execute('SELECT count(*) FROM papers p LEFT JOIN versions v ON v.id=p.id AND v.sha=p.sha WHERE v.id IS NULL').fetchone()[0]
    unimported=c.execute('SELECT count(*) FROM shards WHERE imported=0').fetchone()[0]
    assert missing==0 and unimported==0
    with pg.cursor() as q:
        q.execute('SELECT count(*) FROM voyage4.embeddings');db_count=q.fetchone()[0];assert db_count==total
        q.execute('SELECT count(*) FROM public.papers_paper p LEFT JOIN voyage4.embeddings e ON e.paper_id=p.id WHERE e.paper_id IS NULL');db_missing=q.fetchone()[0];assert db_missing==0
        q.execute('SELECT count(*) FROM voyage4.rolling30');rolling_count=q.fetchone()[0]
        q.execute('SELECT count(*) FROM voyage4.embeddings e LEFT JOIN voyage4.rolling30 r USING(paper_id) WHERE e.created>=now()-interval \'30 days\' AND (r.paper_id IS NULL OR r.abstract_sha<>e.abstract_sha)');assert q.fetchone()[0]==0
        q.execute("SELECT i.relname,x.indisvalid,pg_relation_size(i.oid) FROM pg_class i JOIN pg_index x ON x.indexrelid=i.oid JOIN pg_namespace n ON n.oid=i.relnamespace WHERE n.nspname='voyage4'");indexes=q.fetchall();assert all(x[1] for x in indexes)
    result=dict(passed=True,verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),model=a.MODEL,dimensions=a.DIM,dtype='float32',
                current_papers=total,archived_versions=vectors,shards=len(shards),original_shard_bytes=bytes_saved,
                response_tokens=response_tokens,new_response_tokens=new_tokens,new_list_price_usd=new_tokens/1e6*.12,
                missing_originals=missing,missing_database_papers=db_missing,unimported_shards=unimported,
                rolling_papers=rolling_count,search_copy_samples=samples,indexes=indexes)
    a.atomic_json(root/'archive-audit.json',result);a.event('archive audit passed',**result);c.close();pg.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=a.ROOT);run(p.parse_args().root)
