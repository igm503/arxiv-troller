"""Measure deployed query shapes against exact original-float32 cosine before activation."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import psycopg2
from django.conf import settings
if not settings.configured:settings.configure()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'django'))
from papers.voyage4_search import query_sql,shard_vectors
import archive as a


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=a.ROOT);p.add_argument('--weekly-ef',type=int,default=128);args=p.parse_args()
    pg=psycopg2.connect(dbname='arxiv',options='-c default_transaction_read_only=on -c statement_timeout=120000')
    now=dt.datetime.now(dt.timezone.utc)
    seeds=json.loads((a.PILOT/'reference_metadata.json').read_text())['query_seeds']
    with pg.cursor() as q:
        q.execute('SELECT paper_id,created,categories,vector_send(full_vector) FROM voyage4.rolling30 ORDER BY paper_id')
        rows=q.fetchall()
        ids=np.asarray([r[0] for r in rows],dtype=np.int64)
        dates=[r[1] for r in rows];cats=[r[2] for r in rows]
        matrix=np.stack([np.frombuffer(r[3],dtype='>f4',offset=4).astype('f8') for r in rows]);del rows
        matrix/=np.linalg.norm(matrix,axis=1)[:,None]
        queries=[]
        for seed in seeds:
            q.execute('SELECT archive_path,archive_row FROM voyage4.embeddings WHERE paper_id=%s',(seed['paper_id'],));path,index=q.fetchone()
            queries.append(shard_vectors(path)[index].copy())
        qmat=np.asarray(queries,dtype='f8');qmat/=np.linalg.norm(qmat,axis=1)[:,None]
        sims=matrix@qmat.T;del matrix
        runs=[];summaries=[]
        for days in [7,30]:
            for category in ['', 'cs.LG']:
                scope=f'{days}d'+('_field' if category else '');cutoff=now-dt.timedelta(days=days)
                eligible=np.asarray([d>=cutoff and (not category or category in c) for d,c in zip(dates,cats)])
                for j,(seed,v) in enumerate(zip(seeds,queries)):
                    mask=eligible & (ids!=seed['paper_id']);subset=np.flatnonzero(mask)
                    order=np.lexsort((ids[subset],-sims[subset,j]))[:20];truth=set(map(int,ids[subset[order]]))
                    vector='['+','.join(map(str,v.tolist()))+']';bits=''.join('1' if x else '0' for x in v>0)
                    sql,params=query_sql(rolling=True,cutoff=cutoff,category=category,excluded={seed['paper_id']},limit=20,query_vector=vector,bits=bits)
                    ef=512 if category else args.weekly_ef if days==7 else 128
                    q.execute("SET LOCAL hnsw.iterative_scan='relaxed_order'; SET LOCAL hnsw.max_scan_tuples=20000; SET LOCAL hnsw.ef_search="+str(ef))
                    q.execute(sql,params);q.fetchall()
                    for repeat in range(2):
                        start=time.perf_counter();q.execute(sql,params);found=[x[0] for x in q.fetchall()];lat=(time.perf_counter()-start)*1000
                        assert set(found).issubset(set(map(int,ids[subset]))) and len(set(found))==len(found)
                        recall=len(set(found)&truth)/len(truth) if truth else 1.
                        runs.append(dict(scope=scope,source=seed['paper_id'],repeat=repeat,recall=recall,ms=lat,returned=len(found)))
                group=[r for r in runs if r['scope']==scope]
                result=dict(scope=scope,queries=len(seeds),candidates=int(eligible.sum()),recall=float(np.mean([r['recall'] for r in group])),median_ms=float(np.median([r['ms'] for r in group])),p95_ms=float(np.percentile([r['ms'] for r in group],95)),underfilled=sum(r['returned']<20 for r in group))
                summaries.append(result);a.event('rolling retrieval validation',**result)
    pg.close();a.atomic_json(args.root/f'retrieval-validation-weekly{args.weekly_ef}.json',dict(reference_utc=now.isoformat(),reference='Exhaustive float64 cosine arithmetic on saved original float32 values',summaries=summaries,runs=runs))

if __name__=='__main__':main()
