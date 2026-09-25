"""Full-archive reference for a stratified 12-source check of non-monthly retrieval."""
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

root=a.ROOT;c=a.catalog(root);pg=psycopg2.connect(dbname='arxiv',options='-c default_transaction_read_only=on -c statement_timeout=120000')
now=dt.datetime.now(dt.timezone.utc)
seeds=json.loads((a.PILOT/'reference_metadata.json').read_text())['query_seeds']
selected=[]
for group in dict.fromkeys(s['group'] for s in seeds):selected.extend([s for s in seeds if s['group']==group][:4])
seeds=selected;query_ids=np.asarray([s['paper_id'] for s in seeds]);queries=[]
with pg.cursor() as q:
    for pid in query_ids:
        q.execute('SELECT archive_path,archive_row FROM voyage4.embeddings WHERE paper_id=%s',(int(pid),));path,index=q.fetchone();queries.append(shard_vectors(path)[index].copy())
qmat=np.asarray(queries,dtype='f8');qmat/=np.linalg.norm(qmat,axis=1)[:,None]
scopes=[dict(name=(str(days)+'d' if days else 'all')+('_field' if field else ''),days=days,field=field,
             cutoff=now-dt.timedelta(days=days) if days else None) for days in [90,180,365,None] for field in ['', 'cs.LG']]
reference=root/'general-reference.json'
if reference.exists():
    saved=json.loads(reference.read_text());now=dt.datetime.fromisoformat(saved['reference_utc'])
    for scope in scopes:scope['cutoff']=now-dt.timedelta(days=scope['days']) if scope['days'] else None
    truth=saved['truth']
else:
    best={s['name']:(np.full((20,len(seeds)),-np.inf),np.full((20,len(seeds)),np.iinfo('i8').max,dtype='i8')) for s in scopes}
    a.event('loading compact reference metadata')
    metadata={r[0]:r for r in c.execute('SELECT id,sha,created,categories FROM papers')}
    a.event('compact reference metadata loaded',papers=len(metadata))
    paths=c.execute('SELECT path FROM shards ORDER BY path').fetchall()
    for number,(path,) in enumerate(paths,1):
        with np.load(path,allow_pickle=False) as z:
            ids=z['paper_ids'];hashes=z['text_sha256'];vectors=z['vectors'].astype('f8')
        rows={int(pid):metadata[int(pid)] for pid in ids if int(pid) in metadata}
        keep=np.asarray([i for i,pid in enumerate(ids) if int(pid) in rows and rows[int(pid)][1]==str(hashes[i])])
        if not len(keep):continue
        ids=ids[keep];vectors=vectors[keep];vectors/=np.linalg.norm(vectors,axis=1)[:,None];scores=vectors@qmat.T
        scores[ids[:,None]==query_ids[None,:]]=-np.inf
        dates=[dt.datetime.fromisoformat(rows[int(pid)][2]) for pid in ids];fields=np.asarray(['cs.LG' in json.loads(rows[int(pid)][3]) for pid in ids])
        for scope in scopes:
            mask=np.asarray([not scope['cutoff'] or date>=scope['cutoff'] for date in dates])
            if scope['field']:mask &= fields
            if not mask.any():continue
            old_scores,old_ids=best[scope['name']]
            values=np.concatenate([old_scores,scores[mask]],axis=0)
            candidates=np.concatenate([old_ids,np.broadcast_to(ids[mask,None],(int(mask.sum()),len(seeds)))],axis=0)
            order=np.lexsort((candidates,-values),axis=0)[:20]
            best[scope['name']]=(np.take_along_axis(values,order,axis=0),np.take_along_axis(candidates,order,axis=0))
        if number%500==0:a.event('full archive reference progress',shards=number,total=len(paths))
    truth={scope:{str(pid):ids[:,j].tolist() for j,pid in enumerate(query_ids)} for scope,(scores,ids) in best.items()}
    a.atomic_json(reference,dict(reference_utc=now.isoformat(),sources=seeds,truth=truth))
c.close();runs=[];summary=[]
with pg.cursor() as q:
    for scope in scopes:
        for effort,count in ([(256,500)] if scope['field'] else [(256,200),(1000,500)]):
            name=scope['name']+f'_ef{effort}_k{count}'
            for seed,v in zip(seeds,queries):
                vector='['+','.join(map(str,v.tolist()))+']';bits=''.join('1' if x else '0' for x in v>0)
                sql,params=query_sql(rolling=False,cutoff=scope['cutoff'],category=scope['field'],excluded={seed['paper_id']},limit=20,query_vector=vector,bits=bits)
                params[-3]=count
                q.execute("SET LOCAL hnsw.iterative_scan='relaxed_order';SET LOCAL hnsw.max_scan_tuples=20000;SET LOCAL hnsw.ef_search="+str(effort))
                q.execute(sql,params);q.fetchall()
                for repeat in range(2):
                    start=time.perf_counter();q.execute(sql,params);ids=[x[0] for x in q.fetchall()];ms=(time.perf_counter()-start)*1000
                    recall=len(set(ids)&set(truth[scope['name']][str(seed['paper_id'])]))/20
                    runs.append(dict(config=name,scope=scope['name'],source=seed['paper_id'],recall=recall,ms=ms,returned=len(ids)))
            group=[r for r in runs if r['config']==name]
            result=dict(config=name,recall=float(np.mean([r['recall'] for r in group])),median_ms=float(np.median([r['ms'] for r in group])),p95_ms=float(np.percentile([r['ms'] for r in group],95)),underfilled=sum(r['returned']<20 for r in group))
            summary.append(result);a.event('general retrieval validation',**result)
pg.close();a.atomic_json(root/'general-retrieval-validation.json',dict(reference_utc=now.isoformat(),sources=seeds,summary=summary,runs=runs))
