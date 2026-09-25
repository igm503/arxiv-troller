"""Recall/SQL latency against original float32 vectors in one read-only database snapshot.

Example: python validate_retrieval.py --days 7 30 --queries 36
Use --seeds reference_metadata.json to repeat a previous cohort. No reference cache is reused.
"""
import argparse
import datetime as dt
from itertools import groupby
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from django.conf import settings
if not settings.configured:
    settings.configure()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'django'))
from papers.voyage4_search import query_sql, shard_vectors
import archive as a


def reference(pg, seeds, scopes):
    queries = []
    with pg.cursor() as q:
        for pid in seeds:
            q.execute('SELECT archive_path,archive_row FROM voyage4.embeddings WHERE paper_id=%s', (pid,))
            row = q.fetchone()
            a.require(row is not None, f'Missing query embedding: {pid}')
            queries.append(shard_vectors(row[0])[row[1]].copy())
    matrix = np.asarray(queries, dtype='f8')
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    best = {s['name']: (np.empty((0, len(seeds))), np.empty((0, len(seeds)), dtype='i8')) for s in scopes}
    cutoff = min(s['cutoff'] for s in scopes) if all(s['cutoff'] for s in scopes) else None
    # Archive pointers and metadata come from the same PostgreSQL snapshot as the timed queries.
    with pg.cursor(name='reference_vectors') as q:
        q.itersize = 2048
        q.execute('''SELECT archive_path,archive_row,paper_id,created,categories FROM voyage4.embeddings
            WHERE (%s::timestamptz IS NULL OR created >= %s) ORDER BY archive_path,paper_id''', (cutoff, cutoff))
        for number, (path, group) in enumerate(groupby(q, key=lambda row: row[0]), 1):
            rows = list(group)
            ids = np.asarray([r[2] for r in rows])
            vectors = shard_vectors(path)[[r[1] for r in rows]].astype('f8')
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
            scores = vectors @ matrix.T
            scores[ids[:, None] == np.asarray(seeds)[None, :]] = -np.inf
            for scope in scopes:
                mask = np.asarray([(scope['cutoff'] is None or r[3] >= scope['cutoff']) and
                                   (not scope['field'] or scope['field'] in r[4]) for r in rows])
                values, candidates = best[scope['name']]
                values = np.concatenate([values, scores[mask]])
                candidates = np.concatenate([candidates, np.broadcast_to(ids[mask, None], (int(mask.sum()), len(seeds)))])
                order = np.lexsort((candidates, -values), axis=0)[:20]
                best[scope['name']] = (np.take_along_axis(values, order, axis=0), np.take_along_axis(candidates, order, axis=0))
            if number % 500 == 0:
                a.event('reference progress', shards=number)
    truth = {scope: {pid: set(ids[:, j][np.isfinite(scores[:, j])].tolist()) for j, pid in enumerate(seeds)}
             for scope, (scores, ids) in best.items()}
    return queries, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', nargs='+', type=int, default=[7, 30, 90, 180, 365, 0], help='0 means all time')
    parser.add_argument('--queries', type=int, default=36)
    parser.add_argument('--seeds', type=Path)
    parser.add_argument('--output', type=Path, default=a.ROOT / 'retrieval-validation.json')
    args = parser.parse_args()
    a.require(args.queries > 0 and min(args.days) >= 0, 'Invalid days or query count')
    pg = a.database()
    pg.set_session(isolation_level='REPEATABLE READ', readonly=True)
    with pg, pg.cursor() as q:
        q.execute("SET LOCAL statement_timeout='30min'")
        q.execute('SELECT transaction_timestamp()')
        now = q.fetchone()[0]
        if args.seeds:
            seeds = [s['paper_id'] for s in json.loads(args.seeds.read_text())['query_seeds']][:args.queries]
        else:
            q.execute('''SELECT paper_id FROM voyage4.embeddings TABLESAMPLE SYSTEM(1) REPEATABLE(42)
                ORDER BY md5(paper_id::text) LIMIT %s''', (args.queries,))
            seeds = [r[0] for r in q.fetchall()]
        a.require(seeds, 'No query seeds; provide --seeds for a small corpus')
        scopes = [dict(name=(f'{days}d' if days else 'all') + ('_field' if field else ''), days=days,
                       cutoff=now-dt.timedelta(days=days) if days else None, field=field)
                  for days in args.days for field in ['', 'cs.LG']]
        queries, truth = reference(pg, seeds, scopes)
        runs, summary = [], []
        for scope in scopes:
            rolling = 0 < scope['days'] <= 30
            configs = [(512 if scope['field'] else 128, None)] if rolling else [(256, 500)] if scope['field'] else [(256, 200), (1000, 500)]
            for effort, count in configs:
                config = scope['name'] + f'_ef{effort}_k{count}'
                for pid, vector in zip(seeds, queries):
                    sql, params = query_sql(rolling=rolling, cutoff=scope['cutoff'], category=scope['field'], excluded={pid},
                        limit=20, query_vector='['+','.join(map(str, vector.tolist()))+']', bits=''.join('1' if x > 0 else '0' for x in vector))
                    if count is not None:
                        params[-3] = count
                    q.execute("SET LOCAL hnsw.iterative_scan='relaxed_order'; SET LOCAL hnsw.max_scan_tuples=20000; SET LOCAL hnsw.ef_search="+str(effort))
                    q.execute(sql, params)
                    q.fetchall()
                    for repeat in range(2):
                        start = time.perf_counter()
                        q.execute(sql, params)
                        found = {r[0] for r in q.fetchall()}
                        ms = (time.perf_counter()-start)*1000
                        expected = truth[scope['name']][pid]
                        runs.append(dict(config=config, source=pid, ms=ms, returned=len(found), recall=len(found & expected)/len(expected) if expected else 1.))
                group = [r for r in runs if r['config'] == config]
                result = dict(config=config, recall=float(np.mean([r['recall'] for r in group])),
                              median_ms=float(np.median([r['ms'] for r in group])), p95_ms=float(np.percentile([r['ms'] for r in group], 95)))
                summary.append(result)
                a.event('retrieval validation', **result)
    pg.close()
    a.atomic_json(args.output, dict(reference_utc=now.isoformat(), sources=seeds, summary=summary, runs=runs))


if __name__ == '__main__':
    main()
