"""Voyage 4 retrieval over a maintained 30-day graph and permanent archive search copies."""
import datetime as dt
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
from django.conf import settings
from django.db import connection, transaction, DatabaseError
from django.utils import timezone

logger=logging.getLogger(__name__)
_control_cache=(0.,None)
GENERAL_EF=1000
GENERAL_CANDIDATES=500


@lru_cache(maxsize=16)
def shard_vectors(path):
    # Only database-sourced archive locations are accepted; no request path is used here.
    with np.load(path,allow_pickle=False) as z:
        result=z['vectors']
        if result.dtype!=np.float32 or result.ndim!=2 or result.shape[1]!=2048:
            raise ValueError('Invalid saved Voyage 4 vector array')
        result.flags.writeable=False
        return result


def controls():
    global _control_cache
    now=time.monotonic()
    if now-_control_cache[0]>30:
        with connection.cursor() as q:
            q.execute("SELECT key,value FROM voyage4.control WHERE key IN ('ready','rolling')")
            values={key:json.loads(value) if isinstance(value,str) else value for key,value in q.fetchall()}
        _control_cache=(now,values)
    return _control_cache[1] or {}


def predicate(cutoff,category,excluded,alias='e'):
    terms=[];params=[]
    if cutoff is not None:terms.append(f'{alias}.created >= %s');params.append(cutoff)
    if category:terms.append(f'{alias}.categories @> ARRAY[%s]::varchar[]');params.append(category)
    if excluded:terms.append(f'NOT ({alias}.paper_id = ANY(%s::bigint[]))');params.append(sorted(set(map(int,excluded))))
    return ' AND '.join(terms) or 'TRUE',params


def query_sql(*,rolling,cutoff,category,excluded,limit,query_vector,bits,exact=False):
    where,params=predicate(cutoff,category,excluded)
    table='voyage4.rolling30' if rolling else 'voyage4.embeddings'
    if exact:
        col='full_vector' if rolling else 'vector';cast='vector' if rolling else 'halfvec'
        return (f'SELECT e.paper_id FROM {table} e WHERE {where} ORDER BY (e.{col} <=> %s::{cast}) + 0,e.paper_id LIMIT %s',params+[query_vector,limit])
    if rolling:
        count=max(50,limit) if category else max(limit,20)
        sql=f'''WITH candidates AS MATERIALIZED (
          SELECT e.paper_id,e.vector <-> %s::halfvec AS distance FROM {table} e
          WHERE {where} ORDER BY distance LIMIT %s)
        SELECT e.paper_id FROM candidates c JOIN {table} e USING(paper_id)
        ORDER BY (e.full_vector <=> %s::vector) + 0,e.paper_id LIMIT %s'''
        # Rescoring the unfiltered top-20 also makes relaxed-order results deterministic.
        return sql,[query_vector]+params+[count,query_vector,limit]
    if category:
        # Materialize the eligible binary vectors before sorting, avoiding filtered ANN underfill.
        sql=f'''WITH eligible AS MATERIALIZED (
          SELECT e.paper_id,e.bits FROM {table} e WHERE {where}),
        candidates AS MATERIALIZED (SELECT paper_id FROM eligible ORDER BY (bits <~> %s::bit(2048)) + 0,paper_id LIMIT %s)
        SELECT e.paper_id FROM candidates c JOIN {table} e USING(paper_id)
        ORDER BY (e.vector <=> %s::halfvec) + 0,e.paper_id LIMIT %s'''
        return sql,params+[bits,max(500,limit*10),query_vector,limit]
    sql=f'''WITH candidates AS MATERIALIZED (
      SELECT e.paper_id,e.bits <~> %s::bit(2048) AS distance FROM {table} e
      WHERE {where} ORDER BY distance LIMIT %s)
    SELECT e.paper_id FROM candidates c JOIN {table} e USING(paper_id)
    ORDER BY (e.vector <=> %s::halfvec) + 0,e.paper_id LIMIT %s'''
    return sql,[bits]+params+[max(GENERAL_CANDIDATES,limit*5),query_vector,limit]


def search_ids(paper_id,*,cutoff,category,excluded,limit):
    """Return None to use the intact legacy backend, or an ordered list of V4 IDs."""
    if not getattr(settings,'VOYAGE4_ENABLED',False):return None
    try:
        state=controls()
        if state.get('ready') is not True:return None
        with connection.cursor() as q:
            q.execute('SELECT archive_path,archive_row,vector_sha FROM voyage4.embeddings WHERE paper_id=%s',(paper_id,))
            source=q.fetchone()
        if source is None:return None
        path,index,sha=source;v=shard_vectors(path)[index]
        if hashlib.sha256(v.tobytes()).hexdigest()!=sha:raise ValueError('Saved source vector checksum mismatch')
        query_vector='['+','.join(map(str,v.tolist()))+']';bits=''.join('1' if x else '0' for x in v>0)
        excluded=set(excluded)|{int(paper_id)}
        window=state.get('rolling',{})
        floor=dt.datetime.fromisoformat(window['floor']) if window else None
        rolling=bool(cutoff is not None and floor is not None and cutoff>=floor)
        args=dict(rolling=rolling,cutoff=cutoff,category=category,excluded=excluded,limit=limit,query_vector=query_vector,bits=bits)
        with transaction.atomic(),connection.cursor() as q:
            # SET LOCAL ensures one request cannot leak ANN settings into another.
            q.execute("SET LOCAL hnsw.iterative_scan='relaxed_order'")
            q.execute('SET LOCAL hnsw.ef_search='+str(512 if rolling and category else 128 if rolling else GENERAL_EF))
            q.execute('SET LOCAL hnsw.max_scan_tuples=20000')
            sql,params=query_sql(**args);q.execute(sql,params);ids=[r[0] for r in q.fetchall()]
            if len(ids)<limit:
                # An exhausted iterative scan must not silently hide eligible results.
                sql,params=query_sql(**args,exact=True);q.execute(sql,params);ids=[r[0] for r in q.fetchall()]
        return ids
    except (DatabaseError,OSError,ValueError,KeyError,IndexError):
        logger.exception('Voyage 4 retrieval unavailable; using preserved legacy embeddings')
        return None
