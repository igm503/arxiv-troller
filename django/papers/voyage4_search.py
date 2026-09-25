"""Voyage 4 similarity search.

Windows of up to 30 days search EmbeddingVoyage4Recent, whose HNSW graph holds only recent papers.
Longer windows use binary HNSW over every paper, rescored with half-precision cosine.
"""
from datetime import timedelta

from django.db import connection, transaction
from django.utils import timezone

from .models import EmbeddingVoyage4, EmbeddingVoyage4Recent

ALL_TABLE = EmbeddingVoyage4._meta.db_table
RECENT_TABLE = EmbeddingVoyage4Recent._meta.db_table
# Pruning keeps papers created within RECENT_DAYS of the last embedding run, so any cutoff
# inside this window is fully covered. The "1month" filter (30 days) always qualifies.
RECENT_WINDOW = timedelta(days=EmbeddingVoyage4Recent.RECENT_DAYS)
GENERAL_EF = 1000
GENERAL_CANDIDATES = 500


def predicate(cutoff, category, excluded):
    terms, params = [], []
    if cutoff is not None:
        terms.append("e.created >= %s")
        params.append(cutoff)
    if category:
        terms.append("e.categories @> ARRAY[%s]::varchar[]")
        params.append(category)
    if excluded:
        terms.append("NOT (e.paper_id = ANY(%s::bigint[]))")
        params.append(sorted(set(map(int, excluded))))
    return " AND ".join(terms) or "TRUE", params


def query_sql(*, recent, cutoff, category, excluded, limit, vector, bits, exact=False):
    where, params = predicate(cutoff, category, excluded)
    table = RECENT_TABLE if recent else ALL_TABLE
    if exact:
        sql = f"""SELECT e.paper_id FROM {table} e WHERE {where}
            ORDER BY (e.vector <=> %s::halfvec) + 0, e.paper_id LIMIT %s"""
        return sql, params + [vector, limit]
    if recent:
        count = max(50, limit) if category else max(20, limit)
        sql = f"""WITH candidates AS MATERIALIZED (
              SELECT e.paper_id, e.vector <-> %s::halfvec AS distance FROM {table} e
              WHERE {where} ORDER BY distance LIMIT %s)
            SELECT e.paper_id FROM candidates c JOIN {table} e USING (paper_id)
            ORDER BY (e.vector <=> %s::halfvec) + 0, e.paper_id LIMIT %s"""
        # Rescoring also puts relaxed-order HNSW results back in exact order
        return sql, [vector] + params + [count, vector, limit]
    if category:
        # Materialize the eligible binary vectors before sorting, avoiding filtered ANN underfill
        sql = f"""WITH eligible AS MATERIALIZED (
              SELECT e.paper_id, e.bits FROM {table} e WHERE {where}),
            candidates AS MATERIALIZED (
              SELECT paper_id FROM eligible ORDER BY (bits <~> %s::bit(2048)) + 0, paper_id LIMIT %s)
            SELECT e.paper_id FROM candidates c JOIN {table} e USING (paper_id)
            ORDER BY (e.vector <=> %s::halfvec) + 0, e.paper_id LIMIT %s"""
        return sql, params + [bits, max(500, limit * 10), vector, limit]
    sql = f"""WITH candidates AS MATERIALIZED (
          SELECT e.paper_id, e.bits <~> %s::bit(2048) AS distance FROM {table} e
          WHERE {where} ORDER BY distance LIMIT %s)
        SELECT e.paper_id FROM candidates c JOIN {table} e USING (paper_id)
        ORDER BY (e.vector <=> %s::halfvec) + 0, e.paper_id LIMIT %s"""
    return sql, [bits] + params + [max(GENERAL_CANDIDATES, limit * 5), vector, limit]


def similar_ids(paper_id, *, cutoff, category, excluded, limit):
    """Return the IDs of the most similar papers, nearest first."""
    with connection.cursor() as q:
        q.execute(f"SELECT vector::text, bits::text FROM {ALL_TABLE} WHERE paper_id = %s", [paper_id])
        source = q.fetchone()
    if source is None:
        return []
    vector, bits = source
    recent = cutoff is not None and cutoff >= timezone.now() - RECENT_WINDOW
    args = dict(
        recent=recent,
        cutoff=cutoff,
        category=category,
        excluded=set(excluded) | {int(paper_id)},
        limit=limit,
        vector=vector,
        bits=bits,
    )
    ef_search = (512 if category else 128) if recent else GENERAL_EF
    with transaction.atomic(), connection.cursor() as q:
        # SET LOCAL keeps these search settings from leaking into other requests
        q.execute(
            "SET LOCAL hnsw.iterative_scan = 'relaxed_order';"
            f"SET LOCAL hnsw.ef_search = {ef_search};"
            "SET LOCAL hnsw.max_scan_tuples = 20000"
        )
        q.execute(*query_sql(**args))
        ids = [row[0] for row in q.fetchall()]
        if len(ids) < limit:
            # An exhausted iterative scan must not hide eligible results
            q.execute(*query_sql(**args, exact=True))
            ids = [row[0] for row in q.fetchall()]
    return ids
