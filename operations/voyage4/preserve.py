"""Read-only byte-level inventory of legacy embeddings; permit additions but no changes."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg2
from psycopg2 import sql
import archive as a

TABLES={
 'papers_embeddingvoyagebit2048':'bit_send',
 'papers_embeddingvoyagehalf2048':'halfvec_send',
 'papers_embeddingvoyagehalf256':'halfvec_send',
 'papers_embeddinggeminihalf3072':'halfvec_send',
 'papers_embeddinggeminihalf512':'halfvec_send',
}
class HashSink:
    def __init__(self):self.hash=hashlib.sha256();self.bytes=0
    def write(self,b):self.hash.update(b);self.bytes+=len(b)

def inventory(baseline=None):
    pg=psycopg2.connect(dbname='arxiv',application_name='voyage4_preservation_audit',options='-c default_transaction_read_only=on')
    result={}
    with pg.cursor() as q:
        for table,send in TABLES.items():
            q.execute(sql.SQL('SELECT count(*),max(paper_id) FROM public.{}').format(sql.Identifier(table)))
            count,maximum=q.fetchone();boundary=baseline[table]['max_id'] if baseline else maximum
            sink=HashSink()
            query=sql.SQL('COPY (SELECT paper_id,{}(vector),created_at FROM public.{} WHERE paper_id<=%s ORDER BY paper_id) TO STDOUT WITH(FORMAT BINARY)').format(sql.Identifier(send),sql.Identifier(table))
            q.copy_expert(q.mogrify(query,(boundary or 0,)).decode(),sink)
            result[table]=dict(count=count,max_id=maximum,protected_max_id=boundary,sha256=sink.hash.hexdigest(),bytes=sink.bytes)
            if baseline:
                assert result[table]['sha256']==baseline[table]['sha256'],f'Legacy content changed: {table}'
                assert count>=baseline[table]['count'],f'Legacy rows removed: {table}'
            a.event('legacy table byte audit',table=table,**result[table])
    pg.close();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--root',type=Path,default=a.ROOT);args=p.parse_args()
    path=args.root/'backups'/'legacy-baseline.json'
    if args.verify:
        before=json.loads(path.read_text());result=inventory(before)
        a.atomic_json(args.root/'legacy-preservation-verified.json',dict(passed=True,tables=result))
    elif path.exists():raise RuntimeError('Existing preservation baseline is immutable')
    else:
        result=inventory();a.atomic_json(path,result);path.chmod(0o444)
