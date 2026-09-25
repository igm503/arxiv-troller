import datetime as dt
import importlib.util
import json
import sys
import unittest
from pathlib import Path
import numpy as np
from django.conf import settings
if not settings.configured:settings.configure(VOYAGE4_ENABLED=False)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'django'))
from papers.voyage4_search import query_sql, predicate, search_ids
import storage as s

class SearchTests(unittest.TestCase):
    def test_disabled_backend_is_legacy(self):
        self.assertIsNone(search_ids(1,cutoff=None,category='',excluded=set(),limit=20))
    def test_real_sql_date_category_exclusions_and_float32_rescoring(self):
        pg=s.connect()
        try:
            v=np.zeros((5,2048),dtype='<f4')
            for i in range(5):v[i,0]=1;v[i,1]=i*.1
            v/=np.linalg.norm(v,axis=1)[:,None]
            rows=[(i+1,'abstract','a'*64,10,'2026-09-20T00:00:00+00:00','["cs.LG"]') for i in range(5)]
            rows[1]=(2,'abstract','a'*64,10,'2025-01-01T00:00:00+00:00','["cs.LG"]')
            rows[2]=(3,'abstract','a'*64,10,'2026-09-20T00:00:00+00:00','["math.AG"]')
            with pg.cursor() as q:
                q.execute('CREATE TEMP TABLE test_search_rolling (LIKE voyage4.rolling30) ON COMMIT DROP')
                q.copy_expert('COPY test_search_rolling FROM STDIN WITH(FORMAT BINARY)',s.copy_rows(rows,v,'x',list(range(5)),full=True))
                q.execute('CREATE TEMP TABLE test_search_all (LIKE voyage4.embeddings INCLUDING DEFAULTS) ON COMMIT DROP')
                q.copy_expert(f'COPY test_search_all({s.COLUMNS}) FROM STDIN WITH(FORMAT BINARY)',s.copy_rows(rows,v,'x',list(range(5))))
                for rolling in [False,True]:
                    for exact in [False,True]:
                        for category,expected in [('cs.LG',[4]),('',[3,4])]:
                            sql,args=query_sql(rolling=rolling,cutoff=dt.datetime(2026,9,1,tzinfo=dt.timezone.utc),category=category,
                                  excluded={1,5},limit=20,query_vector='['+','.join(map(str,v[0].tolist()))+']',bits='1'+'0'*2047,exact=exact)
                            sql=sql.replace('voyage4.rolling30','test_search_rolling').replace('voyage4.embeddings','test_search_all')
                            q.execute(sql,args);self.assertEqual([x[0] for x in q.fetchall()],expected,(rolling,exact,category))
        finally:pg.rollback();pg.close()
if __name__=='__main__':unittest.main()
