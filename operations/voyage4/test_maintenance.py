import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import psycopg2
import archive as a
import storage as s

class Cursor:
    def __init__(self,q,schema):self.q=q;self.schema=schema
    def __enter__(self):self.q.__enter__();return self
    def __exit__(self,*args):return self.q.__exit__(*args)
    def execute(self,sql,args=None):
        if isinstance(sql,bytes):sql=sql.decode()
        return self.q.execute(sql.replace('voyage4.',self.schema+'.'),args)
    def copy_expert(self,sql,stream):return self.q.copy_expert(sql.replace('voyage4.',self.schema+'.'),stream)
    def __getattr__(self,key):return getattr(self.q,key)
class Connection:
    def __init__(self,pg,schema):self.pg=pg;self.schema=schema
    def cursor(self):return Cursor(self.pg.cursor(),self.schema)
    def commit(self):return self.pg.commit()
    @property
    def autocommit(self):return self.pg.autocommit
    @autocommit.setter
    def autocommit(self,v):self.pg.autocommit=v

class MaintenanceTests(unittest.TestCase):
    def test_expiry_only_removes_search_cache_and_advances_floor(self):
        schema='v4_maintenance_test_'+uuid.uuid4().hex[:10]
        pg=psycopg2.connect(dbname='arxiv');pg.autocommit=True
        try:
            with pg.cursor() as q:
                q.execute(f'CREATE SCHEMA {schema}')
                q.execute(f'CREATE TABLE {schema}.embeddings (LIKE voyage4.embeddings INCLUDING ALL)')
                q.execute(f'CREATE TABLE {schema}.rolling30 (LIKE voyage4.rolling30 INCLUDING ALL)')
                q.execute(f'CREATE TABLE {schema}.control (LIKE voyage4.control INCLUDING ALL)')
            pg.autocommit=False;wrapped=Connection(pg,schema)
            with tempfile.TemporaryDirectory() as d,patch.object(a,'check_space'):
                root=Path(d);c=a.catalog(root);now=dt.datetime(2026,9,21,tzinfo=dt.timezone.utc)
                v=np.zeros((3,2048),dtype='<f4');v[:,0]=1
                rows=[(i+1,'text',str(i)*64,1,(now-dt.timedelta(days=age)).isoformat(),'["cs.LG"]') for i,age in enumerate([32,15,0])]
                c.executemany('INSERT INTO papers VALUES(?,?,?,?,?,?)',rows);c.commit()
                path=root/'original.npz';np.savez(path,vectors=v);original=a.digest(path)
                with wrapped.cursor() as q:q.copy_expert(f'COPY voyage4.embeddings({s.COLUMNS}) FROM STDIN WITH(FORMAT BINARY)',s.copy_rows(rows,v,path,[0,1,2]))
                wrapped.commit();s.rolling(c,wrapped,root,now)
                with wrapped.cursor() as q:
                    q.execute('SELECT paper_id FROM voyage4.rolling30 ORDER BY paper_id');self.assertEqual(q.fetchall(),[(2,),(3,)])
                wrapped.commit()
                # Metadata-only changes and reverting to an earlier abstract must reuse preserved versions.
                c.execute('INSERT INTO versions VALUES(?,?,?,?)',(3,rows[2][2],str(path),2))
                revised=root/'revised.npz';v2=v.copy();v2[2]=0;v2[2,1]=1;np.savez(revised,vectors=v2)
                revised_hash=a.digest(revised)
                c.execute('INSERT INTO versions VALUES(?,?,?,?)',(3,'9'*64,str(revised),2))
                for sha,cats,expected in [('9'*64,'["math.AG"]',v2[2]),(rows[2][2],'["cs.LG"]',v[2])]:
                    c.execute('UPDATE papers SET sha=?,categories=? WHERE id=3',(sha,cats))
                    c.execute('INSERT OR IGNORE INTO metadata_dirty VALUES(3)');c.commit()
                    s.sync_metadata(c,wrapped);s.rolling(c,wrapped,root,now)
                    with wrapped.cursor() as q:
                        q.execute('SELECT full_vector::text,categories FROM voyage4.rolling30 WHERE paper_id=3')
                        values,categories=q.fetchone()
                        np.testing.assert_array_equal(np.fromstring(values.strip('[]'),sep=',',dtype='<f4'),expected)
                        self.assertEqual(categories,json.loads(cats))
                    wrapped.commit()
                self.assertEqual(a.digest(revised),revised_hash)
                s.rolling(c,wrapped,root,now+dt.timedelta(days=20))
                with wrapped.cursor() as q:
                    q.execute('SELECT paper_id FROM voyage4.rolling30 ORDER BY paper_id');self.assertEqual(q.fetchall(),[(3,)])
                    q.execute('SELECT count(*) FROM voyage4.embeddings');self.assertEqual(q.fetchone()[0],3)
                self.assertEqual(a.digest(path),original);c.close();wrapped.commit()
        finally:
            pg.rollback();pg.autocommit=True
            with pg.cursor() as q:q.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
            pg.close()

if __name__=='__main__':unittest.main()
