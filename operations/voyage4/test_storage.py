import json
import unittest
import numpy as np
import storage as s

class StorageTests(unittest.TestCase):
    def test_binary_copy_preserves_original_float32_and_restricts_role(self):
        pg=s.connect()
        try:
            rng=np.random.default_rng(19);v=rng.normal(size=(2,2048)).astype('<f4');v/=np.linalg.norm(v,axis=1)[:,None]
            rows=[(i+1,'x','a'*64,1,'2026-09-20T00:00:00+00:00',json.dumps(['cs.LG','stat.ML'])) for i in range(2)]
            with pg.cursor() as q:
                q.execute("SELECT has_table_privilege(current_user,'public.papers_embeddingvoyagehalf2048','DELETE'),has_table_privilege(current_user,'public.papers_embeddingvoyagebit2048','UPDATE')")
                self.assertEqual(q.fetchone(),(False,False))
                q.execute('CREATE TEMP TABLE test_rolling (LIKE voyage4.rolling30) ON COMMIT DROP')
                q.copy_expert('COPY test_rolling FROM STDIN WITH(FORMAT BINARY)',s.copy_rows(rows,v,'x',[0,1],full=True))
                q.execute('SELECT paper_id,categories,full_vector::text,vector::text FROM test_rolling ORDER BY paper_id')
                for i,(pid,cats,full,half) in enumerate(q.fetchall()):
                    self.assertEqual(cats,['cs.LG','stat.ML'])
                    self.assertEqual(np.fromstring(full.strip('[]'),sep=',',dtype='<f4').tobytes(),v[i].tobytes())
                    np.testing.assert_array_equal(np.fromstring(half.strip('[]'),sep=',',dtype='<f4'),v[i].astype('f2').astype('f4'))
                q.execute('CREATE TEMP TABLE test_embeddings (LIKE voyage4.embeddings INCLUDING DEFAULTS) ON COMMIT DROP')
                q.copy_expert(f'COPY test_embeddings({s.COLUMNS}) FROM STDIN WITH(FORMAT BINARY)',s.copy_rows(rows,v,'/original.npz',[0,1]))
                q.execute('SELECT bits::text,archive_row FROM test_embeddings ORDER BY paper_id')
                for i,(bits,index) in enumerate(q.fetchall()):
                    self.assertEqual(bits,''.join('1' if x>0 else '0' for x in v[i]));self.assertEqual(index,i)
        finally:pg.rollback();pg.close()

if __name__=='__main__':unittest.main()
