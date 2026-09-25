import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import datetime as dt
import os
import numpy as np
import archive as a

class Response:
    status_code=200
    headers={'x-request-id':'test-request'}
    def __init__(self,body):self.body=body
    def json(self):return self.body
class Client:
    def __init__(self,body):self.body=body;self.calls=0
    def post(self,url,json):
        self.calls+=1
        assert url==a.ENDPOINT and json['output_dtype']=='float' and json['output_dimension']==2048
        assert json['input_type'] is None and json['truncation'] is False
        return Response(self.body)
class Pace:
    def acquire(self,tokens):pass

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(32);self.v=rng.normal(size=(2,2048)).astype('<f4');self.v/=np.linalg.norm(self.v,axis=1)[:,None]
        self.body=dict(model=a.MODEL,usage={'total_tokens':20},data=[dict(index=i,embedding=base64.b64encode(v.tobytes()).decode()) for i,v in enumerate(self.v)])
        self.rows=[(i+1,s,hashlib.sha256(s.encode()).hexdigest(),10,'2026-09-20T00:00:00+00:00','["cs.LG"]') for i,s in enumerate(['Abstract one','Abstract two'])]
    def test_original_bytes_and_resume_without_api(self):
        with tempfile.TemporaryDirectory() as d,patch.object(a,'check_space'):
            root=Path(d);(root/'shards').mkdir();c=a.catalog(root);client=Client(self.body)
            p=a.generate_one(root,'test',self.rows,client,Pace())
            with np.load(p,allow_pickle=False) as z:self.assertEqual(z['vectors'].tobytes(),self.v.tobytes())
            before=a.digest(p)
            a.generate_one(root,'test',self.rows,client,Pace());self.assertEqual(client.calls,1);self.assertEqual(a.digest(p),before)
            a.register(c,p,'standard-api');a.register(c,p,'standard-api')
            self.assertEqual(c.execute('SELECT count(*) FROM versions').fetchone()[0],2)
            self.assertEqual(c.execute('SELECT sum(tokens) FROM shards').fetchone()[0],20)
            c.close()
    def test_reject_wrong_model_and_duplicate_indices(self):
        body=dict(self.body,model='voyage-3-large')
        with self.assertRaises(ValueError):a.decode(body,2)
        body=dict(self.body,data=[self.body['data'][0]]*2)
        with self.assertRaises(ValueError):a.decode(body,2)
    def test_plan_skips_matching_pilot_but_not_changed_text(self):
        with tempfile.TemporaryDirectory() as d:
            c=a.catalog(Path(d));c.executemany('INSERT INTO papers VALUES(?,?,?,?,?,?)',self.rows)
            c.execute('INSERT INTO versions VALUES(?,?,?,?)',(1,self.rows[0][2],'/saved/pilot.npz',0));c.commit()
            self.assertEqual([[r[0] for r in rows] for rows in a.plan(c)], [[2]])
            self.assertEqual([[r[0] for r in rows] for rows in a.plan(c)], [[2]])
            c.close()
    def test_invalid_papers_do_not_block_snapshot_and_can_be_retried(self):
        now = dt.datetime.now(dt.timezone.utc)
        pg, tokenizer = MagicMock(), MagicMock()
        pg.cursor.return_value.__enter__.return_value.fetchone.return_value = (3,)
        with tempfile.TemporaryDirectory() as d, patch.object(a, 'database', return_value=pg), patch.object(a, 'Tokenizer') as factory, patch.object(a, 'check_space'):
            factory.from_pretrained.return_value = tokenizer
            c = a.catalog(Path(d))
            for text, lengths, expected in [('', [0, 32001, 3], [3]), ('repaired', [3, 3, 3], [1, 2, 3])]:
                pg.cursor.return_value.fetchmany.side_effect = [[(1, text, now, ['cs.LG'], now), (2, 'long', now, [], now), (3, 'valid', now, [], now)], []]
                tokenizer.encode_batch.return_value = [range(n) for n in lengths]
                a.freeze(c, Path(d))
                self.assertEqual(sorted(r[0] for batch in a.plan(c) for r in batch), expected)
                self.assertTrue(a.state(c, 'snapshot_complete')['complete'])
            self.assertEqual(c.execute('SELECT count(*) FROM failures').fetchone()[0], 0)
            c.close()

    def test_unchanged_maintenance_does_not_rescan_archives(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a, 'freeze', return_value=0), patch.object(a, 'reuse', side_effect=AssertionError('Unnecessary scan')):
            Path(d, 'progress.json').write_text(json.dumps(dict(status='complete')))
            with patch('sys.argv', ['archive.py', 'all', '--root', d]):
                a.main()

    def test_crash_recovery_and_revisions_preserve_saved_bytes(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a, 'check_space'), patch.dict(os.environ, VOYAGE_API_KEY='test'):
            root = Path(d)
            (root / 'shards').mkdir()
            c = a.catalog(root)
            c.executemany('INSERT INTO papers VALUES(?,?,?,?,?,?)', self.rows)
            c.commit()
            client = Client(self.body)
            with patch.object(a.httpx, 'Client') as factory:
                factory.return_value.__enter__.return_value = client
                with patch.object(a, 'register', side_effect=RuntimeError('simulated crash')):
                    with self.assertRaises(RuntimeError):
                        a.generate(c, root, 16000000, 4000, 1)
                saved = next((root / 'shards').glob('*.npz'))
                before = a.digest(saved)
                a.reuse(c, root)
                a.generate(c, root, 16000000, 4000, 1)
                self.assertEqual(client.calls, 1)
                c.execute('UPDATE papers SET abstract=?,sha=? WHERE id=1', ('revision', hashlib.sha256(b'revision').hexdigest()))
                c.commit()
                client.body = dict(self.body, data=self.body['data'][:1])
                a.generate(c, root, 16000000, 4000, 1)
                self.assertEqual(client.calls, 2)
                self.assertEqual(a.digest(saved), before)
                self.assertEqual(c.execute('SELECT count(*) FROM versions').fetchone()[0], 3)
            c.close()

if __name__=='__main__':unittest.main()
