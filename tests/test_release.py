"""Release integration checks for experiment coverage and fresh manifest preparation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.sr_protocol import jobs, prepare_manifest
from scripts.run_experiments import matrix
from scripts.run_sr_diagnostics import pilot_jobs

class ReleaseTests(unittest.TestCase):
    def test_experiment_coverage(self):
        self.assertEqual(len(jobs()),50)
        self.assertEqual(len({j['key'] for j in jobs()}),50)
        self.assertEqual(len(pilot_jobs()),8)
        for suite,count in [('tabular',16),('completion-entry',60),('completion-tube',60),('denoising',50),('scoretr',30)]:
            commands=matrix(suite,Path('/tmp/example'))
            self.assertEqual(len(commands),count)
            self.assertEqual(len({k for k,c in commands}),count)
            self.assertTrue(all(Path(c[1]).is_file() for k,c in commands))
        denoise=matrix('denoising',Path('/tmp/example'))
        self.assertEqual(sum('--guidance' in c for k,c in denoise),14)
        self.assertTrue(all('--noise-protocol' in c and c[c.index('--noise-protocol')+1]=='legacy' for k,c in denoise))

    def test_fresh_manifest_and_mismatch_detection(self):
        rng=np.random.default_rng(1);gt=rng.random((8,8,3),dtype=np.float32)
        r=np.array([[.6,.3,.1],[.1,.3,.6]],dtype=np.float32)
        job=next(j for j in jobs() if j['group']=='WDC')
        with tempfile.TemporaryDirectory() as d, patch('exp_diff_SR.load_scene',return_value=(gt,r,[], 'WDC', [0,0], 'test')):
            out=Path(d)
            path=prepare_manifest(job,out)
            first=json.loads(path.read_text())
            self.assertEqual(first['config']['model']['latent_dim'],256)
            self.assertEqual(first['config']['diffusion']['channels'],3)
            self.assertEqual(first['config']['trainer']['train_num_steps'],10000)
            self.assertEqual(path,prepare_manifest(job,out))
            first['gt_sha256']='wrong';path.write_text(json.dumps(first))
            with self.assertRaises(ValueError):prepare_manifest(job,out)

if __name__=='__main__':unittest.main()
