"""Paper scene matrix and fresh observation manifests; no archived results required."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CASES = ((4, .5), (4, 2.), (8, .5), (8, 2.), (16, .5))
METHODS = ['DiffTR', 'GTNN', 'GNLTN', 'CS2DIPs', 'EDIP-Net']
BUDGETS = {'GTNN': 100, 'GNLTN': 50, 'CS2DIPs': 5000, 'EDIP-Net': 14000}


def jobs():
    return [dict(group=group, name=name, patch=patch, DS=ds, sigma=sigma,
                 key=f'{group}_{name}_patch{patch}_DS{ds}_sigma{sigma:g}')
            for ds, sigma in CASES
            for group, names, patches in [('WDC', ['WDC'], range(4)),
                ('PaviaU', ['PaviaU'], range(2)),
                ('CAVE', ['beers', 'clay', 'flowers', 'balloons'], [0])]
            for name in names for patch in patches]


def prepare_manifest(job, output):
    import numpy as np
    import torch
    import yaml
    from types import SimpleNamespace
    from exp_diff_SR import load_scene, hrhsit_to_lrhsit, array_hash
    from ablation_study.common import write_json
    torch.set_num_threads(2)
    args = SimpleNamespace(**job, protocol='blur', response_key='SRF',
        response_transpose=True, synthetic_wdc_response=False,
        response_path='data/responses/TenFuse/SRF_matrix.mat' if job['group']=='WDC' else None)
    gt, r, *_ = load_scene(args)
    low = hrhsit_to_lrhsit(gt, sigma=job['sigma'], downsample_factor=job['DS'])
    msi = hrhsit_to_lrhsit(gt @ r.T, sigma=job['sigma'], downsample_factor=1)
    config_file = 'SR_MSI_CAVE_diff_conf.yaml' if job['group']=='CAVE' else 'SR_MSI_PaviaU_diff_conf.yaml'
    cfg = yaml.full_load((ROOT/'configs'/config_file).read_text())
    cfg['model']['tensor_shape'] = list(gt.shape[:2])
    cfg['model']['latent_dim'] = 256 if job['group']=='WDC' else 128
    cfg['diffusion']['channels'] = gt.shape[-1]
    cfg['trainer']['results_folder'] = str(output/'DiffTR'/job['key'])
    value = json.loads(json.dumps(dict(**job, protocol='blur', shape=list(gt.shape), config=cfg,
        gt_sha256=array_hash(gt), response_sha256=array_hash(r),
        LR_HSI_sha256=array_hash(low), HR_MSI_sha256=array_hash(msi))))
    path = output/'reference/observations'/f'{job["key"]}.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Observation/configuration changed: use a fresh output directory: {path}')
    write_json(path, value)
    return path
