"""Standalone inference profile, excluding the training optimizer and common loading."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import os
    os.chdir(ROOT)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--key',required=True)
    p.add_argument('--root',type=Path,default=ROOT/'outputs/sr');p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import numpy as np
    import torch
    from scripts.sr_protocol import jobs
    from baseline_sr.protocol import prepare
    from Diff.networks import ScalarINR2D
    from Diff.Diff4SR import GaussianDiffusion1D
    from Diff.sr_reconstruction import reconstruct_sr
    from ablation_study.common import write_json,metrics,sha
    torch.set_num_threads(2);torch.manual_seed(3407)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    j=next(j for j in jobs() if j['key']==a.key)
    folder=a.root/'DiffTR'/a.key
    saved=json.loads((folder/'result.json').read_text());cfg=saved['settings']
    gt,obs,hashes=prepare(j,a.device,a.root/'reference/observations'/f'{a.key}.json')
    for k,v in hashes.items():
        if saved[k]!=v:raise ValueError('Observation mismatch: '+k)
    model=ScalarINR2D(out_dim=obs.c,**cfg['model'])
    diff=GaussianDiffusion1D(model,obs.msi.cpu(),obs.r.cpu(),**cfg['diffusion'])
    ck=torch.load(folder/'model-1.pt',map_location='cpu',weights_only=False)
    diff.load_state_dict(ck['model']);del ck
    diff.to(a.device).eval().requires_grad_(False)
    x=torch.ones(1024,1024,device=a.device);until=time.perf_counter()+3
    while time.perf_counter()<until:y=x@x;torch.cuda.synchronize(a.device)
    del x,y
    torch.cuda.reset_peak_memory_stats(a.device);torch.cuda.synchronize(a.device);start=time.perf_counter()
    rec,info=reconstruct_sr(diff,obs.low,j['sigma'],j['DS'])
    torch.cuda.synchronize(a.device);seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated(a.device)
    write_json(folder/'profile.json',dict(**j,**hashes,**metrics(gt,rec.cpu().numpy()),
        inference_seconds=seconds,reconstruction_seconds=saved['limits']['optimization']+seconds,
        inference_peak_gpu_bytes=peak,combined_peak_gpu_bytes=max(peak,saved['training_peak_gpu_bytes']),
        network_parameters=saved['network_parameters'],checkpoint_sha256=sha(folder/'model-1.pt'),
        device=torch.cuda.get_device_name(a.device),solver=info))
    print(folder/'profile.json')

if __name__=='__main__':main()
