"""Eight diagnostic combinations used to compare the seven SR refinement candidates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sr_protocol import jobs

VARIANTS=['soft100','progressive25','progressive100','progressive100_no_old_guide',
          'spatial1','spatial10','bregman5','soft1000']

def pilot_jobs():
    return [j for j in jobs() if j['patch']==0 and (j['group']!='CAVE' or j['name']=='beers')
            and ((j['DS'] in [4,16] and j['sigma']==.5) or
                 (j['DS']==4 and j['sigma']==2 and j['group']!='CAVE'))]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--key')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--root',type=Path,default=ROOT/'outputs/sr')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/sr_diagnostics')
    a=p.parse_args();os.chdir(ROOT)
    selected=[j for j in pilot_jobs() if a.key is None or j['key']==a.key]
    if not selected:p.error('No matching diagnostic combination')
    for j in selected:print(j['key'],', '.join(VARIANTS))
    if not a.execute:return
    import numpy as np
    import torch
    from baseline_sr.protocol import prepare
    from ablation_study.common import metrics,write_json,sha,provenance
    from Diff.networks import ScalarINR2D
    from Diff.Diff4SR import GaussianDiffusion1D
    from Diff.sr_guidance import sample_observation_guided
    from Diff.sr_subspace import SpectralProximal,estimate_spectral_basis
    from Diff.sr_progressive import ProgressiveSpectralCorrection
    from Diff.sr_spatial import SpatialSpectralProximal
    from Diff.sr_residual import refine_residual
    torch.set_num_threads(2);torch.manual_seed(3407)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for j in selected:
        gt,obs,hashes=prepare(j,a.device,a.root/'reference/observations'/f'{j["key"]}.json')
        folder=a.root/'DiffTR'/j['key'];saved=json.loads((folder/'result.json').read_text())
        for k,v in hashes.items():
            if saved[k]!=v:raise ValueError('Observation mismatch: '+k)
        cfg=saved['settings'];net=ScalarINR2D(out_dim=obs.c,**cfg['model'])
        diff=GaussianDiffusion1D(net,obs.msi.cpu(),obs.r.cpu(),**cfg['diffusion'])
        checkpoint=folder/'model-1.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        diff.load_state_dict(ck['model']);del ck
        diff.to(a.device).eval().requires_grad_(False)
        out=a.output/j['key'];out.mkdir(parents=True,exist_ok=True)
        for variant in VARIANTS:
            path=out/f'{variant}_seed{a.seed}.json'
            if path.exists():raise FileExistsError(path)
            if str(a.device).startswith('cuda'):torch.cuda.synchronize(a.device)
            start=time.perf_counter();basis,bi=estimate_spectral_basis(obs.low,.999)
            cls=SpatialSpectralProximal if variant.startswith('spatial') else SpectralProximal
            prox=cls(obs.msi,obs.low,obs.r,j['sigma'],j['DS'])
            transform=None
            if variant.startswith('progressive'):
                transform=ProgressiveSpectralCorrection(prox,basis,diff.num_timesteps,
                    25. if variant=='progressive25' else 100.)
            raw=sample_observation_guided(diff,obs.low,j['sigma'],j['DS'],'original',
                guidance=0. if variant.endswith('no_old_guide') else 15.,seed=a.seed,
                clean_transform=transform)
            if variant.startswith('spatial'):rec,solver=prox.solve_spatial(raw,basis,float(variant[7:]))
            elif variant=='bregman5':rec,solver=refine_residual(prox,raw,basis,5)
            else:
                rec,solver=prox.solve_spectral(raw,1000. if variant=='soft1000' else 100.,
                    basis,10.,400 if variant=='soft1000' else 200,1e-5)
            if not solver['converged']:raise RuntimeError(solver)
            rec=rec.clamp(0,1)
            if str(a.device).startswith('cuda'):torch.cuda.synchronize(a.device)
            elapsed=time.perf_counter()-start;arr=rec.cpu().numpy()
            np.save(path.with_suffix('.npy'),arr)
            write_json(path,dict(**j,**hashes,variant=variant,seed=a.seed,**metrics(gt,arr),
                seconds=elapsed,basis=bi,solver=solver,checkpoint_sha256=sha(checkpoint),
                source_sha256=provenance(['scripts/run_sr_diagnostics.py','Diff/sr_progressive.py',
                    'Diff/sr_spatial.py','Diff/sr_residual.py','Diff/sr_subspace.py','Diff/sr_guidance.py']),
                timing='Fresh sampling per candidate; setup + sampling + solve; no standalone warmup',
                intermediate=[] if transform is None else transform.records))
            print(j['key'],variant,metrics(gt,arr),flush=True)

if __name__=='__main__':main()
