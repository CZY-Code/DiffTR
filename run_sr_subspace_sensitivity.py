"""Exploratory spectral energy/penalty sensitivity on cached raw proposals."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from baseline_sr.protocol import prepare, ROOT
from scripts.sr_protocol import jobs
from Diff.sr_subspace import SpectralProximal, estimate_spectral_basis
from ablation_study.common import metrics, write_json, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    primary=ROOT/'outputs/subspace';out=primary/'sensitivity';out.mkdir(exist_ok=True)
    plan=json.loads((primary/'plan.json').read_text())
    configs=[('energy999_gamma1',.999,1.),('energy999_gamma100',.999,100.),
             ('energy9999_gamma10',.9999,10.)]
    write_json(out/'plan.json',dict(scenes=plan['scenes'],seed=3407,configs=configs,strength=100,
        selection='Exploratory sensitivity after mixed primary outcomes; no per-scene winner selection',
        source_sha256={'Diff/sr_subspace.py':sha(ROOT/'Diff/sr_subspace.py'),
                       'run_sr_subspace_sensitivity.py':sha(Path(__file__))}))
    for key in plan['scenes']:
        job=next(j for j in jobs() if j['key']==key)
        gt,obs,hashes=prepare(job,args.device)
        path=primary/key/'original_raw_seed3407.npy'
        raw=torch.from_numpy(np.load(path)).to(args.device)
        prox=SpectralProximal(obs.msi,obs.low,obs.r,obs.sigma,obs.scale)
        for name,energy,gamma in configs:
            basis,info=estimate_spectral_basis(obs.low,energy)
            torch.cuda.synchronize(args.device);start=time.perf_counter()
            result,diagnostic=prox.solve_spectral(raw,100.,basis,gamma,100,1e-5)
            torch.cuda.synchronize(args.device);seconds=time.perf_counter()-start
            rec=result.clamp(0,1);arr=rec.cpu().numpy()
            with torch.no_grad():
                low,hr=prox.forward(rec)
                lm=float((low-obs.low).square().mean());hm=float((hr-obs.msi).square().mean())
            d=dict(**job,**hashes,variant=name,seed=3407,basis=info,solver=diagnostic,
                   seconds=seconds,raw_proposal_sha256=sha(path),LR_MSE=lm,MSI_MSE=hm,
                   **metrics(gt,arr))
            write_json(out/key/f'{name}_seed3407.json',d)
            np.save(out/key/f'{name}_seed3407.npy',arr)
            print(key,name,round(d['psnr'],4),info['rank'],diagnostic,flush=True)


if __name__=='__main__':main()
