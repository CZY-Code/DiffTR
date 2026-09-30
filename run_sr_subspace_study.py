"""Paired fixed-checkpoint tests of soft spectral terminal refinement."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scripts.sr_protocol import jobs
from baseline_sr.protocol import prepare, ROOT
from ablation_study.common import metrics, sha, array_sha, write_json, provenance
from Diff.networks import ScalarINR2D
from Diff.Diff4SR import GaussianDiffusion1D
from Diff.sr_guidance import sample_observation_guided
from Diff.sr_proximal import ObservationProximal
from Diff.sr_subspace import SpectralProximal, estimate_spectral_basis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/subspace')
    p.add_argument('--seeds',type=int,nargs='+',default=[3407,3408,3409])
    args=p.parse_args();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    job=next(j for j in jobs() if j['key']==args.key)
    gt,obs,hashes=prepare(job,args.device)
    base='table6_blur_wdc256_landsat8' if job['group']=='WDC' else 'table6_blur'
    folder=ROOT/'outputs/sr/DiffTR'/args.key
    saved=json.loads((folder/'experiment.json').read_text());cfg=saved['config']
    net=ScalarINR2D(out_dim=gt.shape[-1],**cfg['model'])
    diff=GaussianDiffusion1D(net,obs.msi.cpu(),obs.r.cpu(),**cfg['diffusion'])
    ck=torch.load(folder/'model-1.pt',map_location='cpu',weights_only=False)
    diff.load_state_dict(ck['model']);del ck
    diff.to(args.device).eval().requires_grad_(False)
    assert array_sha(diff.HR_MSI.cpu().numpy())==hashes['HR_MSI_sha256']
    assert array_sha(diff.Matrix_R.cpu().numpy())==hashes['response_sha256']
    out=args.output/args.key;out.mkdir(parents=True,exist_ok=True)
    source=provenance(['Diff/sr_subspace.py','Diff/sr_proximal.py','Diff/sr_guidance.py',
                       'Diff/networks.py','Diff/Diff4SR.py','run_sr_subspace_study.py',
                       'baseline_sr/protocol.py'])
    metadata=dict(**job,**hashes,checkpoint_sha256=sha(folder/'model-1.pt'),
                  source_sha256=source,device=args.device,tf32=False,
                  selection='Fixed thresholds, weights, iterations and final checkpoint; GT only for evaluation')
    torch.cuda.synchronize(args.device);start=time.perf_counter()
    basis,basis_info=estimate_spectral_basis(obs.low,.999)
    prox=SpectralProximal(obs.msi,obs.low,obs.r,obs.sigma,obs.scale)
    torch.cuda.synchronize(args.device);setup_seconds=time.perf_counter()-start
    write_json(out/'basis.json',dict(**basis_info,setup_seconds=setup_seconds,**hashes))
    def record(variant,seed,raw,seconds,sample_seconds,solver=None):
        rec=raw.detach().clamp(0,1);arr=rec.cpu().numpy()
        with torch.no_grad():
            lo,hi=prox.forward(rec)
            low_mse=float((lo-obs.low).square().mean());msi_mse=float((hi-obs.msi).square().mean())
        a,b=gt.astype(np.float64).reshape(-1,gt.shape[-1]),arr.astype(np.float64).reshape(-1,gt.shape[-1])
        norm=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);valid=norm>1e-12
        sam=float(np.degrees(np.arccos(np.clip((a[valid]*b[valid]).sum(1)/norm[valid],-1,1))).mean())
        result=dict(**metadata,variant=variant,seed=seed,**metrics(gt,arr),
                    SAM_deg=sam,LR_MSE=low_mse,MSI_MSE=msi_mse,
                    refinement_seconds=seconds,sampling_seconds=sample_seconds,
                    shared_setup_seconds=setup_seconds,solver=solver,basis=basis_info,
                    reconstruction_sha256=array_sha(arr))
        write_json(out/f'{variant}_seed{seed}.json',result)
        if seed==args.seeds[0]:np.save(out/f'{variant}_seed{seed}.npy',arr)
        print(args.key,variant,seed,'PSNR',result['psnr'],'SSIM',result['ssim'],
              'solver',solver,flush=True)
    for seed in args.seeds:
        torch.cuda.synchronize(args.device);start=time.perf_counter()
        raw=sample_observation_guided(diff,obs.low,obs.sigma,obs.scale,'original',
                                      guidance=15.,batch_size=2048,seed=seed)
        torch.cuda.synchronize(args.device);sampling=time.perf_counter()-start
        np.save(out/f'original_raw_seed{seed}.npy',raw.cpu().numpy())
        record('original',seed,raw,0.,sampling)
        for name,tau,gamma in [('dc10',10.,0.),('dc100',100.,0.),('soft100',100.,10.)]:
            torch.cuda.synchronize(args.device);start=time.perf_counter()
            if name=='dc10':
                z,info=prox.solve(raw,tau,20,1e-5)
            else:
                z,info=prox.solve_spectral(raw,tau,basis,gamma,100,1e-5)
            torch.cuda.synchronize(args.device);elapsed=time.perf_counter()-start
            record(name,seed,z,elapsed,sampling,info)
        if seed==args.seeds[0]:
            ref=obs.low.repeat_interleave(obs.scale,0).repeat_interleave(obs.scale,1)
            record('nearest',seed,ref,0.,0.)
            torch.cuda.synchronize(args.device);start=time.perf_counter()
            z,info=prox.solve_spectral(ref,100.,basis,10.,100,1e-5)
            torch.cuda.synchronize(args.device);elapsed=time.perf_counter()-start
            record('nearest_soft100',seed,z,elapsed,0.,info)


if __name__=='__main__':main()
