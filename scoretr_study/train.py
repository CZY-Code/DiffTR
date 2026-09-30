"""ScoreTR Table V: full HSI patches, legacy corruption, official MSI settings."""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE/'vendor'))
import numpy as np
import scipy.io
import tifffile
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from utils.datautil import truncated_linear_stretch
from utils.noiseFun import add_noise
from grid import BoundedScoreMatching

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def reference(group,patch):
    if group=='WDC':
        full=tifffile.imread(ROOT/'data/dc.tif').transpose(1,2,0).astype(np.float32); h=320; count=4
    else:
        full=scipy.io.loadmat(ROOT/'data/PaviaU.mat')['paviaU']; h=305; count=2
    assert 0<=patch<count
    return np.ascontiguousarray(truncated_linear_stretch(full)[patch*h:(patch+1)*h],dtype=np.float32)

def coords(ix,shape):
    return torch.stack([ix//(shape[1]*shape[2]),ix//shape[2]%shape[1],ix%shape[2]],-1)

def save_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['WDC','PaviaU'],required=True)
    p.add_argument('--patch',type=int,required=True)
    p.add_argument('--case',type=int,choices=range(1,6),required=True)
    p.add_argument('--steps',type=int,default=0,help='0: official 100 full epochs; positive: fixed update budget')
    args=p.parse_args();torch.set_num_threads(1)
    out=HERE/'results'/('official100' if args.steps==0 else f'steps{args.steps}')/f'{args.group}_patch{args.patch}_case{args.case}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): return
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407)
    gt=reference(args.group,args.patch)
    noisy=add_noise(gt,args.case,deadline_protocol='legacy').astype(np.float32)
    cfg=yaml.full_load((HERE/'config.yaml').read_text())
    cfg=json.loads(json.dumps(cfg))
    model=BoundedScoreMatching(tensor_shape=list(gt.shape),dtype=torch.FloatTensor,device='cuda:0',**cfg['model']).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['train']['lr'],weight_decay=0.)
    batch=cfg['train']['batch_size']; epoch_steps=math.ceil(gt.size/batch)
    total=args.steps or epoch_steps*cfg['train']['epoch']
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps or cfg['train']['epoch'],eta_min=1e-6)
    source_files=[HERE/'train.py',HERE/'grid.py',HERE/'config.yaml',ROOT/'utils/noiseFun.py',ROOT/'utils/datautil.py']+sorted((HERE/'vendor').rglob('*.py'))
    manifest=dict(**vars(args),seed=3407,shape=list(gt.shape),batch_size=batch,total_steps=total,
        config=cfg,noise_protocol='legacy',gt_sha256=digest(gt),noisy_sha256=digest(noisy),
        selection='max_psnr_checkpoint',source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())==manifest, 'Existing run protocol/source mismatch'
    save_json(out/'manifest.json',manifest)
    values=torch.from_numpy(noisy).flatten().cuda()
    start_step=0;history=[];best=None;elapsed_before=0.;permutation=None
    if (out/'resume.pt').exists():
        state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        start_step=state['step'];history=state['history'];best=state['best'];elapsed_before=state['elapsed']
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        permutation=state['permutation'].cuda() if state['permutation'] is not None else None
    start=time.monotonic()
    def checkpoint(step):
        state=dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            history=history,best=best,elapsed=elapsed_before+time.monotonic()-start,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),permutation=permutation)
        torch.save(state,out/'resume.tmp');(out/'resume.tmp').replace(out/'resume.pt')
    for step in range(start_step,total):
        offset=step%epoch_steps
        if offset==0: permutation=torch.randperm(gt.size,device='cuda')
        ix=permutation[offset*batch:min((offset+1)*batch,gt.size)]
        model.train();opt.zero_grad(set_to_none=True)
        loss=model.anneal_dsm_Sparse(coords(ix,gt.shape),values[ix],step,total)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward();opt.step()
        epoch_end=offset==epoch_steps-1
        if args.steps or epoch_end: sched.step()
        evaluate=((step+1)%1000==0 if args.steps else epoch_end and (step//epoch_steps)%20==0) or step+1==total
        if evaluate:
            model.eval();rec=np.empty(gt.size,dtype=np.float32)
            with torch.no_grad():
                for first in range(0,gt.size,batch):
                    ix=torch.arange(first,min(first+batch,gt.size),device='cuda')
                    rec[first:first+len(ix)]=model.predict(coords(ix,gt.shape),x_range=[0.,1.],epsilon=1/255).flatten().cpu().numpy()
            rec=rec.reshape(gt.shape).clip(0,1)
            metrics=dict(step=step+1,epoch=(step+1)/epoch_steps,
                psnr=float(peak_signal_noise_ratio(gt,rec,data_range=1)),
                ssim=float(structural_similarity(gt,rec,channel_axis=2,data_range=1)),
                nrmse=float(normalized_root_mse(gt,rec)))
            history.append(metrics)
            if best is None or metrics['psnr']>best['psnr']:
                best=metrics.copy();np.save(out/'best_reconstruction.npy',rec)
                torch.save(model.state_dict(),out/'best_model.pt')
            print(json.dumps(metrics),flush=True)
        if (step+1)%100==0 or evaluate:
            progress=dict(step=step+1,total_steps=total,loss=float(loss.detach()),elapsed=elapsed_before+time.monotonic()-start,best=best)
            save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if (step+1)%1000==0 or evaluate: checkpoint(step+1)
    save_json(out/'result.json',dict(**manifest,best=best,history=history,elapsed=elapsed_before+time.monotonic()-start,
        reconstruction_sha256=digest(np.load(out/'best_reconstruction.npy'))))

if __name__=='__main__':main()
