"""Train a tiny CPU SR model and verify observation-guided reconstruction."""
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import numpy as np
    import torch
    from Diff.networks import ScalarINR2D
    from Diff.Diff4SR import GaussianDiffusion1D,Trainer1D
    from Diff.sr_guidance import spatial_forward
    from Diff.sr_reconstruction import reconstruct_sr
    from utils.ob_dataV2 import tensor2data_2Dtube
    torch.set_num_threads(1);torch.manual_seed(3407)
    gt=torch.rand(8,8,3);r=torch.tensor([[.6,.3,.1],[.1,.3,.6]])
    low=spatial_forward(gt,.5,2);msi=spatial_forward(gt@r.T,.5)
    targets=low.repeat_interleave(2,0).repeat_interleave(2,1)
    ds=tensor2data_2Dtube(targets)
    model=ScalarINR2D(tensor_shape=[8,8],out_dim=3,rank=4,h_dim=[16,16],
        latent_dim=8,posdim=8,x_emb_size=8,t_emb_size=8,act='silu',dropout=0.,
        sigma_func='softplus',pooling_method='cat',skip_connection=True,rff_alpha=4.,rff_sigma=4.)
    diff=GaussianDiffusion1D(model,msi,r,seq_length=1,channels=3,
        timesteps=3,sampling_timesteps=3,objective='pred_x0',self_condition=False,channel_first=False)
    with tempfile.TemporaryDirectory(prefix='difftr-smoke-') as folder:
        trainer=Trainer1D(diff,ds['train'],ds['test'],gt_tensor=gt,down_scale=2,is_tube=True,
            train_num_steps=2,train_batch_size=16,test_batch_size=32,num_workers=0,
            save_and_sample_every=2,results_folder=folder,train_lr=.001)
        trainer.test=lambda:None
        trainer.train()
        trained=trainer.accelerator.unwrap_model(trainer.model).eval().requires_grad_(False)
        rec,info=reconstruct_sr(trained,low.to(trainer.device),.5,2,batch_size=32)
        assert tuple(rec.shape)==(8,8,3) and torch.isfinite(rec).all()
        assert info['solver']['converged']
        assert (Path(folder)/'model-1.pt').exists()
        print('PASS: two training updates, saved checkpoint, Soft1000 inference; device',trainer.device)

if __name__=='__main__':main()
