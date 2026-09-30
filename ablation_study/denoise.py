"""Fixed-budget, paired-seed component ablations using the repository's model/loss."""
import argparse
import time
import numpy as np
import scipy.io
import tifffile
import torch
import yaml
from ema_pytorch import EMA
from common import ROOT, seed_all, array_sha, metrics, write_json, provenance
from Diff.networks import ScalarINR3D
from Diff.Diff4ImgD import GaussianDiffusion1D
from utils.datautil import truncated_linear_stretch
from utils.noiseFun import add_noise

VARIANTS = ('full', 'no_condition', 'no_sparse', 'no_smooth', 'stop_gradient')


class ConditionControl(ScalarINR3D):
    """Constant factor condition; same allocated network and initialization."""
    def forward(self, val, ind, t, **kwargs):
        return super().forward(val, torch.zeros_like(ind), t, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', choices=('PaviaU', 'WDC'), required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--case', type=int, choices=range(1, 7), default=3)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--crop', type=int, default=128)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--col', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.steps < 1 or args.crop < 7 or args.batch_size < 1 or min(args.row, args.col) < 0:
        p.error('Invalid steps, crop, batch size, or origin')
    from pathlib import Path
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    source = 'data/PaviaU.mat' if args.scene == 'PaviaU' else 'data/dc.tif'
    full = (scipy.io.loadmat(ROOT / source)['paviaU'] if args.scene == 'PaviaU'
            else tifffile.imread(ROOT / source).transpose(1, 2, 0).astype(np.float32))
    full = truncated_linear_stretch(full).astype(np.float32)
    gt = np.ascontiguousarray(full[args.row:args.row+args.crop, args.col:args.col+args.crop])
    if gt.shape[:2] != (args.crop, args.crop):
        raise ValueError('Crop exceeds scene boundaries')
    seed_all(args.seed)
    noisy = add_noise(gt, args.case, deadline_protocol='lines').astype(np.float32)
    # Corruption generation is independent of model initialization and capacity.
    seed_all(args.seed)
    conf = yaml.full_load((ROOT / 'configs/ImgD_HSI_diff_conf.yaml').read_text())
    conf['model']['tensor_shape'] = list(gt.shape)
    conf['model']['h_dim'] = list(conf['model']['h_dim'])
    conf['diffusion'].update(sparse_enabled=args.variant != 'no_sparse',
                             smooth_weight=0.0 if args.variant == 'no_smooth' else 0.01,
                             smooth_stop_gradient=args.variant == 'stop_gradient')
    cls = ConditionControl if args.variant == 'no_condition' else ScalarINR3D
    model = cls(out_dim=1, **conf['model'])
    diff = GaussianDiffusion1D(model, **conf['diffusion']).to(device)
    ema = EMA(diff, beta=0.995, update_every=2).to(device)
    opt = torch.optim.Adam(diff.parameters(), lr=0.001, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    coords = torch.cartesian_prod(*(torch.arange(s, device=device) for s in gt.shape))
    values = torch.from_numpy(noisy).to(device).reshape(-1, 1, 1)
    history = []
    start = time.perf_counter()
    for step in range(args.steps):
        # Fixed RNG substream per update keeps atom/noise draws paired even when
        # removing the smooth branch changes the number of random draws.
        torch.manual_seed(args.seed * 100000 + step)
        ix = torch.randint(len(coords), (args.batch_size,), device=device)
        loss = diff(values[ix], coords[ix])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at update {step + 1}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff.parameters(), 2.0, norm_type=float('inf'))
        opt.step()
        scheduler.step()
        ema.update()
        if (step + 1) % 500 == 0 or step + 1 == args.steps:
            history.append(dict(step=step+1, loss=float(loss.detach())))
            print(args.scene, args.variant, args.seed, history[-1], flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    # Only the final EMA is evaluated. No clean-target checkpoint selection.
    seed_all(args.seed + 1000000)
    ema.ema_model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        chunks = [ema.ema_model.sample(c, batch_size=len(c), model_forward_kwargs={'sample': False}).cpu()
                  for c in coords.split(8192)]
    rec = torch.cat(chunks).numpy().reshape(gt.shape).clip(0, 1)
    result = dict(task='denoise', scene=args.scene, variant=args.variant, seed=args.seed,
                  case=args.case, shape=list(gt.shape), origin=[args.row, args.col],
                  steps=args.steps, batch_size=args.batch_size, config=conf,
                  optimizer=dict(lr=0.001, betas=[0.9, 0.99], scheduler='cosine', clip_inf=2.0),
                  ema=dict(beta=0.995, update_every=2, update_after_step=100),
                  selection='fixed_final_EMA', gt_sha256=array_sha(gt), noisy_sha256=array_sha(noisy),
                  reconstruction_sha256=array_sha(rec), observed=metrics(gt, noisy), **metrics(gt, rec),
                  train_seconds=train_seconds, sample_seconds=time.perf_counter()-start,
                  parameters=sum(x.numel() for x in model.parameters()), history=history,
                  torch_version=torch.__version__, device=str(device),
                  source_sha256=provenance([source, 'configs/ImgD_HSI_diff_conf.yaml',
                      'ablation_study/denoise.py', 'ablation_study/common.py', 'Diff/Diff4ImgD.py',
                      'Diff/networks.py', 'utils/noiseFun.py', 'utils/datautil.py']))
    write_json(args.output, result)
    np.save(Path(args.output).with_suffix('.npy'), rec)
    print({k: result[k] for k in ('scene', 'variant', 'seed', 'psnr', 'ssim', 'train_seconds')}, flush=True)


if __name__ == '__main__':
    main()
