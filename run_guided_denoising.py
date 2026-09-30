"""Matched denoising experiments; clean references are used only after fitting.

Examples:
  python run_guided_denoising.py --scene WDC --case 1 --variant likelihood \
      --device cuda:1 --output results_imgD/guided_study/example

Modes: legacy BCD baseline; deterministic MSE/Huber controls; censored-likelihood pilot; ordinary
multi-step diffusion fitted to observations (negative control); pilot followed
by pseudo-clean multi-step diffusion. No mode is a Noise2Self implementation.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.io
import tifffile
import torch
import yaml
from ema_pytorch import EMA
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

from Diff.Diff4ImgD import GaussianDiffusion1D
from Diff.networks import ScalarINR3D
from Diff.denoise_guidance import (BlindNoiseModel, detect_dead_columns,
    proximal_clean_guidance, sample_guided, spectral_basis)
from utils.datautil import truncated_linear_stretch
from utils.noiseFun import add_noise

ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def metrics(gt, x):
    return dict(psnr=float(peak_signal_noise_ratio(gt, x, data_range=1)),
                ssim=float(structural_similarity(gt, x, data_range=1, channel_axis=2)),
                nrmse=float(normalized_root_mse(gt, x)))


@torch.no_grad()
def predict_pilot(model, coords, shape, batch=32768):
    model.eval()
    values = []
    for ind in coords.split(batch):
        zero = torch.zeros((len(ind), 1, 1), device=ind.device)
        times = torch.zeros(len(ind), device=ind.device, dtype=torch.long)
        values.append(((model(zero, ind, times, sample=False)+1)/2).flatten())
    return torch.cat(values).reshape(shape).clamp(0, 1)


def fit(observed, args, output):
    """No GT argument: all fitting, calibration and guidance use observations."""
    device = torch.device(args.device)
    y = torch.as_tensor(observed, device=device)
    valid = detect_dead_columns(y)
    coords = torch.cartesian_prod(*(torch.arange(s, device=device) for s in y.shape))
    values = y.reshape(-1, 1, 1)
    mask = valid.flatten()
    seed_all(args.seed)
    conf = yaml.full_load((ROOT/'configs/ImgD_HSI_diff_conf.yaml').read_text())
    conf['model']['tensor_shape'] = list(y.shape)
    conf['model']['h_dim'] = list(conf['model']['h_dim'])
    conf['model']['rank'] = args.rank
    model = ScalarINR3D(out_dim=1, **conf['model']).to(device)
    diffusion_conf = copy.deepcopy(conf['diffusion'])
    if args.variant in ('noisy_multi', 'staged'):
        diffusion_conf.update(timesteps=args.diffusion_steps, sampling_timesteps=args.sample_steps)
    diffusion = GaussianDiffusion1D(model, **diffusion_conf).to(device)
    ema = EMA(diffusion, beta=.995, update_every=2).to(device)
    noise_model = BlindNoiseModel(y, valid).to(device)
    initial_sigma = noise_model.sigma.detach().cpu().tolist()
    is_likelihood = args.variant in ('likelihood', 'staged')
    is_pilot = args.variant in ('likelihood', 'staged', 'mse', 'huber')
    pilot_steps = (int(args.steps*args.pilot_fraction) if args.variant == 'staged'
                   else args.steps if is_pilot else 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99))
    noise_optimizer = torch.optim.Adam(noise_model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=pilot_steps or args.steps, eta_min=1e-6)
    history, pilot = [], None
    clock = time.perf_counter()
    for step in range(args.steps):
        # Every variant receives exactly the same coordinate substream.
        torch.manual_seed(args.seed*100000 + step)
        ix = torch.randint(len(coords), (args.batch_size,), device=device)
        ind, val = coords[ix], values[ix]
        optimizer.zero_grad(set_to_none=True)
        noise_optimizer.zero_grad(set_to_none=True)
        if is_pilot and step < pilot_steps:
            ts = torch.zeros(len(ix), device=device, dtype=torch.long)
            clean = ((model(torch.zeros_like(val), ind, ts)+1)/2).flatten()
            if args.variant == 'mse':
                point_loss = (clean-val.flatten()).square()
                loss = (point_loss*mask[ix]).sum()/mask[ix].sum().clamp_min(1)
            elif args.variant == 'huber':
                point_loss = torch.nn.functional.huber_loss(clean, val.flatten(),
                    reduction='none', delta=.02)
                loss = (point_loss*mask[ix]).sum()/mask[ix].sum().clamp_min(1)
            else:
                loss = noise_model(clean, val.flatten(), ind[:, 2], mask[ix])
            loss = loss + 10 * (clean-clean.clamp(0, 1)).square().mean()
        elif args.variant == 'baseline':
            loss = diffusion(val, ind)
        else:
            # min-SNR weighting, bounded at 5, avoids huge low-time gradients.
            target = pilot.flatten()[ix].reshape_as(val) if pilot is not None else val
            ts = torch.randint(diffusion.num_timesteps, (len(ix),), device=device)
            target = diffusion.normalize(target)
            per_item = diffusion.p_losses(target, ind, ts, return_reduced_loss=False)
            snr = diffusion.loss_weight[ts].reshape(-1, 1, 1)
            loss = (per_item * snr.clamp(max=5)/snr).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {step+1}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., norm_type=float('inf'))
        optimizer.step()
        # Warmup avoids explaining initial reconstruction error as sensor noise.
        if is_likelihood and min(500, pilot_steps//4) <= step < pilot_steps:
            noise_optimizer.step()
        scheduler.step()
        ema.update()
        if (step+1) % 500 == 0 or step+1 in (pilot_steps, args.steps):
            entry = dict(step=step+1, loss=float(loss.detach()),
                         sigma_mean=float(noise_model.sigma.mean()),
                         outlier_mean=float(noise_model.probability.mean()),
                         seconds=time.perf_counter()-clock)
            history.append(entry)
            print(args.scene, args.case, args.variant, entry, flush=True)
            write_json(output/'progress.json', history)
        if args.variant == 'staged' and step+1 == pilot_steps:
            pilot = predict_pilot(ema.ema_model.model, coords, y.shape).detach()
            model.load_state_dict(ema.ema_model.model.state_dict())
            # New EMA and optimizer start the second, fixed-target phase.
            ema = EMA(diffusion, beta=.995, update_every=2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=args.steps-pilot_steps, eta_min=1e-6)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-clock
    ema.ema_model.eval()
    torch.save(dict(model=ema.ema_model.state_dict(), noise=noise_model.state_dict(),
                    config=conf, diffusion_config=diffusion_conf, arguments=vars(args)),
               output/'final.pt')
    reconstructions = {}
    sample_clock = time.perf_counter()
    if args.variant in ('likelihood', 'mse', 'huber'):
        pilot = predict_pilot(ema.ema_model.model, coords, y.shape).detach()
        reconstructions['pilot'] = pilot
        basis = spectral_basis(pilot, args.spectral_rank)
        # This is a post-fit control, not a diffusion result.
        for strength in args.guidance:
            if strength:
                refined = proximal_clean_guidance(pilot, y, valid, noise_model.sigma,
                    noise_model.probability, basis, strength=strength, iterations=args.prox_iterations)
                reconstructions[f'prox_{strength:g}'] = refined.clamp(0, 1)
    elif args.variant == 'baseline':
        seed_all(args.seed+1000000)
        with torch.no_grad():
            rec = torch.cat([ema.ema_model.sample(ind, batch_size=len(ind),
                model_forward_kwargs={'sample': False}).flatten() for ind in coords.split(32768)])
        reconstructions['baseline'] = rec.reshape_as(y).clamp(0, 1)
    else:
        # Pilot basis adds no information from the clean evaluation reference.
        basis = spectral_basis(pilot if pilot is not None else y, args.spectral_rank)
        if pilot is not None:
            reconstructions['pilot'] = pilot
        for strength in args.guidance:
            reconstructions[f'ddim_{strength:g}'] = sample_guided(ema.ema_model, coords,
                y, valid, noise_model.sigma, noise_model.probability, basis,
                steps=args.sample_steps, strength=strength, iterations=args.prox_iterations,
                seed=args.seed+1000000, pixel_batch=args.pixel_batch)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    diagnostics = dict(config=conf, diffusion_config=diffusion_conf, history=history,
        train_seconds=train_seconds, sample_seconds=time.perf_counter()-sample_clock,
        parameters=sum(p.numel() for p in model.parameters()),
        optimized_noise_parameters=sum(p.numel() for p in noise_model.parameters()) if is_likelihood else 0,
        missing_fraction=float((~valid).float().mean()), sigma_initial=initial_sigma,
        sigma_final=noise_model.sigma.detach().cpu().tolist(),
        outlier_probability=noise_model.probability.detach().cpu().tolist(),
        noise_calibration=('observation_only_joint_censored_mixture_MLE' if is_likelihood
                           else 'observation_only_Haar_MAD_initialization; not fitted'),
        selection='fixed_final_EMA; guidance grid reported in full; no GT selection')
    return {name: rec.detach().cpu().numpy() for name, rec in reconstructions.items()}, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['WDC', 'PaviaU'], required=True)
    parser.add_argument('--case', type=int, choices=range(1, 7), required=True)
    parser.add_argument('--variant', choices=['baseline', 'mse', 'huber', 'likelihood', 'noisy_multi', 'staged'], required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--crop', type=int, default=96, help='square crop; 0 uses a complete table-5 patch')
    parser.add_argument('--patch', type=int, default=0)
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--col', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--spectral-rank', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--pilot-fraction', type=float, default=.75)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--sample-steps', type=int, default=20)
    parser.add_argument('--pixel-batch', type=int, default=128)
    parser.add_argument('--guidance', type=float, nargs='+', default=[0., 1., 4.])
    parser.add_argument('--prox-iterations', type=int, default=10)
    parser.add_argument('--noise-protocol', choices=['legacy', 'lines'], default='legacy')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 2 or args.batch_size < 1 or args.rank < 1 or args.lr <= 0
            or not 0 < args.pilot_fraction < 1 or not 1 <= args.sample_steps <= args.diffusion_steps
            or (args.crop != 0 and args.crop < 7) or min(args.row, args.col, args.patch) < 0
            or args.spectral_rank < 1 or args.prox_iterations < 0 or min(args.guidance) < 0):
        parser.error('invalid experiment parameters')
    if args.variant == 'staged' and not 1 <= int(args.steps*args.pilot_fraction) < args.steps:
        parser.error('both stages need at least one update')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source_files = ['run_guided_denoising.py', 'Diff/denoise_guidance.py', 'Diff/Diff4ImgD.py',
        'Diff/networks.py', 'Diff/modules.py', 'utils/datautil.py', 'utils/noiseFun.py',
        'configs/ImgD_HSI_diff_conf.yaml']
    hashes = {}
    for name in source_files:
        contents = (ROOT/name).read_bytes()
        hashes[name] = hashlib.sha256(contents).hexdigest()
        snapshot = output/'source'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(contents)
    parameters = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}
    write_json(output/'arguments.json', parameters)
    if args.scene == 'PaviaU':
        full = scipy.io.loadmat(ROOT/'data/PaviaU.mat')['paviaU']
        height, patches = 305, 2
    else:
        full = tifffile.imread(ROOT/'data/dc.tif').transpose(1, 2, 0).astype(np.float32)
        height, patches = 320, 4
    if args.patch >= patches:
        raise ValueError('invalid patch')
    full = truncated_linear_stretch(full)
    patch = full[height*args.patch:height*(args.patch+1)]
    gt = patch if args.crop == 0 else patch[args.row:args.row+args.crop, args.col:args.col+args.crop]
    gt = np.ascontiguousarray(gt, dtype=np.float32)
    if args.crop and gt.shape[:2] != (args.crop, args.crop):
        raise ValueError('crop outside patch')
    seed_all(args.seed)
    observed = add_noise(gt, args.case, deadline_protocol=args.noise_protocol).astype(np.float32)
    # All clean-reference evaluation is deliberately below the fit() boundary.
    reconstructions, diagnostics = fit(observed, args, output)
    result = dict(arguments=parameters, shape=list(gt.shape), gt_sha256=array_sha(gt),
        noisy_sha256=array_sha(observed), source_sha256=hashes, torch_version=torch.__version__,
        observed=metrics(gt, observed), diagnostics=diagnostics, reconstructions={})
    for name, rec in reconstructions.items():
        np.save(output/f'{name}.npy', rec)
        result['reconstructions'][name] = dict(**metrics(gt, rec), sha256=array_sha(rec))
    write_json(output/'summary.json', result)
    print(json.dumps(result['reconstructions'], indent=2), flush=True)


if __name__ == '__main__':
    main()
