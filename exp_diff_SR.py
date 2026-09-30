"""Scene-specific DiffTR HSI fusion experiments (legacy or matched paper protocol)."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import tifffile
import torch
import yaml
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity

from utils.ob_dataV2 import tensor2data_2Dtube
from utils.datautil import load_grayscale_images_from_directory, truncated_linear_stretch
from Diff.Diff4SR import GaussianDiffusion1D, Trainer1D
from Diff.networks import ScalarINR2D


def signal_noise_ratio(corrupted, original, eps=1e-8):
    noise_power = np.sum(np.square(original - corrupted), dtype=np.float64)
    if noise_power < eps:
        return float('inf')
    return float(10 * np.log10(np.sum(np.square(original), dtype=np.float64) / noise_power))


def hrhsit_to_lrhsit(X, gaussian_kernel_size=7, sigma=2, downsample_factor=4, return_snr=False):
    """Replicate-boundary Gaussian blur followed by block-average downsampling."""
    if X.ndim != 3 or X.shape[0] % downsample_factor or X.shape[1] % downsample_factor:
        raise ValueError('Input must be HWC with spatial dimensions divisible by DS')
    blurred = np.stack([
        cv2.GaussianBlur(X[..., c].astype(np.float32), (gaussian_kernel_size,) * 2,
                         sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        for c in range(X.shape[-1])], axis=-1)
    blur_snr = signal_noise_ratio(blurred, X)
    print('After Gaussian Blur SNR:', blur_snr, flush=True)
    if downsample_factor > 1:
        h, w, c = X.shape
        blurred = blurred.reshape(h // downsample_factor, downsample_factor,
                                  w // downsample_factor, downsample_factor, c).mean(axis=(1, 3))
    result = blurred.astype(np.float32)
    return (result, blur_snr) if return_snr else result


def add_measurement_noise(clean, snr_db, rng):
    """Gaussian-generated noise rescaled to the requested whole-cube SNR; no clipping."""
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    signal_power = np.mean(np.square(clean), dtype=np.float64)
    noise_power = np.mean(np.square(noise), dtype=np.float64)
    noise *= np.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power))
    noisy = clean + noise
    return noisy.astype(np.float32), signal_noise_ratio(noisy, clean)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_scene(args):
    if args.name in ('beers', 'clay', 'flowers', 'balloons'):
        if args.patch != 0:
            raise ValueError('CAVE uses whole images, patch must be 0')
        gt = load_grayscale_images_from_directory(os.path.join('data/MSIs', args.name))
        response = sio.loadmat('response_coefficient.mat')['A']
        return gt.astype(np.float32), response.astype(np.float32), [30, 16, 5], 'CAVE', [0, 0], 'response_coefficient.mat:A'
    if args.name == 'PaviaU':
        full = truncated_linear_stretch(sio.loadmat('data/PaviaU.mat')['paviaU']).astype(np.float32)
        if args.protocol == 'legacy':
            gt, origin = full[:256, :256], [0, 0]
        else:
            if args.patch not in (0, 1):
                raise ValueError('PaviaU patch must be 0 or 1')
            # Original cube is only 610 rows: the two 320-row patches overlap by 30 rows.
            row = (0, full.shape[0] - 320)[args.patch]
            gt, origin = full[row:row + 320, :320], [row, 0]
        response = sio.loadmat('IKONOS_Spectral_Responses.mat')['Spectral_Responses_Matrix'][:, 4:107]
        response = response / response.sum(axis=1, keepdims=True)
        return gt, response.astype(np.float32), [60, 30, 10], 'PaviaU', origin, 'IKONOS_Spectral_Responses.mat:columns[4:107],row-normalized'
    if args.name in ('WDC', 'DCMall'):
        if args.patch not in range(4):
            raise ValueError('WDC patch must be in 0..3')
        full = truncated_linear_stretch(tifffile.imread('data/dc.tif').transpose(1, 2, 0).astype(np.float32)).astype(np.float32)
        row = args.patch * 320
        gt, origin = full[row:row + 320, :256], [row, 0]
        if args.response_path:
            response = np.load(args.response_path) if args.response_path.endswith('.npy') else sio.loadmat(args.response_path)[args.response_key]
            if getattr(args, 'response_transpose', False):
                response = response.T
            description = f'{args.response_path}:{args.response_key}' + (';transposed' if getattr(args, 'response_transpose', False) else '')
        elif args.synthetic_wdc_response:
            template = sio.loadmat('IKONOS_Spectral_Responses.mat')['Spectral_Responses_Matrix']
            response = np.stack([np.interp(np.linspace(0, 1, 191), np.linspace(0, 1, 138), r) for r in template])
            response /= response.sum(axis=1, keepdims=True)
            description = 'synthetic:IKONOS-template interpolated over normalized band index 138->191; no physical wavelength calibration'
        else:
            raise ValueError('WDC requires --response-path or explicit --synthetic-wdc-response')
        if response.ndim != 2 or response.shape[1] != 191:
            raise ValueError('WDC response must have shape (MSI_channels, 191)')
        return gt, response.astype(np.float32), [100, 60, 30], 'WDC', origin, description
    raise ValueError(f'Unsupported scene: {args.name}')


def main():
    parser = argparse.ArgumentParser(description='DiffTR hyperspectral super-resolution')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--DS', type=int, choices=(4, 8, 16), default=16)
    parser.add_argument('--sigma', type=float, default=None, help='Gaussian blur sigma; defaults to 0.5 legacy/blur or 2 paper')
    parser.add_argument('--type', choices=('MSI',), default='MSI')
    parser.add_argument('--name', default='balloons')
    parser.add_argument('--dev', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--protocol', choices=('legacy', 'blur', 'paper'), default='legacy',
                        help='blur: original degradation with table patches, no measurement noise; paper: archived AWGN protocol')
    parser.add_argument('--snr', type=float, default=30, help='Paper protocol: realized AWGN SNR for both LR-HSI and HR-MSI')
    parser.add_argument('--patch', type=int, default=0)
    parser.add_argument('--response-path')
    parser.add_argument('--response-key', default='R')
    parser.add_argument('--response-transpose', action='store_true', help='Transpose the matrix stored in the response file')
    parser.add_argument('--synthetic-wdc-response', action='store_true')
    parser.add_argument('--results-folder', help='Unique checkpoint directory; use ./results explicitly for archived legacy weights')
    parser.add_argument('--summary-path')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--latent-dim', type=int, default=None, help='Optional capacity diagnostic; default follows YAML')
    parser.add_argument('--test-batch-size', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--guidance-scale', type=float, default=15.0)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--sampling-steps', type=int, default=None)
    parser.add_argument('--reconstruction-path', help='Save full float32 HSI for refinement experiments')
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.dev))
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.sigma = args.sigma if args.sigma is not None else (2.0 if args.protocol == 'paper' else 0.5)
    gt, response, show_channels, group, origin, response_description = load_scene(args)
    conf_path = 'configs/SR_MSI_CAVE_diff_conf.yaml' if group == 'CAVE' else 'configs/SR_MSI_PaviaU_diff_conf.yaml'
    with open(conf_path) as f:
        conf = OmegaConf.create(yaml.full_load(f))
    conf.model.tensor_shape = list(gt.shape[:2])
    conf.diffusion.channels = gt.shape[-1]
    if args.latent_dim is not None:
        conf.model.latent_dim = args.latent_dim
    if args.steps is not None:
        conf.trainer.train_num_steps = args.steps
    # One fixed final checkpoint, never selected with HR-HSI ground truth.
    conf.trainer.save_and_sample_every = conf.trainer.train_num_steps
    if args.sampling_steps is not None:
        conf.diffusion.sampling_timesteps = args.sampling_steps
    lr_clean, hsi_blur_snr = hrhsit_to_lrhsit(gt, sigma=args.sigma, downsample_factor=args.DS, return_snr=True)
    hr_clean = gt @ response.T
    if args.protocol == 'paper':
        rng = np.random.default_rng(args.seed)
        lr_low, lr_snr = add_measurement_noise(lr_clean, args.snr, rng)
        hr_msi, hr_snr = add_measurement_noise(hr_clean, args.snr, rng)
        msi_blur_snr = None
    else:
        lr_low, lr_snr = lr_clean, None
        hr_msi, msi_blur_snr = hrhsit_to_lrhsit(hr_clean, sigma=args.sigma, downsample_factor=1, return_snr=True)
        hr_snr = None
    lr_up = np.repeat(np.repeat(lr_low, args.DS, axis=0), args.DS, axis=1)
    degradation_key = f'snr{args.snr:g}' if args.protocol == 'paper' else f'sigma{args.sigma:g}'
    key = f'{group}_{args.name}_patch{args.patch}_DS{args.DS}_{degradation_key}_{args.protocol}'
    conf.trainer.results_folder = args.results_folder or f'results_SR/{key}'
    Path(conf.trainer.results_folder).mkdir(parents=True, exist_ok=True)
    data = tensor2data_2Dtube(torch.from_numpy(lr_up))
    model = ScalarINR2D(out_dim=conf.diffusion.channels, **conf.model)
    print('Model params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M', flush=True)
    diffusion = GaussianDiffusion1D(model, torch.from_numpy(hr_msi), torch.from_numpy(response),
                                     guidance_scale=args.guidance_scale, **conf.diffusion)
    trainer = Trainer1D(diffusion, data['train'], data['test'], gt_tensor=torch.from_numpy(gt),
                        down_scale=args.DS, is_tube=True, datatype=args.type, showChannel=show_channels,
                        dataname=args.name if args.protocol == 'legacy' else key,
                        test_batch_size=args.test_batch_size, num_workers=args.num_workers,
                        use_ema=args.use_ema, repeats=args.repeats, **conf.trainer)
    metadata = {
        'group': group, 'name': args.name, 'patch': args.patch, 'origin': origin,
        'shape': list(gt.shape), 'DS': args.DS, 'seed': args.seed, 'protocol': args.protocol,
        'sigma': args.sigma, 'kernel_size': 7, 'boundary': 'replicate', 'downsampling': 'block_mean',
        'prior_upsampling': 'nearest', 'response': response_description, 'response_sha256': array_hash(response),
        'response_shape': list(response.shape),
        'response_path': args.response_path, 'response_key': args.response_key,
        'response_transpose': args.response_transpose,
        'response_file_sha256': hashlib.sha256(Path(args.response_path).read_bytes()).hexdigest() if args.response_path else None,
        'LR_HSI_SNR': lr_snr, 'HR_MSI_SNR': hr_snr, 'nominal_snr': args.snr if args.protocol == 'paper' else None,
        'HSI_blur_SNR': hsi_blur_snr, 'MSI_blur_SNR': msi_blur_snr,
        'SNR_semantics': 'measurement_noise' if args.protocol == 'paper' else 'blur_error_before_downsampling',
        'LR_HSI_sha256': array_hash(lr_low), 'HR_MSI_sha256': array_hash(hr_msi), 'gt_sha256': array_hash(gt),
        'use_ema': args.use_ema, 'repeats': args.repeats, 'guidance_scale': args.guidance_scale,
        'test_batch_size': args.test_batch_size, 'config': OmegaConf.to_container(conf),
        'LR_HSI_metrics': {'psnr': float(peak_signal_noise_ratio(gt, lr_up, data_range=1)),
                           'ssim': float(structural_similarity(gt, lr_up, data_range=1, channel_axis=2)),
                           'nrmse': float(normalized_root_mse(gt, lr_up))},
    }
    manifest = Path(conf.trainer.results_folder) / 'experiment.json'
    if args.train:
        if (Path(conf.trainer.results_folder) / 'model-1.pt').exists():
            raise FileExistsError('Checkpoint already exists; use test mode or a new results folder')
        manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        trainer.train()
    else:
        if manifest.exists() or args.protocol != 'legacy':
            saved = json.loads(manifest.read_text(encoding='utf-8'))
            for field in ('group', 'name', 'patch', 'origin', 'shape', 'DS', 'seed', 'protocol', 'sigma',
                          'LR_HSI_sha256', 'HR_MSI_sha256', 'response_sha256', 'gt_sha256'):
                if metadata[field] != saved[field]:
                    raise ValueError(f'Checkpoint experiment mismatch: {field}')
        trainer.load('1', verify_observations=manifest.exists() or args.protocol != 'legacy')
        trainer.test()
        metadata['checkpoint_manifest_verified'] = manifest.exists()
        metadata['loaded_HR_MSI_sha256'] = array_hash(trainer.accelerator.unwrap_model(trainer.model).HR_MSI.cpu().numpy())
    if args.reconstruction_path:
        path = Path(args.reconstruction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, trainer.final_rec)
    metadata.update(trainer.last_metric)
    metadata['steps'] = trainer.step
    if args.summary_path:
        path = Path(args.summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with open('SR_result_log.txt', 'a', encoding='utf-8') as f:
        f.write(json.dumps({k: metadata[k] for k in ('name', 'patch', 'DS', 'protocol', 'sigma', 'nominal_snr', 'HSI_blur_SNR', 'MSI_blur_SNR', 'steps', 'psnr', 'ssim', 'nrmse')}) + '\n')


if __name__ == '__main__':
    main()
