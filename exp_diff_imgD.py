import os
import json
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
import argparse
import yaml
import cv2
import scipy.io
import tifffile
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity

from utils.noiseFun import add_noise
from utils.ob_dataV2 import tensor2data
from utils.datautil import load_grayscale_images_from_directory, truncated_linear_stretch
from Diff.Diff4ImgD import GaussianDiffusion1D, Trainer1D
from Diff.networks import ScalarINR2D, ScalarINR3D
dtype = torch.FloatTensor

def main():
    parser = argparse.ArgumentParser(description='DiffTR tensor denoising')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--case', type=int, choices=range(1, 7), default=1, help='Denoise case')
    parser.add_argument('--type', choices=['MSI', 'HSI'], default='MSI', help='legacy modality argument; group selects the default configuration')
    parser.add_argument('--name', type=str, default='beads', help='name')
    parser.add_argument('--group', choices=['CAVE', 'PaviaU', 'WDC'], default='CAVE', help='Table-V dataset group')
    parser.add_argument('--patch', type=int, default=0, help='zero-based PaviaU/WDC spatial patch')
    parser.add_argument('--dev', type=int, default=0, help='CUDA ID')
    parser.add_argument('--summary-path', type=str, default=None, help='write the final best metric as JSON')
    parser.add_argument('--save-checkpoints', action='store_true', help='save periodic model checkpoints')
    parser.add_argument('--save-images', action='store_true', help='save periodic preview images')
    parser.add_argument('--profile', choices=['corrected', 'legacy'], default='corrected')
    parser.add_argument('--config', help='override the dataset-specific YAML configuration')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--sparse-mode', choices=['auto', 'on', 'off'], default=None,
                        help='default follows YAML; auto disables separation in Gaussian Case 1')
    parser.add_argument('--sparse-threshold', type=float, help='threshold in normalized model units')
    parser.add_argument('--smooth-weight', type=float)
    parser.add_argument('--smooth-scope', choices=['spatial', 'all', 'legacy'])
    parser.add_argument('--smooth-std', type=float)
    parser.add_argument('--noise-protocol', choices=['lines', 'legacy'])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--experiment-tag', default='', help='optional label; configuration hash is always added')
    args = parser.parse_args()

    # Each experiment uses one GPU; avoid 24 CPU threads per process fighting
    # over vectorized indexing when several experiments share a host.
    torch.set_num_threads(1)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.dev))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # read config
    modality = 'HSI' if args.group != 'CAVE' and args.profile != 'legacy' else 'MSI'
    conf_path = args.config or f'./configs/ImgD_{modality}_diff_conf.yaml'

    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)
    if args.profile == 'legacy':
        conf.diffusion.update({'sparse_threshold': 0.04, 'smooth_weight': 0.1,
                               'smooth_scope': 'legacy', 'smooth_std': 1.0})
    # Clipped Gaussian observations benefited from robust residual handling in
    # the matched study. Keep it by default; auto/off remain explicit controls.
    sparse_mode = args.sparse_mode or ('on' if args.profile == 'legacy' or conf.diffusion.get('sparse_enabled', True) else 'off')
    conf.diffusion.sparse_enabled = sparse_mode == 'on' or (sparse_mode == 'auto' and args.case != 1)
    noise_protocol = args.noise_protocol or ('legacy' if args.profile == 'legacy' else 'lines')
    if args.rank is not None:
        conf.model.rank = args.rank
    for name in ('sparse_threshold', 'smooth_weight', 'smooth_scope', 'smooth_std'):
        value = getattr(args, name)
        if value is not None:
            conf.diffusion[name] = value
    for argument, field in (('steps', 'train_num_steps'), ('eval_every', 'save_and_sample_every'),
                            ('batch_size', 'train_batch_size')):
        value = getattr(args, argument)
        if value is not None:
            if value <= 0:
                parser.error(f'--{argument.replace("_", "-")} must be positive')
            conf.trainer[field] = value
    if conf.model.rank <= 0:
        parser.error('--rank must be positive')
    if min(conf.diffusion.sparse_threshold, conf.diffusion.smooth_weight, conf.diffusion.smooth_std) < 0:
        parser.error('threshold, weight and perturbation radius must be nonnegative')
    if args.eval_every is None:
        conf.trainer.save_and_sample_every = min(conf.trainer.save_and_sample_every, conf.trainer.train_num_steps)
    if conf.trainer.train_num_steps % conf.trainer.save_and_sample_every:
        parser.error('training steps must be a multiple of the evaluation interval (so the final step is evaluated)')
    if args.experiment_tag and any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.experiment_tag):
        parser.error('--experiment-tag may only contain letters, digits, underscores and hyphens')

    # read data
    # name, datatype = 'toys', 'MSI' #['balloons', 'beads', 'flowers', 'fruits']

    if args.group == 'CAVE':
        showChannel = [15, 25, 30]
        file_path = os.path.join('data/MSIs', args.name)
        MSI_ori_gt = load_grayscale_images_from_directory(file_path)
        MSI_gt = np.zeros((256, 256, 31), dtype=MSI_ori_gt.dtype)
        for i in range(MSI_ori_gt.shape[-1]):
            MSI_gt[:, :, i] = cv2.resize(MSI_ori_gt[:, :, i], (256, 256), interpolation=cv2.INTER_LINEAR)
        dataname = args.name
    elif args.group == 'PaviaU':
        if not 0 <= args.patch < 2:
            raise ValueError('PaviaU patch must be 0 or 1')
        full = scipy.io.loadmat('data/PaviaU.mat')['paviaU']
        full = truncated_linear_stretch(full)
        MSI_gt = full[args.patch * 305:(args.patch + 1) * 305, :, :]
        showChannel = [60, 30, 10]
        dataname = f'PaviaU_patch{args.patch}'
    elif args.group == 'WDC':
        if not 0 <= args.patch < 4:
            raise ValueError('WDC patch must be in 0..3')
        full = tifffile.imread('data/dc.tif').transpose(1, 2, 0).astype(np.float32)
        full = truncated_linear_stretch(full)
        MSI_gt = full[args.patch * 320:(args.patch + 1) * 320, :, :]
        showChannel = [100, 60, 30]
        dataname = f'WDC_patch{args.patch}'
    else:
        raise NotImplementedError

    MSI_gt = np.ascontiguousarray(MSI_gt, dtype=np.float32)
    conf.model.tensor_shape = tuple(MSI_gt.shape)
    resolved_config = OmegaConf.to_container(conf, resolve=True)
    resolved_config['trainer'].pop('results_folder', None)
    protocol = {'profile': args.profile, 'deadline_protocol': noise_protocol,
                'sparse_mode': sparse_mode, 'configuration': resolved_config,
                'selection': 'max_psnr_checkpoint', 'seed': args.seed}
    source_hash = hashlib.sha256()
    for source in ('exp_diff_imgD.py', 'Diff/Diff4ImgD.py', 'Diff/networks.py',
                   'utils/noiseFun.py', 'utils/ob_dataV2.py', 'utils/datautil.py'):
        source_hash.update(Path(source).read_bytes())
    protocol['source_hash'] = source_hash.hexdigest()
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    suffix = f'{args.experiment_tag}_' if args.experiment_tag else ''
    job_name = f'{args.group}_{dataname}_case{args.case}_seed{args.seed}_{suffix}{config_hash}'
    results_root = args.results_root or Path('results_imgD') / ('denoising_v2' if args.profile == 'corrected' else 'legacy_reruns')
    conf.trainer.results_folder = str(results_root / job_name)
    metrics_log_path = str(Path(conf.trainer.results_folder) / 'metrics.txt')
    if Path(metrics_log_path).exists() or (args.summary_path and Path(args.summary_path).exists()):
        raise FileExistsError('existing results will not be overwritten; use another experiment tag or summary path')
    Path(conf.trainer.results_folder).mkdir(parents=True, exist_ok=True)
    (Path(conf.trainer.results_folder) / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    Path(metrics_log_path).write_text('')
    X = torch.from_numpy(MSI_gt).type(dtype)
    MSI_gt_noise = add_noise(MSI_gt, args.case, deadline_protocol=noise_protocol)
    X_noise = torch.from_numpy(MSI_gt_noise).type(dtype)
    data_set = tensor2data(X_noise)
    model_Unet = ScalarINR3D(out_dim=conf.diffusion.channels, **conf.model)
    
    # ob_psnr = peak_signal_noise_ratio(MSI_gt, MSI_gt_noise, data_range=1.0)
    # ob_ssim = structural_similarity(MSI_gt, MSI_gt_noise, data_range=1.0, channel_axis=2)
    # ob_nrmse = normalized_root_mse(MSI_gt, MSI_gt_noise)
    # print('case:', args.case, args.name, 
    #       'OB_PSNR: {:.3f}, OB_SSIM: {:.3f}, OB_NRMSE: {:.3f}'.format(ob_psnr, ob_ssim, ob_nrmse))
    
    print('Model params: ', sum(param.numel() for param in model_Unet.parameters())/1e6, 'M')

    diffusion_model = GaussianDiffusion1D(model_Unet, **conf.diffusion)
    trainer = Trainer1D(diffusion_model, data_set['train'], data_set['test'], 
                        gt_tensor=X, denoising_case=args.case, 
                        datatype='MSI' if args.group == 'CAVE' else 'HSI', showChannel=showChannel, dataname=dataname,
                        save_checkpoints=args.save_checkpoints, save_images=args.save_images,
                        metrics_log_path=metrics_log_path,
                        **conf.trainer)
    trainer.train()

    summary = {
        'group': args.group, 'name': dataname, 'patch': args.patch,
        'case': args.case, 'seed': args.seed, 'rank': int(conf.model.rank),
        'protocol': protocol, 'configuration_hash': config_hash,
        **trainer.best_metric,
    }
    summary_path = Path(args.summary_path) if args.summary_path else Path(conf.trainer.results_folder) / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    with (results_root / 'result_log.jsonl').open('a') as file:
        file.write(json.dumps(summary) + '\n')

if __name__ == "__main__":
    main()
