"""Verify archived observations and ablate DDIM steps/guidance at fixed weights."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT, seed_all, array_sha, metrics, write_json, provenance, sha
from exp_diff_SR import load_scene, hrhsit_to_lrhsit
from Diff.networks import ScalarINR2D
from Diff.Diff4SR import GaussianDiffusion1D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seeds', type=int, nargs='+', default=[3407, 3408, 3409])
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    folder = ROOT / args.checkpoint_dir
    saved = json.loads((folder / 'experiment.json').read_text())
    if saved['protocol'] != 'blur':
        raise ValueError('This runner reproduces the blur protocol only')
    scene = SimpleNamespace(name=saved['name'], patch=saved['patch'], protocol='blur',
                            response_path=saved.get('response_path'), response_key=saved.get('response_key', 'R'),
                            response_transpose=saved.get('response_transpose', False), synthetic_wdc_response=False)
    gt, response, *_ = load_scene(scene)
    low = hrhsit_to_lrhsit(gt, sigma=saved['sigma'], downsample_factor=saved['DS'])
    hr = hrhsit_to_lrhsit(gt @ response.T, sigma=saved['sigma'], downsample_factor=1)
    for key, value in [('gt_sha256', gt), ('LR_HSI_sha256', low), ('HR_MSI_sha256', hr), ('response_sha256', response)]:
        if array_sha(value) != saved[key]:
            raise ValueError(f'Observation hash mismatch: {key}')
    cfg = saved['config']
    model = ScalarINR2D(out_dim=gt.shape[-1], **cfg['model'])
    diff = GaussianDiffusion1D(model, torch.from_numpy(hr), torch.from_numpy(response), **cfg['diffusion'])
    checkpoint = torch.load(folder / 'model-1.pt', map_location='cpu', weights_only=False)
    for key, value in [('HR_MSI', hr), ('Matrix_R', response)]:
        if not np.array_equal(checkpoint['model'][key].numpy(), value):
            raise ValueError(f'Checkpoint buffer mismatch: {key}')
    diff.load_state_dict(checkpoint['model'], strict=True)
    diff.to(args.device).eval()
    # Model weights are fixed, but sampling needs derivatives with respect to x_t.
    for param in diff.parameters():
        param.requires_grad_(False)
    coords = torch.cartesian_prod(torch.arange(gt.shape[0], device=args.device),
                                  torch.arange(gt.shape[1], device=args.device))
    checkpoint_hash = sha(folder / 'model-1.pt')
    src = provenance(['ablation_study/superres.py', 'ablation_study/common.py',
                      'Diff/Diff4SR.py', 'Diff/networks.py', 'exp_diff_SR.py'])
    # Calling ddim_sample explicitly also at K=40 keeps sampler family fixed.
    variants = [('steps_5', 5, 15.0), ('steps_10', 10, 15.0), ('steps_20', 20, 15.0),
                ('full', 40, 15.0), ('guidance_0', 40, 0.0), ('guidance_1', 40, 1.0),
                ('guidance_5', 40, 5.0)]
    for seed in args.seeds:
        for name, steps, guidance in variants:
            output = Path(args.output_dir) / f"sr_{saved['name']}_{name}_{seed}.json"
            if output.exists():
                raise FileExistsError(output)
            seed_all(seed)
            diff.sampling_timesteps = steps
            diff.guidance_scale = guidance
            # Warmup is excluded and reseeded; each batch has its own noise stream.
            diff.ddim_sample(coords[:8], (8, 1, gt.shape[-1])).detach()
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            chunks = []
            for batch_index, c in enumerate(coords.split(args.batch_size)):
                seed_all(seed * 1000 + batch_index)
                chunks.append(diff.ddim_sample(c, (len(c), 1, gt.shape[-1]),
                              model_forward_kwargs={'sample': False}).detach().cpu())
            seconds = time.perf_counter() - start
            rec = torch.cat(chunks).numpy().reshape(gt.shape).clip(0, 1)
            result = dict(task='superres', scene=saved['name'], variant=name, seed=seed,
                          sampling_steps=steps, guidance_scale=guidance, sampler='DDIM', eta=0,
                          batch_size=args.batch_size, shape=list(gt.shape), sample_seconds=seconds,
                          checkpoint=str(folder.relative_to(ROOT)), checkpoint_sha256=checkpoint_hash,
                          checkpoint_step=int(checkpoint['step']), selection='fixed_archived_raw_weights',
                          observation_hashes_verified=True, checkpoint_buffers_verified=True,
                          experiment=saved, source_sha256=src, torch_version=torch.__version__,
                          reconstruction_sha256=array_sha(rec), **metrics(gt, rec))
            write_json(output, result)
            print(saved['name'], name, seed, result['psnr'], seconds, flush=True)


if __name__ == '__main__':
    main()
