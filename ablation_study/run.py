"""Print or execute the complete experiment matrix on specified devices."""
import argparse
import concurrent.futures
import os
import json
from pathlib import Path
import shlex
import subprocess
import sys
from common import ROOT, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpus', nargs='+', default=['1', '2', '3'])
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--crop', type=int, default=128)
    p.add_argument('--seeds', nargs='+', type=int, default=[3407, 3408, 3409])
    p.add_argument('--output-root', default='ablation_study/results/main')
    p.add_argument('--tasks', choices=['all', 'denoise', 'superres'], default='all')
    p.add_argument('--resume', action='store_true', help='Resume a matching saved matrix')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        p.error('Require 0 <= shard-index < shard-count')
    out = ROOT / args.output_root
    jobs = []
    if args.tasks in ('all', 'denoise'):
        for scene in ('PaviaU', 'WDC'):
            for seed in args.seeds:
                for variant in ('full', 'no_condition', 'no_sparse', 'no_smooth', 'stop_gradient'):
                    name = f'denoise_{scene}_{variant}_{seed}'
                    cmd = [sys.executable, str(ROOT/'ablation_study/denoise.py'), '--scene', scene,
                           '--variant', variant, '--seed', str(seed), '--steps', str(args.steps),
                           '--crop', str(args.crop), '--output', str(out/f'{name}.json')]
                    jobs.append((name, cmd))
    if args.tasks in ('all', 'superres'):
        for scene, parent in [('PaviaU', 'table6_blur'), ('WDC', 'table6_blur_wdc256_landsat8')]:
            folder = f'outputs/sr/DiffTR/{scene}_{scene}_patch0_DS4_sigma0.5'
            cmd = [sys.executable, str(ROOT/'ablation_study/superres.py'), '--checkpoint-dir', folder,
                   '--seeds', *map(str, args.seeds), '--output-dir', str(out)]
            jobs.insert(0, (f'sr_{scene}', cmd))
    jobs = jobs[args.shard_index::args.shard_count]
    for name, cmd in jobs:
        print(shlex.join(cmd), flush=True)
    if not args.execute:
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out/'run_manifest.json').exists():
        if not args.resume:
            raise FileExistsError('Use --resume or a new output root')
        saved = json.loads((out/'run_manifest.json').read_text())['arguments']
        for key in ('steps', 'crop', 'seeds', 'tasks'):
            if saved[key] != getattr(args, key):
                raise ValueError(f'Resume mismatch: {key}')
    else:
        write_json(out/'run_manifest.json', dict(arguments=vars(args), jobs=jobs))
    def worker(gpu, queue):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        for name, cmd in queue:
            if args.resume and name.startswith('denoise_') and (out/f'{name}.json').exists():
                continue
            if args.resume and name.startswith('sr_'):
                variants = ('steps_5', 'steps_10', 'steps_20', 'full', 'guidance_0', 'guidance_1', 'guidance_5')
                if all((out/f'{name}_{v}_{s}.json').exists() for v in variants for s in args.seeds):
                    continue
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print('Completed', name, 'GPU', gpu, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu, jobs[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
