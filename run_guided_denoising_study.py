"""Run independent GPU jobs with fixed configurations and immutable outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', default=['WDC', 'PaviaU'])
    p.add_argument('--cases', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--variants', nargs='+', default=['baseline', 'likelihood', 'staged'])
    p.add_argument('--seeds', nargs='+', type=int, default=[3407])
    p.add_argument('--devices', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--crop', type=int, default=96)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--noise-protocol', choices=['legacy', 'lines'], default='legacy')
    p.add_argument('--guidance', nargs='+', type=float, default=[0., 1., 4.])
    p.add_argument('--skip', nargs='*', default=[], help='job names already run separately')
    p.add_argument('--wait-for', type=Path, nargs='*', default=[],
                   help='wait for own preceding jobs to finish before allocating their GPUs')
    p.add_argument('--wait-timeout', type=float, default=3600)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in args.seeds:
        for scene in args.scenes:
            for case in args.cases:
                for variant in args.variants:
                    name = f'{scene}_c{case}_{variant}_s{seed}'
                    if name in args.skip:
                        continue
                    jobs.append(dict(name=name, scene=scene, case=case, variant=variant, seed=seed))
    (args.root/'plan.json').write_text(json.dumps(jobs, indent=2)+'\n')
    wait_start = time.monotonic()
    while not all(path.is_file() for path in args.wait_for):
        if time.monotonic()-wait_start > args.wait_timeout:
            raise TimeoutError('preceding experiments did not finish; no GPU jobs launched')
        time.sleep(2)
    # Static queues avoid allocating two workers to the same GPU.
    queues = [jobs[i::len(args.devices)] for i in range(len(args.devices))]

    def worker(device, queue):
        results = []
        for job in queue:
            output = args.root/job['name']
            command = [sys.executable, str(ROOT/'run_guided_denoising.py'),
                '--scene', job['scene'], '--case', str(job['case']), '--variant', job['variant'],
                '--seed', str(job['seed']), '--device', f'cuda:{device}',
                '--crop', str(args.crop), '--steps', str(args.steps),
                '--batch-size', str(args.batch_size), '--noise-protocol', args.noise_protocol,
                '--rank', str(args.rank),
                '--guidance', *map(str, args.guidance), '--output', str(output)]
            start = time.time()
            print('START', job['name'], f'GPU {device}', flush=True)
            with (args.root/f'{job["name"]}.log').open('x') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result = dict(**job, command=command, returncode=process.returncode,
                          elapsed_seconds=time.time()-start)
            results.append(result)
            (args.root/f'worker_{device}.json').write_text(json.dumps(results, indent=2)+'\n')
            print('DONE', job['name'], result['returncode'], round(result['elapsed_seconds'], 1), flush=True)
            if process.returncode:
                raise RuntimeError(f'job failed; see {args.root}/{job["name"]}.log')
        return results

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        tasks = [pool.submit(worker, d, q) for d, q in zip(args.devices, queues)]
        results = [r for task in tasks for r in task.result()]
    (args.root/'completed.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
