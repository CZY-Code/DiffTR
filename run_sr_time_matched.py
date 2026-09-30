"""Paper SR comparison with measured optimization budgets and Soft1000 inference.

Clock: observations already on device; include method initialization/optimization.
Exclude common input loading, checkpoint I/O, metric calculation and output I/O.
DiffTR also measures three individual sampling+DC calls (not their sum).
Baselines retain the latest valid result completed before each fixed deadline.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from baseline_sr.protocol import ROOT, prepare
from baseline_sr.time_budget import TimeBudget, parameter_count
from ablation_study.common import metrics, write_json
from scripts.sr_protocol import jobs as selected_jobs, METHODS, BUDGETS

REFERENCE = ROOT/'outputs/sr/reference'
DEFAULT_OUT = ROOT/'outputs/sr'
SOURCES = ['run_sr_time_matched.py', 'baseline_sr/time_budget.py',
           'baseline_sr/protocol.py', 'baseline_sr/cs2dips.py',
           'baseline_sr/edip.py', 'baseline_sr/tensor_methods.py',
           'Diff/Diff4SR.py', 'Diff/networks.py', 'Diff/sr_guidance.py',
           'Diff/sr_proximal.py', 'Diff/sr_subspace.py', 'Diff/sr_reconstruction.py', 'scripts/sr_protocol.py']


def telemetry():
    fields = 'index,uuid,name,temperature.gpu,clocks.sm,utilization.gpu,memory.used'
    return subprocess.check_output(['nvidia-smi', '--query-gpu='+fields,
                                    '--format=csv,noheader'], text=True).strip()


def initialize(out):
    from scripts.sr_protocol import prepare_manifest
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'plan.json', dict(jobs=selected_jobs(), methods=METHODS, refinement='soft1000'))
    for job in selected_jobs():
        prepare_manifest(job, out)


def run(a):
    out = a.output_dir.resolve()
    dest = out/a.method/a.key
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'result.json').exists():
        print('SKIP', dest, flush=True)
        return
    import fcntl
    lock = (dest/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = {'CS2DIPs': 766, 'EDIP-Net': 30}.get(a.method, 3407)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    j = next(j for j in selected_jobs() if j['key'] == a.key)
    gt, obs, hashes = prepare(j, 'cuda:0', REFERENCE/'observations'/(a.key+'.json'))
    # Common fixed warm-up, with no RNG consumption or model updates.
    x = torch.ones(1024, 1024, device='cuda:0')
    until = time.perf_counter()+3
    while time.perf_counter() < until:
        _ = x @ x
        torch.cuda.synchronize()
    del x, _
    snapshot = dest/'source'; snapshot.mkdir(exist_ok=True)
    source_hashes = {}
    for rel in SOURCES:
        src = ROOT/rel
        source_hashes[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
        dst = snapshot/rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    before = telemetry()
    torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()

    def report(step, values):
        record = dict(step=step, wall_seconds=time.perf_counter()-run_start, **values)
        with (dest/'trace.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    base = dict(**j, method=a.method, seed=seed, **hashes,
                physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_name=torch.cuda.get_device_name(), cpu_threads=2,
                torch_version=torch.__version__, source_sha256=source_hashes)
    if a.method == 'DiffTR':
        from Diff.networks import ScalarINR2D
        from Diff.Diff4SR import GaussianDiffusion1D, Trainer1D
        from Diff.sr_reconstruction import reconstruct_sr
        from utils.ob_dataV2 import tensor2data_2Dtube
        saved = json.loads((REFERENCE/'observations'/(a.key+'.json')).read_text())
        cfg = copy.deepcopy(saved['config'])
        cfg['trainer'].update(train_num_steps=10000, save_and_sample_every=10000,
                              results_folder=str(dest))
        torch.cuda.synchronize(); start = time.perf_counter()
        ds = tensor2data_2Dtube(obs.low.cpu().repeat_interleave(j['DS'], 0).repeat_interleave(j['DS'], 1))
        net = ScalarINR2D(out_dim=obs.c, **cfg['model'])
        parameters = parameter_count(net)
        diff = GaussianDiffusion1D(net, obs.msi.cpu(), obs.r.cpu(), **cfg['diffusion'])
        trainer = Trainer1D(diff, ds['train'], ds['test'], gt_tensor=torch.from_numpy(gt),
                            down_scale=j['DS'], is_tube=True, test_batch_size=2048,
                            num_workers=0, **cfg['trainer'])
        assert trainer.device.type == 'cuda'
        trainer.test = lambda: None
        # Checkpoint I/O is outside the timed solver for every method.
        original_save = trainer.save
        trainer.save = lambda milestone: None
        torch.cuda.synchronize(); training_start = time.perf_counter()
        trainer.train()
        torch.cuda.synchronize(); trained = time.perf_counter()
        setup_s = training_start-start; train_s = trained-training_start
        optimization_s = trained-start
        training_peak = torch.cuda.max_memory_allocated()
        report(10000, dict(stage='trained', setup_seconds=setup_s,
                           training_seconds=train_s, optimization_seconds=optimization_s))
        diff = trainer.accelerator.unwrap_model(trainer.model).eval().requires_grad_(False)
        records = []
        for s in [3407, 3408, 3409]:
            torch.cuda.synchronize(); t = time.perf_counter()
            rec_gpu, diagnostics = reconstruct_sr(diff, obs.low, j['sigma'], j['DS'], seed=s)
            diag = diagnostics['solver']
            torch.cuda.synchronize(); sampling_s = time.perf_counter()-t
            if s == 3407:
                first_reconstruction_peak = torch.cuda.max_memory_allocated()
            rec = rec_gpu.cpu().numpy()
            np.save(dest/f'reconstruction_seed{s}.npy', rec)
            record = dict(seed=s, sampling_seconds=sampling_s, **metrics(gt, rec), solver=diag)
            records.append(record); report(10000, dict(stage='sampled', **record))
        one_sample_s = float(np.median([r['sampling_seconds'] for r in records]))
        limits = dict(optimization=optimization_s, end_to_end=optimization_s+one_sample_s)
        scores = {k:float(np.mean([r[k] for r in records])) for k in ['psnr','ssim','nrmse']}
        original_save(1)
        write_json(dest/'experiment.json', {**saved, 'config': cfg, 'protocol': 'blur',
                   'response_path': 'data/responses/TenFuse/SRF_matrix.mat' if j['group']=='WDC' else None,
                   'response_key': 'SRF', 'response_transpose': True, 'synthetic_wdc_response': False})
        result = dict(**base, **scores, limits=limits, setup_seconds=setup_s,
                      training_seconds=train_s, sampling_seconds_median=one_sample_s,
                      network_parameters=parameters, samples=records,
                      training_peak_gpu_bytes=training_peak,
                      single_reconstruction_peak_gpu_bytes=first_reconstruction_peak,
                      settings=cfg, steps=10000, refinement='soft1000')
    else:
        reference = json.loads((out/'DiffTR'/a.key/'result.json').read_text())
        limits = reference['limits']
        if a.method == 'CS2DIPs':
            from baseline_sr.cs2dips import solve
        elif a.method == 'EDIP-Net':
            from baseline_sr.edip import solve
        else:
            from baseline_sr.tensor_methods import solve
        budget = TimeBudget(limits, obs.low.device)
        if a.method == 'EDIP-Net' and getattr(a, 'edip_stage2_fraction', None) is not None:
            budget.metadata['edip_stage2_fraction'] = a.edip_stage2_fraction
        if a.method in ['CS2DIPs', 'EDIP-Net']:
            _, settings = solve(obs, BUDGETS[a.method], report, budget=budget)
        else:
            _, settings = solve(obs, BUDGETS[a.method], report, a.method, j['group'], budget=budget)
        elapsed = budget.elapsed()
        measured = {}
        for name, limit in limits.items():
            if name not in budget.snapshots:
                measured[name] = dict(available=False, limit_seconds=limit,
                                      reason='No complete HR reconstruction within deadline')
                continue
            record = dict(budget.snapshots[name])
            rec = record.pop('reconstruction').cpu().numpy()
            assert rec.shape == gt.shape and np.isfinite(rec).all()
            assert record['seconds'] <= limit
            np.save(dest/f'reconstruction_{name}.npy', rec)
            measured[name] = dict(available=True, limit_seconds=limit, **record,
                                  **metrics(gt, np.clip(rec, 0, 1)))
        result = dict(**base, limits=limits, measured=measured,
                      actual_solver_seconds=elapsed,
                      deadline_overrun_seconds=max(0.,elapsed-max(limits.values())),
                      settings=settings, parameters=budget.metadata)
    result.update(peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  gpu_before=before, gpu_after=telemetry(), complete=True)
    write_json(dest/'result.json', result)
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--initialize', action='store_true')
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--key')
    p.add_argument('--job-file', type=Path)
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--edip-stage2-fraction', type=float)
    a = p.parse_args()
    if a.reference_dir is not None:
        REFERENCE = a.reference_dir.resolve()
    if a.job_file is not None:
        supplied_jobs = json.loads(a.job_file.read_text())['jobs']
        selected_jobs = lambda: supplied_jobs
    if a.edip_stage2_fraction is not None and not 0 < a.edip_stage2_fraction < 1:
        p.error('--edip-stage2-fraction must be between zero and one')
    if a.initialize:
        initialize(a.output_dir.resolve())
    else:
        if not a.method or not a.key:
            p.error('--method and --key required')
        run(a)
