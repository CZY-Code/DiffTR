"""Validate complete paired results and produce machine-readable and LaTeX tables."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import ROOT, sha, write_json

DEN = ('full', 'no_condition', 'no_sparse', 'no_smooth', 'stop_gradient')
SR = ('steps_5', 'steps_10', 'steps_20', 'full', 'guidance_0', 'guidance_1', 'guidance_5')
LABEL = dict(full='Full implementation', no_condition='Constant condition', no_sparse='Without sparse separation',
             no_smooth='Without spatial consistency', stop_gradient='Stop-gradient target',
             steps_5='5 steps', steps_10='10 steps', steps_20='20 steps',
             guidance_0='Guidance 0', guidance_1='Guidance 1', guidance_5='Guidance 5')


def aggregate(folder, seeds):
    records = []
    for path in sorted(folder.glob('*.json')):
        r = json.loads(path.read_text())
        if r.get('task') in ('denoise', 'superres'):
            r['result_file'] = str(path.relative_to(ROOT))
            r['result_sha256'] = sha(path)
            records.append(r)
    by_key = {}
    for r in records:
        key = (r['task'], r['scene'], r['variant'], r['seed'])
        if key in by_key:
            raise ValueError(f'Duplicate: {key}')
        by_key[key] = r
    expected = {(task, scene, variant, seed) for task, variants in [('denoise', DEN), ('superres', SR)]
                for scene in ('PaviaU', 'WDC') for variant in variants for seed in seeds}
    if set(by_key) != expected:
        raise ValueError(f'Matrix mismatch. Missing: {sorted(expected-set(by_key))}; extra: {sorted(set(by_key)-expected)}')
    groups = defaultdict(list)
    verified_hashes = {}
    for key, r in by_key.items():
        baseline = by_key[(r['task'], r['scene'], 'full', r['seed'])]
        same = ('gt_sha256', 'noisy_sha256', 'steps', 'batch_size', 'shape', 'origin', 'case', 'source_sha256') if r['task'] == 'denoise' else ('checkpoint_sha256', 'batch_size', 'source_sha256')
        for field in same:
            if r[field] != baseline[field]:
                raise ValueError(f'Unmatched pair: {key}, {field}')
        if r['task'] == 'superres' and not (r['observation_hashes_verified'] and r['checkpoint_buffers_verified']):
            raise ValueError('Unverified checkpoint')
        # Refuse stale results after any change to a recorded source/dependency.
        for source, expected_hash in r['source_sha256'].items():
            if source not in verified_hashes:
                verified_hashes[source] = sha(ROOT/source)
            if verified_hashes[source] != expected_hash:
                raise ValueError(f'Source changed since experiment: {source}')
        if r['task'] == 'superres':
            checkpoint = str(Path(r['checkpoint'])/'model-1.pt')
            if checkpoint not in verified_hashes:
                verified_hashes[checkpoint] = sha(ROOT/checkpoint)
            if verified_hashes[checkpoint] != r['checkpoint_sha256']:
                raise ValueError(f'Checkpoint changed: {checkpoint}')
        groups[key[:3]].append(r)
    summary = []
    for key, group in sorted(groups.items()):
        row = dict(zip(('task', 'scene', 'variant'), key))
        row['n'] = len(group)
        for metric in ('psnr', 'ssim', 'nrmse', 'sample_seconds'):
            values = [r[metric] for r in group]
            row[metric+'_mean'] = float(np.mean(values))
            row[metric+'_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        delta = [r['psnr']-by_key[(r['task'], r['scene'], 'full', r['seed'])]['psnr'] for r in group]
        row['paired_psnr_delta_mean'] = float(np.mean(delta))
        row['paired_psnr_delta_sd'] = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        row['paired_psnr_deltas'] = delta
        summary.append(row)
    return records, summary


def table(summary, task, variants):
    lookup = {(r['scene'], r['variant']): r for r in summary if r['task'] == task}
    lines = [r'\begin{tabular}{lcccc}', r'\toprule',
             r'& \multicolumn{2}{c}{PaviaU} & \multicolumn{2}{c}{WDC} \\',
             r'Variant & PSNR (dB) & SSIM & PSNR (dB) & SSIM \\', r'\midrule']
    for v in variants:
        label = '40 steps, guidance 15' if task == 'superres' and v == 'full' else LABEL[v]
        cells = [label]
        for scene in ('PaviaU', 'WDC'):
            r = lookup[(scene, v)]
            cells.extend([f"${r['psnr_mean']:.3f}\\pm{r['psnr_sd']:.3f}$", f"${r['ssim_mean']:.4f}\\pm{r['ssim_sd']:.4f}$"])
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    return '\n'.join(lines)+'\n'


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)
    lookup = {(r['scene'], r['variant']): r for r in summary if r['task']=='superres'}
    for scene, marker in [('PaviaU', 'o'), ('WDC', 's')]:
        for ax, variants, xs in [(axes[0], ['steps_5', 'steps_10', 'steps_20', 'full'], [5, 10, 20, 40]),
                                 (axes[1], ['guidance_0', 'guidance_1', 'guidance_5', 'full'], [0, 1, 5, 15])]:
            ax.errorbar(xs, [lookup[(scene,v)]['psnr_mean'] for v in variants],
                        yerr=[lookup[(scene,v)]['psnr_sd'] for v in variants], marker=marker,
                        capsize=3, label=scene, linewidth=1.2)
            ax.grid(alpha=0.25)
            ax.set_ylabel('PSNR (dB)')
    axes[0].set_xlabel('DDIM sampling steps (guidance = 15)')
    axes[1].set_xlabel('Guidance scale (40 DDIM steps)')
    axes[0].legend(frameon=False)
    fig.savefig(out/'sampling_guidance.pdf')
    fig.savefig(out/'sampling_guidance.png', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='ablation_study/results/main')
    p.add_argument('--output', default='ablation_study/analysis')
    p.add_argument('--seeds', type=int, nargs='+', default=[3407,3408,3409])
    args = p.parse_args()
    out = ROOT/args.output
    records, summary = aggregate(ROOT/args.results, args.seeds)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'summary.json', summary)
    write_json(out/'verification.json', dict(count=len(records), seeds=args.seeds,
               checks=['complete matrix', 'unique keys', 'paired observations/checkpoints', 'current source hashes'],
               results=[dict(file=r['result_file'], sha256=r['result_sha256']) for r in records]))
    with (out/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out/'denoise_table.tex').write_text(table(summary, 'denoise', DEN))
    (out/'superres_table.tex').write_text(table(summary, 'superres', SR))
    lines = ['| Task | Scene | Variant | PSNR mean ± SD | SSIM mean ± SD | Paired PSNR change |',
             '|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['task']} | {r['scene']} | {r['variant']} | {r['psnr_mean']:.3f} ± {r['psnr_sd']:.3f} | {r['ssim_mean']:.4f} ± {r['ssim_sd']:.4f} | {r['paired_psnr_delta_mean']:+.3f} ± {r['paired_psnr_delta_sd']:.3f} |")
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    plot(summary, out)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
