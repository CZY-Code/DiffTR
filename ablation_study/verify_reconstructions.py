"""Independently recompute denoising metrics and render a matched qualitative grid."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.io
import tifffile
from common import ROOT, seed_all, array_sha, metrics, write_json
from utils.datautil import truncated_linear_stretch
from utils.noiseFun import add_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='ablation_study/results/main')
    parser.add_argument('--output', default='ablation_study/analysis')
    args = parser.parse_args()
    folder, out = ROOT/args.results, ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.5), constrained_layout=True)
    titles = ['Clean', 'Observed', 'Full', 'Constant condition', 'No sparse separation', 'No consistency']
    verified = []
    for row, scene in enumerate(('PaviaU', 'WDC')):
        full = (scipy.io.loadmat(ROOT/'data/PaviaU.mat')['paviaU'] if scene == 'PaviaU'
                else tifffile.imread(ROOT/'data/dc.tif').transpose(1,2,0).astype(np.float32))
        full = truncated_linear_stretch(full).astype(np.float32)
        for path in sorted(folder.glob(f'denoise_{scene}_*.json')):
            r = json.loads(path.read_text())
            y, x = r['origin']
            h, w, _ = r['shape']
            gt = np.ascontiguousarray(full[y:y+h,x:x+w])
            seed_all(r['seed'])
            noisy = add_noise(gt, r['case'], deadline_protocol='lines').astype(np.float32)
            rec = np.load(path.with_suffix('.npy'))
            for key, data in [('gt_sha256',gt), ('noisy_sha256',noisy), ('reconstruction_sha256',rec)]:
                if array_sha(data) != r[key]:
                    raise ValueError(f'Hash mismatch: {path}, {key}')
            for key, val in metrics(gt,rec).items():
                if not np.isclose(val,r[key],rtol=0,atol=1e-7):
                    raise ValueError(f'Metric mismatch: {path}, {key}')
            verified.append(path.name)
        ref = json.loads((folder/f'denoise_{scene}_full_3407.json').read_text())
        y, x = ref['origin']; h,w,_ = ref['shape']
        gt = full[y:y+h,x:x+w]
        seed_all(3407)
        observed = add_noise(gt, ref['case'], deadline_protocol='lines').astype(np.float32)
        images = [gt,observed] + [np.load(folder/f'denoise_{scene}_{v}_3407.npy')
                  for v in ('full','no_condition','no_sparse','no_smooth')]
        bands = [60,30,10] if scene=='PaviaU' else [100,60,30]
        for col, img in enumerate(images):
            axes[row,col].imshow(np.sqrt(img[...,bands].clip(0,1)))
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            axes[row,col].set_title(titles[col],fontsize=10)
        axes[row,0].set_ylabel(scene)
    fig.suptitle('Case 3, seed 3407; identical display range and gamma 0.5 for every image',fontsize=11)
    fig.savefig(out/'denoising_controls.png',dpi=180)
    fig.savefig(out/'denoising_controls.pdf')
    plt.close(fig)
    if len(verified) != 30:
        raise ValueError(f'Expected 30 reconstructions, got {len(verified)}')
    write_json(out/'reconstruction_verification.json',dict(count=len(verified),
               checks=['regenerated clean/noisy hashes','saved reconstruction hashes','recomputed PSNR/SSIM/NRMSE'],
               files=verified))
    print('Verified',len(verified),'reconstructions and metrics')


if __name__ == '__main__':
    main()
