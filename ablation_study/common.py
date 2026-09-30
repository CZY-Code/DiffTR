"""Shared provenance, deterministic seeding and image metrics."""
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics(gt, rec):
    return dict(psnr=float(peak_signal_noise_ratio(gt, rec, data_range=1)),
                ssim=float(structural_similarity(gt, rec, data_range=1, channel_axis=2)),
                nrmse=float(normalized_root_mse(gt, rec)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def provenance(files):
    return {name: sha(ROOT / name) for name in files}
