"""Reuse the exact DiffTR scene preparation; keep truth out of solvers."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from exp_diff_SR import load_scene, hrhsit_to_lrhsit, array_hash
from Diff.sr_guidance import spatial_forward

ROOT = Path(__file__).resolve().parents[1]

def prepare(job, device, reference_path=None):
    args = SimpleNamespace(**job, protocol='blur', response_path=None,
                           response_key='SRF', response_transpose=True,
                           synthetic_wdc_response=False)
    if job['group'] == 'WDC':
        args.response_path = 'data/responses/TenFuse/SRF_matrix.mat'
    gt, r, _, _, _, _ = load_scene(args)
    low = hrhsit_to_lrhsit(gt, sigma=job['sigma'], downsample_factor=job['DS'])
    msi = hrhsit_to_lrhsit(gt @ r.T, sigma=job['sigma'], downsample_factor=1)
    hashes = dict(gt_sha256=array_hash(gt), response_sha256=array_hash(r),
                  LR_HSI_sha256=array_hash(low), HR_MSI_sha256=array_hash(msi))
    folder = 'table6_blur_wdc256_landsat8' if job['group'] == 'WDC' else 'table6_blur'
    reference = (Path(reference_path) if reference_path is not None else
                 ROOT / 'outputs/sr/reference/observations' / (job['key'] + '.json'))
    expected = json.loads(reference.read_text())
    for k, v in hashes.items():
        if expected[k] != v:
            raise ValueError(f'Observation mismatch: {job["key"]} {k}')
    tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
    return gt, Observations(tensor(low), tensor(msi), tensor(r), job['sigma'], job['DS']), hashes

class Observations:
    def __init__(self, low, msi, response, sigma, scale):
        self.low, self.msi, self.r = low, msi, response
        self.sigma, self.scale = sigma, scale
        self.h, self.w = msi.shape[:2]
        self.c = low.shape[-1]

    def blur(self, x):
        return spatial_forward(x, self.sigma)

    def lr(self, x):
        return spatial_forward(x, self.sigma, self.scale)

    def hr(self, x):
        return self.blur(x @ self.r.T)

    def interpolate(self, x):
        return F.interpolate(x.permute(2, 0, 1)[None], size=(self.h, self.w),
                             mode='bicubic', align_corners=False)[0].permute(1, 2, 0)

    def adjoint(self, function, y, channels):
        # Autograd computes the actual replicate-boundary transpose, not another blur.
        with torch.enable_grad():
            x = torch.zeros(self.h, self.w, channels, device=y.device,
                            dtype=y.dtype, requires_grad=True)
            return torch.autograd.grad(function(x), x, y)[0].detach()

def cg(matvec, b, x, tol=1e-6, maximum=150):
    r = b - matvec(x)
    p = r.clone()
    rr = (r*r).sum()
    target = tol*torch.linalg.vector_norm(b)
    count = 0
    for count in range(maximum):
        if rr.sqrt() <= target:
            break
        ap = matvec(p)
        alpha = rr / (p*ap).sum().clamp_min(1e-30)
        x = x + alpha*p
        r = r - alpha*ap
        new = (r*r).sum()
        p = r + (new/rr)*p
        rr = new
    residual = float(torch.linalg.vector_norm(b-matvec(x))/torch.linalg.vector_norm(b).clamp_min(1e-30))
    return x, dict(iterations=count+1, relative_residual=residual)
