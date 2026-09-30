"""Observation adapters for the existing coordinate-conditioned HSI diffusion.

No learned parameters or training changes. Full-image observation gradients are
chained through the pixel-wise denoiser in chunks (two-pass vector-Jacobian product).
"""
import cv2
import torch
import torch.nn.functional as F


def spatial_forward(x, sigma, scale=1):
    """HWC -> HWC: exact 7x7 replicate Gaussian + nonoverlapping block mean."""
    if x.ndim != 3 or x.shape[0] % scale or x.shape[1] % scale:
        raise ValueError('Expected HWC with spatial dimensions divisible by scale')
    c = x.shape[-1]
    kernel = torch.as_tensor(cv2.getGaussianKernel(7, sigma).ravel(),
                             dtype=x.dtype, device=x.device)
    y = x.permute(2, 0, 1).unsqueeze(0)
    y = F.conv2d(F.pad(y, (3, 3, 0, 0), mode='replicate'),
                 kernel.reshape(1, 1, 1, 7).expand(c, 1, 1, 7), groups=c)
    y = F.conv2d(F.pad(y, (0, 0, 3, 3), mode='replicate'),
                 kernel.reshape(1, 1, 7, 1).expand(c, 1, 7, 1), groups=c)
    return F.avg_pool2d(y, scale).squeeze(0).permute(1, 2, 0)


def observation_loss(x, hr_msi, lr_hsi, response, sigma, scale, variant,
                     lr_weight=1.0):
    """Physical [0,1] units. Preserve the original normalized MSI sum loss.

    LR weight is scaled by number(MSI elements)/number(LR-HSI elements),
    so lr_weight=1 assigns equal weights to their mean squared residuals.
    The factor 4 matches normalize(y)=2*y-1 in the original sampler.
    """
    if variant not in ('original', 'matched_msi', 'dual'):
        raise ValueError(variant)
    pred = x @ response.T
    if variant != 'original':
        pred = spatial_forward(pred, sigma)
    msi = (pred - hr_msi).square().sum()
    loss = 4 * msi
    if variant == 'dual':
        lr = (spatial_forward(x, sigma, scale) - lr_hsi).square().sum()
        loss = loss + 4 * lr_weight * hr_msi.numel() / lr_hsi.numel() * lr
    return loss


def dual_guidance_factor(hr_shape, lr_shape, response, sigma, scale, lr_weight=1.):
    """Conservative step rescaling from original R to the stacked dual operator.

    For the nonnegative spatial operators, max(A.T @ A @ ones) bounds
    ||A||_2^2. R's norm is exact. This controls the observation quadratic,
    not the denoiser Jacobian, and is not a convergence guarantee for DPS.
    """
    h, w, m = hr_shape
    spectral_norm2 = torch.linalg.matrix_norm(response, ord=2).square()
    x = torch.ones(h, w, 1, device=response.device, dtype=response.dtype,
                   requires_grad=True)
    g = spatial_forward(x, sigma)
    dg = spatial_forward(x, sigma, scale)
    ratio = h*w*m / (lr_shape[0]*lr_shape[1]*lr_shape[2])
    quadratic = .5 * spectral_norm2 * g.square().sum() + .5 * lr_weight * ratio * dg.square().sum()
    bound = torch.autograd.grad(quadratic, x)[0].max()
    return float((spectral_norm2 / torch.maximum(bound, spectral_norm2)).detach())


def sample_observation_guided(diff, lr_hsi, sigma, scale, variant='dual',
                              guidance=15., lr_weight=1., batch_size=2048,
                              seed=3407, trace=None, clean_transform=None):
    """Full-step DDPM, matching the original posterior mean and noise schedule.

    Same full-image random stream for every variant; batch size does not change
    it. Requires an eval-mode, atom-wise deterministic denoiser, no self-condition,
    and auto_normalize=True (as in all table-VI checkpoints).
    Optional clean_transform(physical_clean_cube, t) modifies the clean estimate
    used by q_posterior. The original guidance gradient remains evaluated at the
    unmodified prediction. The callback must not consume the sampling RNG.
    """
    if diff.training or diff.self_condition:
        raise ValueError('Use eval mode and self_condition=False')
    device = diff.betas.device
    h, w, _ = diff.HR_MSI.shape
    c = diff.channels
    coords = torch.cartesian_prod(torch.arange(h, device=device),
                                 torch.arange(w, device=device))
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((h*w, 1, c), generator=gen, device=device)
    for t in reversed(range(diff.num_timesteps)):
        # Pass 1 obtains the whole clean cube without retaining network graphs.
        with torch.no_grad():
            clean = torch.empty_like(x)
            for start in range(0, len(coords), batch_size):
                sl = slice(start, start + batch_size)
                ts = torch.full((len(coords[sl]),), t, dtype=torch.long, device=device)
                clean[sl] = diff.model_predictions(x[sl], coords[sl], ts,
                    clip_x_start=True, model_forward_kwargs={'sample': False}).pred_x_start
        if guidance:
            clean_leaf = clean.detach().requires_grad_(True)
            physical = diff.unnormalize(clean_leaf).reshape(h, w, c)
            loss = observation_loss(physical, diff.HR_MSI, lr_hsi, diff.Matrix_R,
                                sigma, scale, variant, lr_weight)
            clean_grad = torch.autograd.grad(loss, clean_leaf)[0].detach()
            if trace is not None:
                trace.append({'t': t, 'normalized_observation_loss':
                          float(loss.detach()) / (4 * diff.HR_MSI.numel())})
            del loss, clean_leaf, physical
        posterior_clean = clean
        if clean_transform is not None:
            with torch.no_grad():
                transformed = clean_transform(diff.unnormalize(clean).reshape(h,w,c), t)
                if transformed.shape != (h,w,c) or not torch.isfinite(transformed).all():
                    raise ValueError('Invalid transformed clean cube')
                posterior_clean = diff.normalize(transformed).reshape_as(clean)
        # Pass 2 applies J_denoiser^T to the global observation gradient.
        noise = torch.randn(x.shape, generator=gen, device=device) if t > 0 else None
        next_x = torch.empty_like(x)
        for start in range(0, len(coords), batch_size):
            sl = slice(start, start + batch_size)
            xt = x[sl].detach().requires_grad_(bool(guidance))
            ts = torch.full((len(coords[sl]),), t, dtype=torch.long, device=device)
            if guidance:
                pred = diff.model_predictions(xt, coords[sl], ts, clip_x_start=True,
                         model_forward_kwargs={'sample': False}).pred_x_start
                grad = torch.autograd.grad(pred, xt, grad_outputs=clean_grad[sl])[0]
            with torch.no_grad():
                mean, _, logvar = diff.q_posterior(posterior_clean[sl], xt, ts)
                next_x[sl] = mean
                if guidance:
                    next_x[sl] -= guidance * diff.alphas_cumprod[t] * grad
                if noise is not None:
                    next_x[sl] += (0.5 * logvar).exp() * noise[sl]
        x = next_x
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'Non-finite sample at t={t}')
    return diff.unnormalize(x).reshape(h, w, c).detach()
