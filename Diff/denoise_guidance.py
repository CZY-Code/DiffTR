"""Observation-only noise calibration and proximal clean-space DDIM guidance.

The contamination likelihood is an approximation to the repository's mixed
noise, not an exact model of its successive clipping operations. No function
in this module accepts a clean reference or synthetic corruption mask.
"""
import math

import torch
from torch import nn


def detect_dead_columns(observed, zero_fraction=0.95):
    """An entire zero column is missing, not evidence for zero clean radiance."""
    if observed.ndim != 3 or not 0 < zero_fraction <= 1:
        raise ValueError('expected H/W/C observations and a fraction in (0, 1]')
    return ((observed <= 0).float().mean(0) < zero_fraction).unsqueeze(0).expand_as(observed)


def estimate_noise(observed, valid):
    """Robust 2x2 Haar MAD initializer (clipping can bias this estimate).

    Sigma is subsequently calibrated by the censored likelihood during fitting.
    Masking missing columns prevents their boundaries from inflating the MAD.
    """
    d = (observed[1:, 1:] - observed[1:, :-1]
         - observed[:-1, 1:] + observed[:-1, :-1]) / 2
    ok = valid[1:, 1:] & valid[1:, :-1] & valid[:-1, 1:] & valid[:-1, :-1]
    values = []
    for b in range(observed.shape[-1]):
        v = d[..., b][ok[..., b]]
        med = v.median() if v.numel() else d.new_tensor(0.)
        mad = (v-med).abs().median() if v.numel() else d.new_tensor(.1 * .67448975)
        values.append((mad / .67448975).clamp(.03, .4))
    return torch.stack(values)


def censored_mixture_nll(clean, observed, sigma, outlier_probability):
    """Elementwise NLL in [0, 1] units, stable even at censored boundaries.

    Inliers: clip(N(clean, sigma**2), 0, 1). Outliers: a clipped uniform
    distribution on [-.5, 1.5] (density .5 and boundary masses .25).
    Densities and masses are only compared within the same observation event.
    """
    sigma = torch.as_tensor(sigma, device=clean.device, dtype=clean.dtype)
    p = torch.as_tensor(outlier_probability, device=clean.device, dtype=clean.dtype)
    if torch.any(sigma <= 0) or torch.any((p < 0) | (p >= 1)):
        raise ValueError('sigma must be positive and contamination in [0, 1)')
    z = (observed-clean) / sigma
    log_in = -.5*z.square() - sigma.log() - .5*math.log(2*math.pi)
    log_in = torch.where(observed <= 0, torch.special.log_ndtr(-clean/sigma), log_in)
    log_in = torch.where(observed >= 1, torch.special.log_ndtr((clean-1)/sigma), log_in)
    log_out = torch.where((observed <= 0) | (observed >= 1),
                          clean.new_tensor(math.log(.25)), clean.new_tensor(math.log(.5)))
    return -torch.logaddexp(torch.log1p(-p) + log_in, p.log() + log_out)


class BlindNoiseModel(nn.Module):
    """Per-band Gaussian scale and contamination fitted without clean targets."""
    def __init__(self, observed, valid):
        super().__init__()
        initial = estimate_noise(observed, valid)
        self.log_sigma = nn.Parameter(initial.log())
        self.outlier_logit = nn.Parameter(torch.full_like(initial, -2.944439))

    @property
    def sigma(self):
        return self.log_sigma.exp().clamp(.02, .5)

    @property
    def probability(self):
        return self.outlier_logit.sigmoid().clamp(.001, .4)

    def forward(self, clean, observed, bands, valid):
        nll = censored_mixture_nll(clean, observed, self.sigma[bands], self.probability[bands])
        return (nll * valid).sum() / valid.sum().clamp_min(1)


def spectral_basis(pilot, rank=8):
    """Estimate a spectral basis from a fitted pilot, never from the reference."""
    if pilot.ndim != 3 or not 1 <= rank <= pilot.shape[-1]:
        raise ValueError('invalid pilot or spectral rank')
    flat = pilot.reshape(-1, pilot.shape[-1])
    mean = flat.mean(0)
    centered = flat - mean
    _, vectors = torch.linalg.eigh(centered.T @ centered / max(1, len(flat)))
    return mean, vectors[:, -rank:]


def proximal_clean_guidance(clean, observed, valid, sigma, probability, basis,
                            strength=1., iterations=5, step_size=.2):
    """Refine clean spectra in a pilot-derived subspace with a quadratic tether.

    This is x0-space proximal guidance, not a DPS gradient through the network.
    Per-band sigma**2 scales the likelihood and a unit quadratic coefficient
    penalty limits displacement. This is a heuristic proximal energy, not an
    exact Bayesian posterior or an exact projection onto observations.
    Caller schedules strength over time.
    """
    if strength < 0 or iterations < 0 or not 0 < step_size <= 1:
        raise ValueError('invalid proximal guidance parameters')
    if strength == 0 or iterations == 0:
        return clean.detach()
    sigma = sigma.detach()
    probability = probability.detach()
    _, vectors = basis
    reference = clean.detach()
    with torch.enable_grad():
        coefficients = torch.zeros((*clean.shape[:-1], vectors.shape[1]),
                                   device=clean.device, dtype=clean.dtype)
        for _ in range(iterations):
            coefficients = coefficients.detach().requires_grad_(True)
            estimate = reference + coefficients @ vectors.T
            nll = censored_mixture_nll(estimate, observed, sigma, probability)
            # Sigma scaling bounds the inlier gradient curvature near one.
            objective = (strength * nll * sigma.square() * valid).sum()
            objective = objective + .5 * coefficients.square().sum()
            grad, = torch.autograd.grad(objective, coefficients)
            coefficients = coefficients - step_size / (1 + strength) * grad
        return (reference + coefficients.detach() @ vectors.T).detach()


@torch.no_grad()
def sample_guided(diffusion, coords, observed, valid, sigma, probability, basis,
                  steps=20, strength=0., iterations=5, seed=3407, pixel_batch=128):
    """Deterministic DDIM, explicit even when sample steps equal training steps.

    Whole spectra share each chunk so guidance cannot depend on voxel batching.
    Initial noise is generated once, making sampling invariant to pixel_batch.
    """
    if not 1 <= steps <= diffusion.num_timesteps or pixel_batch < 1:
        raise ValueError('invalid DDIM steps or batch size')
    bands = observed.shape[-1]
    flat_ob = observed.reshape(-1, bands)
    flat_mask = valid.reshape_as(flat_ob)
    ind = coords.reshape(-1, bands, 3)
    generator = torch.Generator(device=observed.device).manual_seed(seed)
    state = torch.randn((len(flat_ob), bands), device=observed.device, generator=generator)
    times = torch.linspace(-1, diffusion.num_timesteps-1, steps+1).int().tolist()[::-1]
    for time, next_time in zip(times[:-1], times[1:]):
        for start in range(0, len(flat_ob), pixel_batch):
            stop = min(start+pixel_batch, len(flat_ob))
            xt = state[start:stop].reshape(-1, 1, 1)
            ts = torch.full((len(xt),), time, device=xt.device, dtype=torch.long)
            prediction = diffusion.model_predictions(xt, ind[start:stop].reshape(-1, 3), ts,
                clip_x_start=True, rederive_pred_noise=True, model_forward_kwargs={'sample': False})
            clean = diffusion.unnormalize(prediction.pred_x_start).reshape(-1, bands)
            alpha = diffusion.alphas_cumprod[time]
            clean = proximal_clean_guidance(clean, flat_ob[start:stop], flat_mask[start:stop],
                sigma, probability, basis, strength=strength*float(alpha), iterations=iterations)
            x0 = diffusion.normalize(clean.clamp(0, 1)).reshape_as(xt)
            if next_time < 0:
                updated = x0
            else:
                alpha_next = diffusion.alphas_cumprod[next_time]
                updated = alpha_next.sqrt()*x0 + (1-alpha_next).sqrt()*prediction.pred_noise
            state[start:stop] = updated.reshape(-1, bands)
    return diffusion.unnormalize(state).reshape_as(observed).clamp(0, 1)
