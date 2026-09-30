"""Sparse late-step observation/subspace correction of clean DDPM predictions.

No retraining, no ground-truth access, no additional random draws. A final
Soft100 solve is applied by the caller to permit paired inference comparisons.
"""
import torch


class ProgressiveSpectralCorrection:
    def __init__(self, proximal, basis, timesteps, maximum_strength=100.,
                 max_iter=30, tolerance=1e-4):
        if timesteps < 4 or maximum_strength <= 0:
            raise ValueError('Need >=4 diffusion steps and positive strength')
        steps = sorted({timesteps//2, timesteps//4, 1}, reverse=True)
        self.schedule = {t: maximum_strength * 2.**(i-len(steps)+1)
                         for i,t in enumerate(steps)}
        self.proximal, self.basis = proximal, basis
        self.max_iter, self.tolerance = max_iter, tolerance
        self.records = []

    @torch.no_grad()
    def __call__(self, clean, t):
        if t not in self.schedule:
            return clean
        strength = self.schedule[t]
        result, info = self.proximal.solve_spectral(
            clean, strength, self.basis, strength*.1,
            self.max_iter, self.tolerance)
        self.records.append(dict(t=int(t), **info))
        # Soft constraint: do not clip or hard-project the corrected spectrum.
        return result
