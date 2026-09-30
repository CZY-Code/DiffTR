"""Likelihood, missing-data and sampler invariants for denoising experiments."""
import math
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import ndtr
import torch
from torch import nn

from Diff.Diff4ImgD import GaussianDiffusion1D
from Diff.denoise_guidance import (BlindNoiseModel, censored_mixture_nll,
    detect_dead_columns, estimate_noise, proximal_clean_guidance,
    sample_guided, spectral_basis)


class TinyModel(nn.Module):
    def forward(self, x, ind, times, **kwargs):
        return .4*x + .03*ind[:, 2].reshape(-1, 1, 1)


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_likelihood_matches_censored_gaussian_probabilities(self):
        x = torch.tensor([.3, .3, .3], dtype=torch.float64)
        y = torch.tensor([0., .4, 1.], dtype=torch.float64)
        actual = censored_mixture_nll(x, y, .2, 0.)
        probability = [ndtr(-1.5), math.exp(-.5*.5**2)/(.2*math.sqrt(2*math.pi)), ndtr(-3.5)]
        np.testing.assert_allclose(actual.numpy(), -np.log(probability), rtol=1e-12)

    def test_censored_gradient_finite_difference_and_boundary_stability(self):
        x = torch.tensor([-.5, .3, 1.5], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([1., .4, 0.], dtype=torch.float64)
        sigma = torch.tensor(.05, dtype=torch.float64, requires_grad=True)
        p = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b, c: censored_mixture_nll(a, y, b, c), (x, sigma, p)))

    def test_missing_mask_detects_columns_not_isolated_clipped_pixels(self):
        y = torch.ones(20, 10, 3)*.4
        y[:, 2, 1] = 0
        y[0, 4, 2] = 0
        mask = detect_dead_columns(y)
        self.assertFalse(mask[:, 2, 1].any())
        self.assertTrue(mask[:, 4, 2].all())

    def test_mad_initializer_on_unclipped_white_noise(self):
        x = .5 + .1*torch.randn(200, 200, 3)
        estimate = estimate_noise(x, torch.ones_like(x, dtype=torch.bool))
        torch.testing.assert_close(estimate, torch.full_like(estimate, .1), rtol=.03, atol=.001)

    def test_masked_likelihood_has_zero_gradient_at_missing_values(self):
        y = torch.rand(5, 10, 3)
        noise = BlindNoiseModel(y, torch.ones_like(y, dtype=torch.bool))
        clean = torch.full((3,), .4, requires_grad=True)
        loss = noise(clean, torch.tensor([.1, 0., .7]), torch.arange(3),
                     torch.tensor([True, False, True]))
        loss.backward()
        self.assertEqual(float(clean.grad[1]), 0.)

    def test_proximal_step_reduces_energy_and_ignores_missing_observations(self):
        x = torch.tensor([[.4, .5, .6]], dtype=torch.float64)
        y = torch.tensor([[.2, .3, .9]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        sigma = torch.ones(3, dtype=torch.float64)*.1
        p = torch.zeros(3, dtype=torch.float64)
        basis = (torch.zeros(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64))
        refined = proximal_clean_guidance(x, y, mask, sigma, p, basis, strength=1, iterations=100)
        # Analytic Gaussian proximal optimum balances data and quadratic tether.
        torch.testing.assert_close(refined, torch.tensor([[.3, .4, .6]], dtype=torch.float64), atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(proximal_clean_guidance(x, y, mask, sigma, p, basis, strength=0), x)

    def test_ddim_chunk_invariance_and_guidance_changes_output(self):
        y = torch.rand(3, 4, 3)
        mask = torch.ones_like(y, dtype=torch.bool)
        basis = spectral_basis(y, 2)
        d = GaussianDiffusion1D(TinyModel(), channels=1, seq_length=1,
            timesteps=4, sampling_timesteps=4, objective='pred_x0',
            self_condition=False, channel_first=False)
        coords = torch.cartesian_prod(*(torch.arange(s) for s in y.shape))
        kwargs = dict(steps=4, strength=2., iterations=3, seed=19)
        a = sample_guided(d, coords, y, mask, torch.ones(3)*.1, torch.ones(3)*.1,
                          basis, pixel_batch=12, **kwargs)
        b = sample_guided(d, coords, y, mask, torch.ones(3)*.1, torch.ones(3)*.1,
                          basis, pixel_batch=2, **kwargs)
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
        kwargs['strength'] = 0.
        unguided = sample_guided(d, coords, y, mask, torch.ones(3)*.1, torch.ones(3)*.1,
                                basis, pixel_batch=2, **kwargs)
        self.assertGreater(float((a-unguided).abs().max()), 1e-4)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(((a >= 0) & (a <= 1)).all())

    def test_all_training_modes_complete_without_clean_reference(self):
        from run_guided_denoising import fit
        observations = np.random.default_rng(7).uniform(.1, .9, (8, 8, 3)).astype('float32')
        for variant in ('baseline', 'mse', 'huber', 'likelihood', 'noisy_multi', 'staged'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                args = Namespace(device='cpu', seed=3, variant=variant, rank=2,
                    steps=8, pilot_fraction=.5, lr=.001, batch_size=32, scene='synthetic',
                    case=1, diffusion_steps=4, sample_steps=2, spectral_rank=2,
                    guidance=[0., 1.], prox_iterations=2, pixel_batch=8)
                outputs, diagnostics = fit(observations, args, Path(directory))
                self.assertTrue((Path(directory)/'final.pt').exists())
                for value in outputs.values():
                    self.assertEqual(value.shape, observations.shape)
                    self.assertTrue(np.isfinite(value).all())
                self.assertGreater(diagnostics['train_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
