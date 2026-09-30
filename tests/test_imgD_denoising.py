"""Check corruption structure and the loss branches that change the protocol."""
import random
import unittest

import numpy as np
import torch
from torch import nn

from Diff.Diff4ImgD import GaussianDiffusion1D
from utils.noiseFun import add_deadline_noise, add_noise


class ProbeModel(nn.Module):
    tensor_shape = (5, 10, 3)

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.2))
        self.calls = 0

    def forward(self, val, ind, t, **kwargs):
        self.calls += 1
        return torch.zeros_like(val) + self.bias


def diffusion(**kwargs):
    return GaussianDiffusion1D(ProbeModel(), seq_length=1, channels=1,
                               timesteps=1, objective='pred_x0',
                               self_condition=False, channel_first=False, **kwargs)


class DenoisingChecks(unittest.TestCase):
    def test_dead_lines_are_complete_and_present_in_every_band(self):
        x = np.ones((5, 10, 3), dtype=np.float32)
        np.random.seed(3407)
        noisy = add_deadline_noise(x, 0.2)
        self.assertTrue(np.all(x == 1))
        for band in range(3):
            dead = noisy[:, :, band] == 0
            self.assertTrue(np.all(dead == dead[0:1]))
            self.assertEqual(int(dead[0].sum()), 2)

    def test_legacy_deadlines_reproduce_old_random_mask(self):
        x = np.ones((5, 10, 3))
        np.random.seed(7)
        expected = x.copy()
        expected[np.random.rand(*x.shape) < 0.2] = 0
        np.random.seed(7)
        np.testing.assert_array_equal(add_deadline_noise(x, 0.2, 'legacy'), expected)

    def test_protocol_does_not_change_gaussian_or_sparse_only_noise(self):
        x = np.full((5, 10, 3), 0.5)
        for case in (1, 2, 4, 6):
            np.random.seed(7); random.seed(7)
            legacy = add_noise(x, case, 'legacy')
            np.random.seed(7); random.seed(7)
            corrected = add_noise(x, case, 'lines')
            np.testing.assert_array_equal(legacy, corrected)
        self.assertTrue(np.all(x == 0.5))

    def test_spatial_perturbation_preserves_spectral_axis_and_bounds(self):
        d = diffusion(smooth_scope='spatial', smooth_std=100)
        ind = torch.tensor([[0, 0, 0], [4, 9, 2]]).repeat(100, 1)
        torch.manual_seed(7)
        perturbed = d.perturb_coordinates(ind)
        self.assertTrue(torch.equal(perturbed[:, 2], ind[:, 2]))
        self.assertTrue(torch.all(perturbed >= 0))
        self.assertTrue(torch.all(perturbed < torch.tensor(d.model.tensor_shape)))
        self.assertTrue(torch.any(perturbed[:, :2] != ind[:, :2]))

    def test_zero_radius_preserves_coordinates(self):
        ind = torch.tensor([[0, 1, 2], [4, 9, 0]])
        self.assertTrue(torch.equal(diffusion(smooth_scope='all', smooth_std=0).perturb_coordinates(ind), ind))

    def test_legacy_perturbation_keeps_old_rng_and_truncation(self):
        ind = torch.tensor([[0, 1, 2], [4, 9, 0]])
        torch.manual_seed(7)
        expected = torch.normal(ind.float(), torch.ones_like(ind)).long()
        torch.manual_seed(7)
        self.assertTrue(torch.equal(diffusion(smooth_scope='legacy').perturb_coordinates(ind), expected))

    def test_disabled_sparse_branch_fits_observations_and_has_gradient(self):
        d = diffusion(sparse_enabled=False, smooth_weight=0)
        values = torch.full((2, 1, 1), 0.8)
        ind = torch.tensor([[0, 0, 0], [4, 9, 2]])
        t = torch.zeros(2, dtype=torch.long)
        loss = d.p_losses_BCD(values, ind, t, noise=torch.zeros_like(values))
        expected = (0.8 - 0.2) ** 2 * d.loss_weight[0]
        torch.testing.assert_close(loss, expected)
        self.assertEqual(d.model.calls, 1)
        loss.backward()
        self.assertLess(float(d.model.bias.grad), 0)

    def test_sparse_threshold_controls_target_clipping(self):
        values = torch.full((2, 1, 1), 0.8)
        ind = torch.tensor([[0, 0, 0], [4, 9, 2]])
        for threshold in (0.04, 0.2):
            d = diffusion(sparse_threshold=threshold, smooth_weight=0)
            loss = d.p_losses_BCD(values, ind, torch.zeros(2, dtype=torch.long), noise=torch.zeros_like(values))
            torch.testing.assert_close(loss, threshold ** 2 * d.loss_weight[0])
            self.assertEqual(d.model.calls, 2)


if __name__ == '__main__':
    unittest.main()
