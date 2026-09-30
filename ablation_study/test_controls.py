"""Check that the interventions change the intended information/gradient path."""
import unittest
import torch
from torch import nn
from common import ROOT
from Diff.Diff4ImgD import GaussianDiffusion1D
from denoise import ConditionControl
import yaml


class CoordinateProbe(nn.Module):
    tensor_shape = [8, 8, 4]
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))
    def forward(self, val, ind, t, **kwargs):
        return (self.weight * ind[:, 0].float()).reshape(-1, 1, 1)


class Controls(unittest.TestCase):
    def make(self, stop, smooth=0.1):
        return GaussianDiffusion1D(CoordinateProbe(), seq_length=1, channels=1,
            timesteps=1, objective='pred_x0', self_condition=False, channel_first=False,
            sparse_enabled=False, smooth_scope='spatial', smooth_weight=smooth,
            smooth_stop_gradient=stop)

    def test_stop_gradient_changes_backward_only(self):
        vals = torch.zeros(4, 1, 1)
        ind = torch.tensor([[1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0]])
        losses, grads = [], []
        for stop in (False, True):
            d = self.make(stop)
            d.perturb_coordinates = lambda x: x + torch.tensor([1, 0, 0])
            loss = d.p_losses_BCD(vals, ind, torch.zeros(4, dtype=torch.long), noise=vals)
            loss.backward()
            losses.append(loss.detach())
            grads.append(d.model.weight.grad)
        torch.testing.assert_close(losses[0], losses[1])
        self.assertFalse(torch.allclose(grads[0], grads[1]))

    def test_no_smooth_makes_stop_gradient_irrelevant(self):
        ind = torch.tensor([[1, 1, 0], [2, 2, 0]])
        grads = []
        for stop in (False, True):
            d = self.make(stop, smooth=0)
            d(torch.zeros(2, 1, 1), ind).backward()
            grads.append(d.model.weight.grad)
        torch.testing.assert_close(*grads)

    def test_constant_condition_removes_coordinate_information(self):
        cfg = yaml.full_load((ROOT/'configs/ImgD_HSI_diff_conf.yaml').read_text())['model']
        cfg['h_dim'] = list(cfg['h_dim'])
        model = ConditionControl(out_dim=1, **cfg).eval()
        val, t = torch.zeros(2, 1, 1), torch.zeros(2, dtype=torch.long)
        a = model(val, torch.tensor([[1, 2, 3], [10, 20, 30]]), t)
        torch.testing.assert_close(a[0], a[1])


if __name__ == '__main__':
    unittest.main()
