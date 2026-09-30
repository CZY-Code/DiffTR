"""Numerical checks for the observation operator and chunked sampling gradient."""
import unittest
import cv2
import numpy as np
import torch
from torch import nn
from Diff.Diff4SR import GaussianDiffusion1D
from Diff.sr_guidance import spatial_forward, observation_loss, sample_observation_guided, dual_guidance_factor


class TinyDenoiser(nn.Module):
    def forward(self, x, coords, times, **kwargs):
        return (0.4 * x + 0.1 * x.mean(-1, keepdim=True)).tanh() + 0.


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.hr = torch.rand(8, 8, 2)
        self.low = torch.rand(4, 4, 3)
        self.response = torch.rand(2, 3)
        self.response /= self.response.sum(-1, keepdim=True)
        self.diff = GaussianDiffusion1D(TinyDenoiser(), self.hr, self.response,
            seq_length=1, channels=3, self_condition=False, timesteps=3,
            objective='pred_x0', guidance_scale=.2).eval()

    def test_operator_matches_opencv_including_boundaries(self):
        x = np.random.default_rng(10).random((16, 12, 3)).astype(np.float32)
        for sigma in (.5, 2.):
            blur = np.stack([cv2.GaussianBlur(x[..., c], (7, 7), sigma,
                borderType=cv2.BORDER_REPLICATE) for c in range(3)], -1)
            for scale in (1, 4):
                expected = blur.reshape(16//scale, scale, 12//scale, scale, 3).mean((1, 3))
                actual = spatial_forward(torch.tensor(x), sigma, scale).numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_spatial_gradient_finite_difference(self):
        x = torch.rand(8, 8, 3, dtype=torch.float64, requires_grad=True)
        loss = observation_loss(x, self.hr.double(), self.low.double(),
                                self.response.double(), 2., 2, 'dual')
        g = torch.autograd.grad(loss, x)[0]
        direction = torch.randn_like(x)
        eps = 1e-5
        def f(z):
            return observation_loss(z, self.hr.double(), self.low.double(),
                                    self.response.double(), 2., 2, 'dual')
        numerical = (f(x.detach()+eps*direction)-f(x.detach()-eps*direction))/(2*eps)
        torch.testing.assert_close((g*direction).sum(), numerical, rtol=1e-6, atol=1e-6)

    def test_original_matches_existing_sampler(self):
        coords = torch.cartesian_prod(torch.arange(8), torch.arange(8))
        torch.manual_seed(27)
        expected = self.diff.p_sample_loop(coords, (64, 1, 3),
                      model_forward_kwargs={'sample': False}).detach().reshape(8, 8, 3)
        actual = sample_observation_guided(self.diff, self.low, .5, 2,
                  'original', guidance=.2, batch_size=17, seed=27)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_chunked_dual_matches_full_graph(self):
        coords = torch.cartesian_prod(torch.arange(8), torch.arange(8))
        gen = torch.Generator().manual_seed(27)
        x = torch.randn((64, 1, 3), generator=gen)
        for t in reversed(range(3)):
            x = x.detach().requires_grad_(True)
            ts = torch.full((64,), t, dtype=torch.long)
            clean = self.diff.model_predictions(x, coords, ts, clip_x_start=True).pred_x_start
            loss = observation_loss(self.diff.unnormalize(clean).reshape(8, 8, 3),
                     self.hr, self.low, self.response, 2., 2, 'dual')
            grad = torch.autograd.grad(loss, x)[0]
            with torch.no_grad():
                mean, _, logvar = self.diff.q_posterior(clean, x, ts)
                x = mean - .2 * self.diff.alphas_cumprod[t] * grad
                if t:
                    x += (.5*logvar).exp() * torch.randn(x.shape, generator=gen)
        expected = self.diff.unnormalize(x).reshape(8, 8, 3)
        actual = sample_observation_guided(self.diff, self.low, 2., 2,
                  'dual', guidance=.2, batch_size=17, seed=27)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_guidance_factor_bounds_stacked_operator(self):
        # Explicit small operator validates the bound including replicate edges.
        response = self.response.double()
        zero = torch.zeros(4, 4, 3, dtype=torch.float64, requires_grad=True)
        ratio = (4*4*2)/(2*2*3)
        def forward(x):
            return torch.cat([spatial_forward(x @ response.T, 2.).reshape(-1),
                ratio**.5 * spatial_forward(x, 2., 2).reshape(-1)])
        matrix = torch.autograd.functional.jacobian(forward, zero).reshape(-1, 48)
        actual_norm2 = torch.linalg.matrix_norm(matrix, ord=2).square()
        factor = dual_guidance_factor((4, 4, 2), (2, 2, 3), response, 2., 2)
        original_norm2 = torch.linalg.matrix_norm(response, ord=2).square()
        self.assertGreater(factor, 0)
        self.assertLessEqual(factor, 1)
        self.assertGreaterEqual(float(original_norm2/factor) + 1e-10, float(actual_norm2))

    def test_clean_transform_identity_preserves_random_stream(self):
        for guidance in (0., .2):
            kwargs=dict(variant='original',guidance=guidance,seed=83,batch_size=17)
            expected=sample_observation_guided(self.diff,self.low,.5,2,**kwargs)
            actual=sample_observation_guided(self.diff,self.low,.5,2,
                clean_transform=lambda cube,t:cube,**kwargs)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_clean_transform_enters_posterior_with_correct_units(self):
        # At t=0 the DDPM posterior is the clean prediction. Overriding it with
        # a physical-space constant must produce that constant without guidance.
        actual=sample_observation_guided(self.diff,self.low,.5,2,'original',
            guidance=0.,seed=91,clean_transform=lambda cube,t:
            torch.full_like(cube,.37) if t==0 else cube)
        torch.testing.assert_close(actual,torch.full_like(actual,.37),atol=1e-6,rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
