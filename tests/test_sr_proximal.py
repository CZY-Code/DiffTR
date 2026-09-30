"""Validate true boundary transpose, proximal CG, and DDPM integration."""
import unittest
import torch
from Diff.sr_proximal import ObservationProximal, sample_proximal
from Diff.sr_guidance import sample_observation_guided
from Diff.Diff4SR import GaussianDiffusion1D


class TinyDenoiser(torch.nn.Module):
    def forward(self, x, coords, t, **kwargs):
        return (.4*x).tanh()


class ProximalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_adjoint_and_cg_against_dense_solve(self):
        for sigma in (.5, 2.):
            r = torch.rand(2, 3, dtype=torch.float64)
            r /= r.sum(1, keepdim=True)
            low = torch.rand(2, 2, 3, dtype=torch.float64)
            hr = torch.rand(4, 4, 2, dtype=torch.float64)
            prox = ObservationProximal(hr, low, r, sigma, 2)
            x = torch.rand(4, 4, 3, dtype=torch.float64)
            a,b = prox.forward(x)
            torch.testing.assert_close((a*low).sum()+(b*hr).sum(),
                                       (x*prox.adjoint(low,hr)).sum())
            def forward(z):
                a,b = prox.forward(z)
                return torch.cat([a.flatten()/prox.low_bound**.5,
                                  b.flatten()/prox.hr_bound**.5])
            A = torch.autograd.functional.jacobian(forward, x).reshape(-1,x.numel())
            y = torch.cat([low.flatten()/prox.low_bound**.5,hr.flatten()/prox.hr_bound**.5])
            strength = 10.
            expected = torch.linalg.solve(torch.eye(x.numel(),dtype=x.dtype)+strength*A.T@A,
                                          x.flatten()+strength*A.T@y).reshape_as(x)
            actual, info = prox.solve(x,strength,max_iter=100,tolerance=1e-10)
            torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-9)
            self.assertLess(prox.objective(actual,x,strength),prox.objective(x,x,strength))

    def test_zero_strength_matches_unguided_ddpm_and_chunking(self):
        hr, low = torch.rand(8,8,2), torch.rand(4,4,3)
        response = torch.rand(2,3);response /= response.sum(1,keepdim=True)
        diff = GaussianDiffusion1D(TinyDenoiser(),hr,response,seq_length=1,
            channels=3,self_condition=False,timesteps=3,objective='pred_x0').eval()
        expected = sample_observation_guided(diff,low,2.,2,'original',
                                             guidance=0,batch_size=64,seed=27)
        actual = sample_proximal(diff,low,2.,2,strength=0,batch_size=17,seed=27)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        a = sample_proximal(diff,low,2.,2,strength=10,batch_size=64,seed=27)
        b = sample_proximal(diff,low,2.,2,strength=10,batch_size=17,seed=27)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
