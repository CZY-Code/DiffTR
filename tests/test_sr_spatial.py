import unittest
import torch
from Diff.sr_spatial import SpatialSpectralProximal


class SpatialTests(unittest.TestCase):
    def test_matches_dense_solution_and_zero_strength_control(self):
        torch.set_num_threads(1);torch.manual_seed(78);dtype=torch.float64
        r=torch.rand(2,3,dtype=dtype);r/=r.sum(1,keepdim=True)
        low=torch.rand(2,2,3,dtype=dtype);hr=torch.rand(4,4,2,dtype=dtype)
        p=SpatialSpectralProximal(hr,low,r,2.,2)
        x=torch.rand(4,4,3,dtype=dtype);b=torch.linalg.qr(torch.randn(3,2,dtype=dtype)).Q
        def forward(z):
            lo,hi=p.forward(z)
            return torch.cat([lo.flatten()/p.low_bound**.5,hi.flatten()/p.hr_bound**.5])
        a=torch.autograd.functional.jacobian(forward,x).reshape(-1,x.numel())
        y=torch.cat([low.flatten()/p.low_bound**.5,hr.flatten()/p.hr_bound**.5])
        s=torch.autograd.functional.jacobian(lambda z:p.spatial_normal(z,b),x).reshape(48,48)
        torch.testing.assert_close(s,s.T,atol=1e-12,rtol=1e-12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(s).min()),-1e-12)
        self.assertLessEqual(float(torch.linalg.eigvalsh(s).max()),1+1e-12)
        proj=torch.kron(torch.eye(16,dtype=dtype),torch.eye(3,dtype=dtype)-b@b.T)
        for lam in [0.,1.,10.]:
            expected=torch.linalg.solve(torch.eye(48,dtype=dtype)+100*a.T@a+10*proj+lam*s,x.flatten()+100*a.T@y).reshape_as(x)
            actual,info=p.solve_spatial(x,b,lam,max_iter=300,tolerance=1e-10)
            self.assertTrue(info['converged'])
            torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-8)
            if lam==0:
                original,_=p.solve_spectral(x,100,b,10,300,1e-10)
                torch.testing.assert_close(actual,original,atol=1e-8,rtol=1e-8)

    def test_constant_coefficients_unpenalized(self):
        torch.set_num_threads(1)
        p=SpatialSpectralProximal(torch.rand(4,4,2),torch.rand(2,2,3),torch.ones(2,3)/3,.5,2)
        b=torch.linalg.qr(torch.randn(3,2)).Q
        x=torch.ones(4,4,1)*torch.rand(1,1,3)
        torch.testing.assert_close(p.spatial_normal(x,b),torch.zeros_like(x),atol=1e-7,rtol=0)


if __name__=='__main__':unittest.main()
