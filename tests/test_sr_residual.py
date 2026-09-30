import unittest
import torch
from Diff.sr_subspace import SpectralProximal
from Diff.sr_residual import refine_residual


class ResidualTests(unittest.TestCase):
    def test_first_step_equivalence_and_dense_dual_updates(self):
        torch.set_num_threads(1);torch.manual_seed(32);d=torch.float64
        r=torch.rand(2,3,dtype=d);r/=r.sum(1,keepdim=True)
        low=torch.rand(2,2,3,dtype=d);hr=torch.rand(4,4,2,dtype=d)
        prox=SpectralProximal(hr,low,r,.5,2)
        ref=torch.rand(4,4,3,dtype=d);b=torch.linalg.qr(torch.randn(3,2,dtype=d)).Q
        first,_=refine_residual(prox,ref,b,1,300,1e-10)
        base,_=prox.solve_spectral(ref,100,b,10,300,1e-10)
        torch.testing.assert_close(first,base,atol=1e-8,rtol=1e-8)
        def f(z):
            lo,hi=prox.forward(z)
            return torch.cat([lo.flatten()/prox.low_bound**.5,hi.flatten()/prox.hr_bound**.5])
        a=torch.autograd.functional.jacobian(f,ref).reshape(-1,48)
        y=torch.cat([low.flatten()/prox.low_bound**.5,hr.flatten()/prox.hr_bound**.5])
        p=torch.kron(torch.eye(16,dtype=d),torch.eye(3,dtype=d)-b@b.T)
        normal=torch.eye(48,dtype=d)+100*a.T@a+10*p
        dual=torch.zeros_like(y)
        for _ in range(5):
            expected=torch.linalg.solve(normal,ref.flatten()+100*a.T@(y+dual))
            dual+=y-a@expected
        actual,info=refine_residual(prox,ref,b,5,300,1e-10)
        self.assertTrue(info['converged'])
        torch.testing.assert_close(actual.flatten(),expected,atol=1e-8,rtol=1e-8)
        # No mutation of original observations, including after repeated solves.
        torch.testing.assert_close(prox.low,low,atol=0,rtol=0)
        torch.testing.assert_close(prox.hr,hr,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
