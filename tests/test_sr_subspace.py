import unittest
import torch
from Diff.sr_subspace import SpectralProximal, estimate_spectral_basis


class SubspaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(83)

    def test_soft_constraint_matches_dense_solution(self):
        dtype=torch.float64
        r=torch.rand(2,3,dtype=dtype);r/=r.sum(1,keepdim=True)
        low=torch.rand(2,2,3,dtype=dtype);hr=torch.rand(4,4,2,dtype=dtype)
        prox=SpectralProximal(hr,low,r,2.,2)
        ref=torch.rand(4,4,3,dtype=dtype)
        basis=torch.linalg.qr(torch.randn(3,2,dtype=dtype)).Q
        def forward(x):
            a,b=prox.forward(x)
            return torch.cat([a.flatten()/prox.low_bound**.5,b.flatten()/prox.hr_bound**.5])
        A=torch.autograd.functional.jacobian(forward,ref).reshape(-1,ref.numel())
        y=torch.cat([low.flatten()/prox.low_bound**.5,hr.flatten()/prox.hr_bound**.5])
        P=torch.kron(torch.eye(16,dtype=dtype),torch.eye(3,dtype=dtype)-basis@basis.T)
        for gamma in [0.,10.]:
            expected=torch.linalg.solve(torch.eye(48,dtype=dtype)+100*A.T@A+gamma*P,
                                        ref.flatten()+100*A.T@y).reshape_as(ref)
            actual,info=prox.solve_spectral(ref,100,basis,gamma,300,1e-10)
            torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)
            self.assertTrue(info['converged'])

    def test_basis_recovers_known_observation_subspace(self):
        basis=torch.linalg.qr(torch.randn(6,2,dtype=torch.float64)).Q
        low=(torch.randn(50,2,dtype=torch.float64)@basis.T).reshape(5,10,6)
        actual,info=estimate_spectral_basis(low,.999)
        self.assertEqual(info['rank'],2)
        torch.testing.assert_close(actual@actual.T,basis@basis.T,atol=1e-10,rtol=1e-10)


if __name__=='__main__':unittest.main()
