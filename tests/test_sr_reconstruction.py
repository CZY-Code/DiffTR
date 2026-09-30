import unittest
import torch
from torch import nn
from Diff.Diff4SR import GaussianDiffusion1D
from Diff.sr_reconstruction import reconstruct_sr
from Diff.sr_guidance import sample_observation_guided
from Diff.sr_subspace import SpectralProximal,estimate_spectral_basis


class ToyDenoiser(nn.Module):
    def forward(self,x,coords,t,**kwargs):return (.4*x+.1*x.mean(-1,keepdim=True)).tanh()


class ReconstructionTests(unittest.TestCase):
    def test_api_matches_explicit_reconstruction_for_both_configs(self):
        torch.set_num_threads(1);torch.manual_seed(15)
        hr=torch.rand(8,8,2);low=torch.rand(4,4,3);r=torch.rand(2,3);r/=r.sum(1,keepdim=True)
        diff=GaussianDiffusion1D(ToyDenoiser(),hr,r,seq_length=1,channels=3,
            self_condition=False,timesteps=3,objective='pred_x0').eval()
        raw=sample_observation_guided(diff,low,.5,2,'original',guidance=15.,seed=34,batch_size=19)
        b,_=estimate_spectral_basis(low,.999);prox=SpectralProximal(hr,low,r,.5,2)
        for mode,tau,maximum in [('soft100',100.,200),('soft1000',1000.,400)]:
            expected,info=prox.solve_spectral(raw,tau,b,10.,maximum,1e-5)
            actual,diagnostic=reconstruct_sr(diff,low,.5,2,mode,seed=34,batch_size=19)
            self.assertTrue(info['converged'] and diagnostic['solver']['converged'])
            torch.testing.assert_close(actual,expected.clamp(0,1),atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(diff.HR_MSI,hr,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
