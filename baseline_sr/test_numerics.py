"""Independent numerical checks for nontrivial MATLAB/operator porting."""
import unittest
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from baseline_sr.protocol import Observations, cg
from baseline_sr.tensor_methods import Patches, shrink, kmeans
from baseline_sr.cs2dips import DeformConv2dPack

class Numerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)

    def test_opencv_and_transpose(self):
        for sigma in (.5,2):
            x = torch.randn(16,24,3,dtype=torch.float64)
            o = Observations(torch.zeros(4,6,3),torch.zeros(16,24,2),torch.randn(2,3,dtype=x.dtype),sigma,4)
            expected = np.stack([cv2.GaussianBlur(x[...,i].numpy(),(7,7),sigma,borderType=cv2.BORDER_REPLICATE) for i in range(3)],-1)
            expected = expected.reshape(4,4,6,4,3).mean((1,3))
            np.testing.assert_allclose(o.lr(x),expected,rtol=1e-12,atol=1e-12)
            for op in (o.lr,o.hr):
                y = torch.randn_like(op(x))
                self.assertAlmostEqual(float((op(x)*y).sum()),float((x*o.adjoint(op,y,3)).sum()),places=10)

    def test_cg_against_dense_solve(self):
        x = torch.randn(32,32,dtype=torch.float64)
        a = x.T@x+torch.eye(32,dtype=torch.float64)
        b = torch.randn(32,dtype=torch.float64)
        value,info = cg(lambda t:a@t,b,torch.zeros_like(b),tol=1e-10)
        torch.testing.assert_close(value,torch.linalg.solve(a,b),rtol=1e-8,atol=1e-9)
        self.assertLess(info['relative_residual'],1e-9)

    def test_matlab_patch_order_and_overlap(self):
        x = torch.randn(16,20,3)
        for s in (6,8):
            p = Patches(16,20,s,x.device)
            expected = np.stack([x[i:i+s,j:j+s].numpy().reshape(s*s,3,order='F')
                                 for j in range(0,20-s+1,s-4) for i in range(0,16-s+1,s-4)],-1)
            np.testing.assert_array_equal(p.extract(x),expected)
            torch.testing.assert_close(p.join(p.extract(x)),x,rtol=2e-6,atol=2e-6)

    def test_shrink_against_numpy_reference(self):
        for learned in (False,True):
            for shape in ((2,4,3,5),(2,2,2,7),(2,4,3,1)):
                y = torch.randn(shape,dtype=torch.float64)
                expected = []
                for b in y.numpy():
                    z = np.fft.fft(b,axis=2)
                    u3 = None
                    if learned and b.shape[2]>1:
                        flat = z.reshape(-1,b.shape[2],order='F').T
                        u3,_,_ = np.linalg.svd(flat,full_matrices=flat.shape[0]>flat.shape[1])
                        z = (u3.conj().T@flat).T.reshape(b.shape,order='F')
                    out = np.empty_like(z)
                    for k in range(z.shape[2]):
                        u,s,v = np.linalg.svd(z[:,:,k],full_matrices=False)
                        disc = s*s-4*.3
                        s = np.where(disc>0,(s+np.sqrt(np.maximum(disc,0)))/2,0)
                        out[:,:,k] = (u*s)@v
                    if u3 is not None:
                        flat = out.reshape(-1,b.shape[2],order='F').T
                        out = (u3@flat).T.reshape(b.shape,order='F')
                    expected.append(np.fft.ifft(out,axis=2).real)
                np.testing.assert_allclose(shrink(y,.3,learned).numpy(),expected,rtol=1e-8,atol=1e-8)

    def test_dcn_zero_offset_and_gradient(self):
        layer = DeformConv2dPack(2,3,3,padding=1).double()
        x = torch.randn(1,2,6,7,dtype=torch.float64,requires_grad=True)
        torch.testing.assert_close(layer(x),F.conv2d(x,layer.weight,padding=1))
        layer(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(layer.conv_offset.weight.grad.abs().sum()),0)

    def test_kmeans_nearly_identical_patches(self):
        torch.manual_seed(3407)
        x = 1+1e-5*torch.randn(200,16)
        info = {}
        labels = kmeans(x,12,info)
        centers = torch.stack([x[labels==k].double().mean(0) for k in labels.unique()])
        # Direct differences avoid the cancellation of dot-product distances.
        distances = (x.double()[:,None]-centers[None]).square().sum(-1)
        self.assertTrue(torch.equal(labels,distances.argmin(1)))
        self.assertTrue(info['float64_refinement'])

if __name__ == '__main__':
    unittest.main()
