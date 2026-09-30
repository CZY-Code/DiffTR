"""Observation-derived edge-aware quadratic regularization of spectral maps."""
import torch
from Diff.sr_subspace import SpectralProximal


class SpatialSpectralProximal(SpectralProximal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            horizontal=(self.hr[:,1:]-self.hr[:,:-1]).square().mean(-1)
            vertical=(self.hr[1:]-self.hr[:-1]).square().mean(-1)
            # A fixed observation-only rule. Do not estimate from the HR-HSI.
            scale=torch.quantile(torch.cat([horizontal.flatten(),vertical.flatten()]),.75).clamp_min(1e-8)
            self.wh=torch.exp(-horizontal/(2*scale)).clamp_min(.05)
            self.wv=torch.exp(-vertical/(2*scale)).clamp_min(.05)
            degree=torch.zeros_like(self.hr[...,0])
            degree[:,1:]+=self.wh;degree[:,:-1]+=self.wh
            degree[1:]+=self.wv;degree[:-1]+=self.wv
            self.spatial_bound=float(2*degree.max())
            self.edge_scale=float(scale)

    def spatial_normal(self,x,basis):
        coefficients=x@basis
        dh=(coefficients[:,1:]-coefficients[:,:-1])*self.wh[...,None]
        dv=(coefficients[1:]-coefficients[:-1])*self.wv[...,None]
        lap=torch.zeros_like(coefficients)
        lap[:,1:]+=dh;lap[:,:-1]-=dh
        lap[1:]+=dv;lap[:-1]-=dv
        return (lap@basis.T)/self.spatial_bound

    @torch.no_grad()
    def solve_spatial(self,reference,basis,spatial_strength=1.,strength=100.,
                      gamma=10.,max_iter=200,tolerance=1e-5):
        if spatial_strength < 0 or strength < 0 or gamma < 0:
            raise ValueError('Regularization strengths must be nonnegative')
        if basis is None or max_iter < 1 or tolerance <= 0:
            raise ValueError('Basis and valid CG controls required')
        def outside(x):return x-(x@basis)@basis.T
        def normal(x):
            low,hr=self.forward(x)
            return (x+strength*self.adjoint(low/self.low_bound,hr/self.hr_bound)
                    +gamma*outside(x)+spatial_strength*self.spatial_normal(x,basis))
        low,hr=self.forward(reference)
        rhs=(strength*self.adjoint((self.low-low)/self.low_bound,(self.hr-hr)/self.hr_bound)
             -gamma*outside(reference)-spatial_strength*self.spatial_normal(reference,basis))
        correction=torch.zeros_like(reference);residual=rhs.clone();direction=residual.clone()
        rr=residual.square().sum();initial=rr.sqrt();target=tolerance*initial
        count=0;restarts=0
        for iteration in range(max_iter):
            if rr==0 or rr.sqrt()<=target:break
            ad=normal(direction);curvature=(direction*ad).sum()
            if not torch.isfinite(curvature) or curvature<=0:raise FloatingPointError('Invalid curvature')
            alpha=rr/curvature;correction=correction+alpha*direction
            residual=residual-alpha*ad;updated=residual.square().sum();count=iteration+1
            if updated.sqrt()<=target or count%25==0:
                residual=rhs-normal(correction);updated=residual.square().sum()
                if updated.sqrt()<=target:
                    rr=updated;break
                direction=residual.clone();restarts+=1
            else:direction=residual+(updated/rr)*direction
            rr=updated
        actual=(rhs-normal(correction)).norm();result=reference+correction
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite reconstruction')
        return result,dict(iterations=count,relative_initial_residual=float(actual/initial.clamp_min(1e-30)),
            converged=bool(actual<=target),restarts=restarts,strength=strength,gamma=gamma,
            spatial_strength=spatial_strength,spatial_bound=self.spatial_bound,edge_scale=self.edge_scale,
            max_iter=max_iter,tolerance=tolerance)
