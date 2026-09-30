"""Soft spectral-subspace terminal refinement, estimated only from LR-HSI."""
import torch
from Diff.sr_proximal import ObservationProximal


@torch.no_grad()
def estimate_spectral_basis(low, energy=0.999):
    if not 0 < energy <= 1:
        raise ValueError('energy must be in (0,1]')
    spectra = low.reshape(-1, low.shape[-1]).double()
    gram = (spectra.T @ spectra).cpu()
    values, vectors = torch.linalg.eigh(gram)
    values = values.flip(0).clamp_min(0)
    vectors = vectors.flip(1)
    if values.sum() <= 0:
        raise ValueError('Cannot estimate a subspace from a zero observation')
    cumulative = values.cumsum(0)/values.sum()
    rank = min(int(torch.searchsorted(cumulative, energy))+1, len(values))
    basis = vectors[:, :rank].to(device=low.device, dtype=low.dtype)
    return basis, dict(rank=rank, energy_threshold=energy,
                      retained_energy=float(cumulative[rank-1]),
                      centered=False, source='LR-HSI only',
                      eigenvalues=values.tolist())


class SpectralProximal(ObservationProximal):
    @torch.no_grad()
    def solve_spectral(self, reference, strength=100., basis=None, gamma=0.,
                       max_iter=100, tolerance=1e-5):
        if strength < 0 or gamma < 0 or max_iter < 1 or tolerance <= 0:
            raise ValueError('Invalid solver settings')
        if gamma and basis is None:
            raise ValueError('A spectral basis is required for a nonzero penalty')

        def outside(x):
            return x - (x @ basis) @ basis.T

        def normal(x):
            low, hr = self.forward(x)
            value = x + strength * self.adjoint(low/self.low_bound, hr/self.hr_bound)
            if gamma:
                value = value + gamma * outside(x)
            return value

        # Solve for a correction, avoiding cancellation between large DC terms
        # when measuring a small true residual in single precision.
        low, hr = self.forward(reference)
        rhs = strength * self.adjoint((self.low-low)/self.low_bound,
                                      (self.hr-hr)/self.hr_bound)
        if gamma:
            rhs = rhs - gamma * outside(reference)
        correction = torch.zeros_like(reference)
        residual = rhs.clone(); direction = residual.clone()
        rr = residual.square().sum(); initial = rr.sqrt()
        target = tolerance * initial
        count = 0; restarts = 0
        for iteration in range(max_iter):
            if rr == 0 or rr.sqrt() <= target:
                break
            ad = normal(direction)
            curvature = (direction*ad).sum()
            if not torch.isfinite(curvature) or curvature <= 0:
                raise FloatingPointError('Invalid CG curvature')
            alpha = rr / curvature
            correction = correction + alpha * direction
            residual = residual - alpha * ad
            updated = residual.square().sum(); count = iteration+1
            if updated.sqrt() <= target or count % 25 == 0:
                residual = rhs - normal(correction)
                updated = residual.square().sum()
                if updated.sqrt() <= target:
                    rr = updated
                    break
                direction = residual.clone(); restarts += 1
            else:
                direction = residual + (updated/rr)*direction
            rr = updated
        actual = (rhs-normal(correction)).norm()
        ratio = float(actual/initial.clamp_min(1e-30))
        result = reference + correction
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite terminal reconstruction')
        return result, dict(iterations=count, relative_initial_residual=ratio,
                            converged=bool(actual <= target), restarts=restarts,
                            strength=strength, gamma=gamma,
                            rank=None if basis is None else basis.shape[1],
                            max_iter=max_iter,tolerance=tolerance,
                            formulation='correction system with recomputed true residual')
