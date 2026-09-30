"""Matrix-free, damped observation updates; no additional learned parameters.

The proximal solution corrects predicted x0 in physical units. It is then used
in the existing DDPM posterior mean, rather than substituted for noisy xt.
This is an approximate inverse-problem sampler, not exact posterior sampling.
"""
import torch
from Diff.sr_guidance import spatial_forward


class ObservationProximal:
    def __init__(self, hr_msi, lr_hsi, response, sigma, scale):
        self.hr, self.low, self.response = hr_msi, lr_hsi, response
        self.sigma, self.scale = sigma, scale
        # Exact autograd transposes include replicate-padding boundary sums.
        with torch.enable_grad():
            self.zero = torch.zeros(*hr_msi.shape[:2], lr_hsi.shape[-1],
                                    device=lr_hsi.device, dtype=lr_hsi.dtype,
                                    requires_grad=True)
            self.graph_low, self.graph_hr = self.forward(self.zero)
        ones = torch.ones_like(self.zero)
        with torch.no_grad():
            lo, hi = self.forward(ones)
            # Nonnegative operators: maximum row sum bounds ||A||_2^2.
            # Normalize each observation term by its own bound, not by GT.
            self.low_bound = self.adjoint(lo, None).max().item()
            self.hr_bound = self.adjoint(None, hi).max().item()
            if min(self.low_bound, self.hr_bound) <= 0:
                raise ValueError('Observation operators must be nonzero')
            self.rhs_data = (self.adjoint(lr_hsi, None) / self.low_bound
                             + self.adjoint(None, hr_msi) / self.hr_bound)

    def forward(self, x):
        return (spatial_forward(x, self.sigma, self.scale),
                spatial_forward(x @ self.response.T, self.sigma))

    def adjoint(self, low, hr):
        outputs, gradients = [], []
        if low is not None:
            outputs.append(self.graph_low); gradients.append(low)
        if hr is not None:
            outputs.append(self.graph_hr); gradients.append(hr)
        return torch.autograd.grad(outputs, self.zero, grad_outputs=gradients,
                                   retain_graph=True)[0].detach()

    @torch.no_grad()
    def objective(self, x, reference, strength):
        low, hr = self.forward(x)
        return .5 * ((x-reference).square().sum() + strength * (
            (low-self.low).square().sum()/self.low_bound
            + (hr-self.hr).square().sum()/self.hr_bound))

    @torch.no_grad()
    def solve(self, reference, strength, max_iter=20, tolerance=1e-5):
        if strength < 0 or max_iter < 1 or tolerance <= 0:
            raise ValueError('Invalid proximal solver settings')
        def normal(x):
            lo, hi = self.forward(x)
            return x + strength * self.adjoint(lo/self.low_bound, hi/self.hr_bound)
        rhs = reference + strength * self.rhs_data
        x = reference.clone()
        residual = rhs - normal(x)
        p = residual.clone()
        rr = residual.square().sum()
        # Relative to initial residual: large image DC must not hide correction error.
        initial = rr.sqrt()
        target = tolerance * initial
        count = 0
        for _ in range(max_iter):
            if rr.sqrt() <= target or rr == 0:
                break
            ap = normal(p)
            denom = (p*ap).sum()
            if not torch.isfinite(denom) or denom <= 0:
                raise FloatingPointError('Non-positive/nonfinite CG curvature')
            alpha = rr / denom
            x.add_(p, alpha=alpha.item())
            residual.add_(ap, alpha=-alpha.item())
            updated = residual.square().sum()
            p = residual + (updated/rr)*p
            rr = updated
            count += 1
        true_residual = (rhs-normal(x)).norm()
        return x, dict(iterations=count,
                       relative_initial_residual=float(true_residual/initial.clamp_min(1e-30)),
                       strength=float(strength))


@torch.no_grad()
def sample_proximal(diff, lr_hsi, sigma, scale, strength=10.,
                    cg_steps=20, cg_tolerance=1e-5, batch_size=2048,
                    seed=3407, trace=None):
    if diff.training or diff.self_condition:
        raise ValueError('Requires eval mode and no self-conditioning')
    prox = ObservationProximal(diff.HR_MSI, lr_hsi, diff.Matrix_R, sigma, scale)
    h, w, _ = diff.HR_MSI.shape
    device = diff.betas.device
    coords = torch.cartesian_prod(torch.arange(h, device=device), torch.arange(w, device=device))
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((h*w, 1, diff.channels), generator=gen, device=device)
    for t in reversed(range(diff.num_timesteps)):
        clean = torch.empty_like(x)
        for start in range(0, len(coords), batch_size):
            sl = slice(start, start+batch_size)
            ts = torch.full((len(coords[sl]),), t, dtype=torch.long, device=device)
            clean[sl] = diff.model_predictions(x[sl], coords[sl], ts,
                clip_x_start=True, model_forward_kwargs={'sample': False}).pred_x_start
        physical = diff.unnormalize(clean).reshape(h, w, diff.channels)
        weight = strength * float(diff.alphas_cumprod[t])
        corrected, diagnostic = prox.solve(physical, weight, cg_steps, cg_tolerance)
        # Do not clip the unconstrained proximal solution: clipping changes the solve.
        if trace is not None:
            low, hr = prox.forward(corrected)
            diagnostic.update(t=t, LR_MSE=float((low-lr_hsi).square().mean()),
                              MSI_MSE=float((hr-diff.HR_MSI).square().mean()),
                              low_operator_bound=prox.low_bound, hr_operator_bound=prox.hr_bound)
            trace.append(diagnostic)
        clean = diff.normalize(corrected).reshape_as(x)
        ts = torch.full((h*w,), t, dtype=torch.long, device=device)
        mean, _, logvar = diff.q_posterior(clean, x, ts)
        x = mean
        if t:
            x = x + (.5*logvar).exp()*torch.randn(x.shape, generator=gen, device=device)
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'Non-finite sample at t={t}')
    return diff.unnormalize(x).reshape(h, w, diff.channels)
