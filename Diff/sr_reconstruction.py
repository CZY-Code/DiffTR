"""Reusable fixed-checkpoint SR inference with observation/spectral refinement.

Soft1000 is an experimental, benchmark-selected configuration for the current
known-operator, noise-free fusion protocol. It does not change model training.
The caller supplies only observed LR-HSI and the trained diffusion object,
which holds HR-MSI and the spectral response; no HR-HSI target is accepted.
"""
from Diff.sr_guidance import sample_observation_guided
from Diff.sr_subspace import SpectralProximal, estimate_spectral_basis


def reconstruct_sr(diff, lr_hsi, sigma, scale, refinement='soft1000',
                   seed=3407, batch_size=2048):
    """Return the clipped HWC reconstruction and observation-only diagnostics.

    Use refinement='soft100' to reproduce the preceding paper configuration.
    The original stepwise MSI guide is retained at strength 15 for both rules.
    For the benchmark's memory/timing setup, use diff.eval().requires_grad_(False)
    and place the model and observations on the same device.
    """
    configs={'soft100':(100.,200),'soft1000':(1000.,400)}
    if refinement not in configs:raise ValueError(f'Unknown refinement: {refinement}')
    strength,max_iter=configs[refinement]
    basis,basis_info=estimate_spectral_basis(lr_hsi,.999)
    prox=SpectralProximal(diff.HR_MSI,lr_hsi,diff.Matrix_R,sigma,scale)
    raw=sample_observation_guided(diff,lr_hsi,sigma,scale,'original',guidance=15.,
                                 seed=seed,batch_size=batch_size)
    result,solver=prox.solve_spectral(raw,strength,basis,10.,max_iter,1e-5)
    if not solver['converged']:raise RuntimeError(f'SR terminal solve failed: {solver}')
    return result.clamp(0,1),dict(refinement=refinement,basis=basis_info,solver=solver,
        sampling_seed=seed,guidance=15.,clipping='after terminal refinement')
