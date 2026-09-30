"""Bregman residual reinjection for the existing Soft100 terminal objective.

The diffusion reference and LR-derived spectral basis remain fixed. Only the
observation targets are shifted by accumulated residuals; this targets the
bias of finite-strength data penalties. Intended here for known, noise-free
synthetic observations; a noisy-data discrepancy rule is not implemented.
"""
import copy
import torch


@torch.no_grad()
def refine_residual(prox,reference,basis,outer_steps=5,max_iter=200,tolerance=1e-5):
    if outer_steps < 1:raise ValueError('outer_steps must be positive')
    shifted=copy.copy(prox)
    dual_low=torch.zeros_like(prox.low);dual_hr=torch.zeros_like(prox.hr)
    records=[]
    for step in range(outer_steps):
        shifted.low=prox.low+dual_low;shifted.hr=prox.hr+dual_hr
        result,info=shifted.solve_spectral(reference,100.,basis,10.,max_iter,tolerance)
        if not info['converged']:raise RuntimeError(f'Inner solve failed: {info}')
        low,hr=prox.forward(result)
        residual_low=prox.low-low;residual_hr=prox.hr-hr
        dual_low+=residual_low;dual_hr+=residual_hr
        records.append(dict(outer_step=step+1,lr_mse=float(residual_low.square().mean()),
                            msi_mse=float(residual_hr.square().mean()),**info))
    return result,dict(outer_steps=outer_steps,converged=all(r['converged'] for r in records),
                       iterations=sum(r['iterations'] for r in records),inner=records)
