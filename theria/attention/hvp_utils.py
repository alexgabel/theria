"""
Finite-difference Hessian-vector products for SDPA using explicit Triton backward.

This is a sanity utility for Phase 9: it computes HVPs without autograd-in-backward
by central-differencing the explicit VJP (dQ/dK/dV kernels).

Contract: forward stats (m, l) are recomputed for each perturbed point; this mirrors
the true function f(q, k, v) = sum(triton_sdpa_fused(q, k, v)).
"""

import torch

from theria.attention.triton_qk import triton_sdpa_fused
from theria.attention.triton_sdpa_backward import sdpa_bwd_dq, sdpa_bwd_dk, sdpa_bwd_dv


def _explicit_grads_sum(q, k, v):
    """Return grads of sum(output) w.r.t q,k,v using explicit Triton backward."""
    B, H, T, D = q.shape
    M = k.shape[2]
    scale = 1.0 / (D ** 0.5)
    out, m, l = triton_sdpa_fused(q, k, v, return_stats=True)
    dout = torch.ones_like(out, dtype=q.dtype, device=q.device)
    dq = sdpa_bwd_dq(q, k, v, dout, m, l, scale)
    dk = sdpa_bwd_dk(q, k, v, dout, m, l, scale)
    dv = sdpa_bwd_dv(q, k, dout, m, l, scale)
    return dq, dk, dv


def hvp_fd_vjp(q, k, v, dq, dk, dv, eps=1e-3):
    """
    Finite-difference HVP: central difference of explicit VJP along direction (dq,dk,dv).

    Args:
        q,k,v: tensors on CUDA (float16/float32).
        dq,dk,dv: direction tensors, same shapes.
        eps: finite-difference step.
    Returns:
        hvp_q, hvp_k, hvp_v tensors matching q/k/v shapes.
    """
    with torch.no_grad():
        dq_plus, dk_plus, dv_plus = _explicit_grads_sum(q + eps * dq, k + eps * dk, v + eps * dv)
        dq_minus, dk_minus, dv_minus = _explicit_grads_sum(q - eps * dq, k - eps * dk, v - eps * dv)

    inv_step = 0.5 / eps
    hvp_q = (dq_plus - dq_minus) * inv_step
    hvp_k = (dk_plus - dk_minus) * inv_step
    hvp_v = (dv_plus - dv_minus) * inv_step
    return hvp_q, hvp_k, hvp_v


__all__ = ["hvp_fd_vjp"]
