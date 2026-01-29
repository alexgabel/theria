"""
Custom attention entrypoint for Phase 4 experiments.

Currently forwards to the reference implementation; replace with a
JVP/HVP-aware custom kernel in subsequent steps.
"""

import torch

from .reference import reference_attention
from theria.autograd.sdpa_custom_function import SDPACustom
from theria.attention.triton_qk import triton_qk, triton_qk_fast, triton_qk_softmax


def sdpa_custom(q, k, v, *, backend: str = "reference"):
    """
    Phase 4 public API.

    Args:
        q, k, v: attention inputs (B, H, T, D / Dv)
        backend: "reference" (default) or "custom" (uses SDPACustom.apply)

    The backend switch stays narrow so future kernels can slot in without
    changing call sites.
    """
    if backend == "reference":
        return reference_attention(q, k, v)
    if backend == "custom":
        return SDPACustom.apply(q, k, v)
    if backend in ("triton_qk", "triton_ref", "triton"):
        # Triton for QK^T, PyTorch for softmax and PV
        scores = triton_qk(q, k).to(torch.float32) / (q.shape[-1] ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        v_cast = v.to(probs.dtype)
        out = torch.matmul(probs, v_cast)
        return out.to(q.dtype)
    if backend == "triton_fast":
        # Fast Triton QK^T (tensor-core friendly); expect numerical drift.
        scores = triton_qk_fast(q, k).to(torch.float32) / (q.shape[-1] ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        v_cast = v.to(probs.dtype)
        out = torch.matmul(probs, v_cast)
        return out.to(q.dtype)
    if backend == "triton_fast_fused":
        # Fused QK + scale + softmax in Triton; PV remains in PyTorch. Fast path.
        probs = triton_qk_softmax(q, k).to(torch.float32)
        v_cast = v.to(probs.dtype)
        out = torch.matmul(probs, v_cast)
        return out.to(q.dtype)
    raise ValueError(f"Unsupported sdpa_custom backend: {backend}")


def sdpa_custom_hvp(q, k, v, dq, dk, dv):
    """
    Explicit HVP for the custom backend (public helper for tests and debugging).
    """
    return SDPACustom.hvp(q, k, v, dq, dk, dv)


def sdpa_custom_jvp(q, k, v, dq, dk, dv):
    """
    Explicit JVP for the custom backend (public helper for tests and debugging).
    """
    return SDPACustom.jvp(q, k, v, dq, dk, dv)
