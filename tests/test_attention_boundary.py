# tests/test_attention_boundary.py
"""
Phase 3 CPU boundary witness tests for attention mechanisms.
"""

import pytest
import torch

from tests._graph_inspect import can_double_backward


def _make_qkv(device, dtype, B=1, H=1, T=4, D=8):
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def _hvp_double_backward(loss_fn, q, k, v, dq, dk, dv):
    """
    Returns (hvp_q, hvp_k, hvp_v) via double backward:
      hvp = ∇_{q,k,v} <∇L, (dq,dk,dv)>
    Raises RuntimeError if grad-of-grad is unsupported.
    """
    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True, allow_unused=False)
    dot = (grads[0] * dq).sum() + (grads[1] * dk).sum() + (grads[2] * dv).sum()
    hvp = torch.autograd.grad(dot, (q, k, v), allow_unused=False)
    return hvp


def _reference_attention(q, k, v):
    # (B,H,T,D) -> (B,H,T,D)
    d = q.shape[-1]
    s = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    p = torch.softmax(s, dim=-1)
    return p @ v


def test_boundary_reference_attention_supports_double_backward_cpu():
    device = torch.device("cpu")
    dtype = torch.double
    torch.manual_seed(0)

    q, k, v = _make_qkv(device, dtype)
    dq, dk, dv = (torch.randn_like(q), torch.randn_like(k), torch.randn_like(v))
    # Normalize direction vectors for numerical stability
    eps = 1e-12
    dq = dq / (dq.norm() + eps)
    dk = dk / (dk.norm() + eps)
    dv = dv / (dv.norm() + eps)

    def loss_fn(q, k, v):
        return _reference_attention(q, k, v).sum()

    out = loss_fn(q, k, v)
    assert out.grad_fn is not None
    assert can_double_backward(out, [q, k, v])
    # NOTE: reductions (like .sum()) may introduce benign detached edges in the graph,
    # but these do not affect the semantic correctness of higher-order differentiation.


    hvp_q, hvp_k, hvp_v = _hvp_double_backward(loss_fn, q, k, v, dq, dk, dv)
    assert torch.isfinite(hvp_q).all()
    assert torch.isfinite(hvp_k).all()
    assert torch.isfinite(hvp_v).all()


@pytest.mark.xfail(reason="SDPAFunction backward is opaque; double backward is not supported (expected boundary)")
def test_boundary_sdpa_function_double_backward_fails_cpu():
    """
    This is the *expected* boundary: SDPAFunction backward is opaque,
    so grad-of-grad via double backward fails.
    """
    from theria.autograd.sdpa_function import SDPAFunction

    device = torch.device("cpu")
    dtype = torch.double
    torch.manual_seed(0)

    q, k, v = _make_qkv(device, dtype)
    dq, dk, dv = (torch.randn_like(q), torch.randn_like(k), torch.randn_like(v))
    # Normalize direction vectors for numerical stability
    eps = 1e-12
    dq = dq / (dq.norm() + eps)
    dk = dk / (dk.norm() + eps)
    dv = dv / (dv.norm() + eps)

    def loss_fn(q, k, v):
        return SDPAFunction.apply(q, k, v).sum()

    out = loss_fn(q, k, v)
    assert out.grad_fn is not None
    # Expected boundary: double backward not supported
    assert not can_double_backward(out, [q, k, v])
    with pytest.raises(RuntimeError):
        _hvp_double_backward(loss_fn, q, k, v, dq, dk, dv)
