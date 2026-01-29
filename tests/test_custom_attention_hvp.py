import math

import pytest
import torch

from theria.attention.custom import sdpa_custom, sdpa_custom_hvp


def _finite_difference_hvp(loss_fn, q, k, v, dq, dk, dv, eps=1e-3):
    """
    Central-difference HVP approximation:
        (grad(q+eps*v) - grad(q-eps*v)) / (2*eps)
    applied jointly to (q,k,v) directions.
    """
    def grads_at(q_, k_, v_):
        q_ = q_.detach().requires_grad_(True)
        k_ = k_.detach().requires_grad_(True)
        v_ = v_.detach().requires_grad_(True)
        loss = loss_fn(q_, k_, v_)
        return torch.autograd.grad(loss, (q_, k_, v_), create_graph=False)

    q_plus = q + eps * dq
    k_plus = k + eps * dk
    v_plus = v + eps * dv

    q_minus = q - eps * dq
    k_minus = k - eps * dk
    v_minus = v - eps * dv

    g_plus = grads_at(q_plus, k_plus, v_plus)
    g_minus = grads_at(q_minus, k_minus, v_minus)

    hvp = tuple((gp - gm) / (2 * eps) for gp, gm in zip(g_plus, g_minus))
    return hvp


def _make_inputs(device):
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8, device=device, dtype=torch.double)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)

    eps = 1e-12
    dq = dq / (dq.norm() + eps)
    dk = dk / (dk.norm() + eps)
    dv = dv / (dv.norm() + eps)
    return q, k, v, dq, dk, dv


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_custom_attention_hvp_matches_finite_difference(device):
    device = torch.device(device)
    q, k, v, dq, dk, dv = _make_inputs(device)

    def loss_fn(q, k, v):
        return sdpa_custom(q, k, v, backend="custom").sum()

    hvp_explicit = sdpa_custom_hvp(q, k, v, dq, dk, dv)
    hvp_fd = _finite_difference_hvp(loss_fn, q, k, v, dq, dk, dv, eps=1e-3)

    for explicit, fd in zip(hvp_explicit, hvp_fd):
        torch.testing.assert_close(explicit, fd, rtol=1e-3, atol=1e-3)


def test_custom_attention_hvp_flag_plant_cpu():
    """
    Phase 4 flag-planting test:
    - forward -> first grads (create_graph=True)
    - explicit HVP helper vs finite-difference directional derivative
    """
    device = torch.device("cpu")
    q, k, v, dq, dk, dv = _make_inputs(device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    def loss_fn(q, k, v):
        return sdpa_custom(q, k, v, backend="custom").sum()

    # Autograd double-backward HVP (should exist)
    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
    dot = (grads[0] * dq).sum() + (grads[1] * dk).sum() + (grads[2] * dv).sum()
    hvp_autograd = torch.autograd.grad(dot, (q, k, v), retain_graph=False, allow_unused=False)

    # Explicit analytic HVP helper
    hvp_explicit = sdpa_custom_hvp(q, k, v, dq, dk, dv)

    # Finite-difference reference
    hvp_fd = _finite_difference_hvp(loss_fn, q, k, v, dq, dk, dv, eps=1e-3)

    for t in hvp_autograd + hvp_explicit + hvp_fd:
        assert torch.isfinite(t).all()

    for a, b in zip(hvp_explicit, hvp_fd):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)

    for a, b in zip(hvp_autograd, hvp_fd):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
