import math
import os

import pytest
import torch

from theria.attention.triton_sdpa_backward import sdpa_hvp


@pytest.mark.phase9
def test_triton_sdpa_hvp_matches_autograd_cpu():
    os.environ["THERIA_TRITON_JVP_HVP"] = "1"
    if not hasattr(torch.autograd.functional, "hvp"):
        pytest.skip("torch.autograd.functional.hvp not available")

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64

    B, H, T, D = 1, 1, 4, 4
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)

    vq = torch.randn_like(q)
    vk = torch.randn_like(k)
    vv = torch.randn_like(v)

    # Compute frozen stats from the base point
    with torch.no_grad():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        m = scores.max(dim=-1).values
        l = torch.exp(scores - m.unsqueeze(-1)).sum(dim=-1)

    def loss_fn(q_, k_, v_):
        scores_ = torch.matmul(q_, k_.transpose(-2, -1)) * scale
        p = torch.softmax(scores_, dim=-1)
        out = torch.matmul(p, v_)
        return out.sum()

    _, hvp_auto = torch.autograd.functional.hvp(
        loss_fn, (q, k, v), (vq, vk, vv)
    )

    hvp_explicit = sdpa_hvp(q, k, v, vq, vk, vv, m, l, scale)

    for got, ref in zip(hvp_explicit, hvp_auto):
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.phase9
def test_triton_sdpa_jvp_matches_autograd_cpu():
    os.environ["THERIA_TRITON_JVP_HVP"] = "1"
    if not hasattr(torch.autograd.functional, "jvp"):
        pytest.skip("torch.autograd.functional.jvp not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    B, H, T, D = 1, 1, 4, 4
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)

    with torch.no_grad():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        m = scores.max(dim=-1).values
        l = torch.exp(scores - m.unsqueeze(-1)).sum(dim=-1)

    def f(q_, k_, v_):
        scores_ = torch.matmul(q_, k_.transpose(-2, -1)) * scale
        p = torch.exp(scores_ - m.unsqueeze(-1)) / l.unsqueeze(-1)
        return torch.matmul(p, v_)

    _, jvp_auto = torch.autograd.functional.jvp(f, (q, k, v), (dq, dk, dv))
    from theria.attention.triton_sdpa_backward import sdpa_jvp
    jvp_explicit = sdpa_jvp(q, k, v, dq, dk, dv, m, l, scale)

    torch.testing.assert_close(jvp_explicit, jvp_auto, rtol=1e-6, atol=1e-6)
