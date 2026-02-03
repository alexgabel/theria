import pytest
import torch

from theria.attention.custom import sdpa_custom, sdpa_custom_jvp


def _make_inputs(device):
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

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


def _finite_difference_jvp(f, q, k, v, dq, dk, dv, eps=5e-4):
    return (f(q + eps * dq, k + eps * dk, v + eps * dv) - f(q, k, v)) / eps


def test_custom_attention_jvp_matches_finite_difference_cpu():
    device = torch.device("cpu")
    q, k, v, dq, dk, dv = _make_inputs(device)

    def f(q, k, v):
        return sdpa_custom(q, k, v, backend="custom")

    jvp_explicit = sdpa_custom_jvp(q, k, v, dq, dk, dv)
    jvp_fd = _finite_difference_jvp(f, q, k, v, dq, dk, dv)

    assert torch.isfinite(jvp_explicit).all()
    assert torch.isfinite(jvp_fd).all()
    torch.testing.assert_close(jvp_explicit, jvp_fd, rtol=1e-4, atol=1e-4)


def test_custom_attention_jvp_matches_autograd_cpu():
    device = torch.device("cpu")
    if not hasattr(torch.autograd.functional, "jvp"):
        pytest.skip("torch.autograd.functional.jvp not available")
    q, k, v, dq, dk, dv = _make_inputs(device)

    def f(q, k, v):
        # Use reference backend to keep forward-mode purely in primitives.
        return sdpa_custom(q, k, v, backend="reference")

    _, jvp_auto = torch.autograd.functional.jvp(f, (q, k, v), (dq, dk, dv))
    jvp_explicit = sdpa_custom_jvp(q, k, v, dq, dk, dv)
    jvp_fd = _finite_difference_jvp(f, q, k, v, dq, dk, dv)

    assert torch.isfinite(jvp_auto).all()
    assert torch.isfinite(jvp_explicit).all()
    assert torch.isfinite(jvp_fd).all()
    torch.testing.assert_close(jvp_explicit, jvp_auto, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(jvp_explicit, jvp_fd, rtol=1e-4, atol=1e-4)


def test_custom_attention_jvp_not_vjp_cpu():
    device = torch.device("cpu")
    q, k, v, dq, dk, dv = _make_inputs(device)

    # JVP is output-shaped; VJP is input-shaped. They should not be equal.
    jvp = sdpa_custom_jvp(q, k, v, dq, dk, dv)

    def loss_fn(q, k, v):
        return sdpa_custom(q, k, v, backend="reference").sum()

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    loss = loss_fn(q, k, v)
    grad_q, grad_k, grad_v = torch.autograd.grad(loss, (q, k, v))
    vjp_concat = torch.cat([grad_q.reshape(-1), grad_k.reshape(-1), grad_v.reshape(-1)])
    jvp_flat = jvp.reshape(-1)

    assert jvp_flat.shape != vjp_concat.shape

@pytest.mark.gpu
def test_custom_attention_jvp_matches_finite_difference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU JVP test.")
    device = torch.device("cuda")
    q, k, v, dq, dk, dv = _make_inputs(device)

    def f(q, k, v):
        return sdpa_custom(q, k, v, backend="custom")

    jvp_explicit = sdpa_custom_jvp(q, k, v, dq, dk, dv)
    jvp_fd = _finite_difference_jvp(f, q, k, v, dq, dk, dv)

    assert torch.isfinite(jvp_explicit).all()
    assert torch.isfinite(jvp_fd).all()
    torch.testing.assert_close(jvp_explicit, jvp_fd, rtol=1e-4, atol=1e-4)
