import pytest
import torch

from theria.attention.custom import sdpa_custom, sdpa_custom_jvp
from theria.attention.reference import reference_attention


def _make_inputs(dtype=torch.float16):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    q = torch.randn(1, 1, 4, 8, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _finite_difference_jvp(f, q, k, v, dq, dk, dv, eps=1e-3):
    # Central difference for lower truncation error
    return (f(q + eps * dq, k + eps * dk, v + eps * dv) - f(q - eps * dq, k - eps * dk, v - eps * dv)) / (2 * eps)


@pytest.mark.gpu
def test_triton_forward_matches_reference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dtype = torch.float16
        q, k, v = _make_inputs(dtype)
        out_triton = sdpa_custom(q, k, v, backend="triton_ref").to(torch.float32)
        out_ref = reference_attention(q, k, v).to(torch.float32)
        torch.testing.assert_close(out_triton, out_ref, rtol=2e-2, atol=2e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@pytest.mark.gpu
def test_triton_jvp_matches_finite_difference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")

        dtype = torch.float32
        q, k, v = _make_inputs(dtype)

        dq = torch.randn_like(q)
        dk = torch.randn_like(k)
        dv = torch.randn_like(v)

        # normalize directions for stability
        eps = 1e-6
        dq = dq / (dq.norm() + eps)
        dk = dk / (dk.norm() + eps)
        dv = dv / (dv.norm() + eps)

        def f(q, k, v):
            return sdpa_custom(q, k, v, backend="triton_ref")

        jvp_explicit = sdpa_custom_jvp(q, k, v, dq, dk, dv).to(torch.float32)
        jvp_fd = _finite_difference_jvp(f, q, k, v, dq, dk, dv, eps=1e-3).to(torch.float32)

        assert torch.isfinite(jvp_explicit).all()
        assert torch.isfinite(jvp_fd).all()
        cos = torch.nn.functional.cosine_similarity(jvp_explicit.flatten(), jvp_fd.flatten(), dim=0)
        assert cos > 0.99
        torch.testing.assert_close(jvp_explicit, jvp_fd, rtol=5e-2, atol=5e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
