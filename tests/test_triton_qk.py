import pytest
import torch

from theria.attention.triton_qk import triton_qk
from theria.attention.custom import sdpa_custom


@pytest.mark.gpu
def test_triton_qk_matches_reference_scores_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        q = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True)

        scores_triton = triton_qk(q, k).to(torch.float32)
        scores_ref = torch.matmul(q, k.transpose(-2, -1)).to(torch.float32)

        torch.testing.assert_close(scores_triton, scores_ref, rtol=5e-3, atol=5e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@pytest.mark.gpu
def test_triton_qk_backend_end_to_end_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        q = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_triton = sdpa_custom(q, k, v, backend="triton_qk").to(torch.float32)
        out_ref = sdpa_custom(q, k, v, backend="reference").to(torch.float32)

        torch.testing.assert_close(out_triton, out_ref, rtol=5e-3, atol=5e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
