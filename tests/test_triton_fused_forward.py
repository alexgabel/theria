import pytest
import torch

from theria.attention.custom import sdpa_custom


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape",
    [
        {"B": 1, "H": 1, "T": 32, "M": 32, "D": 32},
        {"B": 4, "H": 8, "T": 128, "M": 128, "D": 64},
    ],
)
def test_triton_fused_forward_matches_reference_cuda(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(0)
        q = torch.randn(shape["B"], shape["H"], shape["T"], shape["D"], device="cuda", dtype=torch.float16)
        k = torch.randn(shape["B"], shape["H"], shape["M"], shape["D"], device="cuda", dtype=torch.float16)
        v = torch.randn(shape["B"], shape["H"], shape["M"], shape["D"], device="cuda", dtype=torch.float16)

        out_fused = sdpa_custom(q, k, v, backend="triton_full_fused").to(torch.float32)
        out_ref = sdpa_custom(q, k, v, backend="reference").to(torch.float32)

        torch.testing.assert_close(out_fused, out_ref, rtol=2e-2, atol=2e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@pytest.mark.gpu
def test_triton_fused_handles_zero_rows():
    """
    Regression: when T << BLOCK_M many rows are masked, so l_i can be zero.
    Kernel clamps l_i; outputs must stay finite and match reference for valid rows.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(0)
        q = torch.randn(1, 1, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_fused = sdpa_custom(q, k, v, backend="triton_full_fused")
        out_ref = sdpa_custom(q, k, v, backend="reference")

        assert torch.isfinite(out_fused).all()
        torch.testing.assert_close(out_fused.to(torch.float32), out_ref.to(torch.float32), rtol=2e-2, atol=2e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
