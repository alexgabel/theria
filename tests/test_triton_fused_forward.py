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
