import pytest
import torch

from theria.attention.custom import sdpa_custom


@pytest.mark.gpu
def test_triton_fast_matches_reference_relaxed_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tensor cores
    try:
        torch.manual_seed(0)
        q = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_fast = sdpa_custom(q, k, v, backend="triton_fast").to(torch.float32)
        out_ref = sdpa_custom(q, k, v, backend="reference").to(torch.float32)

        cos = torch.nn.functional.cosine_similarity(out_fast.flatten(), out_ref.flatten(), dim=0)
        assert cos > 0.99
        torch.testing.assert_close(out_fast, out_ref, rtol=1e-1, atol=1e-1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
