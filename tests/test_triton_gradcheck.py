import pytest
import torch

from theria.attention.triton_qk import TritonQKFunction


@pytest.mark.gpu
def test_triton_qk_gradcheck_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    # Triton kernel is fp16/fp32; use fp32 gradcheck as a coarse sanity check.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q = torch.randn(1, 1, 2, 3, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(1, 1, 2, 3, device="cuda", dtype=torch.float32, requires_grad=True)

        def func(q, k):
            return TritonQKFunction.apply(q, k)

        assert torch.autograd.gradcheck(
            func, (q, k), eps=1e-3, atol=1e-3, rtol=1e-2, fast_mode=True
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
