import pytest
import torch

from theria.attention.custom import sdpa_custom


@pytest.mark.gpu
def test_triton_fused_backward_matches_reference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(0)
        q = torch.randn(1, 1, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        def loss_fn(backend):
            out = sdpa_custom(q, k, v, backend=backend)
            return out.sum()

        loss_ref = loss_fn("reference")
        grads_ref = torch.autograd.grad(loss_ref, (q, k, v), retain_graph=True, create_graph=False)

        loss_fused = loss_fn("triton_full_fused")
        grads_fused = torch.autograd.grad(loss_fused, (q, k, v), retain_graph=True, create_graph=False)

        for g_fused, g_ref in zip(grads_fused, grads_ref):
            torch.testing.assert_close(g_fused, g_ref, rtol=2e-2, atol=2e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
