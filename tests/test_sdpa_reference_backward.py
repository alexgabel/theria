import pytest
import torch

from theria.attention.reference import reference_attention
from theria.attention.reference_backward import sdpa_backward_reference


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sdpa_reference_backward_matches_autograd(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dev = torch.device(device)
    B, H, T, M, D = 1, 1, 8, 8, 16
    q = torch.randn(B, H, T, D, device=dev, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, M, D, device=dev, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, M, D, device=dev, dtype=torch.float32, requires_grad=True)

    def loss_fn(q, k, v):
        return reference_attention(q, k, v).sum()

    loss = loss_fn(q, k, v)
    dq_auto, dk_auto, dv_auto = torch.autograd.grad(loss, (q, k, v), retain_graph=False, create_graph=False)

    dout = torch.ones_like(reference_attention(q, k, v))
    scale = 1.0 / (D**0.5)
    dq_ref, dk_ref, dv_ref = sdpa_backward_reference(q, k, v, dout, scale)

    torch.testing.assert_close(dq_ref, dq_auto, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dk_ref, dk_auto, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dv_ref, dv_auto, rtol=1e-5, atol=1e-5)
