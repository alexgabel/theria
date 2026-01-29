import pytest
import torch

from theria.attention.custom import sdpa_custom


@pytest.mark.gpu
def test_triton_double_backward_exists_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 3, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)
    eps = 1e-6
    dq = dq / (dq.norm() + eps)
    dk = dk / (dk.norm() + eps)
    dv = dv / (dv.norm() + eps)

    def loss_fn(q, k, v):
        return sdpa_custom(q, k, v, backend="triton").sum()

    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
    dot = (grads[0] * dq).sum() + (grads[1] * dk).sum() + (grads[2] * dv).sum()
    hvp = torch.autograd.grad(dot, (q, k, v))

    for t in hvp:
        assert torch.isfinite(t).all()
