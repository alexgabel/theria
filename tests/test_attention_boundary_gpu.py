import pytest
import torch
from theria.attention.custom import sdpa_custom

# CUDA availability guard
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available, skipping GPU boundary tests.", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.boundary]


def test_reference_attention_double_backward_cuda():
    from theria.attention.reference import reference_attention
    from theria.attention.custom import sdpa_custom

    q, k, v, dq, dk, dv = _make_qkv_cuda()

    def loss_fn(q, k, v):
        return reference_attention(q, k, v).sum()

    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)

    dot = (grads[0] * dq).sum() + (grads[1] * dk).sum() + (grads[2] * dv).sum()

    hvp = torch.autograd.grad(dot, (q, k, v))

    for t in hvp:
        assert torch.isfinite(t).all()


def test_custom_attention_double_backward_cuda():
    q, k, v, dq, dk, dv = _make_qkv_cuda()

    def loss_fn(q, k, v):
        return sdpa_custom(q, k, v, backend="custom").sum()

    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)

    dot = (grads[0] * dq).sum() + (grads[1] * dk).sum() + (grads[2] * dv).sum()
    hvp = torch.autograd.grad(dot, (q, k, v))

    for t in hvp:
        assert torch.isfinite(t).all()


@pytest.mark.xfail(reason="FlashAttention does not support double backward")
def test_flash_attention_double_backward_fails():
    from torch.backends.cuda import sdp_kernel
    q, k, v, dq, dk, dv = _make_qkv_cuda()

    def loss_fn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        ).sum()

    with sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
        loss = loss_fn(q, k, v)
        grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)

        dot = (grads[0] * dq).sum() + (grads[1] * dk).sum() + (grads[2] * dv).sum()

        torch.autograd.grad(dot, (q, k, v))


def _make_qkv_cuda(dtype=torch.float64):
    device = torch.device("cuda")
    torch.manual_seed(0)

    q = torch.randn(1, 1, 4, 8, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)

    eps = 1e-12
    dq = dq / (dq.norm() + eps)
    dk = dk / (dk.norm() + eps)
    dv = dv / (dv.norm() + eps)

    return q, k, v, dq, dk, dv
