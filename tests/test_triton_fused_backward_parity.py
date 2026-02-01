import pytest
import torch

from theria.attention.custom import sdpa_custom
from theria.attention.reference import reference_attention
from theria.attention.reference_backward_sdpa import sdpa_backward_reference


class RefSDPAExplicit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        scale = 1.0 / (q.shape[-1] ** 0.5)
        out = reference_attention(q, k, v) * 1.0  # reference_attention already applies scaling internally
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = sdpa_backward_reference(q, k, v, grad_out, ctx.scale)
        return dq, dk, dv


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape",
    [
        {"B": 1, "H": 1, "T": 32, "M": 32, "D": 32},
        {"B": 2, "H": 4, "T": 128, "M": 128, "D": 64},
    ],
)
def test_triton_fused_backward_matches_reference_cuda(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    B, H, T, M, D = shape["B"], shape["H"], shape["T"], shape["M"], shape["D"]
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16, requires_grad=True)

    def grad_tuple(backend):
        out = sdpa_custom(q, k, v, backend=backend)
        loss = out.sum()
        return torch.autograd.grad(loss, (q, k, v), retain_graph=True, create_graph=False)

    grads_ref = grad_tuple("reference")
    grads_tri = grad_tuple("triton_full_fused")
    for g_tri, g_ref in zip(grads_tri, grads_ref):
        torch.testing.assert_close(g_tri, g_ref, rtol=2e-2, atol=2e-2)


def test_reference_backward_gradcheck_cpu():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 3, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 1, 2, 3, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 1, 2, 3, dtype=torch.double, requires_grad=True)

    def func(q, k, v):
        return RefSDPAExplicit.apply(q, k, v)

    assert torch.autograd.gradcheck(func, (q, k, v), eps=1e-6, atol=1e-6, rtol=1e-4)
