import torch
from torch.autograd import Function
from theria.attention.reference import sdpa_reference


def sdpa(q, k, v):
    """
    Public SDPA entry point.
    All callers should use this, not SDPAFunction.apply directly.
    """
    return SDPAFunction.apply(q, k, v)


class SDPAFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        ctx.save_for_backward(q, k, v)
        return sdpa_reference(q, k, v)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        with torch.enable_grad():
            out = sdpa_reference(q, k, v)

        grads = torch.autograd.grad(
            out,
            (q, k, v),
            grad_out,
            retain_graph=False,
            allow_unused=False,
        )
        return grads