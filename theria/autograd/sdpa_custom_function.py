"""
Custom autograd.Function for Phase 4 attention experiments.

Designed for correctness and higher-order autodiff (JVP/HVP) — not performance.
"""

import math
import torch

from theria.attention.reference_hvp_sdpa import sdpa_hvp
from theria.attention.reference_jvp_sdpa import sdpa_jvp


class SDPACustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        """
        Forward pass: explicit softmax attention.
        Saves only inputs (q, k, v) and scale; backward recomputes intermediates
        under autograd to keep higher-order gradients valid.
        """
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)

        ctx.scale = scale
        ctx.save_for_backward(q, k, v)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale

        # NOTE: This is intentionally autograd-in-backward. It is correctness
        # scaffolding to preserve higher-order gradients during Phase 4 and
        # will be replaced by an explicit backward/JVP/HVP implementation later.
        with torch.enable_grad():
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)

        grad_q, grad_k, grad_v = torch.autograd.grad(
            outputs=out,
            inputs=(q, k, v),
            grad_outputs=grad_out,
            retain_graph=True,   # allow double backward
            create_graph=True,   # keep backward differentiable for HVP
            allow_unused=False,
        )

        return grad_q, grad_k, grad_v

    @staticmethod
    def hvp(q, k, v, dq, dk, dv):
        """
        Explicit Hessian–vector product helper for tests.
        Uses the analytic reference HVP to stay aligned with the math contract.
        """
        return sdpa_hvp(q, k, v, dq, dk, dv)

    @staticmethod
    def jvp(q, k, v, dq, dk, dv):
        """
        Explicit Jacobian–vector product helper for tests.
        Uses the analytic reference JVP to stay aligned with the math contract.
        """
        scale = 1.0 / (q.shape[-1] ** 0.5)
        return sdpa_jvp(q, k, v, dq, dk, dv, scale)
