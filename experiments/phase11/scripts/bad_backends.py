"""
Phase 11 bad backend variants (experiment-only).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.autograd.function import once_differentiable


def detach_attention_output(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Return an attention wrapper that detaches the SDPA output.
    """

    def _wrapped(*args, **kwargs) -> torch.Tensor:
        out = sdpa_fn(*args, **kwargs)
        # Keep tensor usage in-graph while zeroing the attention gradient path.
        return out.detach() + (out * 0.0)

    return _wrapped


def no_grad_attention(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Return an attention wrapper that runs SDPA under no_grad.
    """

    def _wrapped(*args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            out = sdpa_fn(*args, **kwargs)
        return out

    return _wrapped


def once_differentiable_sim(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Wrap sdpa_fn so first-order grads work but gradgrad through attention is blocked.
    """

    def _wrapped(*args, **kwargs) -> torch.Tensor:
        class _OnceDiffSDPA(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *f_args):
                tensor_positions = []
                tensor_requires_grad = []
                tensor_values = []
                non_tensor_values = {}
                for i, a in enumerate(f_args):
                    if torch.is_tensor(a):
                        tensor_positions.append(i)
                        tensor_requires_grad.append(bool(a.requires_grad))
                        tensor_values.append(a)
                    else:
                        non_tensor_values[i] = a
                ctx.tensor_positions = tuple(tensor_positions)
                ctx.tensor_requires_grad = tuple(tensor_requires_grad)
                ctx.non_tensor_values = non_tensor_values
                ctx.kwargs = kwargs
                ctx.save_for_backward(*tensor_values)
                return sdpa_fn(*f_args, **kwargs)

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_out):
                saved_tensors = list(ctx.saved_tensors)
                rebuilt_args = []
                tensor_idx = 0
                diff_inputs = []
                diff_input_positions = []
                total_args = len(ctx.tensor_positions) + len(ctx.non_tensor_values)
                for i in range(total_args):
                    if i in ctx.non_tensor_values:
                        rebuilt_args.append(ctx.non_tensor_values[i])
                        continue
                    a = saved_tensors[tensor_idx].detach()
                    if ctx.tensor_requires_grad[tensor_idx]:
                        a = a.requires_grad_(True)
                        diff_inputs.append(a)
                        diff_input_positions.append(i)
                    rebuilt_args.append(a)
                    tensor_idx += 1

                with torch.enable_grad():
                    out = sdpa_fn(*rebuilt_args, **ctx.kwargs)
                    diff_grads = torch.autograd.grad(
                        out,
                        diff_inputs,
                        grad_outputs=grad_out,
                        allow_unused=True,
                        retain_graph=False,
                        create_graph=False,
                    )

                grad_map = dict(zip(diff_input_positions, diff_grads))
                full_grads = []
                for i in range(total_args):
                    full_grads.append(grad_map.get(i, None))
                return tuple(full_grads)

        return _OnceDiffSDPA.apply(*args)

    return _wrapped


def stats_detach_logits_sdpa(
    _sdpa_ignored: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., torch.Tensor]:
    """
    Torch SDPA that detaches logits before softmax (strong stats detach).
    Ignores the provided sdpa_fn; implemented inline in torch.
    """

    def _sdpa(q, k, v, *, backend=None, scale=None):
        D = q.shape[-1]
        if scale is None:
            scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        scores = scores.detach() + scores * 0.0  # keep shape, kill logits grad path
        p = torch.softmax(scores, dim=-1)
        return torch.matmul(p, v)

    return _sdpa


def stats_detach_softmax_output_sdpa(
    _sdpa_ignored: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., torch.Tensor]:
    """
    Torch SDPA that detaches softmax output P (even stronger stats detach).
    Ignores the provided sdpa_fn; implemented inline in torch.
    """

    def _sdpa(q, k, v, *, backend=None, scale=None):
        D = q.shape[-1]
        if scale is None:
            scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        p = torch.softmax(scores, dim=-1)
        p = p.detach() + p * 0.0  # zero softmax curvature contribution
        return torch.matmul(p, v)

    return _sdpa
