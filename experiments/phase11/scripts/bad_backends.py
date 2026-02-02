"""
Phase 11 bad backend variants (experiment-only).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint


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


# -----------------------------
# Partial input detaches (Q/K/V)
# -----------------------------

def detach_q_input(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Detach queries before SDPA (kills curvature from Q path).
    """

    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        q2 = q.detach() + q * 0.0
        return sdpa_fn(q2, k, v, **kw)

    return _wrapped


def detach_k_input(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Detach keys before SDPA (kills curvature from K path).
    """

    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        k2 = k.detach() + k * 0.0
        return sdpa_fn(q, k2, v, **kw)

    return _wrapped


def detach_v_input(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Detach values before SDPA (kills curvature from V path).
    """

    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        v2 = v.detach() + v * 0.0
        return sdpa_fn(q, k, v2, **kw)

    return _wrapped


# Strict detaches (no residual edge); use for validation only.
def detach_q_input_strict(sdpa_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        q2 = q.detach()  # no graph edge retained
        return sdpa_fn(q2, k, v, **kw)
    return _wrapped


def detach_k_input_strict(sdpa_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        k2 = k.detach()
        return sdpa_fn(q, k2, v, **kw)
    return _wrapped


def detach_v_input_strict(sdpa_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        v2 = v.detach()
        return sdpa_fn(q, k, v2, **kw)
    return _wrapped


# -----------------------------
# Checkpoint variants
# -----------------------------

def checkpoint_attention(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Recompute SDPA under checkpoint (grad-enabled).
    """

    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        def fn(q_, k_, v_):
            return sdpa_fn(q_, k_, v_, **kw)
        return checkpoint(fn, q, k, v, use_reentrant=False)

    return _wrapped


def checkpoint_no_grad(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Checkpoint SDPA but recompute under no_grad (expected to break gradgrad).
    """

    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        def fn(q_, k_, v_):
            with torch.no_grad():
                return sdpa_fn(q_, k_, v_, **kw)
        return checkpoint(fn, q, k, v, use_reentrant=False)

    return _wrapped


def checkpoint_detach_recompute(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Checkpoint SDPA then detach the recomputed output (silent curvature collapse).
    """

    def _wrapped(q, k, v, **kw) -> torch.Tensor:
        def fn(q_, k_, v_):
            return sdpa_fn(q_, k_, v_, **kw)
        out = checkpoint(fn, q, k, v, use_reentrant=False)
        return out.detach() + out * 0.0

    return _wrapped


# -----------------------------
# Recompute / stats variants
# -----------------------------


def recompute_logits_no_grad_sdpa(
    _sdpa_ignored: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., torch.Tensor]:
    """
    Torch SDPA that recomputes logits under no_grad (kills curvature from Q/K path).
    """

    def _sdpa(q, k, v, *, backend=None, scale=None):
        D = q.shape[-1]
        if scale is None:
            scale = 1.0 / (D ** 0.5)
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            p = torch.softmax(scores, dim=-1)
        return torch.matmul(p, v)

    return _sdpa


def backward_detach_logits_sim(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Custom Function: forward uses sdpa_fn; backward recomputes logits then detaches them,
    returning first-order grads but suppressing higher-order curvature.
    """

    class _DetachLogits(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *f_args):
            ctx.kwargs = {}
            ctx.save_for_backward(*f_args)
            return sdpa_fn(*f_args)

        @staticmethod
        def backward(ctx, grad_out):
            args = list(ctx.saved_tensors)
            q, k, v = args
            D = q.shape[-1]
            scale = 1.0 / (D ** 0.5)
            with torch.enable_grad():
                # recompute with detached logits
                scores = torch.matmul(q, k.transpose(-1, -2)) * scale
                scores = scores.detach() + scores * 0.0
                p = torch.softmax(scores, dim=-1)
                out = torch.matmul(p, v)
                grads = torch.autograd.grad(
                    out,
                    (q, k, v),
                    grad_outputs=grad_out,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
            # pad None to match args
            full = list(grads)
            return tuple(full)

    def _wrapped(q, k, v, **kw):
        return _DetachLogits.apply(q, k, v)

    return _wrapped
