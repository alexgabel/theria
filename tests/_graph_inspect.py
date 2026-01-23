"""
Autograd graph inspection utilities.

These helpers are test-only and used to diagnose
where higher-order differentiation breaks.
"""

import torch
from typing import Set, List, Optional


def _walk_grad_fn(fn, seen: Set[int], depth: int = 0, max_depth: Optional[int] = None):
    if fn is None or id(fn) in seen:
        return
    if max_depth is not None and depth > max_depth:
        return
    seen.add(id(fn))

    indent = "  " * depth
    print(f"{indent}{type(fn).__name__}")

    for next_fn, _ in fn.next_functions:
        _walk_grad_fn(next_fn, seen, depth + 1, max_depth=max_depth)


def print_autograd_graph(tensor: torch.Tensor, max_depth: Optional[int] = None):
    """
    Pretty-print the autograd graph rooted at tensor.
    """
    if tensor.grad_fn is None:
        print("Tensor has no grad_fn (leaf or detached).")
        return
    _walk_grad_fn(tensor.grad_fn, seen=set(), max_depth=max_depth)


def can_double_backward(loss: torch.Tensor, inputs: List[torch.Tensor]) -> bool:
    """
    Returns True if grad-of-grad exists for loss w.r.t inputs.
    """
    grads = torch.autograd.grad(
        loss,
        inputs,
        create_graph=True,
        allow_unused=False,
    )

    try:
        dummy = sum(g.sum() for g in grads)
        torch.autograd.grad(dummy, inputs, allow_unused=False)
        return True
    except RuntimeError:
        return False


def list_saved_tensors(grad_fn):
    """
    Print tensors saved by a grad_fn (if any).
    Useful to see whether backward saved differentiable tensors.
    """
    if not hasattr(grad_fn, "saved_tensors"):
        print("No saved_tensors attribute.")
        return

    for i, t in enumerate(grad_fn.saved_tensors):
        print(
            f"[{i}] shape={tuple(t.shape)}, "
            f"requires_grad={t.requires_grad}, "
            f"is_leaf={t.is_leaf}"
        )


def find_detached_nodes(tensor: torch.Tensor):
    """
    Walk the autograd graph and report nodes that have no grad_fn.
    """
    detached = []

    def walk(fn):
        if fn is None:
            detached.append("None")
            return
        for next_fn, _ in fn.next_functions:
            if next_fn is None:
                detached.append("Detached edge")
            else:
                walk(next_fn)

    if tensor.grad_fn is None:
        detached.append("Root tensor has no grad_fn")
    else:
        walk(tensor.grad_fn)

    return detached
