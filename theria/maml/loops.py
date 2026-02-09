# theria/maml/loops.py
"""
Phase 2 MAML loops.

Design constraints:
- Full MAML (second-order)
- No mutation of model parameters
- Uses torch.func.functional_call
- Attention implementation must support grad-of-grad

This module is correctness-first and intentionally unoptimized.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, OrderedDict as TypedOrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from theria.tasks.synthetic_seqcls import TaskBatch


Params = TypedOrderedDict[str, torch.Tensor]
Buffers = TypedOrderedDict[str, torch.Tensor]


def named_params(model: nn.Module) -> Params:
    return OrderedDict((k, v) for k, v in model.named_parameters())


def named_buffers(model: nn.Module) -> Buffers:
    return OrderedDict((k, v) for k, v in model.named_buffers())


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


def inner_adapt(
    model: nn.Module,
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    task: TaskBatch,
    *,
    inner_lr: float = 0.1,
    inner_steps: int = 1,
    fo: bool = False,
) -> Params:
    """
    Full MAML inner loop:
      phi^{k+1} = phi^k - alpha * grad_{phi^k} L_support(phi^k)

    create_graph=True is the key: it retains the graph so meta-grad exists.
    """
    phi = OrderedDict((k, v) for k, v in params.items())

    for _ in range(inner_steps):
        logits_s = functional_call(model, (phi, buffers), (task.x_s,))
        loss_s = loss_fn(logits_s, task.y_s)

        grads = torch.autograd.grad(
            loss_s,
            tuple(phi.values()),
            create_graph=not fo, # REQUIRED for meta-gradient
            retain_graph=True,
            allow_unused=False,
        )
        # NOTE: create_graph=True is what enables second-order meta-gradients.
        # Removing this turns full MAML into FO-MAML.
        phi = OrderedDict(
            (name, p - inner_lr * g)
            for (name, p), g in zip(phi.items(), grads)
        )

    return phi


def outer_loss(
    model: nn.Module,
    phi: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    task: TaskBatch,
) -> torch.Tensor:
    logits_q = functional_call(model, (phi, buffers), (task.x_q,))
    return loss_fn(logits_q, task.y_q)


def meta_loss_on_tasks(
    model: nn.Module,
    tasks: list[TaskBatch],
    *,
    inner_lr: float = 0.1,
    inner_steps: int = 1,
    fo: bool = False,
    fo_strict: bool = False,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """
    Compute meta-loss across a batch/list of tasks:
      J(theta) = mean_t L_query(phi_K(theta; support_t); query_t)
    """
    params = named_params(model)
    buffers = named_buffers(model)

    losses = []
    accs = []
    for task in tasks:
        phi = inner_adapt(model, params, buffers, task, inner_lr=inner_lr, inner_steps=inner_steps, fo=fo)
        if fo_strict:
            phi = OrderedDict((k, v.detach().clone()) for k, v in phi.items())
        logits_q = functional_call(model, (phi, buffers), (task.x_q,))
        post_loss = loss_fn(logits_q, task.y_q)
        post_acc = (logits_q.argmax(dim=-1) == task.y_q).float().mean()
        losses.append(post_loss)
        accs.append(post_acc)
    outer = torch.stack(losses).mean()
    if return_metrics:
        mean_acc = torch.stack(accs).mean()
        return outer, {"post_adapt_loss": outer.item(), "post_adapt_acc": mean_acc.item()}
    return outer


def meta_loss_on_tasks_full_frozen(
    model: nn.Module,
    tasks: list[TaskBatch],
    *,
    inner_lr: float = 0.1,
    inner_steps: int = 1,
    return_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, float], list[torch.Tensor]]:
    """
    Frozen-stats full meta-gradient using explicit HVP via autograd.functional.hvp.

    This avoids create_graph in the inner loop and instead applies the
    v <- v - alpha * H_s * v recursion using HVPs of the support loss.
    Intended for small shapes / correctness validation.
    """
    params = named_params(model)
    buffers = named_buffers(model)

    meta_grads = [torch.zeros_like(p) for p in params.values()]
    losses = []
    accs = []

    for task in tasks:
        # Inner loop (FO): keep intermediate phis as leaf tensors
        phi = OrderedDict((k, v.detach().requires_grad_(True)) for k, v in params.items())
        phis = [phi]
        for _ in range(inner_steps):
            logits_s = functional_call(model, (phi, buffers), (task.x_s,))
            loss_s = loss_fn(logits_s, task.y_s)
            grads = torch.autograd.grad(
                loss_s,
                tuple(phi.values()),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            phi = OrderedDict(
                (name, p - inner_lr * g)
                for (name, p), g in zip(phi.items(), grads)
            )
            phi = OrderedDict((k, v.detach().requires_grad_(True)) for k, v in phi.items())
            phis.append(phi)

        # Query loss at phi_K
        logits_q = functional_call(model, (phi, buffers), (task.x_q,))
        loss_q = loss_fn(logits_q, task.y_q)
        acc_q = (logits_q.argmax(dim=-1) == task.y_q).float().mean()
        losses.append(loss_q)
        accs.append(acc_q)

        # v = grad L_q w.r.t. phi_K
        v = torch.autograd.grad(
            loss_q,
            tuple(phi.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        # Reverse recursion through inner steps
        for t in reversed(range(inner_steps)):
            phi_t = phis[t]

            def support_loss_fn(*phi_values):
                phi_dict = OrderedDict(
                    (name, val) for name, val in zip(phi_t.keys(), phi_values)
                )
                logits_s_t = functional_call(model, (phi_dict, buffers), (task.x_s,))
                return loss_fn(logits_s_t, task.y_s)

            _, hvp = torch.autograd.functional.hvp(
                support_loss_fn,
                tuple(phi_t.values()),
                v,
            )
            v = tuple(v_i - inner_lr * h_i for v_i, h_i in zip(v, hvp))

        for i, g in enumerate(v):
            meta_grads[i] = meta_grads[i] + g / len(tasks)

    outer = torch.stack(losses).mean()
    mean_acc = torch.stack(accs).mean()
    metrics = {"post_adapt_loss": outer.item(), "post_adapt_acc": mean_acc.item()}
    return outer, metrics, meta_grads
