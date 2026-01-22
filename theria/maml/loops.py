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
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from theria.tasks.synthetic_seqcls import TaskBatch


Params = OrderedDict[str, torch.Tensor]
Buffers = OrderedDict[str, torch.Tensor]


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
            create_graph=True, # REQUIRED for meta-gradient
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
) -> torch.Tensor:
    """
    Compute meta-loss across a batch/list of tasks:
      J(theta) = mean_t L_query(phi_K(theta; support_t); query_t)
    """
    params = named_params(model)
    buffers = named_buffers(model)

    losses = []
    for task in tasks:
        phi = inner_adapt(model, params, buffers, task, inner_lr=inner_lr, inner_steps=inner_steps)
        losses.append(outer_loss(model, phi, buffers, task))

    return torch.stack(losses).mean()