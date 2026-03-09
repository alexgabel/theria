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
import os
import time
from typing import Mapping, OrderedDict as TypedOrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from theria.tasks.synthetic_seqcls import TaskBatch


Params = TypedOrderedDict[str, torch.Tensor]
Buffers = TypedOrderedDict[str, torch.Tensor]
_MAML_PROFILE_ENV = "THERIA_MAML_PROFILE"
_MAML_PROFILE_COUNTERS = {
    "n_tasks": 0,
    "n_inner_steps": 0,
    "n_support_forward_calls": 0,
    "n_support_grad_calls": 0,
    "n_support_grad_create_graph_calls": 0,
    "n_support_grad_no_graph_calls": 0,
    "n_param_update_calls": 0,
    "n_query_forward_calls": 0,
    "support_forward_time_s": 0.0,
    "support_grad_time_s": 0.0,
    "support_grad_create_graph_time_s": 0.0,
    "support_grad_no_graph_time_s": 0.0,
    "param_update_time_s": 0.0,
    "query_forward_time_s": 0.0,
}


def reset_maml_profile_counters() -> None:
    for key, value in list(_MAML_PROFILE_COUNTERS.items()):
        _MAML_PROFILE_COUNTERS[key] = 0.0 if isinstance(value, float) else 0


def get_maml_profile_counters(*, reset: bool = False) -> dict[str, float]:
    out = {
        key: (float(val) if isinstance(val, float) else int(val))
        for key, val in _MAML_PROFILE_COUNTERS.items()
    }
    if reset:
        reset_maml_profile_counters()
    return out


def _maml_profile_enabled() -> bool:
    return os.getenv(_MAML_PROFILE_ENV, "0") == "1"


def _profile_start(device: torch.device) -> float:
    if device.type == "cuda" and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_stop(counter_name: str, start_t: float, device: torch.device) -> float:
    if device.type == "cuda" and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_t
    _MAML_PROFILE_COUNTERS[counter_name] += elapsed
    return elapsed


def named_params(model: nn.Module) -> Params:
    return OrderedDict((k, v) for k, v in model.named_parameters())


def named_buffers(model: nn.Module) -> Buffers:
    return OrderedDict((k, v) for k, v in model.named_buffers())


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


def _assert_finite(t: torch.Tensor, *, name: str, ctx: str) -> None:
    if torch.isfinite(t).all():
        return
    bad = int((~torch.isfinite(t)).sum().item())
    total = int(t.numel())
    raise RuntimeError(f"NONFINITE {name} {ctx} bad={bad}/{total}")


def _ordered_params_from_names(
    names: tuple[str, ...],
    values: tuple[torch.Tensor, ...],
) -> Params:
    return OrderedDict(zip(names, values))


def _update_ordered_params_inplace(
    mapping: Params | None,
    names: tuple[str, ...],
    values: tuple[torch.Tensor, ...],
) -> Params:
    if mapping is None:
        return _ordered_params_from_names(names, values)
    for name, value in zip(names, values):
        mapping[name] = value
    return mapping


def inner_adapt(
    model: nn.Module,
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    task: TaskBatch,
    *,
    inner_lr: float = 0.1,
    inner_steps: int = 1,
    fo: bool = False,
    meta_last_n_inner: int = 0,
    check_finite: bool = False,
    finite_prefix: str = "",
) -> Params:
    """
    Full MAML inner loop:
      phi^{k+1} = phi^k - alpha * grad_{phi^k} L_support(phi^k)

    create_graph=True is the key: it retains the graph so meta-grad exists.
    """
    param_names = tuple(params.keys())
    phi_values = tuple(params.values())
    phi_create_graph_map: Params | None = None
    profile_enabled = _maml_profile_enabled()
    device = task.x_s.device

    for step_idx in range(inner_steps):
        # Keep the no-graph path unchanged. Only reuse the parameter mapping on
        # create-graph steps where Python/container overhead matters most.
        use_create_graph = not fo
        if use_create_graph and meta_last_n_inner > 0 and meta_last_n_inner < inner_steps:
            use_create_graph = step_idx >= (inner_steps - meta_last_n_inner)
        if use_create_graph:
            phi_for_call = _update_ordered_params_inplace(
                phi_create_graph_map,
                param_names,
                phi_values,
            )
            phi_create_graph_map = phi_for_call
        else:
            phi_for_call = _ordered_params_from_names(param_names, phi_values)

        if profile_enabled:
            _MAML_PROFILE_COUNTERS["n_inner_steps"] += 1
            _MAML_PROFILE_COUNTERS["n_support_forward_calls"] += 1
            support_forward_t0 = _profile_start(device)
        logits_s = functional_call(
            model,
            (phi_for_call, buffers),
            (task.x_s,),
        )
        if check_finite:
            _assert_finite(
                logits_s,
                name="support_logits",
                ctx=f"{finite_prefix} inner_step={step_idx}",
            )
        loss_s = loss_fn(logits_s, task.y_s)
        if profile_enabled:
            _profile_stop("support_forward_time_s", support_forward_t0, device)
        if check_finite:
            _assert_finite(
                loss_s,
                name="support_loss",
                ctx=f"{finite_prefix} inner_step={step_idx}",
            )

        if profile_enabled:
            _MAML_PROFILE_COUNTERS["n_support_grad_calls"] += 1
            if use_create_graph:
                _MAML_PROFILE_COUNTERS["n_support_grad_create_graph_calls"] += 1
            else:
                _MAML_PROFILE_COUNTERS["n_support_grad_no_graph_calls"] += 1
            support_grad_t0 = _profile_start(device)
        grads = torch.autograd.grad(
            loss_s,
            phi_values,
            create_graph=use_create_graph, # REQUIRED for meta-gradient
            retain_graph=use_create_graph,
            allow_unused=False,
        )
        if profile_enabled:
            elapsed = _profile_stop("support_grad_time_s", support_grad_t0, device)
            if use_create_graph:
                _MAML_PROFILE_COUNTERS["support_grad_create_graph_time_s"] += elapsed
            else:
                _MAML_PROFILE_COUNTERS["support_grad_no_graph_time_s"] += elapsed
        if check_finite:
            for pname, g in zip(param_names, grads):
                _assert_finite(
                    g,
                    name=f"support_grad[{pname}]",
                    ctx=f"{finite_prefix} inner_step={step_idx}",
                )
        # NOTE: create_graph=True is what enables second-order meta-gradients.
        # Removing this turns full MAML into FO-MAML.
        if profile_enabled:
            _MAML_PROFILE_COUNTERS["n_param_update_calls"] += 1
            param_update_t0 = _profile_start(device)
        phi_values = tuple(
            p - inner_lr * g
            for p, g in zip(phi_values, grads)
        )
        if profile_enabled:
            _profile_stop("param_update_time_s", param_update_t0, device)

    if phi_create_graph_map is not None:
        return _update_ordered_params_inplace(phi_create_graph_map, param_names, phi_values)
    return _ordered_params_from_names(param_names, phi_values)


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
    meta_last_n_inner: int = 0,
    check_finite: bool = False,
    finite_prefix: str = "",
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """
    Compute meta-loss across a batch/list of tasks:
      J(theta) = mean_t L_query(phi_K(theta; support_t); query_t)
    """
    params = named_params(model)
    buffers = named_buffers(model)

    loss_sum: torch.Tensor | None = None
    acc_sum = 0.0
    n_tasks = 0
    profile_enabled = _maml_profile_enabled()
    for task_idx, task in enumerate(tasks):
        if profile_enabled:
            _MAML_PROFILE_COUNTERS["n_tasks"] += 1
        task_prefix = f"{finite_prefix} task={task_idx}".strip()
        phi = inner_adapt(
            model,
            params,
            buffers,
            task,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            fo=fo,
            meta_last_n_inner=meta_last_n_inner,
            check_finite=check_finite,
            finite_prefix=task_prefix,
        )
        if fo_strict:
            phi = OrderedDict((k, v.detach().clone()) for k, v in phi.items())
        if profile_enabled:
            _MAML_PROFILE_COUNTERS["n_query_forward_calls"] += 1
            query_forward_t0 = _profile_start(task.x_q.device)
        logits_q = functional_call(model, (phi, buffers), (task.x_q,))
        if check_finite:
            _assert_finite(
                logits_q,
                name="query_logits",
                ctx=task_prefix,
            )
        post_loss = loss_fn(logits_q, task.y_q)
        if profile_enabled:
            _profile_stop("query_forward_time_s", query_forward_t0, task.x_q.device)
        if check_finite:
            _assert_finite(
                post_loss,
                name="query_loss",
                ctx=task_prefix,
            )
        if return_metrics:
            post_acc = (logits_q.argmax(dim=-1) == task.y_q).float().mean()
            if check_finite:
                _assert_finite(
                    post_acc,
                    name="query_acc",
                    ctx=task_prefix,
                )
            acc_sum += float(post_acc.item())
        loss_sum = post_loss if loss_sum is None else (loss_sum + post_loss)
        n_tasks += 1
    assert loss_sum is not None and n_tasks > 0, "meta_loss_on_tasks requires non-empty tasks"
    outer = loss_sum / n_tasks
    if return_metrics:
        mean_acc = acc_sum / n_tasks
        return outer, {"post_adapt_loss": outer.item(), "post_adapt_acc": mean_acc}
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
