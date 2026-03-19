"""
Phase 12 behavior runner (outer-loop aggregation).

Patch P3 focus:
- Track per-step outer loss / post-adaptation accuracy
- Aggregate final metrics from the last 20 outer steps
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
from pathlib import Path
import sys
import time
from typing import Any, Literal

import torch

# Allow direct script execution from repo root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase10.scripts.run_maml_backend_compare import (
    Phase10TinyAttentionModel,
    set_attention_backend,
)
from theria.attention.custom import sdpa_custom
from theria.attention.triton_qk import (
    get_triton_meta_bwd_counters,
    reset_triton_meta_bwd_counters,
)
from theria.attention.triton_sdpa_backward import (
    get_triton_sdpa_debug_counters,
    reset_triton_sdpa_debug_counters,
)
from theria.maml.loops import meta_loss_on_tasks, meta_loss_on_tasks_full_frozen
from theria.maml.loops import get_maml_profile_counters, reset_maml_profile_counters
from theria.tasks.synthetic_seqcls import DatasetSplit, TaskBatch, task_sampler
from experiments.phase11.scripts.run_bad_backend_diagnostics import _attention_second_order_ok


Mode = Literal["FULL", "FO", "FO_STRICT", "FULL_FROZEN", "FULL_HYBRID"]
EXPERIMENTAL_BACKENDS = {"triton_fused_meta"}
CUDA_GRAPH_STATIC_BACKEND = "triton_fused_meta_strict"
CUDA_GRAPH_WARMUP_STEPS = 2
EVAL_SPLIT_NUM_META_BATCHES = 4


def _mean_last(values: list[float], n: int) -> float:
    if not values:
        return float("nan")
    take = values[-min(n, len(values)) :]
    return float(sum(take) / len(take))


def _mean_first(values: list[float], n: int) -> float:
    if not values:
        return float("nan")
    take = values[: min(n, len(values))]
    return float(sum(take) / len(take))


def _profile_start(device: torch.device) -> float:
    if device.type == "cuda" and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_elapsed(start_t: float, device: torch.device) -> float:
    if device.type == "cuda" and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize(device)
    return time.perf_counter() - start_t


def _clone_task_batch(task: TaskBatch) -> TaskBatch:
    return TaskBatch(
        x_s=task.x_s.detach().clone(),
        y_s=task.y_s.detach().clone(),
        x_q=task.x_q.detach().clone(),
        y_q=task.y_q.detach().clone(),
    )


def _copy_task_batch_(dst: TaskBatch, src: TaskBatch) -> None:
    dst.x_s.copy_(src.x_s)
    dst.y_s.copy_(src.y_s)
    dst.x_q.copy_(src.x_q)
    dst.y_q.copy_(src.y_q)


def _copy_task_batches_(dst_tasks: list[TaskBatch], src_tasks: list[TaskBatch]) -> None:
    for dst_task, src_task in zip(dst_tasks, src_tasks):
        _copy_task_batch_(dst_task, src_task)


def _task_batch_to_device(task: TaskBatch, device: torch.device) -> TaskBatch:
    return TaskBatch(
        x_s=task.x_s.to(device=device, non_blocking=False),
        y_s=task.y_s.to(device=device, non_blocking=False),
        x_q=task.x_q.to(device=device, non_blocking=False),
        y_q=task.y_q.to(device=device, non_blocking=False),
    )


@contextlib.contextmanager
def _preserve_rng_state(device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if device.type == "cuda" and cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _sample_tasks_for_split(
    *,
    split: DatasetSplit,
    n_tasks: int,
    seq_len: int,
    d_model: int,
    num_signal_positions: int,
    sampler_device: torch.device,
    device: torch.device,
    dataset_seed: int,
    episode_seed: int | None = None,
) -> list[TaskBatch]:
    with _preserve_rng_state(device):
        if episode_seed is not None:
            torch.manual_seed(episode_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(episode_seed)
        sampled_tasks = [
            task_sampler(
                T=seq_len,
                D=d_model,
                num_signal_positions=num_signal_positions,
                device=sampler_device,
                split=split,
                dataset_seed=dataset_seed,
            )
            for _ in range(n_tasks)
        ]
    if sampler_device != device:
        return [_task_batch_to_device(task, device) for task in sampled_tasks]
    return sampled_tasks


def _evaluate_meta_split(
    *,
    model: torch.nn.Module,
    split: DatasetSplit,
    n_meta_batches: int,
    meta_batch_size: int,
    inner_lr: float,
    inner_steps: int,
    seq_len: int,
    num_signal_positions: int,
    sampler_device: torch.device,
    device: torch.device,
    dataset_seed: int,
    fail_on_nonfinite: bool,
    autocast_enabled: bool,
    cast_dtype: torch.dtype,
) -> tuple[float, float]:
    losses: list[float] = []
    accs: list[float] = []
    for batch_idx in range(n_meta_batches):
        tasks = _sample_tasks_for_split(
            split=split,
            n_tasks=meta_batch_size,
            seq_len=seq_len,
            d_model=model.cfg.d_model,
            num_signal_positions=num_signal_positions,
            sampler_device=sampler_device,
            device=device,
            dataset_seed=dataset_seed,
            episode_seed=dataset_seed + 10_000 * (1 + batch_idx) + (100 if split == "val" else 200),
        )
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=cast_dtype,
        ):
            outer_loss, metrics = meta_loss_on_tasks(
                model=model,
                tasks=tasks,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                fo=True,
                fo_strict=False,
                meta_last_n_inner=0,
                check_finite=fail_on_nonfinite,
                finite_prefix=f"eval_split={split}",
                return_metrics=True,
            )
        losses.append(float(outer_loss.item()))
        accs.append(float(metrics["post_adapt_acc"]))
    return float(sum(losses) / len(losses)), float(sum(accs) / len(accs))


def _snapshot_model_state(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    param_state = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }
    buffer_state = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
    }
    return param_state, buffer_state


def _restore_model_state_(
    model: torch.nn.Module,
    param_state: dict[str, torch.Tensor],
    buffer_state: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(param_state[name])
        for name, buf in model.named_buffers():
            buf.copy_(buffer_state[name])


def _snapshot_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> dict[int, dict[str, Any]]:
    snapshot: dict[int, dict[str, Any]] = {}
    for param, state in optimizer.state.items():
        param_snapshot: dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                param_snapshot[key] = value.detach().clone()
            else:
                param_snapshot[key] = value
        snapshot[id(param)] = param_snapshot
    return snapshot


def _restore_optimizer_state_(
    optimizer: torch.optim.Optimizer,
    snapshot: dict[int, dict[str, Any]],
) -> None:
    for param, state in optimizer.state.items():
        param_snapshot = snapshot.get(id(param))
        if param_snapshot is None:
            for key, value in state.items():
                if torch.is_tensor(value):
                    value.zero_()
                elif isinstance(value, bool):
                    state[key] = False
                elif isinstance(value, int):
                    state[key] = 0
                elif isinstance(value, float):
                    state[key] = 0.0
            continue
        for key, value in state.items():
            snapshot_value = param_snapshot[key]
            if torch.is_tensor(value) and torch.is_tensor(snapshot_value):
                value.copy_(snapshot_value)
            else:
                state[key] = snapshot_value


def _ensure_grad_buffers_(trainable_params: list[torch.nn.Parameter]) -> None:
    for param in trainable_params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        else:
            param.grad.zero_()


def _capture_static_hybrid_graph(
    *,
    model: torch.nn.Module,
    trainable_params: list[torch.nn.Parameter],
    tasks: list[TaskBatch],
    inner_lr: float,
    inner_steps: int,
    fo_step: bool,
    fo_strict_step: bool,
    meta_last_n_inner_step: int,
    fail_on_nonfinite: bool,
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    static_tasks = [_clone_task_batch(task) for task in tasks]

    def _run_once() -> tuple[torch.Tensor, torch.Tensor]:
        _ensure_grad_buffers_(trainable_params)
        outer_loss, metrics = meta_loss_on_tasks(
            model=model,
            tasks=static_tasks,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            fo=fo_step,
            fo_strict=fo_strict_step,
            meta_last_n_inner=meta_last_n_inner_step,
            check_finite=fail_on_nonfinite,
            finite_prefix="cuda_graph_capture",
            return_metrics=True,
            return_metrics_tensors=True,
        )
        grads = torch.autograd.grad(
            outer_loss,
            trainable_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        for param, grad in zip(trainable_params, grads):
            if grad is None:
                param.grad.zero_()
            else:
                param.grad.copy_(grad)
        return outer_loss, metrics["post_adapt_acc"]

    capture_t0 = _profile_start(device)
    _ensure_grad_buffers_(trainable_params)
    warmup_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(CUDA_GRAPH_WARMUP_STEPS):
            _run_once()
    current_stream.wait_stream(warmup_stream)
    torch.cuda.synchronize(device)
    _ensure_grad_buffers_(trainable_params)

    graph = torch.cuda.CUDAGraph()
    loss_out = torch.zeros((), device=device)
    acc_out = torch.zeros((), device=device)
    with torch.cuda.graph(graph):
        outer_loss, post_acc = _run_once()
        loss_out.copy_(outer_loss)
        acc_out.copy_(post_acc)
    torch.cuda.synchronize(device)
    _ensure_grad_buffers_(trainable_params)

    return (
        {
            "graph": graph,
            "static_tasks": static_tasks,
            "loss_out": loss_out,
            "acc_out": acc_out,
        },
        _profile_elapsed(capture_t0, device),
    )


def run_behavior(
    *,
    attention_backend: str,
    mode: Mode,
    seed: int,
    outer_steps: int,
    meta_batch_size: int,
    inner_steps: int,
    inner_lr: float,
    outer_lr: float,
    seq_len: int = 32,
    num_signal_positions: int = 4,
    device: torch.device,
    autocast_enabled: bool = False,
    meta_every_n_outer: int = 2,
    meta_last_n_inner: int = 0,
    profile_meta_bwd: bool = False,
    fail_on_nonfinite: bool = True,
    allow_experimental_backends: bool = False,
    grad_eps: float | None = None,
    rel_diff_probe: bool = True,
    cuda_graph_static: bool = False,
) -> dict[str, float | int | str]:
    if attention_backend in EXPERIMENTAL_BACKENDS and not allow_experimental_backends:
        raise ValueError(
            f"Backend '{attention_backend}' is experimental. "
            "Pass --allow-experimental-backends to opt in."
        )
    if num_signal_positions >= seq_len:
        raise ValueError("num_signal_positions must be < seq_len (position 0 is reserved).")
    if mode == "FULL_HYBRID" and meta_every_n_outer < 1:
        raise ValueError("meta_every_n_outer must be >= 1 for FULL_HYBRID")
    if meta_last_n_inner < 0:
        raise ValueError("meta_last_n_inner must be >= 0")
    if meta_last_n_inner > inner_steps:
        raise ValueError("meta_last_n_inner must be <= inner_steps")
    reset_triton_meta_bwd_counters()
    reset_triton_sdpa_debug_counters()
    reset_maml_profile_counters()
    prev_meta_profile_env = os.environ.get("THERIA_TRITON_META_PROFILE")
    prev_maml_profile_env = os.environ.get("THERIA_MAML_PROFILE")
    if profile_meta_bwd:
        os.environ["THERIA_TRITON_META_PROFILE"] = "1"
        os.environ["THERIA_MAML_PROFILE"] = "1"
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Phase10TinyAttentionModel().to(device)
    backend_for_run = attention_backend
    if mode == "FULL_FROZEN":
        if attention_backend == "triton_fused":
            backend_for_run = "triton_frozen_stats"
            if device.type != "cuda":
                raise ValueError("FULL_FROZEN with triton_fused requires CUDA device")
        elif attention_backend == "reference":
            backend_for_run = "reference_frozen_stats"
        else:
            raise ValueError("FULL_FROZEN only supports reference or triton_fused backends")
    set_attention_backend(model, backend_for_run)
    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    dtype_name = str(next(model.parameters()).dtype).replace("torch.", "")
    if autocast_enabled and device.type == "cuda":
        compute_dtype = "fp16"
    elif autocast_enabled:
        compute_dtype = "bf16"
    else:
        compute_dtype = "fp32"

    fo = mode not in {"FULL", "FULL_FROZEN", "FULL_HYBRID"}
    fo_strict = mode == "FO_STRICT"
    cast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    if grad_eps is None:
        grad_eps = 1e-6 if (autocast_enabled and device.type == "cuda") else 1e-8

    cuda_graph_static_requested = int(bool(cuda_graph_static))
    cuda_graph_static_used = 0
    cuda_graph_capture_time_s = 0.0
    cuda_graph_replay_time_s = 0.0
    cuda_graph_meta_capture_ok = 0
    cuda_graph_fo_capture_ok = 0
    cuda_graph_meta_replays = 0
    cuda_graph_fo_replays = 0
    cuda_graph_static_reason = ""
    cuda_graph_supported = (
        cuda_graph_static
        and device.type == "cuda"
        and attention_backend == CUDA_GRAPH_STATIC_BACKEND
        and mode == "FULL_HYBRID"
        and not autocast_enabled
        and not profile_meta_bwd
    )
    if cuda_graph_static and not cuda_graph_supported:
        if device.type != "cuda":
            cuda_graph_static_reason = "requires_cuda"
        elif attention_backend != CUDA_GRAPH_STATIC_BACKEND:
            cuda_graph_static_reason = "requires_triton_fused_meta_strict"
        elif mode != "FULL_HYBRID":
            cuda_graph_static_reason = "requires_full_hybrid"
        elif autocast_enabled:
            cuda_graph_static_reason = "autocast_not_supported"
        elif profile_meta_bwd:
            cuda_graph_static_reason = "profile_meta_bwd_not_supported"
    cuda_graph_bundles: dict[str, dict[str, Any]] = {}
    sampler_device = torch.device("cpu") if cuda_graph_supported else device

    outer_losses: list[float] = []
    outer_accs: list[float] = []
    q_grad_norms: list[float] = []
    k_grad_norms: list[float] = []
    v_grad_norms: list[float] = []
    peak_cuda_mem_bytes: int = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    backend_map = {
        "reference": "reference",
        "custom": "custom",
        "triton_fused": "triton_full_fused",
        "triton_fused_meta": "triton_full_fused",
        "triton_fused_meta_strict": "reference",
        "triton_full_autograd": "triton_full_fused",
        "triton_frozen_stats": "triton_full_fused",
        "reference_frozen_stats": "reference",
    }
    probe_backend = backend_map[attention_backend]

    def sdpa_for_probe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return sdpa_custom(q, k, v, backend=probe_backend)

    sdpa_input_gradgrad_ok = _attention_second_order_ok(
        sdpa_fn=sdpa_for_probe,
        device=device,
    )
    t0 = time.perf_counter()
    replay_t_total = 0.0

    n_hybrid_meta_steps = 0
    n_hybrid_fo_steps = 0
    meta_loss_time_s = 0.0
    outer_backward_time_s = 0.0
    optimizer_step_time_s = 0.0
    hybrid_meta_meta_loss_time_s = 0.0
    hybrid_fo_meta_loss_time_s = 0.0
    hybrid_meta_outer_backward_time_s = 0.0
    hybrid_fo_outer_backward_time_s = 0.0
    hybrid_meta_optimizer_step_time_s = 0.0
    hybrid_fo_optimizer_step_time_s = 0.0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    for step_idx in range(outer_steps):
        tasks = _sample_tasks_for_split(
            split="train",
            n_tasks=meta_batch_size,
            seq_len=seq_len,
            d_model=model.cfg.d_model,
            num_signal_positions=num_signal_positions,
            sampler_device=sampler_device,
            device=device,
            dataset_seed=seed,
        )
        if not cuda_graph_supported:
            optimizer.zero_grad(set_to_none=True)
        # Keep explicit autocast flag in logs even when fixed for this phase.
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=cast_dtype,
        ):
            is_hybrid_meta_step = False
            if mode == "FULL_FROZEN":
                meta_loss_t0 = _profile_start(device) if profile_meta_bwd else 0.0
                outer_loss, metrics, meta_grads = meta_loss_on_tasks_full_frozen(
                    model=model,
                    tasks=tasks,
                    inner_lr=inner_lr,
                    inner_steps=inner_steps,
                    return_metrics=True,
                )
                if profile_meta_bwd:
                    meta_loss_time_s += _profile_elapsed(meta_loss_t0, device)
                for (_, p), g in zip(model.named_parameters(), meta_grads):
                    p.grad = g
                outer_loss_value = float(outer_loss.item())
                outer_acc_value = float(metrics["post_adapt_acc"])
            else:
                fo_step = fo
                fo_strict_step = fo_strict
                if mode == "FULL_HYBRID":
                    use_meta_step = (step_idx % meta_every_n_outer) == 0
                    is_hybrid_meta_step = use_meta_step
                    fo_step = not use_meta_step
                    fo_strict_step = False
                    if use_meta_step:
                        n_hybrid_meta_steps += 1
                    else:
                        n_hybrid_fo_steps += 1
                meta_last_n_inner_step = meta_last_n_inner if not fo_step else 0
                outer_loss_value: float
                outer_acc_value: float
                used_cuda_graph_step = False
                if cuda_graph_supported and not is_hybrid_meta_step:
                    graph_key = "fo"
                    try:
                        if graph_key not in cuda_graph_bundles:
                            bundle, capture_elapsed = _capture_static_hybrid_graph(
                                model=model,
                                trainable_params=trainable_params,
                                tasks=tasks,
                                inner_lr=inner_lr,
                                inner_steps=inner_steps,
                                fo_step=fo_step,
                                fo_strict_step=fo_strict_step,
                                meta_last_n_inner_step=meta_last_n_inner_step,
                                fail_on_nonfinite=fail_on_nonfinite,
                                device=device,
                            )
                            cuda_graph_bundles[graph_key] = bundle
                            cuda_graph_capture_time_s += capture_elapsed
                            cuda_graph_fo_capture_ok = 1
                        bundle = cuda_graph_bundles[graph_key]
                        _copy_task_batches_(bundle["static_tasks"], tasks)
                        replay_t0 = _profile_start(device)
                        bundle["graph"].replay()
                        replay_elapsed = _profile_elapsed(replay_t0, device)
                        replay_t_total += replay_elapsed
                        cuda_graph_replay_time_s += replay_elapsed
                        cuda_graph_static_used = 1
                        used_cuda_graph_step = True
                        cuda_graph_fo_replays += 1
                        outer_loss = bundle["loss_out"]
                        outer_loss_value = float(outer_loss.item())
                        outer_acc_value = float(bundle["acc_out"].item())
                    except Exception as graph_error:
                        cuda_graph_supported = False
                        cuda_graph_bundles.clear()
                        cuda_graph_static_reason = (
                            f"fo_graph_failed:{graph_error!r}"
                        )
                        optimizer.zero_grad(set_to_none=True)
                elif cuda_graph_supported and is_hybrid_meta_step and not cuda_graph_static_reason:
                    cuda_graph_static_reason = "fo_only_graph_path"
                if not used_cuda_graph_step:
                    meta_loss_t0 = _profile_start(device) if profile_meta_bwd else 0.0
                    outer_loss, metrics = meta_loss_on_tasks(
                        model=model,
                        tasks=tasks,
                        inner_lr=inner_lr,
                        inner_steps=inner_steps,
                        fo=fo_step,
                        fo_strict=fo_strict_step,
                        meta_last_n_inner=meta_last_n_inner_step,
                        check_finite=fail_on_nonfinite,
                        finite_prefix=f"outer_step={step_idx}",
                        return_metrics=True,
                    )
                    outer_loss_value = float(outer_loss.item())
                    outer_acc_value = float(metrics["post_adapt_acc"])
                    if profile_meta_bwd:
                        meta_loss_elapsed = _profile_elapsed(meta_loss_t0, device)
                        meta_loss_time_s += meta_loss_elapsed
                        if mode == "FULL_HYBRID":
                            if is_hybrid_meta_step:
                                hybrid_meta_meta_loss_time_s += meta_loss_elapsed
                            else:
                                hybrid_fo_meta_loss_time_s += meta_loss_elapsed
                    if fail_on_nonfinite and not torch.isfinite(outer_loss):
                        raise RuntimeError(
                            f"NONFINITE outer_loss outer_step={step_idx}"
                        )
                    backward_t0 = _profile_start(device) if profile_meta_bwd else 0.0
                    outer_grads = torch.autograd.grad(
                        outer_loss,
                        trainable_params,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    for p, g in zip(trainable_params, outer_grads):
                        p.grad = g
                    if profile_meta_bwd:
                        backward_elapsed = _profile_elapsed(backward_t0, device)
                        outer_backward_time_s += backward_elapsed
                        if mode == "FULL_HYBRID":
                            if is_hybrid_meta_step:
                                hybrid_meta_outer_backward_time_s += backward_elapsed
                            else:
                                hybrid_fo_outer_backward_time_s += backward_elapsed

        qg = model.q_proj.weight.grad
        kg = model.k_proj.weight.grad
        vg = model.v_proj.weight.grad
        if fail_on_nonfinite:
            if qg is not None and not torch.isfinite(qg).all():
                raise RuntimeError(f"NONFINITE grad[q_proj] outer_step={step_idx}")
            if kg is not None and not torch.isfinite(kg).all():
                raise RuntimeError(f"NONFINITE grad[k_proj] outer_step={step_idx}")
            if vg is not None and not torch.isfinite(vg).all():
                raise RuntimeError(f"NONFINITE grad[v_proj] outer_step={step_idx}")
        q_grad_norms.append(float(qg.norm().item()) if qg is not None else 0.0)
        k_grad_norms.append(float(kg.norm().item()) if kg is not None else 0.0)
        v_grad_norms.append(float(vg.norm().item()) if vg is not None else 0.0)

        if not cuda_graph_supported or not cuda_graph_static_used:
            optimizer_step_t0 = _profile_start(device) if profile_meta_bwd else 0.0
            optimizer.step()
            if profile_meta_bwd:
                optimizer_elapsed = _profile_elapsed(optimizer_step_t0, device)
                optimizer_step_time_s += optimizer_elapsed
                if mode == "FULL_HYBRID":
                    if is_hybrid_meta_step:
                        hybrid_meta_optimizer_step_time_s += optimizer_elapsed
                    else:
                        hybrid_fo_optimizer_step_time_s += optimizer_elapsed

        outer_losses.append(outer_loss_value)
        outer_accs.append(outer_acc_value)

    final_loss = _mean_last(outer_losses, 20)
    final_acc = _mean_last(outer_accs, 20)
    attn_grad_norm_q = _mean_last(q_grad_norms, 20)
    attn_grad_norm_k = _mean_last(k_grad_norms, 20)
    attn_grad_norm_v = _mean_last(v_grad_norms, 20)
    attn_grad_present = (
        (attn_grad_norm_q > grad_eps)
        or (attn_grad_norm_k > grad_eps)
        or (attn_grad_norm_v > grad_eps)
    )
    convergence_delta = _mean_first(outer_losses, 20) - _mean_last(outer_losses, 20)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_mem_bytes = int(torch.cuda.max_memory_allocated(device))
    wall_time_total_s = time.perf_counter() - t0
    if cuda_graph_static_used:
        graph_steady_state_time_s = max(wall_time_total_s - cuda_graph_capture_time_s, 0.0)
        mean_outer_step_time_s = graph_steady_state_time_s / max(outer_steps, 1)
    else:
        mean_outer_step_time_s = wall_time_total_s / max(outer_steps, 1)
    # Capture counters for the actual training loop only; reset before optional probes.
    meta_bwd_counts = get_triton_meta_bwd_counters(reset=True)
    sdpa_debug_counts = get_triton_sdpa_debug_counters(reset=True)
    maml_profile_counts = get_maml_profile_counters(reset=True)

    val_loss, val_acc = _evaluate_meta_split(
        model=model,
        split="val",
        n_meta_batches=EVAL_SPLIT_NUM_META_BATCHES,
        meta_batch_size=meta_batch_size,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        seq_len=seq_len,
        num_signal_positions=num_signal_positions,
        sampler_device=sampler_device,
        device=device,
        dataset_seed=seed,
        fail_on_nonfinite=fail_on_nonfinite,
        autocast_enabled=autocast_enabled,
        cast_dtype=cast_dtype,
    )
    test_loss, test_acc = _evaluate_meta_split(
        model=model,
        split="test",
        n_meta_batches=EVAL_SPLIT_NUM_META_BATCHES,
        meta_batch_size=meta_batch_size,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        seq_len=seq_len,
        num_signal_positions=num_signal_positions,
        sampler_device=sampler_device,
        device=device,
        dataset_seed=seed,
        fail_on_nonfinite=fail_on_nonfinite,
        autocast_enabled=autocast_enabled,
        cast_dtype=cast_dtype,
    )
    reset_triton_meta_bwd_counters()
    reset_triton_sdpa_debug_counters()
    reset_maml_profile_counters()

    # Sparse rel_diff probe (once per run): FULL vs FO on one fresh task
    rel_diff_probe_val = float("nan")
    if rel_diff_probe:
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=cast_dtype,
        ):
            probe_task = task_sampler(
                T=seq_len,
                D=model.cfg.d_model,
                num_signal_positions=num_signal_positions,
                device=sampler_device,
                split="val",
                dataset_seed=seed,
            )
            if sampler_device != device:
                probe_task = _task_batch_to_device(probe_task, device)
            params = [p for p in model.parameters() if p.requires_grad]
            outer_full = meta_loss_on_tasks(
                model=model,
                tasks=[probe_task],
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                fo=False,
                fo_strict=False,
                meta_last_n_inner=meta_last_n_inner,
                check_finite=fail_on_nonfinite,
                finite_prefix="probe_full",
                return_metrics=False,
            )
            outer_fo = meta_loss_on_tasks(
                model=model,
                tasks=[probe_task],
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                fo=True,
                fo_strict=False,
                meta_last_n_inner=0,
                check_finite=fail_on_nonfinite,
                finite_prefix="probe_fo",
                return_metrics=False,
            )
            eps_probe = 1e-9
            grads_full = torch.autograd.grad(outer_full, params, retain_graph=False, allow_unused=True)
            grads_fo = torch.autograd.grad(outer_fo, params, retain_graph=False, allow_unused=True)
            def _vec(grads):
                return torch.cat([g.flatten() for g in grads if g is not None]) if any(g is not None for g in grads) else None
            v_full = _vec(grads_full)
            v_fo = _vec(grads_fo)
            if v_full is not None and v_fo is not None and v_full.norm().item() > eps_probe:
                rel_diff_probe_val = (v_full - v_fo).norm().item() / (v_full.norm().item() + eps_probe)
        # Keep counters scoped to training path; probe calls are diagnostic only.
        reset_triton_meta_bwd_counters()
        reset_triton_sdpa_debug_counters()
        reset_maml_profile_counters()

    fallback_bwd_count = int(meta_bwd_counts.get("n_fallback_bwd", 0))
    row = {
        "backend": attention_backend,
        "mode": mode,
        "seed": seed,
        "outer_steps": outer_steps,
        "meta_batch": meta_batch_size,
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
        "outer_lr": outer_lr,
        "seq_len": seq_len,
        "num_signal_positions": num_signal_positions,
        "dtype": dtype_name,
        "compute_dtype": compute_dtype,
        "autocast": int(bool(autocast_enabled)),
        "meta_every_n_outer": int(meta_every_n_outer if mode == "FULL_HYBRID" else 0),
        "meta_last_n_inner": int(meta_last_n_inner if mode in {"FULL", "FULL_HYBRID"} else 0),
        "n_hybrid_meta_steps": int(n_hybrid_meta_steps),
        "n_hybrid_fo_steps": int(n_hybrid_fo_steps),
        "final_loss": final_loss,
        "final_acc": final_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "attn_grad_norm_q": attn_grad_norm_q,
        "attn_grad_norm_k": attn_grad_norm_k,
        "attn_grad_norm_v": attn_grad_norm_v,
        "attn_grad_present": str(bool(attn_grad_present)),
        "sdpa_input_gradgrad_ok": str(bool(sdpa_input_gradgrad_ok)),
        "rel_diff_probe": rel_diff_probe_val,
        "convergence_delta": convergence_delta,
        "wall_time_total_s": wall_time_total_s,
        "mean_outer_step_time_s": mean_outer_step_time_s,
        "peak_cuda_mem_bytes": peak_cuda_mem_bytes,
        "cuda_graph_static_requested": cuda_graph_static_requested,
        "cuda_graph_static_used": int(bool(cuda_graph_static_used)),
        "cuda_graph_static_reason": cuda_graph_static_reason,
        "cuda_graph_capture_time_s": cuda_graph_capture_time_s,
        "cuda_graph_replay_time_s": cuda_graph_replay_time_s,
        "cuda_graph_meta_capture_ok": int(cuda_graph_meta_capture_ok),
        "cuda_graph_fo_capture_ok": int(cuda_graph_fo_capture_ok),
        "cuda_graph_meta_replays": int(cuda_graph_meta_replays),
        "cuda_graph_fo_replays": int(cuda_graph_fo_replays),
        "meta_loss_time_s": meta_loss_time_s,
        "outer_backward_time_s": outer_backward_time_s,
        "optimizer_step_time_s": optimizer_step_time_s,
        "meta_loss_time_per_outer_step_s": meta_loss_time_s / max(outer_steps, 1),
        "outer_backward_time_per_outer_step_s": outer_backward_time_s / max(outer_steps, 1),
        "optimizer_step_time_per_outer_step_s": optimizer_step_time_s / max(outer_steps, 1),
        "hybrid_meta_meta_loss_time_s": hybrid_meta_meta_loss_time_s,
        "hybrid_fo_meta_loss_time_s": hybrid_fo_meta_loss_time_s,
        "hybrid_meta_outer_backward_time_s": hybrid_meta_outer_backward_time_s,
        "hybrid_fo_outer_backward_time_s": hybrid_fo_outer_backward_time_s,
        "hybrid_meta_optimizer_step_time_s": hybrid_meta_optimizer_step_time_s,
        "hybrid_fo_optimizer_step_time_s": hybrid_fo_optimizer_step_time_s,
        "hybrid_meta_meta_loss_time_per_meta_step_s": hybrid_meta_meta_loss_time_s
        / max(n_hybrid_meta_steps, 1),
        "hybrid_fo_meta_loss_time_per_fo_step_s": hybrid_fo_meta_loss_time_s
        / max(n_hybrid_fo_steps, 1),
        "hybrid_meta_outer_backward_time_per_meta_step_s": hybrid_meta_outer_backward_time_s
        / max(n_hybrid_meta_steps, 1),
        "hybrid_fo_outer_backward_time_per_fo_step_s": hybrid_fo_outer_backward_time_s
        / max(n_hybrid_fo_steps, 1),
        "hybrid_meta_optimizer_step_time_per_meta_step_s": hybrid_meta_optimizer_step_time_s
        / max(n_hybrid_meta_steps, 1),
        "hybrid_fo_optimizer_step_time_per_fo_step_s": hybrid_fo_optimizer_step_time_s
        / max(n_hybrid_fo_steps, 1),
        "n_fast_bwd": int(meta_bwd_counts["n_fast_bwd"]),
        "n_meta_bwd": int(meta_bwd_counts["n_meta_bwd"]),
        "n_fallback_bwd": fallback_bwd_count,
        "fallback_incident": int(fallback_bwd_count > 0),
        "fallback_bwd_per_outer_step": float(fallback_bwd_count) / max(outer_steps, 1),
        "fast_bwd_time_s": float(meta_bwd_counts.get("fast_bwd_time_s", 0.0)),
        "meta_bwd_time_s": float(meta_bwd_counts.get("meta_bwd_time_s", 0.0)),
        "meta_recompute_time_s": float(meta_bwd_counts.get("meta_recompute_time_s", 0.0)),
        "fallback_bwd_time_s": float(meta_bwd_counts.get("fallback_bwd_time_s", 0.0)),
        "fast_bwd_time_per_outer_step_s": float(meta_bwd_counts.get("fast_bwd_time_s", 0.0))
        / max(outer_steps, 1),
        "meta_bwd_time_per_outer_step_s": float(meta_bwd_counts.get("meta_bwd_time_s", 0.0))
        / max(outer_steps, 1),
        "meta_recompute_time_per_outer_step_s": float(
            meta_bwd_counts.get("meta_recompute_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "fallback_bwd_time_per_outer_step_s": float(
            meta_bwd_counts.get("fallback_bwd_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "n_tasks_profiled": int(maml_profile_counts.get("n_tasks", 0)),
        "n_inner_steps_profiled": int(maml_profile_counts.get("n_inner_steps", 0)),
        "n_support_forward_calls": int(maml_profile_counts.get("n_support_forward_calls", 0)),
        "n_support_grad_calls": int(maml_profile_counts.get("n_support_grad_calls", 0)),
        "n_support_grad_create_graph_calls": int(
            maml_profile_counts.get("n_support_grad_create_graph_calls", 0)
        ),
        "n_support_grad_no_graph_calls": int(
            maml_profile_counts.get("n_support_grad_no_graph_calls", 0)
        ),
        "n_param_update_calls": int(maml_profile_counts.get("n_param_update_calls", 0)),
        "n_query_forward_calls": int(maml_profile_counts.get("n_query_forward_calls", 0)),
        "support_forward_time_s": float(maml_profile_counts.get("support_forward_time_s", 0.0)),
        "support_grad_time_s": float(maml_profile_counts.get("support_grad_time_s", 0.0)),
        "support_grad_create_graph_time_s": float(
            maml_profile_counts.get("support_grad_create_graph_time_s", 0.0)
        ),
        "support_grad_no_graph_time_s": float(
            maml_profile_counts.get("support_grad_no_graph_time_s", 0.0)
        ),
        "param_update_time_s": float(maml_profile_counts.get("param_update_time_s", 0.0)),
        "query_forward_time_s": float(maml_profile_counts.get("query_forward_time_s", 0.0)),
        "support_forward_time_per_outer_step_s": float(
            maml_profile_counts.get("support_forward_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "support_grad_time_per_outer_step_s": float(
            maml_profile_counts.get("support_grad_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "support_grad_create_graph_time_per_outer_step_s": float(
            maml_profile_counts.get("support_grad_create_graph_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "support_grad_no_graph_time_per_outer_step_s": float(
            maml_profile_counts.get("support_grad_no_graph_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "param_update_time_per_outer_step_s": float(
            maml_profile_counts.get("param_update_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "query_forward_time_per_outer_step_s": float(
            maml_profile_counts.get("query_forward_time_s", 0.0)
        )
        / max(outer_steps, 1),
        "sdpa_debug_n_nonfinite_m": int(sdpa_debug_counts.get("n_nonfinite_m", 0)),
        "sdpa_debug_n_nonfinite_l": int(sdpa_debug_counts.get("n_nonfinite_l", 0)),
        "sdpa_debug_n_tiny_l": int(sdpa_debug_counts.get("n_tiny_l", 0)),
        "sdpa_debug_n_nonfinite_p": int(sdpa_debug_counts.get("n_nonfinite_p", 0)),
        "sdpa_debug_n_extreme_p": int(sdpa_debug_counts.get("n_extreme_p", 0)),
        "sdpa_debug_n_nonfinite_dp": int(sdpa_debug_counts.get("n_nonfinite_dp", 0)),
        "sdpa_debug_n_extreme_dp": int(sdpa_debug_counts.get("n_extreme_dp", 0)),
        "sdpa_debug_n_nonfinite_ds": int(sdpa_debug_counts.get("n_nonfinite_ds", 0)),
        "sdpa_debug_n_extreme_ds": int(sdpa_debug_counts.get("n_extreme_ds", 0)),
        "sdpa_debug_n_nonfinite_dq": int(sdpa_debug_counts.get("n_nonfinite_dq", 0)),
        "sdpa_debug_n_nonfinite_dk": int(sdpa_debug_counts.get("n_nonfinite_dk", 0)),
        "sdpa_debug_n_nonfinite_dv": int(sdpa_debug_counts.get("n_nonfinite_dv", 0)),
        "sdpa_debug_max_abs_p": float(sdpa_debug_counts.get("max_abs_p", 0.0)),
        "sdpa_debug_max_p_row_sum_err": float(
            sdpa_debug_counts.get("max_p_row_sum_err", 0.0)
        ),
        "sdpa_debug_max_abs_dp": float(sdpa_debug_counts.get("max_abs_dp", 0.0)),
        "sdpa_debug_max_abs_ds": float(sdpa_debug_counts.get("max_abs_ds", 0.0)),
    }
    if profile_meta_bwd:
        if prev_meta_profile_env is None:
            os.environ.pop("THERIA_TRITON_META_PROFILE", None)
        else:
            os.environ["THERIA_TRITON_META_PROFILE"] = prev_meta_profile_env
        if prev_maml_profile_env is None:
            os.environ.pop("THERIA_MAML_PROFILE", None)
        else:
            os.environ["THERIA_MAML_PROFILE"] = prev_maml_profile_env
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=[
            "reference",
            "custom",
            "triton_fused",
            "triton_fused_meta",
            "triton_fused_meta_strict",
            "triton_full_autograd",
            "triton_frozen_stats",
            "reference_frozen_stats",
        ],
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["FULL", "FO", "FO_STRICT", "FULL_FROZEN", "FULL_HYBRID"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-steps", type=int, default=500)
    parser.add_argument("--meta-batch-size", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument(
        "--meta-every-n-outer",
        type=int,
        default=8,
        help="FULL_HYBRID only: run FULL step every N outer steps; FO otherwise.",
    )
    parser.add_argument(
        "--meta-last-n-inner",
        type=int,
        default=0,
        help=(
            "For FULL/FULL_HYBRID only: keep full second-order graph for the last N "
            "inner updates (0 disables truncation)."
        ),
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-signal-positions", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--autocast", action="store_true", help="Enable torch.autocast during training")
    parser.add_argument(
        "--profile-meta-bwd",
        action="store_true",
        help=(
            "Enable Triton meta backward and MAML loop timing counters via "
            "THERIA_TRITON_META_PROFILE=1 and THERIA_MAML_PROFILE=1."
        ),
    )
    parser.add_argument(
        "--no-fail-on-nonfinite",
        action="store_true",
        help="Disable hard failure on non-finite logits/loss/grads (debug only).",
    )
    parser.add_argument(
        "--allow-experimental-backends",
        action="store_true",
        help="Opt in to experimental backend(s) such as triton_fused_meta.",
    )
    parser.add_argument(
        "--cuda-graph-static",
        action="store_true",
        help=(
            "Enable the static-shape CUDA-graph path for the stable practical "
            "workload (CUDA + triton_fused_meta_strict FULL_HYBRID only)."
        ),
    )
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    try:
        row = run_behavior(
            attention_backend=args.backend,
            mode=args.mode,  # type: ignore[arg-type]
            seed=args.seed,
            outer_steps=args.outer_steps,
            meta_batch_size=args.meta_batch_size,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            outer_lr=args.outer_lr,
            meta_every_n_outer=args.meta_every_n_outer,
            meta_last_n_inner=args.meta_last_n_inner,
            profile_meta_bwd=args.profile_meta_bwd,
            fail_on_nonfinite=not args.no_fail_on_nonfinite,
            allow_experimental_backends=args.allow_experimental_backends,
            seq_len=args.seq_len,
            num_signal_positions=args.num_signal_positions,
            device=torch.device(args.device),
            autocast_enabled=args.autocast,
            cuda_graph_static=args.cuda_graph_static,
        )
        status, error = "OK", ""
    except Exception as e:
        row = {
            "backend": args.backend,
            "mode": args.mode,
            "seed": args.seed,
            "outer_steps": args.outer_steps,
            "meta_batch": args.meta_batch_size,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "outer_lr": args.outer_lr,
            "seq_len": args.seq_len,
            "num_signal_positions": args.num_signal_positions,
            "dtype": "NA",
            "compute_dtype": "NA",
            "autocast": int(bool(args.autocast)),
            "meta_every_n_outer": int(args.meta_every_n_outer if args.mode == "FULL_HYBRID" else 0),
            "meta_last_n_inner": int(
                args.meta_last_n_inner if args.mode in {"FULL", "FULL_HYBRID"} else 0
            ),
            "n_hybrid_meta_steps": 0,
            "n_hybrid_fo_steps": 0,
            "final_loss": float("nan"),
            "final_acc": float("nan"),
            "val_loss": float("nan"),
            "val_acc": float("nan"),
            "test_loss": float("nan"),
            "test_acc": float("nan"),
            "attn_grad_present": "False",
            "attn_grad_norm_q": float("nan"),
            "attn_grad_norm_k": float("nan"),
            "attn_grad_norm_v": float("nan"),
            "sdpa_input_gradgrad_ok": "False",
            "rel_diff_probe": float("nan"),
            "wall_time_total_s": float("nan"),
            "mean_outer_step_time_s": float("nan"),
            "peak_cuda_mem_bytes": 0,
            "cuda_graph_static_requested": int(bool(args.cuda_graph_static)),
            "cuda_graph_static_used": 0,
            "cuda_graph_static_reason": "",
            "cuda_graph_capture_time_s": float("nan"),
            "cuda_graph_replay_time_s": float("nan"),
            "cuda_graph_meta_capture_ok": 0,
            "cuda_graph_fo_capture_ok": 0,
            "cuda_graph_meta_replays": 0,
            "cuda_graph_fo_replays": 0,
            "convergence_delta": float("nan"),
            "n_fast_bwd": 0,
            "n_meta_bwd": 0,
            "n_fallback_bwd": 0,
            "fallback_incident": 0,
            "fallback_bwd_per_outer_step": float("nan"),
            "fast_bwd_time_s": float("nan"),
            "meta_bwd_time_s": float("nan"),
            "meta_recompute_time_s": float("nan"),
            "fallback_bwd_time_s": float("nan"),
            "fast_bwd_time_per_outer_step_s": float("nan"),
            "meta_bwd_time_per_outer_step_s": float("nan"),
            "meta_recompute_time_per_outer_step_s": float("nan"),
            "fallback_bwd_time_per_outer_step_s": float("nan"),
            "n_tasks_profiled": 0,
            "n_inner_steps_profiled": 0,
            "n_support_forward_calls": 0,
            "n_support_grad_calls": 0,
            "n_support_grad_create_graph_calls": 0,
            "n_support_grad_no_graph_calls": 0,
            "n_param_update_calls": 0,
            "n_query_forward_calls": 0,
            "meta_loss_time_s": float("nan"),
            "outer_backward_time_s": float("nan"),
            "optimizer_step_time_s": float("nan"),
            "meta_loss_time_per_outer_step_s": float("nan"),
            "outer_backward_time_per_outer_step_s": float("nan"),
            "optimizer_step_time_per_outer_step_s": float("nan"),
            "hybrid_meta_meta_loss_time_s": float("nan"),
            "hybrid_fo_meta_loss_time_s": float("nan"),
            "hybrid_meta_outer_backward_time_s": float("nan"),
            "hybrid_fo_outer_backward_time_s": float("nan"),
            "hybrid_meta_optimizer_step_time_s": float("nan"),
            "hybrid_fo_optimizer_step_time_s": float("nan"),
            "hybrid_meta_meta_loss_time_per_meta_step_s": float("nan"),
            "hybrid_fo_meta_loss_time_per_fo_step_s": float("nan"),
            "hybrid_meta_outer_backward_time_per_meta_step_s": float("nan"),
            "hybrid_fo_outer_backward_time_per_fo_step_s": float("nan"),
            "hybrid_meta_optimizer_step_time_per_meta_step_s": float("nan"),
            "hybrid_fo_optimizer_step_time_per_fo_step_s": float("nan"),
            "support_forward_time_s": float("nan"),
            "support_grad_time_s": float("nan"),
            "support_grad_create_graph_time_s": float("nan"),
            "support_grad_no_graph_time_s": float("nan"),
            "param_update_time_s": float("nan"),
            "query_forward_time_s": float("nan"),
            "support_forward_time_per_outer_step_s": float("nan"),
            "support_grad_time_per_outer_step_s": float("nan"),
            "support_grad_create_graph_time_per_outer_step_s": float("nan"),
            "support_grad_no_graph_time_per_outer_step_s": float("nan"),
            "param_update_time_per_outer_step_s": float("nan"),
            "query_forward_time_per_outer_step_s": float("nan"),
            "sdpa_debug_n_nonfinite_m": 0,
            "sdpa_debug_n_nonfinite_l": 0,
            "sdpa_debug_n_tiny_l": 0,
            "sdpa_debug_n_nonfinite_p": 0,
            "sdpa_debug_n_extreme_p": 0,
            "sdpa_debug_n_nonfinite_dp": 0,
            "sdpa_debug_n_extreme_dp": 0,
            "sdpa_debug_n_nonfinite_ds": 0,
            "sdpa_debug_n_extreme_ds": 0,
            "sdpa_debug_n_nonfinite_dq": 0,
            "sdpa_debug_n_nonfinite_dk": 0,
            "sdpa_debug_n_nonfinite_dv": 0,
            "sdpa_debug_max_abs_p": float("nan"),
            "sdpa_debug_max_p_row_sum_err": float("nan"),
            "sdpa_debug_max_abs_dp": float("nan"),
            "sdpa_debug_max_abs_ds": float("nan"),
        }
        err_str = repr(e)
        if "NONFINITE" in err_str or "non_finite" in err_str:
            status, error = "HARD_FAIL_NONFINITE", err_str
        else:
            status, error = "HARD_FAIL_OTHER", err_str

    print(
        f"backend={row['backend']} mode={row['mode']} seed={row['seed']} "
        f"final_loss={row['final_loss']} final_acc={row['final_acc']} "
        f"val_loss={row.get('val_loss','NA')} val_acc={row.get('val_acc','NA')} "
        f"test_loss={row.get('test_loss','NA')} test_acc={row.get('test_acc','NA')} "
        f"attn_grad_norm_q={row['attn_grad_norm_q']} "
        f"attn_grad_norm_k={row['attn_grad_norm_k']} "
        f"attn_grad_norm_v={row['attn_grad_norm_v']} "
        f"attn_grad_present={row['attn_grad_present']} "
        f"rel_diff_probe={row.get('rel_diff_probe','NA')} "
        f"convergence_delta={row['convergence_delta']} "
        f"wall_time_total_s={row['wall_time_total_s']} "
        f"mean_outer_step_time_s={row['mean_outer_step_time_s']} "
        f"cuda_graph_static_used={row.get('cuda_graph_static_used',0)} "
        f"meta_last_n_inner={row.get('meta_last_n_inner',0)} "
        f"n_fast_bwd={row.get('n_fast_bwd',0)} "
        f"n_meta_bwd={row.get('n_meta_bwd',0)} "
        f"n_fallback_bwd={row.get('n_fallback_bwd',0)} "
        f"meta_loss_time_s={row.get('meta_loss_time_s',0.0)} "
        f"outer_backward_time_s={row.get('outer_backward_time_s',0.0)} "
        f"meta_bwd_time_s={row.get('meta_bwd_time_s',0.0)} "
        f"meta_recompute_time_s={row.get('meta_recompute_time_s',0.0)} "
        f"support_grad_time_s={row.get('support_grad_time_s',0.0)} "
        f"query_forward_time_s={row.get('query_forward_time_s',0.0)} "
        f"fallback_bwd_time_s={row.get('fallback_bwd_time_s',0.0)} "
        f"status={status}"
    )

    if args.csv_out:
        path = Path(args.csv_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "backend",
                    "mode",
                    "inner_steps",
                    "seed",
                    "meta_batch",
                    "outer_lr",
                    "outer_steps",
                    "inner_lr",
                    "seq_len",
                    "num_signal_positions",
                    "dtype",
                    "compute_dtype",
                    "autocast",
                    "meta_every_n_outer",
                    "meta_last_n_inner",
                    "n_hybrid_meta_steps",
                    "n_hybrid_fo_steps",
                    "final_loss",
                    "final_acc",
                    "val_loss",
                    "val_acc",
                    "test_loss",
                    "test_acc",
                    "attn_grad_present",
                    "attn_grad_norm_q",
                    "attn_grad_norm_k",
                    "attn_grad_norm_v",
                    "sdpa_input_gradgrad_ok",
                    "rel_diff_probe",
                    "wall_time_total_s",
                    "mean_outer_step_time_s",
                    "peak_cuda_mem_bytes",
                    "cuda_graph_static_requested",
                    "cuda_graph_static_used",
                    "cuda_graph_static_reason",
                    "cuda_graph_capture_time_s",
                    "cuda_graph_replay_time_s",
                    "cuda_graph_meta_capture_ok",
                    "cuda_graph_fo_capture_ok",
                    "cuda_graph_meta_replays",
                    "cuda_graph_fo_replays",
                    "meta_loss_time_s",
                    "outer_backward_time_s",
                    "optimizer_step_time_s",
                    "meta_loss_time_per_outer_step_s",
                    "outer_backward_time_per_outer_step_s",
                    "optimizer_step_time_per_outer_step_s",
                    "hybrid_meta_meta_loss_time_s",
                    "hybrid_fo_meta_loss_time_s",
                    "hybrid_meta_outer_backward_time_s",
                    "hybrid_fo_outer_backward_time_s",
                    "hybrid_meta_optimizer_step_time_s",
                    "hybrid_fo_optimizer_step_time_s",
                    "hybrid_meta_meta_loss_time_per_meta_step_s",
                    "hybrid_fo_meta_loss_time_per_fo_step_s",
                    "hybrid_meta_outer_backward_time_per_meta_step_s",
                    "hybrid_fo_outer_backward_time_per_fo_step_s",
                    "hybrid_meta_optimizer_step_time_per_meta_step_s",
                    "hybrid_fo_optimizer_step_time_per_fo_step_s",
                    "n_fast_bwd",
                    "n_meta_bwd",
                    "n_fallback_bwd",
                    "fallback_incident",
                    "fallback_bwd_per_outer_step",
                    "fast_bwd_time_s",
                    "meta_bwd_time_s",
                    "meta_recompute_time_s",
                    "fallback_bwd_time_s",
                    "fast_bwd_time_per_outer_step_s",
                    "meta_bwd_time_per_outer_step_s",
                    "meta_recompute_time_per_outer_step_s",
                    "fallback_bwd_time_per_outer_step_s",
                    "n_tasks_profiled",
                    "n_inner_steps_profiled",
                    "n_support_forward_calls",
                    "n_support_grad_calls",
                    "n_support_grad_create_graph_calls",
                    "n_support_grad_no_graph_calls",
                    "n_param_update_calls",
                    "n_query_forward_calls",
                    "support_forward_time_s",
                    "support_grad_time_s",
                    "support_grad_create_graph_time_s",
                    "support_grad_no_graph_time_s",
                    "param_update_time_s",
                    "query_forward_time_s",
                    "support_forward_time_per_outer_step_s",
                    "support_grad_time_per_outer_step_s",
                    "support_grad_create_graph_time_per_outer_step_s",
                    "support_grad_no_graph_time_per_outer_step_s",
                    "param_update_time_per_outer_step_s",
                    "query_forward_time_per_outer_step_s",
                    "sdpa_debug_n_nonfinite_m",
                    "sdpa_debug_n_nonfinite_l",
                    "sdpa_debug_n_tiny_l",
                    "sdpa_debug_n_nonfinite_p",
                    "sdpa_debug_n_extreme_p",
                    "sdpa_debug_n_nonfinite_dp",
                    "sdpa_debug_n_extreme_dp",
                    "sdpa_debug_n_nonfinite_ds",
                    "sdpa_debug_n_extreme_ds",
                    "sdpa_debug_n_nonfinite_dq",
                    "sdpa_debug_n_nonfinite_dk",
                    "sdpa_debug_n_nonfinite_dv",
                    "sdpa_debug_max_abs_p",
                    "sdpa_debug_max_p_row_sum_err",
                    "sdpa_debug_max_abs_dp",
                    "sdpa_debug_max_abs_ds",
                    "status",
                    "error",
                    "convergence_delta",
                ],
            )
            if write_header:
                w.writeheader()
            w.writerow({**row, "status": status, "error": error, "convergence_delta": row.get("convergence_delta", float("nan"))})


if __name__ == "__main__":
    main()
