#!/usr/bin/env python3
"""
Minimal CUDA-graph capture probe for the stable practical workload.

This isolates capture viability away from the frontier runners.
It intentionally targets the FO-equivalent path only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.func import functional_call

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase10.scripts.run_maml_backend_compare import (  # noqa: E402
    Phase10TinyAttentionModel,
    set_attention_backend,
)
from theria.attention.custom import sdpa_custom  # noqa: E402
from theria.maml.loops import (  # noqa: E402
    inner_adapt,
    loss_fn,
    meta_loss_on_tasks,
    named_buffers,
    named_params,
)
from theria.tasks.synthetic_seqcls import TaskBatch, task_sampler  # noqa: E402


def _task_batch_to_device(task: TaskBatch, device: torch.device) -> TaskBatch:
    return TaskBatch(
        x_s=task.x_s.to(device=device, non_blocking=False),
        y_s=task.y_s.to(device=device, non_blocking=False),
        x_q=task.x_q.to(device=device, non_blocking=False),
        y_q=task.y_q.to(device=device, non_blocking=False),
    )


def _build_static_tasks(
    *,
    model: Phase10TinyAttentionModel,
    meta_batch_size: int,
    seq_len: int,
    num_signal_positions: int,
    device: torch.device,
) -> list[TaskBatch]:
    cpu_device = torch.device("cpu")
    return [
        _task_batch_to_device(
            task_sampler(
                T=seq_len,
                D=model.cfg.d_model,
                num_signal_positions=num_signal_positions,
                device=cpu_device,
            ),
            device,
        )
        for _ in range(meta_batch_size)
    ]


def _isfinite_scalar(t: torch.Tensor) -> int:
    return int(bool(torch.isfinite(t).item()))


def _adapt_probe_scalar(phi: dict[str, torch.Tensor]) -> torch.Tensor:
    probe: torch.Tensor | None = None
    for value in phi.values():
        term = value.float().square().mean()
        probe = term if probe is None else (probe + term)
    assert probe is not None
    return probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "fo_loss",
            "fo_loss_and_outer_grad",
            "fo_support_only",
            "fo_query_only",
            "query_forward_only",
            "query_loss_only",
            "direct_forward_only",
            "attention_only",
            "attention_explicit_only",
            "attention_sdpa_reference_only",
            "qk_matmul_only",
            "qk_matmul_static_buffers_only",
            "qk_matmul_transpose_view_only",
            "softmax_only",
            "pv_matmul_only",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--num-signal-positions", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("run_phase12_cuda_graph_capture_debug.py requires --device cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = Phase10TinyAttentionModel().to(device)
    set_attention_backend(model, "triton_fused_meta_strict")
    static_tasks = _build_static_tasks(
        model=model,
        meta_batch_size=args.meta_batch_size,
        seq_len=args.seq_len,
        num_signal_positions=args.num_signal_positions,
        device=device,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    params = named_params(model)
    buffers = named_buffers(model)
    grad_buffers = [torch.zeros_like(p) for p in trainable_params]
    loss_out = torch.zeros((), device=device)
    acc_out = torch.zeros((), device=device)
    query_task = static_tasks[0]
    logits_static = functional_call(model, (params, buffers), (query_task.x_q,))
    logits_static = logits_static.detach().clone()
    q_attn = model.q_proj(query_task.x_q[:, :1, :]).unsqueeze(1).float().detach().clone()
    k_attn = model.k_proj(query_task.x_q).unsqueeze(1).float().detach().clone()
    v_attn = model.v_proj(query_task.x_q).unsqueeze(1).float().detach().clone()
    attn_scale = 1.0 / (q_attn.shape[-1] ** 0.5)
    k_attn_t_static = k_attn.transpose(-2, -1).contiguous()
    scores_static = (torch.matmul(q_attn, k_attn.transpose(-2, -1)) * attn_scale).detach().clone()
    probs_static = torch.softmax(scores_static, dim=-1).detach().clone()

    def _run_stage() -> None:
        if args.stage in {"fo_loss", "fo_loss_and_outer_grad"}:
            outer_loss, metrics = meta_loss_on_tasks(
                model=model,
                tasks=static_tasks,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                fo=True,
                fo_strict=False,
                meta_last_n_inner=0,
                check_finite=True,
                finite_prefix=f"cuda_graph_debug stage={args.stage}",
                return_metrics=True,
                return_metrics_tensors=True,
            )
            loss_out.copy_(outer_loss)
            acc_out.copy_(metrics["post_adapt_acc"])
        elif args.stage == "fo_support_only":
            probe_total: torch.Tensor | None = None
            for task_idx, task in enumerate(static_tasks):
                phi = inner_adapt(
                    model,
                    params,
                    buffers,
                    task,
                    inner_lr=args.inner_lr,
                    inner_steps=args.inner_steps,
                    fo=True,
                    meta_last_n_inner=0,
                    check_finite=True,
                    finite_prefix=f"cuda_graph_debug stage={args.stage} task={task_idx}",
                )
                probe = _adapt_probe_scalar(phi)
                probe_total = probe if probe_total is None else (probe_total + probe)
            assert probe_total is not None
            loss_out.copy_(probe_total / len(static_tasks))
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "fo_query_only":
            loss_sum: torch.Tensor | None = None
            acc_sum: torch.Tensor | None = None
            for task_idx, task in enumerate(static_tasks):
                logits_q = functional_call(model, (params, buffers), (task.x_q,))
                if not torch.isfinite(logits_q).all():
                    raise RuntimeError(
                        f"NONFINITE query_logits cuda_graph_debug stage={args.stage} task={task_idx}"
                    )
                post_loss = loss_fn(logits_q, task.y_q)
                post_acc = (logits_q.argmax(dim=-1) == task.y_q).float().mean()
                loss_sum = post_loss if loss_sum is None else (loss_sum + post_loss)
                acc_sum = post_acc if acc_sum is None else (acc_sum + post_acc)
            assert loss_sum is not None and acc_sum is not None
            outer_loss = loss_sum / len(static_tasks)
            loss_out.copy_(outer_loss)
            acc_out.copy_(acc_sum / len(static_tasks))
        elif args.stage == "query_forward_only":
            logits_q = functional_call(model, (params, buffers), (query_task.x_q,))
            if not torch.isfinite(logits_q).all():
                raise RuntimeError(
                    "NONFINITE query_logits cuda_graph_debug stage=query_forward_only"
                )
            loss_out.copy_(logits_q.float().square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "query_loss_only":
            post_loss = loss_fn(logits_static, query_task.y_q)
            post_acc = (logits_static.argmax(dim=-1) == query_task.y_q).float().mean()
            loss_out.copy_(post_loss)
            acc_out.copy_(post_acc)
            outer_loss = post_loss
        elif args.stage == "direct_forward_only":
            logits_q = model(query_task.x_q)
            if not torch.isfinite(logits_q).all():
                raise RuntimeError(
                    "NONFINITE direct_logits cuda_graph_debug stage=direct_forward_only"
                )
            loss_out.copy_(logits_q.float().square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "attention_only":
            q = model.q_proj(query_task.x_q[:, :1, :]).unsqueeze(1)
            k = model.k_proj(query_task.x_q).unsqueeze(1)
            v = model.v_proj(query_task.x_q).unsqueeze(1)
            h = sdpa_custom(q, k, v, backend="triton_fused_meta_strict")
            if not torch.isfinite(h).all():
                raise RuntimeError(
                    "NONFINITE attention_out cuda_graph_debug stage=attention_only"
                )
            loss_out.copy_(h.float().square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "attention_explicit_only":
            q = model.q_proj(query_task.x_q[:, :1, :]).unsqueeze(1).float()
            k = model.k_proj(query_task.x_q).unsqueeze(1).float()
            v = model.v_proj(query_task.x_q).unsqueeze(1).float()
            scale = 1.0 / (q.shape[-1] ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores - scores.max(dim=-1, keepdim=True).values
            probs = torch.softmax(scores, dim=-1)
            h = torch.matmul(probs, v)
            if not torch.isfinite(h).all():
                raise RuntimeError(
                    "NONFINITE attention_out cuda_graph_debug stage=attention_explicit_only"
                )
            loss_out.copy_(h.square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "attention_sdpa_reference_only":
            q = model.q_proj(query_task.x_q[:, :1, :]).unsqueeze(1)
            k = model.k_proj(query_task.x_q).unsqueeze(1)
            v = model.v_proj(query_task.x_q).unsqueeze(1)
            h = sdpa_custom(q, k, v, backend="reference")
            if not torch.isfinite(h).all():
                raise RuntimeError(
                    "NONFINITE attention_out cuda_graph_debug stage=attention_sdpa_reference_only"
                )
            loss_out.copy_(h.float().square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "qk_matmul_only":
            scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * attn_scale
            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    "NONFINITE scores cuda_graph_debug stage=qk_matmul_only"
                )
            loss_out.copy_(scores.square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "qk_matmul_static_buffers_only":
            scores = torch.matmul(q_attn, k_attn_t_static) * attn_scale
            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    "NONFINITE scores cuda_graph_debug stage=qk_matmul_static_buffers_only"
                )
            loss_out.copy_(scores.square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "qk_matmul_transpose_view_only":
            k_t = k_attn.transpose(-2, -1)
            scores = torch.matmul(q_attn, k_t) * attn_scale
            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    "NONFINITE scores cuda_graph_debug stage=qk_matmul_transpose_view_only"
                )
            loss_out.copy_(scores.square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "softmax_only":
            probs = torch.softmax(scores_static, dim=-1)
            if not torch.isfinite(probs).all():
                raise RuntimeError(
                    "NONFINITE probs cuda_graph_debug stage=softmax_only"
                )
            loss_out.copy_(probs.square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        elif args.stage == "pv_matmul_only":
            h = torch.matmul(probs_static, v_attn)
            if not torch.isfinite(h).all():
                raise RuntimeError(
                    "NONFINITE pv_out cuda_graph_debug stage=pv_matmul_only"
                )
            loss_out.copy_(h.square().mean())
            acc_out.zero_()
            outer_loss = loss_out
        else:  # pragma: no cover - argparse choices keep this unreachable
            raise ValueError(f"Unknown stage: {args.stage}")

        if args.stage == "fo_loss_and_outer_grad":
            grads = torch.autograd.grad(
                outer_loss,
                trainable_params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            for buf, grad in zip(grad_buffers, grads):
                if grad is None:
                    buf.zero_()
                else:
                    buf.copy_(grad)

    capture_ok = 0
    replay_ok = 0
    loss_isfinite = 0
    acc_isfinite = 0
    all_grads_finite = 1 if args.stage == "fo_loss" else 0
    error = ""

    warmup_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            _run_stage()
    current_stream.wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _run_stage()
        capture_ok = 1
    except Exception as e:  # pragma: no cover - GPU-only failure path
        error = repr(e)

    if capture_ok:
        try:
            graph.replay()
            torch.cuda.synchronize(device)
            replay_ok = 1
            loss_isfinite = _isfinite_scalar(loss_out)
            acc_isfinite = _isfinite_scalar(acc_out)
            if args.stage == "fo_loss_and_outer_grad":
                all_grads_finite = int(
                    all(bool(torch.isfinite(buf).all().item()) for buf in grad_buffers)
                )
        except Exception as e:  # pragma: no cover - GPU-only failure path
            error = repr(e)

    status = "OK"
    if not (capture_ok and replay_ok and loss_isfinite and acc_isfinite and all_grads_finite):
        status = "FAIL"

    row = {
        "stage": args.stage,
        "backend": "triton_fused_meta_strict",
        "device": str(device),
        "seed": args.seed,
        "seq_len": args.seq_len,
        "meta_batch_size": args.meta_batch_size,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "outer_lr": args.outer_lr,
        "num_signal_positions": args.num_signal_positions,
        "status": status,
        "capture_ok": capture_ok,
        "replay_ok": replay_ok,
        "loss_isfinite": loss_isfinite,
        "acc_isfinite": acc_isfinite,
        "all_grads_finite": all_grads_finite,
        "error": error,
    }

    print(
        " ".join(
            [
                f"stage={row['stage']}",
                f"status={row['status']}",
                f"capture_ok={row['capture_ok']}",
                f"replay_ok={row['replay_ok']}",
                f"loss_isfinite={row['loss_isfinite']}",
                f"acc_isfinite={row['acc_isfinite']}",
                f"all_grads_finite={row['all_grads_finite']}",
                f"error={row['error']}",
            ]
        )
    )

    if args.csv_out:
        path = Path(args.csv_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
