#!/usr/bin/env python3
"""
Deterministic instability reproducer and fused-vs-strict divergence tracer.

Runs one outer-step at a time on identical tasks/model init for:
  - triton_fused_meta
  - triton_fused_meta_strict

Logs first divergence with attention-level diagnostics:
  - logits / softmax stats
  - dQ/dK/dV stats (via q_proj/k_proj/v_proj output grad hooks)
  - projection weight gradient stats
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from experiments.phase10.scripts.run_maml_backend_compare import (
    Phase10TinyAttentionModel,
    set_attention_backend,
)
from theria.attention.triton_qk import (
    get_triton_meta_bwd_counters,
    get_triton_sdpa_fwd_debug_artifact,
    reset_triton_meta_bwd_counters,
    reset_triton_sdpa_fwd_debug_artifacts,
)
from theria.attention.triton_sdpa_backward import (
    get_triton_sdpa_debug_artifact,
    reset_triton_sdpa_debug_artifact,
)
from theria.maml.loops import meta_loss_on_tasks
from theria.tasks.synthetic_seqcls import task_sampler


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float()
    finite_mask = torch.isfinite(xf)
    finite_frac = float(finite_mask.float().mean().item())
    safe = xf[finite_mask]
    if safe.numel() == 0:
        return {
            "absmax": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "l2": float("nan"),
            "finite_frac": 0.0,
        }
    return {
        "absmax": float(safe.abs().max().item()),
        "mean": float(safe.mean().item()),
        "std": float(safe.std(unbiased=False).item()),
        "min": float(safe.min().item()),
        "max": float(safe.max().item()),
        "l2": float(safe.norm().item()),
        "finite_frac": finite_frac,
    }


def _agg_max(entries: list[dict[str, float]], key: str) -> float:
    vals = [float(e[key]) for e in entries if key in e and math.isfinite(float(e[key]))]
    return float(max(vals)) if vals else float("nan")


def _agg_mean(entries: list[dict[str, float]], key: str) -> float:
    vals = [float(e[key]) for e in entries if key in e and math.isfinite(float(e[key]))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


class AttentionProbe:
    def __init__(self, model: Phase10TinyAttentionModel) -> None:
        self.model = model
        self._handles = []
        self._pending_q: list[torch.Tensor] = []
        self.q_entries: list[dict[str, float]] = []
        self.k_entries: list[dict[str, float]] = []
        self.v_entries: list[dict[str, float]] = []
        self.logits_entries: list[dict[str, float]] = []
        self.softmax_entries: list[dict[str, float]] = []

    def clear(self) -> None:
        self._pending_q.clear()
        self.q_entries.clear()
        self.k_entries.clear()
        self.v_entries.clear()
        self.logits_entries.clear()
        self.softmax_entries.clear()

    def attach(self) -> None:
        self.detach()
        self._handles = [
            self.model.q_proj.register_forward_hook(self._q_hook),
            self.model.k_proj.register_forward_hook(self._k_hook),
            self.model.v_proj.register_forward_hook(self._v_hook),
        ]

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _register_grad_hook(self, output: torch.Tensor, entry: dict[str, float]) -> None:
        if not output.requires_grad:
            return

        def _capture_grad(g: torch.Tensor) -> None:
            stats = _tensor_stats(g)
            entry["grad_absmax"] = stats["absmax"]
            entry["grad_l2"] = stats["l2"]
            entry["grad_finite_frac"] = stats["finite_frac"]

        output.register_hook(_capture_grad)

    def _q_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        entry = _tensor_stats(output)
        self._register_grad_hook(output, entry)
        self.q_entries.append(entry)
        self._pending_q.append(output.detach())

    def _k_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        entry = _tensor_stats(output)
        self._register_grad_hook(output, entry)
        self.k_entries.append(entry)
        if self._pending_q:
            q = self._pending_q.pop(0)
            k = output.detach()
            scale = 1.0 / (k.shape[-1] ** 0.5)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            probs = torch.softmax(logits, dim=-1)
            self.logits_entries.append(_tensor_stats(logits))
            self.softmax_entries.append(_tensor_stats(probs))

    def _v_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        entry = _tensor_stats(output)
        self._register_grad_hook(output, entry)
        self.v_entries.append(entry)

    def summary(self) -> dict[str, float]:
        return {
            "logits_absmax": _agg_max(self.logits_entries, "absmax"),
            "logits_mean": _agg_mean(self.logits_entries, "mean"),
            "softmax_min": _agg_mean(self.softmax_entries, "min"),
            "softmax_max": _agg_mean(self.softmax_entries, "max"),
            "softmax_std": _agg_mean(self.softmax_entries, "std"),
            "dq_absmax": _agg_max(self.q_entries, "grad_absmax"),
            "dk_absmax": _agg_max(self.k_entries, "grad_absmax"),
            "dv_absmax": _agg_max(self.v_entries, "grad_absmax"),
            "dq_l2": _agg_mean(self.q_entries, "grad_l2"),
            "dk_l2": _agg_mean(self.k_entries, "grad_l2"),
            "dv_l2": _agg_mean(self.v_entries, "grad_l2"),
        }


def _sample_tasks(
    *,
    seed: int,
    step_idx: int,
    n_tasks: int,
    seq_len: int,
    d_model: int,
    num_signal_positions: int,
    device: torch.device,
) -> list[Any]:
    # Make task stream deterministic per outer-step and independent of backend.
    step_seed = seed * 100_000 + step_idx
    torch.manual_seed(step_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(step_seed)
    return [
        task_sampler(
            T=seq_len,
            D=d_model,
            num_signal_positions=num_signal_positions,
            device=device,
        )
        for _ in range(n_tasks)
    ]


def _proj_grad_stats(model: Phase10TinyAttentionModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in (
        ("q_proj_grad", model.q_proj.weight.grad),
        ("k_proj_grad", model.k_proj.weight.grad),
        ("v_proj_grad", model.v_proj.weight.grad),
    ):
        if p is None:
            out[f"{name}_absmax"] = float("nan")
            out[f"{name}_l2"] = float("nan")
            out[f"{name}_finite_frac"] = 0.0
            continue
        stats = _tensor_stats(p)
        out[f"{name}_absmax"] = stats["absmax"]
        out[f"{name}_l2"] = stats["l2"]
        out[f"{name}_finite_frac"] = stats["finite_frac"]
    return out


def _run_one_step(
    *,
    model: Phase10TinyAttentionModel,
    optimizer: torch.optim.Optimizer,
    probe: AttentionProbe,
    tasks: list[Any],
    mode: str,
    inner_lr: float,
    inner_steps: int,
    meta_every_n_outer: int,
    meta_last_n_inner: int,
    outer_step_idx: int,
) -> dict[str, float | str | int]:
    probe.clear()
    reset_triton_meta_bwd_counters()
    optimizer.zero_grad(set_to_none=True)

    fo_step = mode != "FULL"
    if mode == "FULL_HYBRID":
        fo_step = (outer_step_idx % max(meta_every_n_outer, 1)) != 0
    meta_last_n_inner_step = meta_last_n_inner if not fo_step else 0

    try:
        outer_loss, metrics = meta_loss_on_tasks(
            model=model,
            tasks=tasks,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            fo=fo_step,
            fo_strict=False,
            meta_last_n_inner=meta_last_n_inner_step,
            check_finite=True,
            finite_prefix=f"outer_step={outer_step_idx}",
            return_metrics=True,
        )
        if not torch.isfinite(outer_loss):
            raise RuntimeError(f"NONFINITE outer_loss outer_step={outer_step_idx}")
        outer_loss.backward()
        for pname, p in model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(
                    f"NONFINITE grad[{pname}] outer_step={outer_step_idx}"
                )
        optimizer.step()
        status = "OK"
        error = ""
        final_loss = float(outer_loss.item())
        final_acc = float(metrics["post_adapt_acc"])
    except Exception as e:
        status = "HARD_FAIL_NONFINITE" if "NONFINITE" in repr(e) else "HARD_FAIL_OTHER"
        error = repr(e)
        final_loss = float("nan")
        final_acc = float("nan")

    out = {
        "status": status,
        "failure_stage": _classify_failure_stage(error),
        "error": error,
        "final_loss": final_loss,
        "final_acc": final_acc,
    }
    out.update(probe.summary())
    out.update(_proj_grad_stats(model))
    counts = get_triton_meta_bwd_counters(reset=True)
    out["n_fast_bwd"] = int(counts.get("n_fast_bwd", 0))
    out["n_meta_bwd"] = int(counts.get("n_meta_bwd", 0))
    out["n_fallback_bwd"] = int(counts.get("n_fallback_bwd", 0))
    return out


def _abs_diff(a: float | int | str, b: float | int | str) -> float:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return float("nan")
    if not (math.isfinite(af) and math.isfinite(bf)):
        return float("nan")
    return abs(af - bf)


def _classify_failure_stage(error: str) -> str:
    if not error:
        return "none"
    if (
        "fast_input[" in error
        or "fast_grad[" in error
        or "in_triton_fused_meta" in error
        or "context=fast_path" in error
        or "context=output[dq]" in error
        or "context=output[dk]" in error
        or "context=output[dv]" in error
    ):
        return "fast_backward"
    if (
        "in_meta_recompute" in error
        or "fallback_input[" in error
        or "fallback_recompute[" in error
        or "fallback_grad[" in error
    ):
        return "recompute"
    if (
        "support_logits" in error
        or "support_loss" in error
        or "support_grad[" in error
        or "query_logits" in error
        or "query_loss" in error
        or "query_acc" in error
        or "outer_loss" in error
        or "grad[" in error
    ):
        return "outer_model_grad"
    return "other"


def _classify_row_root_cause(
    *,
    delta_m: float,
    delta_l_rel: float,
    row_sum_err_fused: float,
    m_abs_tol: float,
    l_rel_tol: float,
    row_sum_err_tol: float,
) -> str:
    if abs(delta_m) > m_abs_tol:
        return "m_mismatch"
    if delta_l_rel > l_rel_tol:
        return "l_mismatch"
    if row_sum_err_fused > row_sum_err_tol:
        return "reconstruction_precision"
    return "mixed"


def _python_online_softmax_trace(
    scores_ref: torch.Tensor,
    *,
    block_n: int,
) -> tuple[list[dict[str, float | int]], float, float]:
    scores_ref = scores_ref.detach().float()
    final_m_ref = float(scores_ref.max().item())
    final_shifted = scores_ref - final_m_ref
    final_l_ref = float(torch.exp(final_shifted).sum().item())
    running_m = float("-inf")
    running_l = 0.0
    trace_rows: list[dict[str, float | int]] = []
    for block_rank, block_start in enumerate(range(0, int(scores_ref.numel()), block_n)):
        block_end = min(block_start + block_n, int(scores_ref.numel()))
        block_scores = scores_ref[block_start:block_end]
        block_max = float(block_scores.max().item())
        if math.isinf(running_m) and running_m < 0:
            running_m_new = block_max
            alpha = 0.0
        else:
            running_m_new = max(running_m, block_max)
            alpha = math.exp(running_m - running_m_new)
        block_exp = torch.exp(block_scores - running_m_new)
        running_l = running_l * alpha + float(block_exp.sum().item())
        running_m = running_m_new

        prefix_scores = scores_ref[:block_end]
        prefix_m_ref = float(prefix_scores.max().item())
        prefix_l_ref = float(torch.exp(prefix_scores - prefix_m_ref).sum().item())
        delta_m_prefix = running_m - prefix_m_ref
        delta_l_prefix = running_l - prefix_l_ref
        delta_l_prefix_rel = abs(delta_l_prefix) / max(abs(prefix_l_ref), 1e-12)
        trace_rows.append(
            {
                "block_rank": block_rank,
                "block_start": block_start,
                "block_end": block_end,
                "block_max_ref": block_max,
                "running_m_py": running_m,
                "prefix_m_ref": prefix_m_ref,
                "delta_m_prefix": delta_m_prefix,
                "running_l_py": running_l,
                "prefix_l_ref": prefix_l_ref,
                "delta_l_prefix": delta_l_prefix,
                "delta_l_prefix_rel": delta_l_prefix_rel,
                "final_m_ref": final_m_ref,
                "final_l_ref": final_l_ref,
            }
        )
    return trace_rows, running_m, running_l


def _reference_row_debug_entries(
    *,
    artifact: dict[str, object],
    seed: int,
    mode: str,
    outer_step: int,
    fused_backend: str,
    strict_backend: str,
    failure_stage: str,
    failure_reason: str,
    m_abs_tol: float,
    l_rel_tol: float,
    row_sum_err_tol: float,
    block_n: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    q = artifact["q"].float()
    k = artifact["k"].float()
    scale = float(artifact["scale"])
    forward_debug_call_index = int(artifact.get("forward_debug_call_index", -1))
    forward_artifact = (
        get_triton_sdpa_fwd_debug_artifact(call_index=forward_debug_call_index)
        if forward_debug_call_index >= 0
        else {}
    )
    m_write_all = forward_artifact.get("m")
    l_write_all = forward_artifact.get("l")
    m_dtype_write = str(forward_artifact.get("m_dtype_write", ""))
    l_dtype_write = str(forward_artifact.get("l_dtype_write", ""))
    m_dtype_read = str(artifact.get("m_dtype_read", ""))
    l_dtype_read = str(artifact.get("l_dtype_read", ""))
    rows = artifact.get("rows", [])
    out: list[dict[str, float | int | str]] = []
    trace_out: list[dict[str, float | int | str]] = []
    for row in rows:
        b_idx = int(row["batch_idx"])
        h_idx = int(row["head_idx"])
        t_idx = int(row["query_row_idx"])
        scores_ref = torch.matmul(q[b_idx, h_idx, t_idx], k[b_idx, h_idx].transpose(-2, -1)) * scale
        m_ref = float(scores_ref.max().item())
        shifted = scores_ref - m_ref
        exp_shifted = torch.exp(shifted)
        l_ref = float(exp_shifted.sum().item())
        p_ref = exp_shifted / max(l_ref, 1e-12)
        m_fused = float(row["m_fused"])
        l_fused = float(row["l_fused"])
        p_fused_from_saved = torch.exp(scores_ref - m_fused) / max(l_fused, 1e-12)
        row_sum_ref = float(p_ref.sum().item())
        row_sum_err_ref = abs(row_sum_ref - 1.0)
        trace_rows, m_py_blockwise, l_py_blockwise = _python_online_softmax_trace(
            scores_ref,
            block_n=block_n,
        )
        delta_m = m_fused - m_ref
        delta_l = l_fused - l_ref
        delta_l_rel = abs(delta_l) / max(abs(l_ref), 1e-12)
        delta_m_py_blockwise = m_py_blockwise - m_ref
        delta_l_py_blockwise = l_py_blockwise - l_ref
        delta_l_py_blockwise_rel = abs(delta_l_py_blockwise) / max(abs(l_ref), 1e-12)
        forward_oracle_guess = (
            "recurrence_matches_exact"
            if abs(delta_m_py_blockwise) <= 1e-6 and delta_l_py_blockwise_rel <= 1e-6
            else "recurrence_mismatch"
        )
        if isinstance(m_write_all, torch.Tensor):
            m_write = float(m_write_all[b_idx, h_idx, t_idx].item())
        else:
            m_write = float("nan")
        if isinstance(l_write_all, torch.Tensor):
            l_write = float(l_write_all[b_idx, h_idx, t_idx].item())
        else:
            l_write = float("nan")
        delta_m_write = m_write - m_ref if math.isfinite(m_write) else float("nan")
        delta_l_write = l_write - l_ref if math.isfinite(l_write) else float("nan")
        delta_l_write_rel = (
            abs(delta_l_write) / max(abs(l_ref), 1e-12)
            if math.isfinite(delta_l_write)
            else float("nan")
        )
        m_write_read_abs_diff = (
            abs(m_write - m_fused) if math.isfinite(m_write) else float("nan")
        )
        l_write_read_abs_diff = (
            abs(l_write - l_fused) if math.isfinite(l_write) else float("nan")
        )
        row_sum_fused = float(row["row_sum_fused"])
        row_sum_err_fused = float(row["row_sum_err_fused"])
        root_cause_guess = _classify_row_root_cause(
            delta_m=delta_m,
            delta_l_rel=delta_l_rel,
            row_sum_err_fused=row_sum_err_fused,
            m_abs_tol=m_abs_tol,
            l_rel_tol=l_rel_tol,
            row_sum_err_tol=row_sum_err_tol,
        )
        out.append(
            {
                "seed": seed,
                "mode": mode,
                "outer_step": outer_step,
                "fused_backend": fused_backend,
                "strict_backend": strict_backend,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "failure_context": str(artifact.get("context", "")),
                "audit_call_index": int(artifact.get("audit_call_index", 0)),
                "failing_row_rank": int(row["failing_row_rank"]),
                "batch_idx": b_idx,
                "head_idx": h_idx,
                "query_row_idx": t_idx,
                "key_len": int(scores_ref.shape[-1]),
                "scale": scale,
                "score_min_ref": float(scores_ref.min().item()),
                "score_max_ref": float(scores_ref.max().item()),
                "m_fused": m_fused,
                "m_ref": m_ref,
                "delta_m": delta_m,
                "l_fused": l_fused,
                "l_ref": l_ref,
                "delta_l": delta_l,
                "delta_l_rel": delta_l_rel,
                "row_sum_fused": row_sum_fused,
                "row_sum_err_fused": row_sum_err_fused,
                "row_sum_ref": row_sum_ref,
                "row_sum_err_ref": row_sum_err_ref,
                "p_min_fused": float(row["p_min_fused"]),
                "p_max_fused": float(row["p_max_fused"]),
                "p_min_ref": float(p_ref.min().item()),
                "p_max_ref": float(p_ref.max().item()),
                "root_cause_guess": root_cause_guess,
                "n_negative_p_fused": int(float(row.get("n_negative_p_fused", 0))),
                "n_p_gt_1_fused": int(float(row.get("n_p_gt_1_fused", 0))),
                "forward_debug_call_index": forward_debug_call_index,
                "m_dtype_write": m_dtype_write,
                "m_dtype_read": m_dtype_read,
                "l_dtype_write": l_dtype_write,
                "l_dtype_read": l_dtype_read,
                "m_write": m_write,
                "delta_m_write": delta_m_write,
                "l_write": l_write,
                "delta_l_write": delta_l_write,
                "delta_l_write_rel": delta_l_write_rel,
                "m_write_read_abs_diff": m_write_read_abs_diff,
                "l_write_read_abs_diff": l_write_read_abs_diff,
                "m_py_blockwise": m_py_blockwise,
                "delta_m_py_blockwise": delta_m_py_blockwise,
                "l_py_blockwise": l_py_blockwise,
                "delta_l_py_blockwise": delta_l_py_blockwise,
                "delta_l_py_blockwise_rel": delta_l_py_blockwise_rel,
                "forward_oracle_guess": forward_oracle_guess,
            }
        )
        for trace in trace_rows:
            trace_out.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "outer_step": outer_step,
                    "fused_backend": fused_backend,
                    "strict_backend": strict_backend,
                    "failure_stage": failure_stage,
                    "failure_reason": failure_reason,
                    "failing_row_rank": int(row["failing_row_rank"]),
                    "batch_idx": b_idx,
                    "head_idx": h_idx,
                    "query_row_idx": t_idx,
                    **trace,
                }
            )
    return out, trace_out


def _top_row_root_cause(row_debug_rows: list[dict[str, float | int | str]]) -> str:
    if not row_debug_rows:
        return ""
    counts: dict[str, int] = {}
    for row in row_debug_rows:
        key = str(row.get("root_cause_guess", ""))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-backend", type=str, default="triton_fused_meta")
    parser.add_argument("--strict-backend", type=str, default="triton_fused_meta_strict")
    parser.add_argument("--mode", type=str, default="FULL", choices=["FULL", "FO", "FULL_HYBRID"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=60)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--meta-every-n-outer", type=int, default=8)
    parser.add_argument("--meta-last-n-inner", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-signal-positions", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--divergence-tol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for loss/grad stat diffs before declaring divergence.",
    )
    parser.add_argument("--m-abs-tol", type=float, default=1e-5)
    parser.add_argument("--l-rel-tol", type=float, default=1e-4)
    parser.add_argument("--row-sum-err-tol", type=float, default=1e-3)
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help="Exit non-zero when a first divergence is detected.",
    )
    parser.add_argument("--csv-out", type=str, default=None)
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--row-debug-csv-out", type=str, default=None)
    parser.add_argument("--forward-trace-csv-out", type=str, default=None)
    args = parser.parse_args()
    if args.row_debug_csv_out and "THERIA_TRITON_FWD_DEBUG" not in os.environ:
        os.environ["THERIA_TRITON_FWD_DEBUG"] = "1"

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    base_model = Phase10TinyAttentionModel().to(device)
    init_state = deepcopy(base_model.state_dict())

    model_fused = Phase10TinyAttentionModel().to(device)
    model_fused.load_state_dict(deepcopy(init_state))
    set_attention_backend(model_fused, args.fused_backend)
    opt_fused = torch.optim.Adam(model_fused.parameters(), lr=args.outer_lr)
    probe_fused = AttentionProbe(model_fused)
    probe_fused.attach()

    model_strict = Phase10TinyAttentionModel().to(device)
    model_strict.load_state_dict(deepcopy(init_state))
    set_attention_backend(model_strict, args.strict_backend)
    opt_strict = torch.optim.Adam(model_strict.parameters(), lr=args.outer_lr)
    probe_strict = AttentionProbe(model_strict)
    probe_strict.attach()

    rows: list[dict[str, float | int | str]] = []
    first_divergence_step = None
    divergence_reason = "none"
    divergence_row: dict[str, float | int | str] | None = None
    row_debug_rows: list[dict[str, float | int | str]] = []
    forward_trace_rows: list[dict[str, float | int | str]] = []
    reset_triton_sdpa_debug_artifact()
    reset_triton_sdpa_fwd_debug_artifacts()

    for step_idx in range(args.outer_steps):
        tasks = _sample_tasks(
            seed=args.seed,
            step_idx=step_idx,
            n_tasks=args.meta_batch_size,
            seq_len=args.seq_len,
            d_model=model_fused.cfg.d_model,
            num_signal_positions=args.num_signal_positions,
            device=device,
        )

        fused_row = _run_one_step(
            model=model_fused,
            optimizer=opt_fused,
            probe=probe_fused,
            tasks=tasks,
            mode=args.mode,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            meta_every_n_outer=args.meta_every_n_outer,
            meta_last_n_inner=args.meta_last_n_inner,
            outer_step_idx=step_idx,
        )
        strict_row = _run_one_step(
            model=model_strict,
            optimizer=opt_strict,
            probe=probe_strict,
            tasks=tasks,
            mode=args.mode,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            meta_every_n_outer=args.meta_every_n_outer,
            meta_last_n_inner=args.meta_last_n_inner,
            outer_step_idx=step_idx,
        )

        step_row: dict[str, float | int | str] = {"outer_step": step_idx}
        for key, val in fused_row.items():
            step_row[f"fused_{key}"] = val
        for key, val in strict_row.items():
            step_row[f"strict_{key}"] = val
        for key in (
            "final_loss",
            "final_acc",
            "logits_absmax",
            "softmax_max",
            "softmax_min",
            "dq_l2",
            "dk_l2",
            "dv_l2",
            "q_proj_grad_l2",
            "k_proj_grad_l2",
            "v_proj_grad_l2",
        ):
            step_row[f"abs_diff_{key}"] = _abs_diff(
                fused_row.get(key, float("nan")),
                strict_row.get(key, float("nan")),
            )
        rows.append(step_row)

        fused_status = str(fused_row["status"])
        strict_status = str(strict_row["status"])
        print(
            f"step={step_idx:03d} "
            f"fused(status={fused_status},loss={fused_row['final_loss']},acc={fused_row['final_acc']},fallback={fused_row['n_fallback_bwd']}) "
            f"strict(status={strict_status},loss={strict_row['final_loss']},acc={strict_row['final_acc']}) "
            f"diff(loss)={step_row['abs_diff_final_loss']}"
        )

        diverged = False
        reason = ""
        if fused_status != "OK" or strict_status != "OK":
            diverged = True
            reason = f"status fused={fused_status} strict={strict_status}"
        else:
            candidate_diffs = [
                float(step_row["abs_diff_final_loss"]),
                float(step_row["abs_diff_dk_l2"]),
                float(step_row["abs_diff_k_proj_grad_l2"]),
                float(step_row["abs_diff_softmax_max"]),
            ]
            if any(math.isfinite(v) and v > args.divergence_tol for v in candidate_diffs):
                diverged = True
                reason = (
                    "metric_diff "
                    f"(loss={step_row['abs_diff_final_loss']}, "
                    f"dk_l2={step_row['abs_diff_dk_l2']}, "
                    f"k_proj_grad_l2={step_row['abs_diff_k_proj_grad_l2']})"
                )
        if diverged:
            first_divergence_step = step_idx
            divergence_reason = reason
            divergence_row = dict(step_row)
            artifact = get_triton_sdpa_debug_artifact(reset=True)
            fused_failure_stage = str(fused_row.get("failure_stage", ""))
            fused_error = str(fused_row.get("error", ""))
            if (
                artifact
                and str(artifact.get("context", "")) == "fast_path"
                and "row_sum_err" in fused_error
            ):
                row_debug_rows, forward_trace_rows = _reference_row_debug_entries(
                    artifact=artifact,
                    seed=args.seed,
                    mode=args.mode,
                    outer_step=step_idx,
                    fused_backend=args.fused_backend,
                    strict_backend=args.strict_backend,
                    failure_stage=fused_failure_stage,
                    failure_reason=fused_error,
                    m_abs_tol=args.m_abs_tol,
                    l_rel_tol=args.l_rel_tol,
                    row_sum_err_tol=args.row_sum_err_tol,
                    block_n=64,
                )
            print(f"first_divergence_step={step_idx} reason={reason}")
            print(f"fused_error={fused_row.get('error','')}")
            print(f"strict_error={strict_row.get('error','')}")
            break

    probe_fused.detach()
    probe_strict.detach()

    diag_row = divergence_row if divergence_row is not None else (rows[-1] if rows else None)
    diag_source = "first_divergence" if divergence_row is not None else "final_step"

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["outer_step"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_path}")

    if args.row_debug_csv_out and row_debug_rows:
        row_debug_path = Path(args.row_debug_csv_out)
        row_debug_path.parent.mkdir(parents=True, exist_ok=True)
        row_debug_fieldnames = [
            "seed",
            "mode",
            "outer_step",
            "fused_backend",
            "strict_backend",
            "failure_stage",
            "failure_reason",
            "failure_context",
            "audit_call_index",
            "failing_row_rank",
            "batch_idx",
            "head_idx",
            "query_row_idx",
            "key_len",
            "scale",
            "score_min_ref",
            "score_max_ref",
            "m_fused",
            "m_ref",
            "delta_m",
            "l_fused",
            "l_ref",
            "delta_l",
            "delta_l_rel",
            "row_sum_fused",
            "row_sum_err_fused",
            "row_sum_ref",
            "row_sum_err_ref",
            "p_min_fused",
            "p_max_fused",
            "p_min_ref",
            "p_max_ref",
            "root_cause_guess",
            "n_negative_p_fused",
            "n_p_gt_1_fused",
            "forward_debug_call_index",
            "m_dtype_write",
            "m_dtype_read",
            "l_dtype_write",
            "l_dtype_read",
            "m_write",
            "delta_m_write",
            "l_write",
            "delta_l_write",
            "delta_l_write_rel",
            "m_write_read_abs_diff",
            "l_write_read_abs_diff",
            "m_py_blockwise",
            "delta_m_py_blockwise",
            "l_py_blockwise",
            "delta_l_py_blockwise",
            "delta_l_py_blockwise_rel",
            "forward_oracle_guess",
        ]
        with row_debug_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row_debug_fieldnames)
            w.writeheader()
            w.writerows(row_debug_rows)
        print(f"wrote {row_debug_path}")

    if args.forward_trace_csv_out and forward_trace_rows:
        trace_path = Path(args.forward_trace_csv_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fieldnames = [
            "seed",
            "mode",
            "outer_step",
            "fused_backend",
            "strict_backend",
            "failure_stage",
            "failure_reason",
            "failing_row_rank",
            "batch_idx",
            "head_idx",
            "query_row_idx",
            "block_rank",
            "block_start",
            "block_end",
            "block_max_ref",
            "running_m_py",
            "prefix_m_ref",
            "delta_m_prefix",
            "running_l_py",
            "prefix_l_ref",
            "delta_l_prefix",
            "delta_l_prefix_rel",
            "final_m_ref",
            "final_l_ref",
        ]
        with trace_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trace_fieldnames)
            w.writeheader()
            w.writerows(forward_trace_rows)
        print(f"wrote {trace_path}")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        row_debug_root_cause_top1 = _top_row_root_cause(row_debug_rows)
        summary_row: dict[str, float | int | str] = {
            "seed": args.seed,
            "fused_backend": args.fused_backend,
            "strict_backend": args.strict_backend,
            "mode": args.mode,
            "outer_steps_requested": args.outer_steps,
            "outer_steps_completed": len(rows),
            "meta_batch_size": args.meta_batch_size,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "outer_lr": args.outer_lr,
            "meta_every_n_outer": args.meta_every_n_outer,
            "meta_last_n_inner": args.meta_last_n_inner,
            "seq_len": args.seq_len,
            "num_signal_positions": args.num_signal_positions,
            "device": args.device,
            "divergence_tol": args.divergence_tol,
            "divergence_detected": int(first_divergence_step is not None),
            "first_divergence_step": (
                first_divergence_step if first_divergence_step is not None else -1
            ),
            "divergence_reason": divergence_reason,
            "diagnostics_source": diag_source,
            "diagnostics_step": (
                diag_row["outer_step"] if diag_row is not None else -1
            ),
            "row_debug_csv": args.row_debug_csv_out or "",
            "n_row_debug_rows": len(row_debug_rows),
            "row_debug_root_cause_top1": row_debug_root_cause_top1,
            "forward_trace_csv": args.forward_trace_csv_out or "",
            "n_forward_trace_rows": len(forward_trace_rows),
        }
        if diag_row is not None:
            for key, val in diag_row.items():
                if key == "outer_step":
                    continue
                summary_row[str(key)] = val
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            w.writeheader()
            w.writerow(summary_row)
        print(f"wrote {summary_path}")

    if first_divergence_step is None:
        print("no divergence detected within outer_steps")
    elif args.fail_on_divergence:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
