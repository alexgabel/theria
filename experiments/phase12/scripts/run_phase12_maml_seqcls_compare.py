#!/usr/bin/env python3
"""
Phase 12 MAML seqcls comparison runner.

Runs a grid over (backend, mode, inner_steps, seed) and writes a single CSV with
per-run metrics from run_phase12_behavior.run_behavior. Optional summary CSV
aggregates means/stds per (backend, mode, inner_steps).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

# Allow direct script execution from repo root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase12.scripts.run_phase12_behavior import run_behavior

EXPERIMENTAL_BACKENDS = {"triton_fused_meta"}


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _mean(vals: list[float]) -> float:
    finite_vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.mean(finite_vals)) if finite_vals else float("nan")


def _std(vals: list[float]) -> float:
    finite_vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.pstdev(finite_vals)) if len(finite_vals) > 1 else 0.0


def _resolve_mode_lrs(mode: str, args: argparse.Namespace) -> tuple[float, float]:
    mode_key = mode.upper()
    mode_overrides = {
        "FULL": (args.full_inner_lr, args.full_outer_lr),
        "FO": (args.fo_inner_lr, args.fo_outer_lr),
        "FO_STRICT": (args.fo_strict_inner_lr, args.fo_strict_outer_lr),
        "FULL_FROZEN": (args.full_frozen_inner_lr, args.full_frozen_outer_lr),
        "FULL_HYBRID": (args.full_hybrid_inner_lr, args.full_hybrid_outer_lr),
    }
    inner_override, outer_override = mode_overrides.get(mode_key, (None, None))
    inner_lr = float(inner_override) if inner_override is not None else float(args.inner_lr)
    outer_lr = float(outer_override) if outer_override is not None else float(args.outer_lr)
    return inner_lr, outer_lr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        type=str,
        default="triton_fused_meta_strict",
        help="Comma-separated backends.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="FULL,FULL_HYBRID",
        help="Comma-separated modes (FULL,FO,FO_STRICT,FULL_FROZEN,FULL_HYBRID).",
    )
    parser.add_argument(
        "--inner-steps",
        type=str,
        default="2,5",
        help="Comma-separated inner step counts.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1",
        help="Comma-separated seeds.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=200)
    parser.add_argument(
        "--wall-clock-budget-s",
        type=float,
        default=None,
        help=(
            "If set, estimate per-step time from a pilot run for each "
            "(backend,mode,inner_steps) and choose outer_steps to match this budget."
        ),
    )
    parser.add_argument(
        "--pilot-steps",
        type=int,
        default=30,
        help="Pilot outer steps used to estimate per-step time for wall-clock mode.",
    )
    parser.add_argument(
        "--pilot-seed",
        type=int,
        default=None,
        help="Seed used for pilot timing. Defaults to first entry from --seeds.",
    )
    parser.add_argument(
        "--min-outer-steps",
        type=int,
        default=10,
        help="Lower bound for computed outer_steps in wall-clock mode.",
    )
    parser.add_argument(
        "--max-outer-steps",
        type=int,
        default=2000,
        help="Upper bound for computed outer_steps in wall-clock mode.",
    )
    parser.add_argument("--meta-batch-size", type=int, default=16)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--full-inner-lr", type=float, default=None)
    parser.add_argument("--full-outer-lr", type=float, default=None)
    parser.add_argument("--fo-inner-lr", type=float, default=None)
    parser.add_argument("--fo-outer-lr", type=float, default=None)
    parser.add_argument("--fo-strict-inner-lr", type=float, default=None)
    parser.add_argument("--fo-strict-outer-lr", type=float, default=None)
    parser.add_argument("--full-frozen-inner-lr", type=float, default=None)
    parser.add_argument("--full-frozen-outer-lr", type=float, default=None)
    parser.add_argument("--full-hybrid-inner-lr", type=float, default=None)
    parser.add_argument("--full-hybrid-outer-lr", type=float, default=None)
    parser.add_argument(
        "--meta-every-n-outer",
        type=int,
        default=8,
        help="FULL_HYBRID only: run FULL step every N outer steps.",
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
    parser.add_argument(
        "--profile-meta-bwd",
        action="store_true",
        help="Enable Triton meta backward timing counters during runs.",
    )
    parser.add_argument(
        "--no-fail-on-nonfinite",
        action="store_true",
        help="Pass through to run_behavior (debug only).",
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
            "Enable the static-shape CUDA-graph path for CUDA "
            "triton_fused_meta_strict FULL_HYBRID runs."
        ),
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-signal-positions", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument(
        "--csv-out",
        type=str,
        required=True,
        help="Output CSV path for per-run rows.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Optional summary CSV path (mean/std per backend/mode/inner_steps).",
    )
    args = parser.parse_args()

    backends = _parse_str_list(args.backends)
    modes = _parse_str_list(args.modes)
    inner_steps_list = _parse_int_list(args.inner_steps)
    seeds = _parse_int_list(args.seeds)
    device = torch.device(args.device)
    blocked = [b for b in backends if b in EXPERIMENTAL_BACKENDS]
    if blocked and not args.allow_experimental_backends:
        blocked_str = ",".join(sorted(set(blocked)))
        raise ValueError(
            f"Experimental backend(s) requested: {blocked_str}. "
            "Pass --allow-experimental-backends to opt in."
        )

    fieldnames = [
        "comparison_protocol",
        "backend",
        "mode",
        "inner_steps",
        "seed",
        "repeat",
        "meta_batch",
        "outer_lr",
        "outer_steps",
        "inner_lr",
        "seq_len",
        "num_signal_positions",
        "wall_clock_budget_s",
        "dtype",
        "compute_dtype",
        "autocast",
        "meta_every_n_outer",
        "meta_last_n_inner",
        "n_hybrid_meta_steps",
        "n_hybrid_fo_steps",
        "final_loss",
        "final_acc",
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
    ]

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    rows: list[dict[str, float | int | str]] = []
    outer_steps_map: dict[tuple[str, str, int], int] = {}

    total = (
        len(backends)
        * len(modes)
        * len(inner_steps_list)
        * len(seeds)
        * max(args.repeats, 1)
    )
    idx = 0

    if args.wall_clock_budget_s is not None:
        pilot_seed = seeds[0] if args.pilot_seed is None else args.pilot_seed
        for backend in backends:
            for mode in modes:
                pilot_inner_lr, pilot_outer_lr = _resolve_mode_lrs(mode, args)
                for inner_steps in inner_steps_list:
                    key = (backend, mode, inner_steps)
                    try:
                        pilot = run_behavior(
                            attention_backend=backend,
                            mode=mode,  # type: ignore[arg-type]
                            seed=pilot_seed,
                            outer_steps=args.pilot_steps,
                            meta_batch_size=args.meta_batch_size,
                            inner_steps=inner_steps,
                            inner_lr=pilot_inner_lr,
                            outer_lr=pilot_outer_lr,
                            meta_every_n_outer=args.meta_every_n_outer,
                            meta_last_n_inner=args.meta_last_n_inner,
                            profile_meta_bwd=args.profile_meta_bwd,
                            allow_experimental_backends=args.allow_experimental_backends,
                            seq_len=args.seq_len,
                            num_signal_positions=args.num_signal_positions,
                            device=device,
                            autocast_enabled=args.autocast,
                            cuda_graph_static=args.cuda_graph_static,
                        )
                        step_s = float(pilot["mean_outer_step_time_s"])
                        est_steps = int(args.wall_clock_budget_s / max(step_s, 1e-9))
                        est_steps = max(args.min_outer_steps, est_steps)
                        est_steps = min(args.max_outer_steps, est_steps)
                        outer_steps_map[key] = est_steps
                        print(
                            "pilot "
                            f"backend={backend} mode={mode} k={inner_steps} "
                            f"inner_lr={pilot_inner_lr} outer_lr={pilot_outer_lr} "
                            f"step_s={step_s:.6f} -> outer_steps={est_steps}"
                        )
                    except Exception as e:
                        outer_steps_map[key] = args.outer_steps
                        print(
                            "pilot_fail "
                            f"backend={backend} mode={mode} k={inner_steps} "
                            f"fallback_outer_steps={args.outer_steps} err={e!r}"
                        )
    else:
        for backend in backends:
            for mode in modes:
                for inner_steps in inner_steps_list:
                    outer_steps_map[(backend, mode, inner_steps)] = args.outer_steps

    for backend in backends:
        for mode in modes:
            run_inner_lr, run_outer_lr = _resolve_mode_lrs(mode, args)
            for inner_steps in inner_steps_list:
                run_outer_steps = outer_steps_map[(backend, mode, inner_steps)]
                for seed in seeds:
                    for repeat in range(args.repeats):
                        idx += 1
                        try:
                            row = run_behavior(
                                attention_backend=backend,
                                mode=mode,  # type: ignore[arg-type]
                                seed=seed,
                                outer_steps=run_outer_steps,
                                meta_batch_size=args.meta_batch_size,
                                inner_steps=inner_steps,
                                inner_lr=run_inner_lr,
                                outer_lr=run_outer_lr,
                                meta_every_n_outer=args.meta_every_n_outer,
                                meta_last_n_inner=args.meta_last_n_inner,
                                profile_meta_bwd=args.profile_meta_bwd,
                                fail_on_nonfinite=not args.no_fail_on_nonfinite,
                                allow_experimental_backends=args.allow_experimental_backends,
                                seq_len=args.seq_len,
                                num_signal_positions=args.num_signal_positions,
                                device=device,
                                autocast_enabled=args.autocast,
                                cuda_graph_static=args.cuda_graph_static,
                            )
                            status, error = "OK", ""
                            for key in ("final_loss", "final_acc"):
                                try:
                                    v = float(row.get(key, float("nan")))
                                except (TypeError, ValueError):
                                    v = float("nan")
                                if not math.isfinite(v):
                                    status = "HARD_FAIL_NONFINITE"
                                    error = f"non_finite_{key}"
                                    break
                        except Exception as e:
                            row = {
                                "backend": backend,
                                "mode": mode,
                                "seed": seed,
                                "outer_steps": run_outer_steps,
                                "meta_batch": args.meta_batch_size,
                                "inner_steps": inner_steps,
                                "inner_lr": run_inner_lr,
                                "outer_lr": run_outer_lr,
                                "seq_len": args.seq_len,
                                "num_signal_positions": args.num_signal_positions,
                                "dtype": "NA",
                                "compute_dtype": "NA",
                                "autocast": int(bool(args.autocast)),
                                "meta_every_n_outer": int(
                                    args.meta_every_n_outer if mode == "FULL_HYBRID" else 0
                                ),
                                "meta_last_n_inner": int(
                                    args.meta_last_n_inner
                                    if mode in {"FULL", "FULL_HYBRID"}
                                    else 0
                                ),
                                "n_hybrid_meta_steps": 0,
                                "n_hybrid_fo_steps": 0,
                                "final_loss": float("nan"),
                                "final_acc": float("nan"),
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

                        row = {
                            **row,
                            "comparison_protocol": (
                                "equal_time"
                                if args.wall_clock_budget_s is not None
                                else "equal_step"
                            ),
                            "backend": backend,
                            "mode": mode,
                            "inner_steps": inner_steps,
                            "seed": seed,
                            "repeat": repeat,
                            "meta_batch": args.meta_batch_size,
                            "outer_lr": run_outer_lr,
                            "outer_steps": run_outer_steps,
                            "inner_lr": run_inner_lr,
                            "meta_every_n_outer": (
                                int(args.meta_every_n_outer) if mode == "FULL_HYBRID" else 0
                            ),
                            "meta_last_n_inner": (
                                int(args.meta_last_n_inner)
                                if mode in {"FULL", "FULL_HYBRID"}
                                else 0
                            ),
                            "seq_len": args.seq_len,
                            "num_signal_positions": args.num_signal_positions,
                            "wall_clock_budget_s": (
                                ""
                                if args.wall_clock_budget_s is None
                                else float(args.wall_clock_budget_s)
                            ),
                            "status": status,
                            "error": error,
                        }
                        rows.append(row)
                        print(
                            f"[{idx}/{total}] backend={backend} mode={mode} "
                            f"k={inner_steps} seed={seed} repeat={repeat} "
                            f"steps={run_outer_steps} inner_lr={run_inner_lr} "
                            f"outer_lr={run_outer_lr} status={status} "
                            f"acc={row.get('final_acc','NA')}"
                        )

    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[str, str, int], list[dict[str, float | int | str]]] = defaultdict(list)
        for row in rows:
            key = (str(row["backend"]), str(row["mode"]), int(row["inner_steps"]))
            groups[key].append(row)

        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "backend",
                    "mode",
                    "inner_steps",
                    "n_runs",
                    "comparison_protocol",
                    "inner_lr_mean",
                    "outer_lr_mean",
                    "meta_last_n_inner_mean",
                    "outer_steps_mean",
                    "final_acc_mean",
                    "final_acc_std",
                    "final_loss_mean",
                    "final_loss_std",
                    "mean_outer_step_time_s_mean",
                    "mean_outer_step_time_s_std",
                    "wall_time_total_s_mean",
                    "wall_time_total_s_std",
                    "meta_loss_time_per_outer_step_s_mean",
                    "meta_loss_time_per_outer_step_s_std",
                    "outer_backward_time_per_outer_step_s_mean",
                    "outer_backward_time_per_outer_step_s_std",
                    "optimizer_step_time_per_outer_step_s_mean",
                    "optimizer_step_time_per_outer_step_s_std",
                    "n_fast_bwd_mean",
                    "n_fast_bwd_std",
                    "n_meta_bwd_mean",
                    "n_meta_bwd_std",
                    "n_fallback_bwd_mean",
                    "n_fallback_bwd_std",
                    "fallback_incident_mean",
                    "fallback_incident_std",
                    "fallback_bwd_per_outer_step_mean",
                    "fallback_bwd_per_outer_step_std",
                    "fast_bwd_time_per_outer_step_s_mean",
                    "fast_bwd_time_per_outer_step_s_std",
                    "meta_bwd_time_per_outer_step_s_mean",
                    "meta_bwd_time_per_outer_step_s_std",
                    "meta_recompute_time_per_outer_step_s_mean",
                    "meta_recompute_time_per_outer_step_s_std",
                    "fallback_bwd_time_per_outer_step_s_mean",
                    "fallback_bwd_time_per_outer_step_s_std",
                    "support_forward_time_per_outer_step_s_mean",
                    "support_forward_time_per_outer_step_s_std",
                    "support_grad_time_per_outer_step_s_mean",
                    "support_grad_time_per_outer_step_s_std",
                    "support_grad_create_graph_time_per_outer_step_s_mean",
                    "support_grad_create_graph_time_per_outer_step_s_std",
                    "support_grad_no_graph_time_per_outer_step_s_mean",
                    "support_grad_no_graph_time_per_outer_step_s_std",
                    "param_update_time_per_outer_step_s_mean",
                    "param_update_time_per_outer_step_s_std",
                    "query_forward_time_per_outer_step_s_mean",
                    "query_forward_time_per_outer_step_s_std",
                    "hybrid_meta_meta_loss_time_per_meta_step_s_mean",
                    "hybrid_meta_meta_loss_time_per_meta_step_s_std",
                    "hybrid_fo_meta_loss_time_per_fo_step_s_mean",
                    "hybrid_fo_meta_loss_time_per_fo_step_s_std",
                    "hybrid_meta_outer_backward_time_per_meta_step_s_mean",
                    "hybrid_meta_outer_backward_time_per_meta_step_s_std",
                    "hybrid_fo_outer_backward_time_per_fo_step_s_mean",
                    "hybrid_fo_outer_backward_time_per_fo_step_s_std",
                    "hybrid_meta_optimizer_step_time_per_meta_step_s_mean",
                    "hybrid_meta_optimizer_step_time_per_meta_step_s_std",
                    "hybrid_fo_optimizer_step_time_per_fo_step_s_mean",
                    "hybrid_fo_optimizer_step_time_per_fo_step_s_std",
                    "n_hybrid_meta_steps_mean",
                    "n_hybrid_meta_steps_std",
                    "n_hybrid_fo_steps_mean",
                    "n_hybrid_fo_steps_std",
                ]
            )
            for (backend, mode, inner_steps), items in sorted(groups.items()):
                accs = [float(x["final_acc"]) for x in items if str(x.get("status")) == "OK"]
                losses = [float(x["final_loss"]) for x in items if str(x.get("status")) == "OK"]
                step_times = [
                    float(x["mean_outer_step_time_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                wall_times = [
                    float(x["wall_time_total_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                meta_loss_time_step_vals = [
                    float(x["meta_loss_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                outer_backward_time_step_vals = [
                    float(x["outer_backward_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                optimizer_step_time_step_vals = [
                    float(x["optimizer_step_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                n_fast_bwd_vals = [
                    float(x["n_fast_bwd"]) for x in items if str(x.get("status")) == "OK"
                ]
                n_meta_bwd_vals = [
                    float(x["n_meta_bwd"]) for x in items if str(x.get("status")) == "OK"
                ]
                n_fallback_bwd_vals = [
                    float(x["n_fallback_bwd"]) for x in items if str(x.get("status")) == "OK"
                ]
                fallback_incident_vals = [
                    float(x.get("fallback_incident", 0))
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                fallback_rate_vals = [
                    float(x.get("fallback_bwd_per_outer_step", float("nan")))
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                outer_steps_vals = [
                    float(x["outer_steps"]) for x in items if str(x.get("status")) == "OK"
                ]
                fast_bwd_time_step_vals = [
                    float(x["fast_bwd_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                meta_bwd_time_step_vals = [
                    float(x["meta_bwd_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                meta_recompute_time_step_vals = [
                    float(x["meta_recompute_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                fallback_time_step_vals = [
                    float(x["fallback_bwd_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                support_forward_time_step_vals = [
                    float(x["support_forward_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                support_grad_time_step_vals = [
                    float(x["support_grad_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                support_grad_create_graph_time_step_vals = [
                    float(x["support_grad_create_graph_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                support_grad_no_graph_time_step_vals = [
                    float(x["support_grad_no_graph_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                param_update_time_step_vals = [
                    float(x["param_update_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                query_forward_time_step_vals = [
                    float(x["query_forward_time_per_outer_step_s"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                hybrid_meta_step_vals = [
                    float(x["n_hybrid_meta_steps"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                hybrid_fo_step_vals = [
                    float(x["n_hybrid_fo_steps"])
                    for x in items
                    if str(x.get("status")) == "OK"
                ]
                w.writerow(
                    [
                        backend,
                        mode,
                        inner_steps,
                        len(items),
                        str(items[0].get("comparison_protocol", "")),
                        _mean([float(x["inner_lr"]) for x in items if str(x.get("status")) == "OK"]),
                        _mean([float(x["outer_lr"]) for x in items if str(x.get("status")) == "OK"]),
                        _mean(
                            [
                                float(x.get("meta_last_n_inner", 0))
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _mean(outer_steps_vals),
                        _mean(accs),
                        _std(accs),
                        _mean(losses),
                        _std(losses),
                        _mean(step_times),
                        _std(step_times),
                        _mean(wall_times),
                        _std(wall_times),
                        _mean(meta_loss_time_step_vals),
                        _std(meta_loss_time_step_vals),
                        _mean(outer_backward_time_step_vals),
                        _std(outer_backward_time_step_vals),
                        _mean(optimizer_step_time_step_vals),
                        _std(optimizer_step_time_step_vals),
                        _mean(n_fast_bwd_vals),
                        _std(n_fast_bwd_vals),
                        _mean(n_meta_bwd_vals),
                        _std(n_meta_bwd_vals),
                        _mean(n_fallback_bwd_vals),
                        _std(n_fallback_bwd_vals),
                        _mean(fallback_incident_vals),
                        _std(fallback_incident_vals),
                        _mean(fallback_rate_vals),
                        _std(fallback_rate_vals),
                        _mean(fast_bwd_time_step_vals),
                        _std(fast_bwd_time_step_vals),
                        _mean(meta_bwd_time_step_vals),
                        _std(meta_bwd_time_step_vals),
                        _mean(meta_recompute_time_step_vals),
                        _std(meta_recompute_time_step_vals),
                        _mean(fallback_time_step_vals),
                        _std(fallback_time_step_vals),
                        _mean(support_forward_time_step_vals),
                        _std(support_forward_time_step_vals),
                        _mean(support_grad_time_step_vals),
                        _std(support_grad_time_step_vals),
                        _mean(support_grad_create_graph_time_step_vals),
                        _std(support_grad_create_graph_time_step_vals),
                        _mean(support_grad_no_graph_time_step_vals),
                        _std(support_grad_no_graph_time_step_vals),
                        _mean(param_update_time_step_vals),
                        _std(param_update_time_step_vals),
                        _mean(query_forward_time_step_vals),
                        _std(query_forward_time_step_vals),
                        _mean(
                            [
                                float(x["hybrid_meta_meta_loss_time_per_meta_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _std(
                            [
                                float(x["hybrid_meta_meta_loss_time_per_meta_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _mean(
                            [
                                float(x["hybrid_fo_meta_loss_time_per_fo_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _std(
                            [
                                float(x["hybrid_fo_meta_loss_time_per_fo_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _mean(
                            [
                                float(x["hybrid_meta_outer_backward_time_per_meta_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _std(
                            [
                                float(x["hybrid_meta_outer_backward_time_per_meta_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _mean(
                            [
                                float(x["hybrid_fo_outer_backward_time_per_fo_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _std(
                            [
                                float(x["hybrid_fo_outer_backward_time_per_fo_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _mean(
                            [
                                float(x["hybrid_meta_optimizer_step_time_per_meta_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _std(
                            [
                                float(x["hybrid_meta_optimizer_step_time_per_meta_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _mean(
                            [
                                float(x["hybrid_fo_optimizer_step_time_per_fo_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _std(
                            [
                                float(x["hybrid_fo_optimizer_step_time_per_fo_step_s"])
                                for x in items
                                if str(x.get("status")) == "OK"
                            ]
                        ),
                        _mean(hybrid_meta_step_vals),
                        _std(hybrid_meta_step_vals),
                        _mean(hybrid_fo_step_vals),
                        _std(hybrid_fo_step_vals),
                    ]
                )


if __name__ == "__main__":
    main()
