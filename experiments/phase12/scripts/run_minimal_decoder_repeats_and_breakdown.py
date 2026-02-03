#!/usr/bin/env python3
"""
Repeat minimal-decoder benchmark runs and report robust summary stats.

This script does two things:
1) Runs N repeats of reference vs triton_full_fused and reports median/p90 speed.
2) Re-runs attention-level breakdown (dq/dk/dv + integration_other).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import statistics
import sys

import torch


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    idx = q * (len(xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    alpha = idx - lo
    return xs[lo] * (1.0 - alpha) + xs[hi] * alpha


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": statistics.fmean(values),
        "median": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
    }


def _load_minimal_decoder_module(repo_root: Path):
    mod_path = repo_root / "examples" / "minimal_decoder_transformer.py"
    spec = importlib.util.spec_from_file_location("minimal_decoder_transformer", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_attention_breakdown_onpath(
    *,
    minimal,
    seed: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    vocab_size: int,
    num_classes: int,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    lr: float,
    symmetric_layout: bool,
    device: torch.device,
) -> dict[str, float]:
    from theria.attention.triton_qk import (
        get_triton_sdpa_bwd_profile,
        reset_triton_sdpa_bwd_profile,
    )

    prior_profile_env = os.environ.get("THERIA_SDPA_PROFILE_BWD", "0")
    os.environ["THERIA_SDPA_PROFILE_BWD"] = "0"
    ref = minimal.benchmark_backend(
        backend="reference",
        device=device,
        seed=seed,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=vocab_size,
        num_classes=num_classes,
        seq_len=seq_len,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup,
        bench_steps=steps,
        symmetric_layout=symmetric_layout,
    )

    reset_triton_sdpa_bwd_profile()
    os.environ["THERIA_SDPA_PROFILE_BWD"] = "1"
    tri = minimal.benchmark_backend(
        backend="triton_full_fused",
        device=device,
        seed=seed,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=vocab_size,
        num_classes=num_classes,
        seq_len=seq_len,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup,
        bench_steps=steps,
        symmetric_layout=symmetric_layout,
    )
    profile = get_triton_sdpa_bwd_profile()
    os.environ["THERIA_SDPA_PROFILE_BWD"] = prior_profile_env

    calls = max(int(profile.get("calls", 0)), 1)
    delta_ms = float(profile.get("delta_ms_sum", 0.0)) / calls
    dq_ms = float(profile.get("dq_ms_sum", 0.0)) / calls
    dk_dv_ms = float(profile.get("dk_dv_ms_sum", 0.0)) / calls
    shared_ms = float(profile.get("shared_ms_sum", 0.0)) / calls
    bwd_total_ms = float(profile.get("total_bwd_ms_sum", 0.0)) / calls
    ref_ms = float(ref["ms_per_step"])
    tri_ms = float(tri["ms_per_step"])
    kernel_sum = delta_ms + dq_ms + dk_dv_ms + shared_ms

    return {
        "reference_step_total_ms": ref_ms,
        "triton_step_total_ms": tri_ms,
        "profile_calls": float(calls),
        "bwd_total_ms": bwd_total_ms,
        "delta_path_ms": delta_ms,
        "dq_ms": dq_ms,
        "dk_dv_ms": dk_dv_ms,
        "shared_ms": shared_ms,
        "kernel_sum_ms": kernel_sum,
        "integration_other_ms": tri_ms - kernel_sum,
        "delta_vs_reference_ms": tri_ms - ref_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7, help="Number of full benchmark repeats")
    parser.add_argument("--breakdown-steps", type=int, default=30, help="Timed iterations for breakdown timings")
    parser.add_argument("--breakdown-warmup", type=int, default=10, help="Warmup iterations for breakdown timings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bench-warmup", type=int, default=20)
    parser.add_argument("--bench-steps", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--benchmark-symmetric-layout", action="store_true")
    parser.add_argument("--bwd-shared", type=int, default=0, choices=[0, 1], help="Set THERIA_SDPA_BWD_SHARED")
    parser.add_argument("--bwd-reuse", type=int, default=1, choices=[0, 1], help="Set THERIA_SDPA_BWD_REUSE")
    parser.add_argument("--csv-out", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This script is CUDA-focused. Use --device cuda.")
    os.environ["THERIA_SDPA_BWD_SHARED"] = str(args.bwd_shared)
    os.environ["THERIA_SDPA_BWD_REUSE"] = str(args.bwd_reuse)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    minimal = _load_minimal_decoder_module(repo_root)

    rows: list[dict[str, float | int | str]] = []
    ref_vals: list[float] = []
    tri_vals: list[float] = []
    speedups: list[float] = []

    for i in range(args.repeats):
        ref = minimal.benchmark_backend(
            backend="reference",
            device=device,
            seed=args.seed + i,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.bench_warmup,
            bench_steps=args.bench_steps,
            symmetric_layout=args.benchmark_symmetric_layout,
        )
        tri = minimal.benchmark_backend(
            backend="triton_full_fused",
            device=device,
            seed=args.seed + i,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.bench_warmup,
            bench_steps=args.bench_steps,
            symmetric_layout=args.benchmark_symmetric_layout,
        )
        ref_ms = float(ref["ms_per_step"])
        tri_ms = float(tri["ms_per_step"])
        sp = ref_ms / max(tri_ms, 1e-12)
        ref_vals.append(ref_ms)
        tri_vals.append(tri_ms)
        speedups.append(sp)

        rows.append(
            {
                "repeat": i,
                "reference_ms_per_step": ref_ms,
                "triton_ms_per_step": tri_ms,
                "speedup_ref_over_triton": sp,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "bench_steps": args.bench_steps,
                "bwd_shared": args.bwd_shared,
                "bwd_reuse": args.bwd_reuse,
                "symmetric_layout": int(args.benchmark_symmetric_layout),
            }
        )
        print(
            f"[repeat {i + 1}/{args.repeats}] "
            f"ref={ref_ms:.4f} tri={tri_ms:.4f} speedup={sp:.3f}x"
        )

    ref_s = _summary(ref_vals)
    tri_s = _summary(tri_vals)
    sp_s = _summary(speedups)
    print("\n=== Repeat Summary (ms/step) ===")
    print(
        f"reference median={ref_s['median']:.4f} p90={ref_s['p90']:.4f} "
        f"(mean={ref_s['mean']:.4f})"
    )
    print(
        f"triton    median={tri_s['median']:.4f} p90={tri_s['p90']:.4f} "
        f"(mean={tri_s['mean']:.4f})"
    )
    print(
        f"speedup(ref/tri) median={sp_s['median']:.3f}x p90={sp_s['p90']:.3f}x "
        f"(mean={sp_s['mean']:.3f}x)"
    )

    breakdown = _run_attention_breakdown_onpath(
        minimal=minimal,
        seed=args.seed + 10_000,
        d_model=args.d_model,
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup=args.breakdown_warmup,
        steps=args.breakdown_steps,
        symmetric_layout=args.benchmark_symmetric_layout,
        device=device,
    )
    print("\n=== Attention Breakdown (on-path ms/call) ===")
    print(f"reference step_total_ms={breakdown['reference_step_total_ms']:.4f}")
    print(f"triton step_total_ms={breakdown['triton_step_total_ms']:.4f}")
    print(
        "  delta_ms={dlt:.4f} dq_ms={dq:.4f} dk_dv_ms={dkdv:.4f} shared_ms={sh:.4f}  (sum={sm:.4f})".format(
            dlt=breakdown["delta_path_ms"],
            dq=breakdown["dq_ms"],
            dkdv=breakdown["dk_dv_ms"],
            sh=breakdown["shared_ms"],
            sm=breakdown["kernel_sum_ms"],
        )
    )
    print(f"  bwd_total_ms={breakdown['bwd_total_ms']:.4f} profile_calls={breakdown['profile_calls']:.0f}")
    print(f"  integration_other_ms={breakdown['integration_other_ms']:.4f}")
    print(f"delta_vs_reference_ms={breakdown['delta_vs_reference_ms']:.4f}")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            fieldnames = list(rows[0].keys()) + [
                "summary_ref_median_ms",
                "summary_ref_p90_ms",
                "summary_tri_median_ms",
                "summary_tri_p90_ms",
                "summary_speedup_median",
                "summary_speedup_p90",
                "breakdown_reference_step_total_ms",
                "breakdown_triton_step_total_ms",
                "breakdown_profile_calls",
                "breakdown_bwd_total_ms",
                "breakdown_delta_path_ms",
                "breakdown_dq_ms",
                "breakdown_dk_dv_ms",
                "breakdown_shared_ms",
                "breakdown_kernel_sum_ms",
                "breakdown_integration_other_ms",
                "breakdown_delta_vs_reference_ms",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                enriched = dict(row)
                enriched.update(
                    {
                        "summary_ref_median_ms": ref_s["median"],
                        "summary_ref_p90_ms": ref_s["p90"],
                        "summary_tri_median_ms": tri_s["median"],
                        "summary_tri_p90_ms": tri_s["p90"],
                        "summary_speedup_median": sp_s["median"],
                        "summary_speedup_p90": sp_s["p90"],
                        "breakdown_reference_step_total_ms": breakdown["reference_step_total_ms"],
                        "breakdown_triton_step_total_ms": breakdown["triton_step_total_ms"],
                        "breakdown_profile_calls": breakdown["profile_calls"],
                        "breakdown_bwd_total_ms": breakdown["bwd_total_ms"],
                        "breakdown_delta_path_ms": breakdown["delta_path_ms"],
                        "breakdown_dq_ms": breakdown["dq_ms"],
                        "breakdown_dk_dv_ms": breakdown["dk_dv_ms"],
                        "breakdown_shared_ms": breakdown["shared_ms"],
                        "breakdown_kernel_sum_ms": breakdown["kernel_sum_ms"],
                        "breakdown_integration_other_ms": breakdown["integration_other_ms"],
                        "breakdown_delta_vs_reference_ms": breakdown["delta_vs_reference_ms"],
                    }
                )
                writer.writerow(enriched)
        print(f"\nwrote {out_path}")

    if breakdown["integration_other_ms"] > breakdown["kernel_sum_ms"] * 0.5:
        print(
            "\n[next ROI] integration_other is still large; prioritize "
            "buffer reuse/preallocation in autograd path, then CUDA Graph capture."
        )


if __name__ == "__main__":
    main()
