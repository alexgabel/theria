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
from typing import Callable

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


def _time_cuda_ms(fn: Callable[[], None], steps: int, warmup: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    out: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        out.append(float(start.elapsed_time(end)))
    return out


def _load_minimal_decoder_module(repo_root: Path):
    mod_path = repo_root / "examples" / "minimal_decoder_transformer.py"
    spec = importlib.util.spec_from_file_location("minimal_decoder_transformer", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_attention_breakdown(
    *,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    d_per_head: int,
    dtype: torch.dtype,
    warmup: int,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    from theria.attention.custom import sdpa_custom
    from theria.attention.triton_qk import triton_sdpa_fused
    from theria.attention.triton_sdpa_backward import sdpa_bwd_dk, sdpa_bwd_dq, sdpa_bwd_dv

    q = torch.randn(batch_size, n_heads, seq_len, d_per_head, device=device, dtype=dtype).contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()
    do = torch.randn_like(q).contiguous()
    scale = 1.0 / (d_per_head**0.5)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    q_tri = q.detach().clone().requires_grad_(True)
    k_tri = k.detach().clone().requires_grad_(True)
    v_tri = v.detach().clone().requires_grad_(True)

    def _step_ref() -> None:
        q_ref.grad = None
        k_ref.grad = None
        v_ref.grad = None
        out = sdpa_custom(q_ref, k_ref, v_ref, backend="reference")
        loss = (out * do).sum()
        loss.backward()

    def _step_tri() -> None:
        q_tri.grad = None
        k_tri.grad = None
        v_tri.grad = None
        out = sdpa_custom(q_tri, k_tri, v_tri, backend="triton_full_fused")
        loss = (out * do).sum()
        loss.backward()

    ref_step_ms = _time_cuda_ms(_step_ref, steps=steps, warmup=warmup, device=device)
    tri_step_ms = _time_cuda_ms(_step_tri, steps=steps, warmup=warmup, device=device)

    with torch.no_grad():
        _, m, l = triton_sdpa_fused(q, k, v, return_stats=True)

    def _dq_once() -> None:
        _ = sdpa_bwd_dq(q, k, v, do, m, l, scale)

    def _dk_once() -> None:
        _ = sdpa_bwd_dk(q, k, v, do, m, l, scale)

    def _dv_once() -> None:
        _ = sdpa_bwd_dv(q, k, do, m, l, scale)

    dq_ms = _time_cuda_ms(_dq_once, steps=steps, warmup=warmup, device=device)
    dk_ms = _time_cuda_ms(_dk_once, steps=steps, warmup=warmup, device=device)
    dv_ms = _time_cuda_ms(_dv_once, steps=steps, warmup=warmup, device=device)

    ref_med = _quantile(ref_step_ms, 0.5)
    tri_med = _quantile(tri_step_ms, 0.5)
    dq_med = _quantile(dq_ms, 0.5)
    dk_med = _quantile(dk_ms, 0.5)
    dv_med = _quantile(dv_ms, 0.5)
    kernel_sum = dq_med + dk_med + dv_med

    return {
        "reference_step_total_ms": ref_med,
        "triton_step_total_ms": tri_med,
        "dq_ms": dq_med,
        "dk_ms": dk_med,
        "dv_ms": dv_med,
        "kernel_sum_ms": kernel_sum,
        "integration_other_ms": tri_med - kernel_sum,
        "delta_vs_reference_ms": tri_med - ref_med,
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
    parser.add_argument("--csv-out", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This script is CUDA-focused. Use --device cuda.")
    os.environ["THERIA_SDPA_BWD_SHARED"] = str(args.bwd_shared)

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

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    breakdown = _run_attention_breakdown(
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        d_per_head=args.d_model // args.n_heads,
        dtype=dtype_map[args.dtype],
        warmup=args.breakdown_warmup,
        steps=args.breakdown_steps,
        device=device,
    )
    print("\n=== Attention Breakdown (median ms) ===")
    print(f"reference step_total_ms={breakdown['reference_step_total_ms']:.4f}")
    print(f"triton step_total_ms={breakdown['triton_step_total_ms']:.4f}")
    print(
        "  dq_ms={dq:.4f} dk_ms={dk:.4f} dv_ms={dv:.4f}  (sum={sm:.4f})".format(
            dq=breakdown["dq_ms"],
            dk=breakdown["dk_ms"],
            dv=breakdown["dv_ms"],
            sm=breakdown["kernel_sum_ms"],
        )
    )
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
                "breakdown_dq_ms",
                "breakdown_dk_ms",
                "breakdown_dv_ms",
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
                        "breakdown_dq_ms": breakdown["dq_ms"],
                        "breakdown_dk_ms": breakdown["dk_ms"],
                        "breakdown_dv_ms": breakdown["dv_ms"],
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
