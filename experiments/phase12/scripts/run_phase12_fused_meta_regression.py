#!/usr/bin/env python3
"""
Owned instability regression target for fused meta backward.

This target fixes the known diffusion-like failure configuration and runs the
first-divergence tracer on a fixed seed set. It supports two expectation modes:

  - pre-fix: known bad seeds must fail/diverge for `triton_fused_meta`
  - post-fix: the same seeds must pass cleanly for `triton_fused_meta`

`triton_fused_meta_strict` must remain clean throughout in both modes.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _parse_seeds(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one summary row in {path}, got {len(rows)}")
    return rows[0]


def _observed_fused_state(summary_row: dict[str, str]) -> str:
    divergence = int(summary_row.get("divergence_detected", "0")) != 0
    fused_status = summary_row.get("fused_status", "")
    return "fail" if divergence or fused_status != "OK" else "pass"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-backend", type=str, default="triton_fused_meta")
    parser.add_argument("--strict-backend", type=str, default="triton_fused_meta_strict")
    parser.add_argument("--mode", type=str, default="FULL", choices=["FULL", "FO", "FULL_HYBRID"])
    parser.add_argument("--seeds", type=str, default="1,4")
    parser.add_argument("--expect-fused-meta", type=str, choices=["fail", "pass"], default="fail")
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
    parser.add_argument("--divergence-tol", type=float, default=1e-3)
    parser.add_argument("--cublas-workspace-config", type=str, default=":4096:8")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Leave THERIA_TRITON_META_ENABLE_FALLBACK unchanged. Default keeps fallback disabled.",
    )
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="experiments/phase12/runs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    tag = args.tag or f"phase12_fused_meta_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)

    script_path = repo_root / "experiments/phase12/scripts/run_phase12_fused_meta_instability_diff.py"
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = args.cublas_workspace_config
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if not args.allow_fallback:
        env["THERIA_TRITON_META_ENABLE_FALLBACK"] = "0"

    print("== Phase 12 fused-meta instability regression ==")
    print(f"tag={tag}")
    print(f"expect_fused_meta={args.expect_fused_meta}")
    print(f"seeds={','.join(str(seed) for seed in seeds)}")
    print(f"CUBLAS_WORKSPACE_CONFIG={env['CUBLAS_WORKSPACE_CONFIG']}")
    if not args.allow_fallback:
        print("THERIA_TRITON_META_ENABLE_FALLBACK=0")

    aggregate_rows: list[dict[str, str | int]] = []
    overall_ok = True

    for seed in seeds:
        per_step_csv = out_dir / f"{tag}_seed{seed}.csv"
        per_seed_summary = out_dir / f"{tag}_seed{seed}_summary.csv"
        cmd = [
            sys.executable,
            str(script_path),
            "--fused-backend",
            args.fused_backend,
            "--strict-backend",
            args.strict_backend,
            "--mode",
            args.mode,
            "--seed",
            str(seed),
            "--outer-steps",
            str(args.outer_steps),
            "--meta-batch-size",
            str(args.meta_batch_size),
            "--inner-steps",
            str(args.inner_steps),
            "--inner-lr",
            str(args.inner_lr),
            "--outer-lr",
            str(args.outer_lr),
            "--meta-every-n-outer",
            str(args.meta_every_n_outer),
            "--meta-last-n-inner",
            str(args.meta_last_n_inner),
            "--seq-len",
            str(args.seq_len),
            "--num-signal-positions",
            str(args.num_signal_positions),
            "--device",
            args.device,
            "--divergence-tol",
            str(args.divergence_tol),
            "--fail-on-divergence",
            "--csv-out",
            str(per_step_csv),
            "--summary-out",
            str(per_seed_summary),
        ]
        print()
        print(f"-- seed={seed}")
        print(shlex.join(cmd))
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

        row: dict[str, str | int] = {
            "seed": seed,
            "expect_fused_meta": args.expect_fused_meta,
            "returncode": completed.returncode,
            "per_step_csv": str(per_step_csv),
            "summary_csv": str(per_seed_summary),
        }

        if completed.returncode not in (0, 2):
            row.update(
                {
                    "observed_fused_meta": "error",
                    "strict_status": "UNKNOWN",
                    "fused_status": "UNKNOWN",
                    "strict_failure_stage": "unknown",
                    "fused_failure_stage": "unknown",
                    "divergence_detected": "unknown",
                    "first_divergence_step": "unknown",
                    "divergence_reason": completed.stderr.strip() or completed.stdout.strip(),
                    "regression_pass": 0,
                }
            )
            overall_ok = False
            aggregate_rows.append(row)
            continue

        if not per_seed_summary.exists():
            row.update(
                {
                    "observed_fused_meta": "missing_summary",
                    "strict_status": "UNKNOWN",
                    "fused_status": "UNKNOWN",
                    "strict_failure_stage": "unknown",
                    "fused_failure_stage": "unknown",
                    "divergence_detected": "unknown",
                    "first_divergence_step": "unknown",
                    "divergence_reason": "missing summary output",
                    "regression_pass": 0,
                }
            )
            overall_ok = False
            aggregate_rows.append(row)
            continue

        summary_row = _read_single_csv_row(per_seed_summary)
        observed_fused_meta = _observed_fused_state(summary_row)
        strict_clean = summary_row.get("strict_status", "") == "OK"
        fused_matches_expectation = observed_fused_meta == args.expect_fused_meta
        regression_pass = int(strict_clean and fused_matches_expectation)
        if not regression_pass:
            overall_ok = False
        row.update(
            {
                "observed_fused_meta": observed_fused_meta,
                "divergence_detected": summary_row.get("divergence_detected", ""),
                "first_divergence_step": summary_row.get("first_divergence_step", ""),
                "divergence_reason": summary_row.get("divergence_reason", ""),
                "fused_status": summary_row.get("fused_status", ""),
                "strict_status": summary_row.get("strict_status", ""),
                "fused_failure_stage": summary_row.get("fused_failure_stage", ""),
                "strict_failure_stage": summary_row.get("strict_failure_stage", ""),
                "fused_error": summary_row.get("fused_error", ""),
                "strict_error": summary_row.get("strict_error", ""),
                "regression_pass": regression_pass,
            }
        )
        print(
            "result "
            f"seed={seed} observed={observed_fused_meta} "
            f"strict_status={row['strict_status']} "
            f"fused_stage={row['fused_failure_stage']} "
            f"step={row['first_divergence_step']} "
            f"pass={bool(regression_pass)}"
        )
        aggregate_rows.append(row)

    aggregate_csv = out_dir / f"{tag}_aggregate.csv"
    with aggregate_csv.open("w", newline="") as f:
        fieldnames = list(aggregate_rows[0].keys()) if aggregate_rows else ["seed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print()
    print(f"wrote {aggregate_csv}")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
