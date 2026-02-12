#!/usr/bin/env python3
"""
Run meta-contract checks for both triton_fused_meta and triton_fused_meta_strict,
then print a side-by-side delta table.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECK_SCRIPT = REPO_ROOT / "experiments/phase12/scripts/run_phase12_meta_contract_checks.py"


def _run_check(
    *,
    reference_backend: str,
    candidate_backend: str,
    profiles: str,
    inner_steps: str,
    seeds: str,
    inner_lr: float,
    device: str,
    csv_out: Path,
    summary_out: Path,
    gate_csv_out: Path,
    fail_on_gate: bool,
    min_cosine_full_mean: float,
    min_cosine_fo_mean: float,
    max_rel_diff_abs_error: float,
    require_trend_sign_match: bool,
) -> None:
    cmd = [
        sys.executable,
        str(CHECK_SCRIPT),
        "--reference-backend",
        reference_backend,
        "--candidate-backend",
        candidate_backend,
        "--profiles",
        profiles,
        "--inner-steps",
        inner_steps,
        "--seeds",
        seeds,
        "--inner-lr",
        str(inner_lr),
        "--device",
        device,
        "--csv-out",
        str(csv_out),
        "--summary-out",
        str(summary_out),
        "--gate-csv-out",
        str(gate_csv_out),
        "--min-cosine-full-mean",
        str(min_cosine_full_mean),
        "--min-cosine-fo-mean",
        str(min_cosine_fo_mean),
        "--max-rel-diff-abs-error",
        str(max_rel_diff_abs_error),
    ]
    if fail_on_gate:
        cmd.append("--fail-on-gate")
    if require_trend_sign_match:
        cmd.append("--require-trend-sign-match")
    print(f"\n== Running {candidate_backend} ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _load_summary(path: Path) -> dict[tuple[str, int], dict[str, float | int | str]]:
    rows: dict[tuple[str, int], dict[str, float | int | str]] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (str(r["profile"]), int(r["inner_steps"]))
            rows[key] = {
                **r,
                "inner_steps": int(r["inner_steps"]),
                "cosine_full_mean": float(r["cosine_full_mean"]),
                "rel_diff_ref_mean": float(r["rel_diff_ref_mean"]),
                "rel_diff_candidate_mean": float(r["rel_diff_candidate_mean"]),
            }
    return rows


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-backend", type=str, default="reference")
    parser.add_argument("--meta-backend", type=str, default="triton_fused_meta")
    parser.add_argument("--strict-backend", type=str, default="triton_fused_meta_strict")
    parser.add_argument("--profiles", type=str, default="seqcls_default,diffusion_proxy")
    parser.add_argument("--inner-steps", type=str, default="2,5,10")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="experiments/phase12/runs")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--delta-out", type=str, default=None)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--min-cosine-full-mean", type=float, default=0.999)
    parser.add_argument("--min-cosine-fo-mean", type=float, default=0.999)
    parser.add_argument("--max-rel-diff-abs-error", type=float, default=5e-4)
    parser.add_argument("--require-trend-sign-match", action="store_true")
    args = parser.parse_args()

    if args.device != "cuda" and (
        "triton" in args.meta_backend or "triton" in args.strict_backend
    ):
        raise ValueError(
            "Triton candidate backends require --device cuda for this dual compare runner."
        )

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = out_dir / f"phase12_meta_contract_{tag}_{args.meta_backend}.csv"
    meta_summary = out_dir / f"phase12_meta_contract_{tag}_{args.meta_backend}_summary.csv"
    meta_gate = out_dir / f"phase12_meta_contract_{tag}_{args.meta_backend}_gate.csv"
    strict_csv = out_dir / f"phase12_meta_contract_{tag}_{args.strict_backend}.csv"
    strict_summary = out_dir / f"phase12_meta_contract_{tag}_{args.strict_backend}_summary.csv"
    strict_gate = out_dir / f"phase12_meta_contract_{tag}_{args.strict_backend}_gate.csv"

    _run_check(
        reference_backend=args.reference_backend,
        candidate_backend=args.meta_backend,
        profiles=args.profiles,
        inner_steps=args.inner_steps,
        seeds=args.seeds,
        inner_lr=args.inner_lr,
        device=args.device,
        csv_out=meta_csv,
        summary_out=meta_summary,
        gate_csv_out=meta_gate,
        fail_on_gate=args.fail_on_gate,
        min_cosine_full_mean=args.min_cosine_full_mean,
        min_cosine_fo_mean=args.min_cosine_fo_mean,
        max_rel_diff_abs_error=args.max_rel_diff_abs_error,
        require_trend_sign_match=args.require_trend_sign_match,
    )
    _run_check(
        reference_backend=args.reference_backend,
        candidate_backend=args.strict_backend,
        profiles=args.profiles,
        inner_steps=args.inner_steps,
        seeds=args.seeds,
        inner_lr=args.inner_lr,
        device=args.device,
        csv_out=strict_csv,
        summary_out=strict_summary,
        gate_csv_out=strict_gate,
        fail_on_gate=args.fail_on_gate,
        min_cosine_full_mean=args.min_cosine_full_mean,
        min_cosine_fo_mean=args.min_cosine_fo_mean,
        max_rel_diff_abs_error=args.max_rel_diff_abs_error,
        require_trend_sign_match=args.require_trend_sign_match,
    )

    meta_rows = _load_summary(meta_summary)
    strict_rows = _load_summary(strict_summary)
    keys = sorted(set(meta_rows.keys()) & set(strict_rows.keys()), key=lambda x: (x[0], x[1]))

    delta_out = (
        (REPO_ROOT / args.delta_out).resolve()
        if args.delta_out
        else out_dir / f"phase12_meta_contract_{tag}_delta_table.csv"
    )
    with delta_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "profile",
                "inner_steps",
                "cos_full_meta",
                "cos_full_strict",
                "cos_strict_minus_meta",
                "rel_ref",
                "rel_cand_meta",
                "rel_cand_strict",
                "abs_err_meta",
                "abs_err_strict",
                "abs_err_improvement_strict_minus_meta",
            ]
        )
        for key in keys:
            m = meta_rows[key]
            s = strict_rows[key]
            rel_ref = float(m["rel_diff_ref_mean"])
            rel_meta = float(m["rel_diff_candidate_mean"])
            rel_strict = float(s["rel_diff_candidate_mean"])
            abs_err_meta = abs(rel_meta - rel_ref)
            abs_err_strict = abs(rel_strict - rel_ref)
            w.writerow(
                [
                    key[0],
                    key[1],
                    float(m["cosine_full_mean"]),
                    float(s["cosine_full_mean"]),
                    float(s["cosine_full_mean"]) - float(m["cosine_full_mean"]),
                    rel_ref,
                    rel_meta,
                    rel_strict,
                    abs_err_meta,
                    abs_err_strict,
                    abs_err_meta - abs_err_strict,
                ]
            )

    print("\n== Side-by-side delta table ==")
    header = (
        "profile            k   cos(meta)  cos(strict)  rel_ref   rel_meta  rel_strict  "
        "abs_err_meta  abs_err_strict  improve"
    )
    print(header)
    print("-" * len(header))
    for key in keys:
        m = meta_rows[key]
        s = strict_rows[key]
        rel_ref = float(m["rel_diff_ref_mean"])
        rel_meta = float(m["rel_diff_candidate_mean"])
        rel_strict = float(s["rel_diff_candidate_mean"])
        abs_err_meta = abs(rel_meta - rel_ref)
        abs_err_strict = abs(rel_strict - rel_ref)
        improve = abs_err_meta - abs_err_strict
        print(
            f"{key[0]:<18} {key[1]:>2}  {_fmt(float(m['cosine_full_mean'])):>9}  "
            f"{_fmt(float(s['cosine_full_mean'])):>11}  {_fmt(rel_ref):>8}  "
            f"{_fmt(rel_meta):>8}  {_fmt(rel_strict):>10}  {_fmt(abs_err_meta):>12}  "
            f"{_fmt(abs_err_strict):>14}  {_fmt(improve):>7}"
        )

    print(
        f"\nWrote:\n- {meta_csv}\n- {meta_summary}\n- {meta_gate}\n"
        f"- {strict_csv}\n- {strict_summary}\n- {strict_gate}\n- {delta_out}"
    )


if __name__ == "__main__":
    main()
