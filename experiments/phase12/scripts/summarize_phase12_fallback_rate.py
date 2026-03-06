#!/usr/bin/env python3
"""
Summarize fallback dependence from a Phase 12 per-run CSV.

Tracks:
  - fallback calls per outer step
  - fallback incidence by k (inner_steps)
  - fallback incidence by seed
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    mu = _mean(vals)
    return float((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Per-run CSV from run_phase12_maml_seqcls_compare.py")
    parser.add_argument("--backend", default="triton_fused_meta")
    parser.add_argument("--mode", default="FULL")
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with Path(args.csv).open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("backend") != args.backend:
                continue
            if row.get("mode") != args.mode:
                continue
            if row.get("status") != "OK":
                continue
            rows.append(row)

    if not rows:
        raise SystemExit(
            f"no OK rows found for backend={args.backend} mode={args.mode} in {args.csv}"
        )

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["inner_steps"]), row["seed"])].append(row)

    summary_rows: list[dict[str, str | float | int]] = []
    for (inner_steps, seed), items in sorted(grouped.items()):
        fallback_counts = [float(x.get("n_fallback_bwd", 0.0)) for x in items]
        fallback_rates = [
            float(
                x.get(
                    "fallback_bwd_per_outer_step",
                    float(x.get("n_fallback_bwd", 0.0)) / max(float(x.get("outer_steps", 1.0)), 1.0),
                )
            )
            for x in items
        ]
        fallback_incidence = [float(x.get("fallback_incident", 0.0)) for x in items]
        outer_steps = [float(x.get("outer_steps", 0.0)) for x in items]
        summary_rows.append(
            {
                "backend": args.backend,
                "mode": args.mode,
                "inner_steps": inner_steps,
                "seed": seed,
                "n_runs": len(items),
                "n_fallback_bwd_mean": _mean(fallback_counts),
                "n_fallback_bwd_std": _std(fallback_counts),
                "fallback_bwd_per_outer_step_mean": _mean(fallback_rates),
                "fallback_bwd_per_outer_step_std": _std(fallback_rates),
                "fallback_incident_mean": _mean(fallback_incidence),
                "fallback_incident_std": _std(fallback_incidence),
                "outer_steps_mean": _mean(outer_steps),
            }
        )

    grouped_by_k: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_by_k[int(row["inner_steps"])].append(row)
    for inner_steps, items in sorted(grouped_by_k.items()):
        fallback_counts = [float(x.get("n_fallback_bwd", 0.0)) for x in items]
        fallback_rates = [
            float(
                x.get(
                    "fallback_bwd_per_outer_step",
                    float(x.get("n_fallback_bwd", 0.0)) / max(float(x.get("outer_steps", 1.0)), 1.0),
                )
            )
            for x in items
        ]
        fallback_incidence = [float(x.get("fallback_incident", 0.0)) for x in items]
        outer_steps = [float(x.get("outer_steps", 0.0)) for x in items]
        summary_rows.append(
            {
                "backend": args.backend,
                "mode": args.mode,
                "inner_steps": inner_steps,
                "seed": "ALL",
                "n_runs": len(items),
                "n_fallback_bwd_mean": _mean(fallback_counts),
                "n_fallback_bwd_std": _std(fallback_counts),
                "fallback_bwd_per_outer_step_mean": _mean(fallback_rates),
                "fallback_bwd_per_outer_step_std": _std(fallback_rates),
                "fallback_incident_mean": _mean(fallback_incidence),
                "fallback_incident_std": _std(fallback_incidence),
                "outer_steps_mean": _mean(outer_steps),
            }
        )

    print("Fallback summary by k and seed:")
    for row in summary_rows:
        print(
            f"backend={row['backend']} mode={row['mode']} k={row['inner_steps']} "
            f"seed={row['seed']} calls_mean={row['n_fallback_bwd_mean']:.3f} "
            f"per_outer_mean={row['fallback_bwd_per_outer_step_mean']:.6f} "
            f"incident_mean={row['fallback_incident_mean']:.3f}"
        )

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
