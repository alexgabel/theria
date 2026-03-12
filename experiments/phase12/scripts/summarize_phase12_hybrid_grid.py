#!/usr/bin/env python3
"""
Rank (meta_every_n_outer, meta_last_n_inner) settings from equal-time summaries.

Default target is triton_fused_meta_strict + FULL_HYBRID using Phase-12 summary CSVs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Row:
    n_outer: int
    last_n_inner: int
    inner_steps: int
    acc: float
    source: str


def _safe_float(v: str | None) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _extract_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1))


def _infer_n_outer(summary_row: dict[str, str], source_path: str) -> int | None:
    n_outer = _extract_int(r"_N(\d+)", os.path.basename(source_path))
    if n_outer is not None:
        return n_outer

    meta_steps = _safe_float(summary_row.get("n_hybrid_meta_steps_mean"))
    fo_steps = _safe_float(summary_row.get("n_hybrid_fo_steps_mean"))
    total_steps = meta_steps + fo_steps
    if meta_steps > 0 and total_steps > 0:
        est = int(round(total_steps / meta_steps))
        return est if est >= 1 else None
    return None


def _infer_last_n_inner(summary_row: dict[str, str], source_path: str) -> int | None:
    raw = _safe_float(summary_row.get("meta_last_n_inner_mean"))
    if not math.isnan(raw):
        val = int(round(raw))
        if val >= 0:
            return val
    from_name = _extract_int(r"_L(\d+)", os.path.basename(source_path))
    if from_name is not None:
        return from_name
    # Legacy summaries may not include this column/tag; treat as disabled.
    return 0


def _load_rows(
    paths: Iterable[str],
    *,
    backend: str,
    mode: str,
    protocol: str,
) -> list[Row]:
    out: list[Row] = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("backend") != backend:
                    continue
                if r.get("mode") != mode:
                    continue
                if protocol and r.get("comparison_protocol", "") != protocol:
                    continue
                n_outer = _infer_n_outer(r, path)
                last_n_inner = _infer_last_n_inner(r, path)
                inner_steps = int(float(r["inner_steps"]))
                acc = float(r["final_acc_mean"])
                if n_outer is None or last_n_inner is None:
                    continue
                out.append(
                    Row(
                        n_outer=n_outer,
                        last_n_inner=last_n_inner,
                        inner_steps=inner_steps,
                        acc=acc,
                        source=path,
                    )
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        type=str,
        default="experiments/phase12/runs/phase12_maml_seqcls_equal_time_*_summary.csv",
        help="Glob for summary CSV files.",
    )
    parser.add_argument("--backend", type=str, default="triton_fused_meta_strict")
    parser.add_argument("--mode", type=str, default="FULL_HYBRID")
    parser.add_argument(
        "--protocol",
        type=str,
        default="equal_time",
        help="comparison_protocol filter (empty to disable).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional CSV output with ranked settings.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    rows = _load_rows(paths, backend=args.backend, mode=args.mode, protocol=args.protocol)
    if not rows:
        raise SystemExit(
            "No matching rows found. "
            f"(backend={args.backend}, mode={args.mode}, protocol={args.protocol})"
        )

    # First, merge duplicate runs at fixed (N, L, k) by averaging accuracy.
    by_triplet: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for r in rows:
        by_triplet[(r.n_outer, r.last_n_inner, r.inner_steps)].append(r.acc)

    merged: list[tuple[int, int, int, float]] = []
    for (n_outer, last_n_inner, inner_steps), vals in by_triplet.items():
        merged.append((n_outer, last_n_inner, inner_steps, sum(vals) / len(vals)))

    # Then, rank each (N, L) by its best-k frontier and report mean-k too.
    by_pair: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for n_outer, last_n_inner, inner_steps, acc in merged:
        by_pair[(n_outer, last_n_inner)].append((inner_steps, acc))

    ranking: list[dict[str, float | int]] = []
    for (n_outer, last_n_inner), vals in by_pair.items():
        best_k, best_acc = max(vals, key=lambda x: x[1])
        mean_acc = sum(v for _, v in vals) / len(vals)
        ranking.append(
            {
                "meta_every_n_outer": n_outer,
                "meta_last_n_inner": last_n_inner,
                "best_inner_steps": best_k,
                "best_final_acc_mean": best_acc,
                "mean_final_acc_mean_over_k": mean_acc,
                "n_k_values": len(vals),
            }
        )

    ranking.sort(key=lambda r: float(r["best_final_acc_mean"]), reverse=True)

    print("Ranked (meta_every_n_outer, meta_last_n_inner) by best equal-time final_acc_mean:")
    for i, r in enumerate(ranking, start=1):
        print(
            f"{i:2d}) N={int(r['meta_every_n_outer'])} "
            f"L={int(r['meta_last_n_inner'])} "
            f"best_acc={float(r['best_final_acc_mean']):.6f} "
            f"(k={int(r['best_inner_steps'])}, mean_over_k={float(r['mean_final_acc_mean_over_k']):.6f})"
        )

    if args.csv_out:
        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "meta_every_n_outer",
                    "meta_last_n_inner",
                    "best_inner_steps",
                    "best_final_acc_mean",
                    "mean_final_acc_mean_over_k",
                    "n_k_values",
                ],
            )
            w.writeheader()
            for i, r in enumerate(ranking, start=1):
                w.writerow({"rank": i, **r})
        print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
