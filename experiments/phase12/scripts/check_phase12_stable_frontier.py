#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _load_summary(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["backend"], row["mode"], int(row["inner_steps"]))
        out[key] = row
    return out


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal-step-summary", required=True)
    parser.add_argument("--equal-time-summary", required=True)
    parser.add_argument("--reference-backend", default="reference")
    parser.add_argument("--stable-backend", default="triton_fused_meta_strict")
    parser.add_argument("--full-mode", default="FULL")
    parser.add_argument("--fo-mode", default="FO")
    parser.add_argument("--hybrid-mode", default="FULL_HYBRID")
    parser.add_argument("--full-acc-tol", type=float, default=5e-4)
    parser.add_argument("--full-loss-tol", type=float, default=1e-3)
    parser.add_argument("--hybrid-frontier-acc-tol", type=float, default=1e-2)
    parser.add_argument("--fail-on-check", action="store_true")
    args = parser.parse_args()

    step_rows = _load_summary(Path(args.equal_step_summary))
    time_rows = _load_summary(Path(args.equal_time_summary))

    full_keys = sorted(
        k for k in step_rows if k[0] == args.reference_backend and k[1] == args.full_mode
    )
    if not full_keys:
        raise SystemExit("No reference FULL rows found in equal-step summary")

    full_pass = True
    print("== Equal-step FULL semantics ==")
    for _, _, inner_steps in full_keys:
        ref = step_rows[(args.reference_backend, args.full_mode, inner_steps)]
        cand_key = (args.stable_backend, args.full_mode, inner_steps)
        if cand_key not in step_rows:
            print(f"missing stable FULL row for k={inner_steps}")
            full_pass = False
            continue
        cand = step_rows[cand_key]
        acc_delta = _f(cand, "final_acc_mean") - _f(ref, "final_acc_mean")
        loss_delta = _f(cand, "final_loss_mean") - _f(ref, "final_loss_mean")
        row_pass = abs(acc_delta) <= args.full_acc_tol and abs(loss_delta) <= args.full_loss_tol
        full_pass &= row_pass
        print(
            f"k={inner_steps} acc_delta={acc_delta:+.6f} "
            f"loss_delta={loss_delta:+.6f} pass={row_pass}"
        )

    fo_rows = [
        row for key, row in time_rows.items()
        if key[0] == args.stable_backend and key[1] == args.fo_mode
    ]
    hybrid_rows = [
        row for key, row in time_rows.items()
        if key[0] == args.stable_backend and key[1] == args.hybrid_mode
    ]
    if not fo_rows or not hybrid_rows:
        raise SystemExit("Missing FO or FULL_HYBRID rows in equal-time summary")

    best_fo = max(_f(r, "final_acc_mean") for r in fo_rows)
    best_hybrid = max(_f(r, "final_acc_mean") for r in hybrid_rows)
    frontier_delta = best_hybrid - best_fo
    frontier_pass = frontier_delta + args.hybrid_frontier_acc_tol >= 0.0

    print("\n== Equal-time frontier ==")
    print(
        f"best_fo={best_fo:.6f} best_full_hybrid={best_hybrid:.6f} "
        f"delta={frontier_delta:+.6f} tol={args.hybrid_frontier_acc_tol:.6f} "
        f"pass={frontier_pass}"
    )

    overall_pass = full_pass and frontier_pass
    print(f"\nOVERALL_PASS={overall_pass}")
    if args.fail_on_check and not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
