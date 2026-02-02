"""
Phase 11 checkpoint-variant sweep runner.

Runs a grid over:
- wrappers: checkpoint_attention / checkpoint_no_grad / checkpoint_detach_recompute
- attention backends: reference / custom / triton_fused
- inner_steps: typically 1,5,10,20
- seeds: optional replicate seeds

Writes one consolidated CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

# Allow direct script execution from repo root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from bad_backends import (
    checkpoint_attention,
    checkpoint_no_grad,
    checkpoint_detach_recompute,
)
from run_bad_backend_diagnostics import run_backend


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrappers",
        type=str,
        default="checkpoint_attention,checkpoint_no_grad,checkpoint_detach_recompute",
        help="Comma-separated wrapper names.",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="reference,custom,triton_fused",
        help="Comma-separated attention backends.",
    )
    parser.add_argument(
        "--inner-steps",
        type=str,
        default="1,5,10,20",
        help="Comma-separated inner-step values.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seed values.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="experiments/phase11/runs/checkpoint_sweep.csv",
    )
    args = parser.parse_args()

    wrapper_map = {
        "checkpoint_attention": checkpoint_attention,
        "checkpoint_no_grad": checkpoint_no_grad,
        "checkpoint_detach_recompute": checkpoint_detach_recompute,
    }

    wrappers = _parse_str_list(args.wrappers)
    attention_backends = _parse_str_list(args.attention_backends)
    inner_steps_list = _parse_int_list(args.inner_steps)
    seeds = _parse_int_list(args.seeds)
    device = torch.device(args.device)

    rows: list[dict[str, str | int | float]] = []
    total = len(wrappers) * len(attention_backends) * len(inner_steps_list) * len(seeds)
    idx = 0

    for wrapper_name in wrappers:
        if wrapper_name not in wrapper_map:
            raise ValueError(f"Unknown wrapper: {wrapper_name}")
        wrapper = wrapper_map[wrapper_name]
        for attention_backend in attention_backends:
            for inner_steps in inner_steps_list:
                for seed in seeds:
                    idx += 1
                    row = run_backend(
                        wrapper_name=wrapper_name,
                        wrapper=wrapper,
                        attention_backend=attention_backend,
                        seed=seed,
                        steps=args.steps,
                        inner_steps=inner_steps,
                        inner_lr=args.inner_lr,
                        device=device,
                    )
                    row["inner_steps"] = inner_steps
                    row["seed"] = seed
                    row["steps"] = args.steps
                    rows.append(row)
                    print(
                        f"[{idx}/{total}] wrapper={wrapper_name} "
                        f"attn={attention_backend} k={inner_steps} seed={seed} "
                        f"status={row['status']} rel={row.get('rel_diff_mean', '') or 'NA'}"
                    )

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wrapper",
        "attention_backend",
        "inner_steps",
        "seed",
        "steps",
        "status",
        "second_order_path_all",
        "second_order_path_any",
        "second_order_path_attn_all",
        "second_order_path_attn_any",
        "second_order_path_head_all",
        "second_order_path_head_any",
        "sdpa_input_gradgrad_ok",
        "rel_diff_mean",
        "grad_norm_q_proj",
        "grad_norm_k_proj",
        "grad_norm_v_proj",
        "error",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
