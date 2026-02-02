"""
Phase 11 full taxonomy sweep runner.

Runs all registered wrappers over:
- attention backends (reference/custom/triton_fused)
- inner_steps list (default 1,5,10,20)
- seeds list (default 0)
- steps, inner_lr, device configurable

Writes a consolidated CSV with one row per (wrapper, backend, inner_steps, seed).
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

from run_bad_backend_diagnostics import run_backend
from bad_backends import (
    detach_attention_output,
    detach_q_input,
    detach_k_input,
    detach_v_input,
    detach_q_input_strict,
    detach_k_input_strict,
    detach_v_input_strict,
    checkpoint_attention,
    checkpoint_no_grad,
    checkpoint_detach_recompute,
    recompute_logits_no_grad_sdpa,
    backward_detach_logits_sim,
    no_grad_attention,
    once_differentiable_sim,
    stats_detach_logits_sdpa,
    stats_detach_softmax_output_sdpa,
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="reference,custom,triton_fused",
        help="Comma-separated attention backends.",
    )
    parser.add_argument(
        "--inner-steps-list",
        type=str,
        default="1,5,10,20",
        help='Comma-separated inner-step values, e.g., "1,5,10,20".',
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="experiments/phase11/runs/taxonomy_sweep.csv",
    )
    args = parser.parse_args()

    wrappers = {
        "baseline": None,
        "detach_attention_output": detach_attention_output,
        "detach_q_input": detach_q_input,
        "detach_k_input": detach_k_input,
        "detach_v_input": detach_v_input,
        "detach_q_input_strict": detach_q_input_strict,
        "detach_k_input_strict": detach_k_input_strict,
        "detach_v_input_strict": detach_v_input_strict,
        "checkpoint_attention": checkpoint_attention,
        "checkpoint_no_grad": checkpoint_no_grad,
        "checkpoint_detach_recompute": checkpoint_detach_recompute,
        "recompute_logits_no_grad_sdpa": recompute_logits_no_grad_sdpa,
        "backward_detach_logits_sim": backward_detach_logits_sim,
        "no_grad_attention": no_grad_attention,
        "once_differentiable_sim": once_differentiable_sim,
        "stats_detach_logits_sdpa": stats_detach_logits_sdpa,
        "stats_detach_softmax_output_sdpa": stats_detach_softmax_output_sdpa,
    }

    attention_backends = _parse_str_list(args.attention_backends)
    inner_steps_list = _parse_int_list(args.inner_steps_list)
    seeds = _parse_int_list(args.seeds)
    device = torch.device(args.device)

    rows = []
    total = len(wrappers) * len(attention_backends) * len(inner_steps_list) * len(seeds)
    idx = 0
    for wrapper_name, wrapper in wrappers.items():
        for backend in attention_backends:
            for k in inner_steps_list:
                for seed in seeds:
                    idx += 1
                    row = run_backend(
                        wrapper_name=wrapper_name,
                        wrapper=wrapper,
                        attention_backend=backend,
                        seed=seed,
                        steps=args.steps,
                        inner_steps=k,
                        inner_lr=args.inner_lr,
                        device=device,
                    )
                    row["inner_steps"] = k
                    row["seed"] = seed
                    row["steps"] = args.steps
                    rows.append(row)
                    print(
                        f"[{idx}/{total}] wrapper={wrapper_name} backend={backend} k={k} seed={seed} "
                        f"status={row['status']} rel={row.get('rel_diff_mean','') or 'NA'}"
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

