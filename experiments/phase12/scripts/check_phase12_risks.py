"""
Quick Phase 12 risk checks:
1) Determinism (same config twice, same seed)
2) Reference vs Triton forward parity
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

# Allow direct script execution from repo root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase10.scripts.run_maml_backend_compare import (
    Phase10TinyAttentionModel,
    set_attention_backend,
)
from experiments.phase12.scripts.run_phase12_behavior import run_behavior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--determinism-outer-steps", type=int, default=100)
    parser.add_argument("--out-csv", type=str, default="experiments/phase12/runs/phase12_risk_checks.csv")
    parser.add_argument("--parity-device", type=str, default="cuda")
    parser.add_argument("--determinism-tol", type=float, default=1e-9)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-signal-positions", type=int, default=4)
    args = parser.parse_args()

    rows: list[dict[str, str | float]] = []

    # Check 1: Determinism
    run_a = run_behavior(
        attention_backend="reference",
        mode="FULL",
        seed=0,
        outer_steps=args.determinism_outer_steps,
        meta_batch_size=16,
        inner_steps=5,
        inner_lr=0.4,
        outer_lr=1e-3,
        seq_len=args.seq_len,
        num_signal_positions=args.num_signal_positions,
        device=torch.device("cpu"),
        autocast_enabled=False,
        rel_diff_probe=False,
    )
    run_b = run_behavior(
        attention_backend="reference",
        mode="FULL",
        seed=0,
        outer_steps=args.determinism_outer_steps,
        meta_batch_size=16,
        inner_steps=5,
        inner_lr=0.4,
        outer_lr=1e-3,
        seq_len=args.seq_len,
        num_signal_positions=args.num_signal_positions,
        device=torch.device("cpu"),
        autocast_enabled=False,
        rel_diff_probe=False,
    )
    loss_diff = abs(float(run_a["final_loss"]) - float(run_b["final_loss"]))
    acc_diff = abs(float(run_a["final_acc"]) - float(run_b["final_acc"]))
    det_status = "PASS" if (loss_diff <= args.determinism_tol and acc_diff <= args.determinism_tol) else "WARN"
    rows.append(
        {
            "check": "seed_determinism_cpu_reference_full_k5",
            "status": det_status,
            "value_1": loss_diff,
            "value_2": acc_diff,
            "details": f"tol={args.determinism_tol}",
        }
    )

    # Check 2: Forward parity
    parity_device = torch.device(args.parity_device)
    if parity_device.type == "cuda" and not torch.cuda.is_available():
        rows.append(
            {
                "check": "reference_vs_triton_forward_parity",
                "status": "SKIP",
                "value_1": float("nan"),
                "value_2": float("nan"),
                "details": "CUDA not available",
            }
        )
    else:
        torch.manual_seed(0)
        model = Phase10TinyAttentionModel().to(parity_device).eval()
        x = torch.randn(4, 8, model.cfg.d_model, device=parity_device)
        with torch.no_grad():
            set_attention_backend(model, "reference")
            y_ref = model(x)
            set_attention_backend(model, "triton_fused")
            y_tri = model(x)

        max_abs_err = (y_ref - y_tri).abs().max().item()
        rel_err = (y_ref - y_tri).norm().item() / (y_ref.norm().item() + 1e-9)
        parity_status = "PASS" if max_abs_err < 1e-4 and rel_err < 1e-3 else "WARN"
        rows.append(
            {
                "check": "reference_vs_triton_forward_parity",
                "status": parity_status,
                "value_1": max_abs_err,
                "value_2": rel_err,
                "details": f"device={parity_device}",
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["check", "status", "value_1", "value_2", "details"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    for row in rows:
        print(
            f"{row['check']}: {row['status']} "
            f"value_1={row['value_1']} value_2={row['value_2']} ({row['details']})"
        )


if __name__ == "__main__":
    main()
