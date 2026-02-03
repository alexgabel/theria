"""
Summarize Phase 12 run CSVs and generate plots + markdown.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _as_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load_latest(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", type=str, default=None)
    parser.add_argument("--triton-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="experiments/phase12/runs")
    parser.add_argument("--fig-dir", type=str, default="experiments/phase12/figures")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ref_csv = Path(args.reference_csv) if args.reference_csv else _load_latest(out_dir, "phase12_reference_*.csv")
    tri_csv = Path(args.triton_csv) if args.triton_csv else _load_latest(out_dir, "phase12_triton_*.csv")

    tag = args.tag
    if tag is None:
        tag = ref_csv.stem.split("phase12_reference_")[-1]

    rows = _read_rows(ref_csv) + _read_rows(tri_csv)
    if not rows:
        raise RuntimeError("No rows found in inputs.")

    all_csv = out_dir / f"phase12_all_{tag}.csv"
    with all_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status", "") == "OK"]
    grouped: dict[tuple[str, str, int], list[tuple[float, float, float]]] = defaultdict(list)

    for r in ok_rows:
        key = (r["backend"], r["mode"], int(r["inner_steps"]))
        q = _as_float(r["attn_grad_norm_q"])
        k = _as_float(r["attn_grad_norm_k"])
        v = _as_float(r["attn_grad_norm_v"])
        attn_norm = (q * q + k * k + v * v) ** 0.5
        grouped[key].append((_as_float(r["final_acc"]), _as_float(r["final_loss"]), attn_norm))

    summary_rows: list[dict[str, str | int | float]] = []
    for (backend, mode, inner_steps), vals in sorted(grouped.items()):
        accs = [x[0] for x in vals]
        losses = [x[1] for x in vals]
        attn_norms = [x[2] for x in vals]
        summary_rows.append(
            {
                "backend": backend,
                "mode": mode,
                "inner_steps": inner_steps,
                "n": len(vals),
                "final_acc_mean": mean(accs),
                "final_acc_std": pstdev(accs) if len(accs) > 1 else 0.0,
                "final_loss_mean": mean(losses),
                "attn_grad_norm_mean": mean(attn_norms),
            }
        )

    summary_csv = out_dir / f"phase12_summary_{tag}.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "mode",
                "inner_steps",
                "n",
                "final_acc_mean",
                "final_acc_std",
                "final_loss_mean",
                "attn_grad_norm_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    modes = ["FULL", "FO", "FO_STRICT"]
    inner_steps_list = [1, 5, 10, 20]
    backends = sorted({r["backend"] for r in summary_rows})

    def _summary_lookup(backend: str, mode: str, inner_steps: int) -> dict[str, str | int | float] | None:
        for row in summary_rows:
            if (
                row["backend"] == backend
                and row["mode"] == mode
                and int(row["inner_steps"]) == inner_steps
            ):
                return row
        return None

    for metric, ylabel, filename in [
        ("final_acc_mean", "Final Accuracy (mean over last 20 steps)", f"phase12_acc_vs_inner_steps_{tag}.png"),
        ("attn_grad_norm_mean", "Attention Grad Norm", f"phase12_attn_grad_norm_vs_inner_steps_{tag}.png"),
    ]:
        fig, axes = plt.subplots(1, len(backends), figsize=(6 * len(backends), 4), squeeze=False)
        for col, backend in enumerate(backends):
            ax = axes[0][col]
            for mode in modes:
                ys: list[float] = []
                errs: list[float] = []
                for inner_steps in inner_steps_list:
                    row = _summary_lookup(backend, mode, inner_steps)
                    if row is None:
                        ys.append(float("nan"))
                        errs.append(0.0)
                    else:
                        ys.append(float(row[metric]))
                        errs.append(float(row["final_acc_std"]) if metric == "final_acc_mean" else 0.0)
                if metric == "final_acc_mean":
                    ax.errorbar(inner_steps_list, ys, yerr=errs, marker="o", capsize=3, label=mode)
                else:
                    ax.plot(inner_steps_list, ys, marker="o", label=mode)
            ax.set_title(backend)
            ax.set_xlabel("inner_steps")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[0][0].legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=160)
        plt.close(fig)

    summary_md = out_dir / f"phase12_summary_{tag}.md"
    lines = [
        "# Phase 12 Milestone Summary",
        "",
        f"- Combined CSV: `{all_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "## k=20 headline",
    ]

    for backend in backends:
        full = _summary_lookup(backend, "FULL", 20)
        fo = _summary_lookup(backend, "FO", 20)
        fo_str = _summary_lookup(backend, "FO_STRICT", 20)
        if full and fo and fo_str:
            lines.append(
                "- "
                f"{backend}: FULL={float(full['final_acc_mean']):.4f}, "
                f"FO={float(fo['final_acc_mean']):.4f}, "
                f"FO_STRICT={float(fo_str['final_acc_mean']):.4f}"
            )

    lines.extend(
        [
            "",
            "## Figures",
            f"- `experiments/phase12/figures/phase12_acc_vs_inner_steps_{tag}.png`",
            f"- `experiments/phase12/figures/phase12_attn_grad_norm_vs_inner_steps_{tag}.png`",
        ]
    )

    summary_md.write_text("\n".join(lines))
    print(f"Wrote {all_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
