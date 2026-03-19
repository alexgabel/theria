#!/usr/bin/env python3
"""
Phase 12 inner-step visualizations.

Supports two workflows:
  1. Plot final accuracy vs inner steps from existing Phase 12 summary CSVs.
  2. Meta-train a tiny attention model, then project a held-out task into 2D
     with PCA across selected inner adaptation steps.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
import os
from pathlib import Path
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.func import functional_call


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.phase10.scripts.run_maml_backend_compare import (  # noqa: E402
    Phase10TinyAttentionModel,
    set_attention_backend,
)
from theria.attention.custom import sdpa_custom  # noqa: E402
from theria.maml.loops import meta_loss_on_tasks, named_buffers, named_params  # noqa: E402
from theria.tasks.synthetic_seqcls import TaskBatch, task_sampler  # noqa: E402


MODE_ORDER = ("FULL", "FO", "FULL_HYBRID")
BACKEND_ORDER = ("reference", "triton_fused_meta_strict", "triton_fused_meta")
BACKEND_LABELS = {
    "reference": "Reference",
    "triton_fused_meta_strict": "Strict",
    "triton_fused_meta": "Fused-meta",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _derive_label(path: Path) -> str:
    stem = path.stem
    stem = stem.removeprefix("phase12_maml_seqcls_equal_step_")
    stem = stem.removeprefix("phase12_maml_seqcls_equal_time_")
    stem = stem.removesuffix("_summary")
    return stem.replace("_", " ")


def _protocol_from_name(path: Path) -> str:
    if "equal_step" in path.name:
        return "equal-step"
    if "equal_time" in path.name:
        return "equal-time"
    return "unknown"


def _plot_summary(summary_csv: Path, label: str, out_dir: Path) -> Path:
    rows = _read_rows(summary_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {summary_csv}")

    fig, axes = plt.subplots(1, len(MODE_ORDER), figsize=(15, 4), squeeze=False)
    protocol = _protocol_from_name(summary_csv)

    for col, mode in enumerate(MODE_ORDER):
        ax = axes[0][col]
        mode_rows = [r for r in rows if r["mode"] == mode]
        inner_steps = sorted({int(r["inner_steps"]) for r in mode_rows})
        for backend in BACKEND_ORDER:
            backend_rows = {
                int(r["inner_steps"]): float(r["final_acc_mean"])
                for r in mode_rows
                if r["backend"] == backend
            }
            ys = [backend_rows.get(k, float("nan")) for k in inner_steps]
            ax.plot(
                inner_steps,
                ys,
                marker="o",
                linewidth=2,
                label=BACKEND_LABELS.get(backend, backend),
            )
        ax.set_title(mode)
        ax.set_xlabel("Inner steps")
        ax.set_ylabel("Final accuracy")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} ({protocol})", fontsize=12)
    axes[0][0].legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)

    out_path = out_dir / f"{summary_csv.stem}_inner_steps.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _parse_csv_steps(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("Need at least one inner step")
    if values[0] < 0:
        raise ValueError("Inner steps must be >= 0")
    return values


def _device_from_arg(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _mode_step_flags(
    mode: str,
    outer_step_idx: int,
    *,
    meta_every_n_outer: int,
    meta_last_n_inner: int,
) -> tuple[bool, int]:
    if mode == "FULL":
        return False, meta_last_n_inner
    if mode == "FO":
        return True, 0
    if mode == "FULL_HYBRID":
        use_meta_step = (outer_step_idx % meta_every_n_outer) == 0
        return (not use_meta_step), (meta_last_n_inner if use_meta_step else 0)
    raise ValueError(f"Unsupported mode: {mode}")


def _ordered_params(
    names: tuple[str, ...],
    values: tuple[torch.Tensor, ...],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(zip(names, values))


def _train_meta_init(
    *,
    backend: str,
    mode: str,
    device: torch.device,
    seed: int,
    outer_steps: int,
    inner_steps: int,
    meta_batch_size: int,
    inner_lr: float,
    outer_lr: float,
    seq_len: int,
    num_signal_positions: int,
    meta_every_n_outer: int,
    meta_last_n_inner: int,
) -> Phase10TinyAttentionModel:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = Phase10TinyAttentionModel().to(device)
    set_attention_backend(model, backend)
    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for outer_step_idx in range(outer_steps):
        tasks = [
            task_sampler(
                T=seq_len,
                D=model.cfg.d_model,
                num_signal_positions=num_signal_positions,
                device=device,
                split="train",
                dataset_seed=seed,
            )
            for _ in range(meta_batch_size)
        ]
        fo, meta_last_n_inner_step = _mode_step_flags(
            mode,
            outer_step_idx,
            meta_every_n_outer=meta_every_n_outer,
            meta_last_n_inner=meta_last_n_inner,
        )

        optimizer.zero_grad(set_to_none=True)
        outer_loss, metrics = meta_loss_on_tasks(
            model=model,
            tasks=tasks,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            fo=fo,
            fo_strict=False,
            meta_last_n_inner=meta_last_n_inner_step,
            return_metrics=True,
        )
        grads = torch.autograd.grad(
            outer_loss,
            trainable_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        for param, grad in zip(trainable_params, grads):
            param.grad = grad
        optimizer.step()

        if ((outer_step_idx + 1) % max(outer_steps // 5, 1)) == 0 or outer_step_idx == 0:
            print(
                f"[train {outer_step_idx + 1:04d}/{outer_steps}] "
                f"mode={mode} backend={backend} "
                f"outer_loss={float(outer_loss.item()):.4f} "
                f"query_acc={float(metrics['post_adapt_acc']):.4f}"
            )

    return model


def _compute_embeddings_and_logits(
    model: Phase10TinyAttentionModel,
    params: OrderedDict[str, torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.linear(x[:, :1, :], params["q_proj.weight"])
    k = F.linear(x, params["k_proj.weight"])
    v = F.linear(x, params["v_proj.weight"])
    h = sdpa_custom(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        backend=model._resolve_backend(),
    )
    h = h.squeeze(1).squeeze(1)
    logits = F.linear(h, params["classifier.weight"], params["classifier.bias"])
    return h, logits


def _adapt_snapshots(
    model: Phase10TinyAttentionModel,
    task: TaskBatch,
    *,
    inner_lr: float,
    inner_steps: int,
) -> tuple[dict[int, OrderedDict[str, torch.Tensor]], OrderedDict[str, torch.Tensor]]:
    params = named_params(model)
    buffers = named_buffers(model)
    param_names = tuple(params.keys())
    phi_values = tuple(p.detach().clone().requires_grad_(True) for p in params.values())
    snapshots = {0: _ordered_params(param_names, tuple(v.detach() for v in phi_values))}

    for step_idx in range(inner_steps):
        phi = _ordered_params(param_names, phi_values)
        logits_s = functional_call(model, (phi, buffers), (task.x_s,))
        loss_s = F.cross_entropy(logits_s, task.y_s)
        grads = torch.autograd.grad(
            loss_s,
            phi_values,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        phi_values = tuple(
            (param - inner_lr * grad).detach().clone().requires_grad_(True)
            for param, grad in zip(phi_values, grads)
        )
        snapshots[step_idx + 1] = _ordered_params(
            param_names,
            tuple(v.detach() for v in phi_values),
        )

    return snapshots, buffers


def _pca_project(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.to(dtype=torch.float32, device="cpu")
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    _, _, v = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ v[:, :2]
    return coords, mean.squeeze(0), v[:, :2]


def _coords_from_projection(
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    centered = x.to(dtype=torch.float32, device="cpu") - mean
    return centered @ basis


def _scatter_step_panel(
    ax: plt.Axes,
    *,
    support_coords: torch.Tensor,
    query_coords: torch.Tensor,
    y_s: torch.Tensor,
    y_q: torch.Tensor,
    query_correct: torch.Tensor,
    class_colors,
) -> None:
    for cls in range(len(class_colors)):
        cls_mask_s = y_s == cls
        cls_mask_q = y_q == cls
        if cls_mask_s.any():
            ax.scatter(
                support_coords[cls_mask_s, 0],
                support_coords[cls_mask_s, 1],
                s=48,
                marker="^",
                color=class_colors[cls],
                alpha=0.22,
                linewidths=0.0,
            )
        if cls_mask_q.any():
            correct_mask = cls_mask_q & query_correct
            wrong_mask = cls_mask_q & (~query_correct)
            if correct_mask.any():
                ax.scatter(
                    query_coords[correct_mask, 0],
                    query_coords[correct_mask, 1],
                    s=70,
                    marker="o",
                    color=class_colors[cls],
                    edgecolors="black",
                    linewidths=0.5,
                )
            if wrong_mask.any():
                ax.scatter(
                    query_coords[wrong_mask, 0],
                    query_coords[wrong_mask, 1],
                    s=92,
                    marker="X",
                    color=class_colors[cls],
                    edgecolors="black",
                    linewidths=0.6,
                )


def _plot_task_pca(args: argparse.Namespace) -> Path:
    device = _device_from_arg(args.device)
    plot_steps = _parse_csv_steps(args.plot_steps)
    if plot_steps[-1] > args.inner_steps:
        raise ValueError("--plot-steps cannot exceed --inner-steps")

    model = _train_meta_init(
        backend=args.backend,
        mode=args.mode,
        device=device,
        seed=args.seed,
        outer_steps=args.outer_steps,
        inner_steps=args.inner_steps,
        meta_batch_size=args.meta_batch_size,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        seq_len=args.seq_len,
        num_signal_positions=args.num_signal_positions,
        meta_every_n_outer=args.meta_every_n_outer,
        meta_last_n_inner=args.meta_last_n_inner,
    )

    torch.manual_seed(args.task_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.task_seed)
    task = task_sampler(
        T=args.seq_len,
        D=model.cfg.d_model,
        num_signal_positions=args.num_signal_positions,
        device=device,
        split="test",
        dataset_seed=args.seed,
    )

    snapshots, _ = _adapt_snapshots(
        model,
        task,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
    )

    per_step: dict[int, dict[str, torch.Tensor | float]] = {}
    pca_inputs: list[torch.Tensor] = []
    for step in plot_steps:
        params = snapshots[step]
        with torch.no_grad():
            support_h, _ = _compute_embeddings_and_logits(model, params, task.x_s)
            query_h, query_logits = _compute_embeddings_and_logits(model, params, task.x_q)
            query_pred = query_logits.argmax(dim=-1)
            query_correct = query_pred.eq(task.y_q)
            query_acc = float(query_correct.float().mean().item())
        per_step[step] = {
            "support_h": support_h.detach().cpu(),
            "query_h": query_h.detach().cpu(),
            "query_pred": query_pred.detach().cpu(),
            "query_correct": query_correct.detach().cpu(),
            "query_acc": query_acc,
        }
        pca_inputs.extend([support_h.detach().cpu(), query_h.detach().cpu()])
        print(f"[eval k={step:02d}] query_acc={query_acc:.4f}")

    _, mean, basis = _pca_project(torch.cat(pca_inputs, dim=0))
    raw_support = task.x_s.detach().cpu().flatten(start_dim=1)
    raw_query = task.x_q.detach().cpu().flatten(start_dim=1)
    _, raw_mean, raw_basis = _pca_project(torch.cat([raw_support, raw_query], dim=0))
    class_colors = plt.get_cmap("tab10")(range(model.cfg.num_classes))

    ncols = len(plot_steps)
    fig, axes = plt.subplots(2, ncols, figsize=(4.1 * ncols, 8.6), squeeze=False)
    axes_raw = axes[0]
    axes_feat = axes[1]

    feat_x_limits: list[float] = []
    feat_y_limits: list[float] = []
    for step in plot_steps:
        support_coords = _coords_from_projection(
            per_step[step]["support_h"],  # type: ignore[arg-type]
            mean=mean,
            basis=basis,
        )
        query_coords = _coords_from_projection(
            per_step[step]["query_h"],  # type: ignore[arg-type]
            mean=mean,
            basis=basis,
        )
        per_step[step]["support_coords"] = support_coords
        per_step[step]["query_coords"] = query_coords
        feat_x_limits.extend([support_coords[:, 0].min().item(), support_coords[:, 0].max().item()])
        feat_x_limits.extend([query_coords[:, 0].min().item(), query_coords[:, 0].max().item()])
        feat_y_limits.extend([support_coords[:, 1].min().item(), support_coords[:, 1].max().item()])
        feat_y_limits.extend([query_coords[:, 1].min().item(), query_coords[:, 1].max().item()])

    raw_support_coords = _coords_from_projection(raw_support, mean=raw_mean, basis=raw_basis)
    raw_query_coords = _coords_from_projection(raw_query, mean=raw_mean, basis=raw_basis)
    raw_x_limits = [
        raw_support_coords[:, 0].min().item(),
        raw_support_coords[:, 0].max().item(),
        raw_query_coords[:, 0].min().item(),
        raw_query_coords[:, 0].max().item(),
    ]
    raw_y_limits = [
        raw_support_coords[:, 1].min().item(),
        raw_support_coords[:, 1].max().item(),
        raw_query_coords[:, 1].min().item(),
        raw_query_coords[:, 1].max().item(),
    ]

    feat_pad_x = 0.08 * (max(feat_x_limits) - min(feat_x_limits) + 1e-6)
    feat_pad_y = 0.08 * (max(feat_y_limits) - min(feat_y_limits) + 1e-6)
    feat_xlim = (min(feat_x_limits) - feat_pad_x, max(feat_x_limits) + feat_pad_x)
    feat_ylim = (min(feat_y_limits) - feat_pad_y, max(feat_y_limits) + feat_pad_y)
    raw_pad_x = 0.08 * (max(raw_x_limits) - min(raw_x_limits) + 1e-6)
    raw_pad_y = 0.08 * (max(raw_y_limits) - min(raw_y_limits) + 1e-6)
    raw_xlim = (min(raw_x_limits) - raw_pad_x, max(raw_x_limits) + raw_pad_x)
    raw_ylim = (min(raw_y_limits) - raw_pad_y, max(raw_y_limits) + raw_pad_y)

    y_s_cpu = task.y_s.detach().cpu()
    y_q_cpu = task.y_q.detach().cpu()

    for col, step in enumerate(plot_steps):
        step_data = per_step[step]
        support_coords = step_data["support_coords"]  # type: ignore[assignment]
        query_coords = step_data["query_coords"]  # type: ignore[assignment]
        query_correct = step_data["query_correct"]  # type: ignore[assignment]
        raw_ax = axes_raw[col]
        feat_ax = axes_feat[col]

        _scatter_step_panel(
            raw_ax,
            support_coords=raw_support_coords,
            query_coords=raw_query_coords,
            y_s=y_s_cpu,
            y_q=y_q_cpu,
            query_correct=query_correct,
            class_colors=class_colors,
        )
        _scatter_step_panel(
            feat_ax,
            support_coords=support_coords,
            query_coords=query_coords,
            y_s=y_s_cpu,
            y_q=y_q_cpu,
            query_correct=query_correct,
            class_colors=class_colors,
        )

        raw_ax.set_title(f"k={step}\nquery acc={float(step_data['query_acc']):.3f}")
        raw_ax.set_xlim(*raw_xlim)
        raw_ax.set_ylim(*raw_ylim)
        raw_ax.set_xlabel("Raw PC1")
        raw_ax.grid(True, alpha=0.2)

        feat_ax.set_xlim(*feat_xlim)
        feat_ax.set_ylim(*feat_ylim)
        feat_ax.set_xlabel("Feature PC1")
        feat_ax.grid(True, alpha=0.2)

    for ax in axes_raw:
        ax.set_ylabel("Raw PC2")
    for ax in axes_feat:
        ax.set_ylabel("Feature PC2")

    axes_raw[0].text(
        -0.18,
        0.5,
        "Raw input\n(fixed across k)",
        transform=axes_raw[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
    )
    axes_feat[0].text(
        -0.18,
        0.5,
        "Adapted features",
        transform=axes_feat[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
    )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=class_colors[cls],
            markeredgecolor="black",
            markersize=8,
            label=f"class {cls}",
        )
        for cls in range(model.cfg.num_classes)
    ]
    split_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            linestyle="",
            markersize=8,
            alpha=0.45,
            label="support sample",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="",
            markersize=8,
            markeredgecolor="black",
            label="query correct",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="gray",
            linestyle="",
            markersize=8,
            markeredgecolor="black",
            label="query wrong",
        ),
    ]
    class_legend = fig.legend(
        handles=class_handles,
        title="Color = true class",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.975),
        frameon=False,
    )
    fig.add_artist(class_legend)
    fig.legend(
        handles=split_handles,
        title="Shape = split / outcome",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.71),
        frameon=False,
    )
    fig.text(
        0.995,
        0.58,
        "Top row: raw input PCA\nBottom row: adapted feature PCA",
        ha="right",
        va="top",
        fontsize=9,
    )
    fig.suptitle(
        (
            f"Held-out task PCA | backend={args.backend} | mode={args.mode} | "
            f"T={args.seq_len} | signal={args.num_signal_positions}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.84, left=0.08, right=0.82)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"phase12_task_pca_{args.backend}_{args.mode}_"
        f"T{args.seq_len}_S{args.num_signal_positions}_"
        f"train{args.outer_steps}_task{args.task_seed}"
    )
    out_path = out_dir / f"{tag}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    metrics_path = out_dir / f"{tag}_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["inner_step", "query_acc", "backend", "mode", "seq_len", "num_signal_positions"],
        )
        writer.writeheader()
        for step in plot_steps:
            writer.writerow(
                {
                    "inner_step": step,
                    "query_acc": f"{float(per_step[step]['query_acc']):.6f}",
                    "backend": args.backend,
                    "mode": args.mode,
                    "seq_len": args.seq_len,
                    "num_signal_positions": args.num_signal_positions,
                }
            )
    print(f"wrote {out_path}")
    print(f"wrote {metrics_path}")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Plot summary CSVs vs inner steps.")
    summary_parser.add_argument("summaries", nargs="+", help="Summary CSVs to plot.")
    summary_parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label for a summary CSV; repeat in the same order as summaries.",
    )
    summary_parser.add_argument(
        "--out-dir",
        default="experiments/phase12/figures",
        help="Directory for output figures.",
    )

    pca_parser = subparsers.add_parser(
        "task-pca",
        help="Meta-train a model and plot a held-out task in 2D PCA across inner steps.",
    )
    pca_parser.add_argument("--backend", default="triton_fused_meta_strict")
    pca_parser.add_argument("--mode", choices=MODE_ORDER, default="FULL_HYBRID")
    pca_parser.add_argument("--device", default="auto", help="cuda, cpu, or auto.")
    pca_parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    pca_parser.add_argument("--task-seed", type=int, default=17, help="Held-out task seed.")
    pca_parser.add_argument("--outer-steps", type=int, default=200)
    pca_parser.add_argument("--inner-steps", type=int, default=10)
    pca_parser.add_argument("--plot-steps", default="0,1,2,5,10")
    pca_parser.add_argument("--meta-batch-size", type=int, default=8)
    pca_parser.add_argument("--inner-lr", type=float, default=0.4)
    pca_parser.add_argument("--outer-lr", type=float, default=1e-3)
    pca_parser.add_argument("--seq-len", type=int, default=128)
    pca_parser.add_argument("--num-signal-positions", type=int, default=12)
    pca_parser.add_argument("--meta-every-n-outer", type=int, default=8)
    pca_parser.add_argument("--meta-last-n-inner", type=int, default=2)
    pca_parser.add_argument(
        "--out-dir",
        default="experiments/phase12/figures",
        help="Directory for output figures.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "summary":
        summary_paths = [Path(p) for p in args.summaries]
        labels = args.label or []
        if labels and len(labels) != len(summary_paths):
            raise ValueError("--label must be repeated exactly once per summary CSV")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, summary_path in enumerate(summary_paths):
            label = labels[idx] if labels else _derive_label(summary_path)
            out_path = _plot_summary(summary_path, label, out_dir)
            print(f"wrote {out_path}")
        return

    if args.command == "task-pca":
        _plot_task_pca(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
