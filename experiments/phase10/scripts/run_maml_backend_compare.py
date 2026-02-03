"""
Phase 10 experiment: MAML backend comparison.

Compares:
- Attention backend: reference vs custom vs triton_fused
- Meta-learning mode: full MAML vs FO-MAML

Logs:
- outer loss
- gradient norm

Design rules:
- No changes to theria/ library code
- No kernel tuning
- Minimal logging (stdout / CSV-ready)
"""

from __future__ import annotations

import argparse
import math
import contextlib
import csv
from pathlib import Path
import time
import torch
import torch.nn as nn

from theria.attention.custom import sdpa_custom
from experiments.phase11.scripts.run_bad_backend_diagnostics import _attention_second_order_ok
from theria.models.tiny_attention import TinyAttentionConfig
from theria.tasks.synthetic_seqcls import task_sampler
from theria.maml.loops import meta_loss_on_tasks


# -----------------------------
# Utilities
# -----------------------------

def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


class Phase10TinyAttentionModel(nn.Module):
    """
    Minimal single-head attention classifier with a swappable backend.

    Uses token 0 as a CLS-like query. Attention is delegated to sdpa_custom
    with H=1 and Tq=1.
    """
    def __init__(self, cfg: TinyAttentionConfig | None = None, backend: str = "reference"):
        super().__init__()
        self.cfg = cfg or TinyAttentionConfig()
        self.backend = backend

        D = self.cfg.d_model
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.classifier = nn.Linear(D, self.cfg.num_classes)

    def set_attention_backend(self, backend: str) -> None:
        self.backend = backend

    def _resolve_backend(self) -> str:
        if self.backend == "triton_fused":
            return "triton_full_fused"
        return self.backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        q = self.q_proj(x[:, :1, :])  # (B, 1, D)
        k = self.k_proj(x)            # (B, T, D)
        v = self.v_proj(x)            # (B, T, D)

        # Add head dim for sdpa_custom: (B, H=1, Tq/Tk, D)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        h = sdpa_custom(q, k, v, backend=self._resolve_backend())  # (B, 1, 1, D)
        h = h.squeeze(1).squeeze(1)  # (B, D)
        return self.classifier(h)


def set_attention_backend(model: Phase10TinyAttentionModel, backend: str):
    """Switch attention backend in the model."""
    if backend not in {"reference", "custom", "triton_fused"}:
        raise ValueError(f"Unknown backend: {backend}")
    model.set_attention_backend(backend)


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(
    backend: str,
    fo: bool,
    fo_strict: bool,
    device: torch.device,
    seed: int,
    steps: int,
    inner_steps: int,
    inner_lr: float | None,
    neg_control_detach_inner: bool,
    capture_metrics: bool = False,
):
    if inner_steps > 0 and inner_lr is None:
        raise ValueError("inner_lr must be set when inner_steps > 0")
    torch.manual_seed(seed)

    model = Phase10TinyAttentionModel().to(device)
    set_attention_backend(model, backend)

    if fo_strict:
        mode = "FO_STRICT"
    elif fo:
        mode = "FO"
    else:
        mode = "FULL"

    create_graph = mode == "FULL"
    detach_phi = mode == "FO_STRICT"
    fo_effective = mode != "FULL"
    neg_control_detach_effective = neg_control_detach_inner or (not create_graph)

    for step in range(steps):
        model.zero_grad(set_to_none=True)

        task = task_sampler(device=device)

        # Compute BOTH full and FO meta-losses on the same task for diagnostics.
        outer_full = meta_loss_on_tasks(
            model=model,
            tasks=[task],
            fo=False,
            fo_strict=False,
            inner_steps=inner_steps,
            inner_lr=inner_lr if inner_lr is not None else None,
            return_metrics=capture_metrics,
        )
        if capture_metrics:
            outer_loss_full, metrics_full = outer_full
        else:
            outer_loss_full = outer_full

        outer_fo = meta_loss_on_tasks(
            model=model,
            tasks=[task],
            fo=True,
            fo_strict=detach_phi,
            inner_steps=inner_steps,
            inner_lr=inner_lr if inner_lr is not None else None,
            return_metrics=capture_metrics,
        )
        if capture_metrics:
            outer_loss_fo, metrics_fo = outer_fo
        else:
            outer_loss_fo = outer_fo

        # The loss actually used for training follows the selected mode.
        outer_loss = outer_loss_fo if fo_effective else outer_loss_full

        # Diagnostic: TRUE second-order path probe
        # "Can we differentiate the meta-gradient again w.r.t. initial params?"
        params = [p for p in model.parameters() if p.requires_grad]
        second_order_ok = False
        try:
            g1 = torch.autograd.grad(
                outer_loss,
                params,
                retain_graph=True,
                create_graph=not neg_control_detach_effective,
                allow_unused=True,
            )
            g1_scalar = None
            for gi in g1:
                if gi is None:
                    continue
                term = gi.pow(2).sum()
                g1_scalar = term if g1_scalar is None else (g1_scalar + term)

            if g1_scalar is None:
                second_order_ok = False
            else:
                g2 = torch.autograd.grad(
                    g1_scalar,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                second_order_ok = any(gj is not None for gj in g2)
        except RuntimeError:
            second_order_ok = False

        # Diagnostic: FULL vs FO meta-gradient relative difference.
        eps = 1e-9
        def _vec(grads):
            return torch.cat([g.flatten() for g in grads if g is not None]) if any(g is not None for g in grads) else None
        grads_full = torch.autograd.grad(
            outer_loss_full, params, retain_graph=True, allow_unused=True
        )
        grads_fo = torch.autograd.grad(
            outer_loss_fo, params, retain_graph=True, allow_unused=True
        )
        v_full = _vec(grads_full)
        v_fo = _vec(grads_fo)
        rel_diff = float("nan")
        if v_full is not None and v_full.norm().item() > eps and v_fo is not None:
            rel_diff = (v_full - v_fo).norm().item() / (v_full.norm().item() + eps)

        outer_loss.backward()
        gnorm = grad_norm(model.parameters())

        print(
            f"step={step:03d} "
            f"backend={backend:<12} "
            f"mode={mode:<9} "
            f"loss={outer_loss.item():.6f} "
            f"grad_norm={gnorm:.6f} "
            f"second_order_path={second_order_ok} "
            f"rel_diff_full_vs_fo={rel_diff:.4f}"
        )
    if capture_metrics:
        # Attention-local probe (once per run)
        attn_signal_present = _attention_second_order_ok(sdpa_custom, device=device)
        grad_norms = {}
        for name, p in model.named_parameters():
            if p.grad is not None and any(k in name for k in ["q_proj", "k_proj", "v_proj"]):
                grad_norms[name] = p.grad.detach().abs().sum().item()

        chosen_metrics = metrics_fo if fo_effective else metrics_full
        return {
            "final_loss": outer_loss.item(),
            "final_acc": chosen_metrics.get("post_adapt_acc", None) if capture_metrics else None,
            "attn_signal_present": attn_signal_present,
            "grad_norm_q_proj": grad_norms.get("q_proj.weight", 0.0),
            "grad_norm_k_proj": grad_norms.get("k_proj.weight", 0.0),
            "grad_norm_v_proj": grad_norms.get("v_proj.weight", 0.0),
            "steps": steps,
        }


# -----------------------------
# Benchmark mode
# -----------------------------

def _run_one_step(
    model: Phase10TinyAttentionModel,
    *,
    fo_effective: bool,
    inner_steps: int,
    inner_lr: float,
    device: torch.device,
) -> None:
    model.zero_grad(set_to_none=True)
    task = task_sampler(device=device)
    outer_loss = meta_loss_on_tasks(
        model=model,
        tasks=[task],
        fo=fo_effective,
        fo_strict=fo_effective,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
    )
    outer_loss.backward()


def run_bench(
    backend: str,
    fo: bool,
    fo_strict: bool,
    device: torch.device,
    seed: int,
    inner_steps: int,
    inner_lr: float,
    warmup_steps: int,
    bench_steps: int,
    csv_out: str | None,
) -> None:
    torch.manual_seed(seed)
    model = Phase10TinyAttentionModel().to(device)
    set_attention_backend(model, backend)
    fo_effective = fo or fo_strict
    mode = "FO_STR" if fo_strict else ("FO" if fo else "FULL")

    # Warmup: includes Triton compile and CUDA context initialization.
    for _ in range(warmup_steps):
        _run_one_step(
            model,
            fo_effective=fo_effective,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            device=device,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(bench_steps):
            _run_one_step(
                model,
                fo_effective=fo_effective,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                device=device,
            )
        end.record()
        torch.cuda.synchronize(device)
        total_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(bench_steps):
            _run_one_step(
                model,
                fo_effective=fo_effective,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                device=device,
            )
        total_ms = (time.perf_counter() - t0) * 1000.0

    ms_per_step = total_ms / max(bench_steps, 1)
    print(
        f"bench backend={backend} mode={mode} device={device} "
        f"warmup_steps={warmup_steps} bench_steps={bench_steps} "
        f"total_ms={total_ms:.3f} ms_per_step={ms_per_step:.3f}"
    )

    if csv_out:
        path = Path(csv_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "backend",
                        "mode",
                        "device",
                        "inner_steps",
                        "inner_lr",
                        "warmup_steps",
                        "bench_steps",
                        "total_ms",
                        "ms_per_step",
                    ]
                )
            w.writerow(
                [
                    backend,
                    mode,
                    str(device),
                    inner_steps,
                    inner_lr,
                    warmup_steps,
                    bench_steps,
                    f"{total_ms:.6f}",
                    f"{ms_per_step:.6f}",
                ]
            )


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True,
                        choices=["reference", "custom", "triton_fused"])
    parser.add_argument("--fo", action="store_true",
                        help="Use first-order MAML (FO-MAML)")
    parser.add_argument("--fo-strict", action="store_true",
                        help="Detaches inner gradients (strong FO ablation)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=None,
        help="Inner-loop learning rate (overrides default)",
    )
    parser.add_argument(
        "--neg-control-detach-inner",
        action="store_true",
        help="Negative control: compute first-order meta-grad with create_graph=False to break second-order path probe",
    )
    parser.add_argument("--bench", action="store_true",
                        help="Run timing benchmark mode (no per-step diagnostics)")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup iterations before timed benchmark loop")
    parser.add_argument("--bench-steps", type=int, default=20,
                        help="Timed iterations for benchmark mode")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional CSV output path for benchmark summary row")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)
    if args.bench:
        if args.inner_lr is None:
            raise ValueError("inner_lr must be set in --bench mode")
        run_bench(
            backend=args.backend,
            fo=args.fo,
            fo_strict=args.fo_strict,
            device=device,
            seed=args.seed,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            warmup_steps=args.warmup_steps,
            bench_steps=args.bench_steps,
            csv_out=args.csv_out,
        )
    else:
        run_experiment(
            backend=args.backend,
            fo=args.fo,
            fo_strict=args.fo_strict,
            device=device,
            seed=args.seed,
            steps=args.steps,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            neg_control_detach_inner=args.neg_control_detach_inner,
        )


if __name__ == "__main__":
    main()
