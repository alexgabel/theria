#!/usr/bin/env python3
"""
Deterministic instability reproducer and fused-vs-strict divergence tracer.

Runs one outer-step at a time on identical tasks/model init for:
  - triton_fused_meta
  - triton_fused_meta_strict

Logs first divergence with attention-level diagnostics:
  - logits / softmax stats
  - dQ/dK/dV stats (via q_proj/k_proj/v_proj output grad hooks)
  - projection weight gradient stats
"""

from __future__ import annotations

import argparse
import csv
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from experiments.phase10.scripts.run_maml_backend_compare import (
    Phase10TinyAttentionModel,
    set_attention_backend,
)
from theria.attention.triton_qk import (
    get_triton_meta_bwd_counters,
    reset_triton_meta_bwd_counters,
)
from theria.maml.loops import meta_loss_on_tasks
from theria.tasks.synthetic_seqcls import task_sampler


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float()
    finite_mask = torch.isfinite(xf)
    finite_frac = float(finite_mask.float().mean().item())
    safe = xf[finite_mask]
    if safe.numel() == 0:
        return {
            "absmax": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "l2": float("nan"),
            "finite_frac": 0.0,
        }
    return {
        "absmax": float(safe.abs().max().item()),
        "mean": float(safe.mean().item()),
        "std": float(safe.std(unbiased=False).item()),
        "min": float(safe.min().item()),
        "max": float(safe.max().item()),
        "l2": float(safe.norm().item()),
        "finite_frac": finite_frac,
    }


def _agg_max(entries: list[dict[str, float]], key: str) -> float:
    vals = [float(e[key]) for e in entries if key in e and math.isfinite(float(e[key]))]
    return float(max(vals)) if vals else float("nan")


def _agg_mean(entries: list[dict[str, float]], key: str) -> float:
    vals = [float(e[key]) for e in entries if key in e and math.isfinite(float(e[key]))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


class AttentionProbe:
    def __init__(self, model: Phase10TinyAttentionModel) -> None:
        self.model = model
        self._handles = []
        self._pending_q: list[torch.Tensor] = []
        self.q_entries: list[dict[str, float]] = []
        self.k_entries: list[dict[str, float]] = []
        self.v_entries: list[dict[str, float]] = []
        self.logits_entries: list[dict[str, float]] = []
        self.softmax_entries: list[dict[str, float]] = []

    def clear(self) -> None:
        self._pending_q.clear()
        self.q_entries.clear()
        self.k_entries.clear()
        self.v_entries.clear()
        self.logits_entries.clear()
        self.softmax_entries.clear()

    def attach(self) -> None:
        self.detach()
        self._handles = [
            self.model.q_proj.register_forward_hook(self._q_hook),
            self.model.k_proj.register_forward_hook(self._k_hook),
            self.model.v_proj.register_forward_hook(self._v_hook),
        ]

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _register_grad_hook(self, output: torch.Tensor, entry: dict[str, float]) -> None:
        if not output.requires_grad:
            return

        def _capture_grad(g: torch.Tensor) -> None:
            stats = _tensor_stats(g)
            entry["grad_absmax"] = stats["absmax"]
            entry["grad_l2"] = stats["l2"]
            entry["grad_finite_frac"] = stats["finite_frac"]

        output.register_hook(_capture_grad)

    def _q_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        entry = _tensor_stats(output)
        self._register_grad_hook(output, entry)
        self.q_entries.append(entry)
        self._pending_q.append(output.detach())

    def _k_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        entry = _tensor_stats(output)
        self._register_grad_hook(output, entry)
        self.k_entries.append(entry)
        if self._pending_q:
            q = self._pending_q.pop(0)
            k = output.detach()
            scale = 1.0 / (k.shape[-1] ** 0.5)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            probs = torch.softmax(logits, dim=-1)
            self.logits_entries.append(_tensor_stats(logits))
            self.softmax_entries.append(_tensor_stats(probs))

    def _v_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        entry = _tensor_stats(output)
        self._register_grad_hook(output, entry)
        self.v_entries.append(entry)

    def summary(self) -> dict[str, float]:
        return {
            "logits_absmax": _agg_max(self.logits_entries, "absmax"),
            "logits_mean": _agg_mean(self.logits_entries, "mean"),
            "softmax_min": _agg_mean(self.softmax_entries, "min"),
            "softmax_max": _agg_mean(self.softmax_entries, "max"),
            "softmax_std": _agg_mean(self.softmax_entries, "std"),
            "dq_absmax": _agg_max(self.q_entries, "grad_absmax"),
            "dk_absmax": _agg_max(self.k_entries, "grad_absmax"),
            "dv_absmax": _agg_max(self.v_entries, "grad_absmax"),
            "dq_l2": _agg_mean(self.q_entries, "grad_l2"),
            "dk_l2": _agg_mean(self.k_entries, "grad_l2"),
            "dv_l2": _agg_mean(self.v_entries, "grad_l2"),
        }


def _sample_tasks(
    *,
    seed: int,
    step_idx: int,
    n_tasks: int,
    seq_len: int,
    d_model: int,
    num_signal_positions: int,
    device: torch.device,
) -> list[Any]:
    # Make task stream deterministic per outer-step and independent of backend.
    step_seed = seed * 100_000 + step_idx
    torch.manual_seed(step_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(step_seed)
    return [
        task_sampler(
            T=seq_len,
            D=d_model,
            num_signal_positions=num_signal_positions,
            device=device,
        )
        for _ in range(n_tasks)
    ]


def _proj_grad_stats(model: Phase10TinyAttentionModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in (
        ("q_proj_grad", model.q_proj.weight.grad),
        ("k_proj_grad", model.k_proj.weight.grad),
        ("v_proj_grad", model.v_proj.weight.grad),
    ):
        if p is None:
            out[f"{name}_absmax"] = float("nan")
            out[f"{name}_l2"] = float("nan")
            out[f"{name}_finite_frac"] = 0.0
            continue
        stats = _tensor_stats(p)
        out[f"{name}_absmax"] = stats["absmax"]
        out[f"{name}_l2"] = stats["l2"]
        out[f"{name}_finite_frac"] = stats["finite_frac"]
    return out


def _run_one_step(
    *,
    model: Phase10TinyAttentionModel,
    optimizer: torch.optim.Optimizer,
    probe: AttentionProbe,
    tasks: list[Any],
    mode: str,
    inner_lr: float,
    inner_steps: int,
    meta_every_n_outer: int,
    meta_last_n_inner: int,
    outer_step_idx: int,
) -> dict[str, float | str | int]:
    probe.clear()
    reset_triton_meta_bwd_counters()
    optimizer.zero_grad(set_to_none=True)

    fo_step = mode != "FULL"
    if mode == "FULL_HYBRID":
        fo_step = (outer_step_idx % max(meta_every_n_outer, 1)) != 0
    meta_last_n_inner_step = meta_last_n_inner if not fo_step else 0

    try:
        outer_loss, metrics = meta_loss_on_tasks(
            model=model,
            tasks=tasks,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            fo=fo_step,
            fo_strict=False,
            meta_last_n_inner=meta_last_n_inner_step,
            check_finite=True,
            finite_prefix=f"outer_step={outer_step_idx}",
            return_metrics=True,
        )
        if not torch.isfinite(outer_loss):
            raise RuntimeError(f"NONFINITE outer_loss outer_step={outer_step_idx}")
        outer_loss.backward()
        for pname, p in model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(
                    f"NONFINITE grad[{pname}] outer_step={outer_step_idx}"
                )
        optimizer.step()
        status = "OK"
        error = ""
        final_loss = float(outer_loss.item())
        final_acc = float(metrics["post_adapt_acc"])
    except Exception as e:
        status = "HARD_FAIL_NONFINITE" if "NONFINITE" in repr(e) else "HARD_FAIL_OTHER"
        error = repr(e)
        final_loss = float("nan")
        final_acc = float("nan")

    out = {
        "status": status,
        "error": error,
        "final_loss": final_loss,
        "final_acc": final_acc,
    }
    out.update(probe.summary())
    out.update(_proj_grad_stats(model))
    counts = get_triton_meta_bwd_counters(reset=True)
    out["n_fast_bwd"] = int(counts.get("n_fast_bwd", 0))
    out["n_meta_bwd"] = int(counts.get("n_meta_bwd", 0))
    out["n_fallback_bwd"] = int(counts.get("n_fallback_bwd", 0))
    return out


def _abs_diff(a: float | int | str, b: float | int | str) -> float:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return float("nan")
    if not (math.isfinite(af) and math.isfinite(bf)):
        return float("nan")
    return abs(af - bf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-backend", type=str, default="triton_fused_meta")
    parser.add_argument("--strict-backend", type=str, default="triton_fused_meta_strict")
    parser.add_argument("--mode", type=str, default="FULL", choices=["FULL", "FO", "FULL_HYBRID"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=60)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--meta-every-n-outer", type=int, default=8)
    parser.add_argument("--meta-last-n-inner", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-signal-positions", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--divergence-tol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for loss/grad stat diffs before declaring divergence.",
    )
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    base_model = Phase10TinyAttentionModel().to(device)
    init_state = deepcopy(base_model.state_dict())

    model_fused = Phase10TinyAttentionModel().to(device)
    model_fused.load_state_dict(deepcopy(init_state))
    set_attention_backend(model_fused, args.fused_backend)
    opt_fused = torch.optim.Adam(model_fused.parameters(), lr=args.outer_lr)
    probe_fused = AttentionProbe(model_fused)
    probe_fused.attach()

    model_strict = Phase10TinyAttentionModel().to(device)
    model_strict.load_state_dict(deepcopy(init_state))
    set_attention_backend(model_strict, args.strict_backend)
    opt_strict = torch.optim.Adam(model_strict.parameters(), lr=args.outer_lr)
    probe_strict = AttentionProbe(model_strict)
    probe_strict.attach()

    rows: list[dict[str, float | int | str]] = []
    first_divergence_step = None

    for step_idx in range(args.outer_steps):
        tasks = _sample_tasks(
            seed=args.seed,
            step_idx=step_idx,
            n_tasks=args.meta_batch_size,
            seq_len=args.seq_len,
            d_model=model_fused.cfg.d_model,
            num_signal_positions=args.num_signal_positions,
            device=device,
        )

        fused_row = _run_one_step(
            model=model_fused,
            optimizer=opt_fused,
            probe=probe_fused,
            tasks=tasks,
            mode=args.mode,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            meta_every_n_outer=args.meta_every_n_outer,
            meta_last_n_inner=args.meta_last_n_inner,
            outer_step_idx=step_idx,
        )
        strict_row = _run_one_step(
            model=model_strict,
            optimizer=opt_strict,
            probe=probe_strict,
            tasks=tasks,
            mode=args.mode,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            meta_every_n_outer=args.meta_every_n_outer,
            meta_last_n_inner=args.meta_last_n_inner,
            outer_step_idx=step_idx,
        )

        step_row: dict[str, float | int | str] = {"outer_step": step_idx}
        for key, val in fused_row.items():
            step_row[f"fused_{key}"] = val
        for key, val in strict_row.items():
            step_row[f"strict_{key}"] = val
        for key in (
            "final_loss",
            "final_acc",
            "logits_absmax",
            "softmax_max",
            "softmax_min",
            "dq_l2",
            "dk_l2",
            "dv_l2",
            "q_proj_grad_l2",
            "k_proj_grad_l2",
            "v_proj_grad_l2",
        ):
            step_row[f"abs_diff_{key}"] = _abs_diff(
                fused_row.get(key, float("nan")),
                strict_row.get(key, float("nan")),
            )
        rows.append(step_row)

        fused_status = str(fused_row["status"])
        strict_status = str(strict_row["status"])
        print(
            f"step={step_idx:03d} "
            f"fused(status={fused_status},loss={fused_row['final_loss']},acc={fused_row['final_acc']},fallback={fused_row['n_fallback_bwd']}) "
            f"strict(status={strict_status},loss={strict_row['final_loss']},acc={strict_row['final_acc']}) "
            f"diff(loss)={step_row['abs_diff_final_loss']}"
        )

        diverged = False
        reason = ""
        if fused_status != "OK" or strict_status != "OK":
            diverged = True
            reason = f"status fused={fused_status} strict={strict_status}"
        else:
            candidate_diffs = [
                float(step_row["abs_diff_final_loss"]),
                float(step_row["abs_diff_dk_l2"]),
                float(step_row["abs_diff_k_proj_grad_l2"]),
                float(step_row["abs_diff_softmax_max"]),
            ]
            if any(math.isfinite(v) and v > args.divergence_tol for v in candidate_diffs):
                diverged = True
                reason = (
                    "metric_diff "
                    f"(loss={step_row['abs_diff_final_loss']}, "
                    f"dk_l2={step_row['abs_diff_dk_l2']}, "
                    f"k_proj_grad_l2={step_row['abs_diff_k_proj_grad_l2']})"
                )
        if diverged:
            first_divergence_step = step_idx
            print(f"first_divergence_step={step_idx} reason={reason}")
            print(f"fused_error={fused_row.get('error','')}")
            print(f"strict_error={strict_row.get('error','')}")
            break

    probe_fused.detach()
    probe_strict.detach()

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["outer_step"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_path}")

    if first_divergence_step is None:
        print("no divergence detected within outer_steps")


if __name__ == "__main__":
    main()

