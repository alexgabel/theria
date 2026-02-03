"""
Phase 12 behavior runner (outer-loop aggregation).

Patch P3 focus:
- Track per-step outer loss / post-adaptation accuracy
- Aggregate final metrics from the last 20 outer steps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Literal

import torch

# Allow direct script execution from repo root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase10.scripts.run_maml_backend_compare import (
    Phase10TinyAttentionModel,
    set_attention_backend,
)
from theria.attention.custom import sdpa_custom
from theria.maml.loops import meta_loss_on_tasks
from theria.tasks.synthetic_seqcls import task_sampler
from experiments.phase11.scripts.run_bad_backend_diagnostics import _attention_second_order_ok


Mode = Literal["FULL", "FO", "FO_STRICT"]


def _mean_last(values: list[float], n: int) -> float:
    if not values:
        return float("nan")
    take = values[-min(n, len(values)) :]
    return float(sum(take) / len(take))


def _mean_first(values: list[float], n: int) -> float:
    if not values:
        return float("nan")
    take = values[: min(n, len(values))]
    return float(sum(take) / len(take))


def run_behavior(
    *,
    attention_backend: str,
    mode: Mode,
    seed: int,
    outer_steps: int,
    meta_batch_size: int,
    inner_steps: int,
    inner_lr: float,
    outer_lr: float,
    device: torch.device,
    autocast_enabled: bool = False,
    grad_eps: float | None = None,
    rel_diff_probe: bool = True,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Phase10TinyAttentionModel().to(device)
    set_attention_backend(model, attention_backend)
    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    dtype_name = str(next(model.parameters()).dtype).replace("torch.", "")
    if autocast_enabled and device.type == "cuda":
        compute_dtype = "fp16"
    elif autocast_enabled:
        compute_dtype = "bf16"
    else:
        compute_dtype = "fp32"

    fo = mode != "FULL"
    fo_strict = mode == "FO_STRICT"
    cast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    if grad_eps is None:
        grad_eps = 1e-6 if (autocast_enabled and device.type == "cuda") else 1e-8

    outer_losses: list[float] = []
    outer_accs: list[float] = []
    q_grad_norms: list[float] = []
    k_grad_norms: list[float] = []
    v_grad_norms: list[float] = []
    peak_cuda_mem_bytes: int = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    backend_map = {
        "reference": "reference",
        "custom": "custom",
        "triton_fused": "triton_full_fused",
    }
    probe_backend = backend_map[attention_backend]

    def sdpa_for_probe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return sdpa_custom(q, k, v, backend=probe_backend)

    sdpa_input_gradgrad_ok = _attention_second_order_ok(
        sdpa_fn=sdpa_for_probe,
        device=device,
    )
    t0 = time.perf_counter()

    for _ in range(outer_steps):
        optimizer.zero_grad(set_to_none=True)
        tasks = [task_sampler(device=device) for _ in range(meta_batch_size)]
        # Keep explicit autocast flag in logs even when fixed for this phase.
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=cast_dtype,
        ):
            outer_loss, metrics = meta_loss_on_tasks(
                model=model,
                tasks=tasks,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                fo=fo,
                fo_strict=fo_strict,
                return_metrics=True,
            )
        outer_loss.backward()

        qg = model.q_proj.weight.grad
        kg = model.k_proj.weight.grad
        vg = model.v_proj.weight.grad
        q_grad_norms.append(float(qg.norm().item()) if qg is not None else 0.0)
        k_grad_norms.append(float(kg.norm().item()) if kg is not None else 0.0)
        v_grad_norms.append(float(vg.norm().item()) if vg is not None else 0.0)

        optimizer.step()

        outer_losses.append(float(outer_loss.item()))
        outer_accs.append(float(metrics["post_adapt_acc"]))

    final_loss = _mean_last(outer_losses, 20)
    final_acc = _mean_last(outer_accs, 20)
    attn_grad_norm_q = _mean_last(q_grad_norms, 20)
    attn_grad_norm_k = _mean_last(k_grad_norms, 20)
    attn_grad_norm_v = _mean_last(v_grad_norms, 20)
    attn_grad_present = (
        (attn_grad_norm_q > grad_eps)
        or (attn_grad_norm_k > grad_eps)
        or (attn_grad_norm_v > grad_eps)
    )
    convergence_delta = _mean_first(outer_losses, 20) - _mean_last(outer_losses, 20)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_mem_bytes = int(torch.cuda.max_memory_allocated(device))
    wall_time_total_s = time.perf_counter() - t0
    mean_outer_step_time_s = wall_time_total_s / max(outer_steps, 1)

    # Sparse rel_diff probe (once per run): FULL vs FO on one fresh task
    rel_diff_probe_val = float("nan")
    if rel_diff_probe:
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=cast_dtype,
        ):
            probe_task = task_sampler(device=device)
            params = [p for p in model.parameters() if p.requires_grad]
            outer_full = meta_loss_on_tasks(
                model=model,
                tasks=[probe_task],
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                fo=False,
                fo_strict=False,
                return_metrics=False,
            )
            outer_fo = meta_loss_on_tasks(
                model=model,
                tasks=[probe_task],
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                fo=True,
                fo_strict=False,
                return_metrics=False,
            )
            eps_probe = 1e-9
            grads_full = torch.autograd.grad(outer_full, params, retain_graph=False, allow_unused=True)
            grads_fo = torch.autograd.grad(outer_fo, params, retain_graph=False, allow_unused=True)
            def _vec(grads):
                return torch.cat([g.flatten() for g in grads if g is not None]) if any(g is not None for g in grads) else None
            v_full = _vec(grads_full)
            v_fo = _vec(grads_fo)
            if v_full is not None and v_fo is not None and v_full.norm().item() > eps_probe:
                rel_diff_probe_val = (v_full - v_fo).norm().item() / (v_full.norm().item() + eps_probe)

    return {
        "backend": attention_backend,
        "mode": mode,
        "seed": seed,
        "outer_steps": outer_steps,
        "meta_batch": meta_batch_size,
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
        "outer_lr": outer_lr,
        "dtype": dtype_name,
        "compute_dtype": compute_dtype,
        "autocast": int(bool(autocast_enabled)),
        "final_loss": final_loss,
        "final_acc": final_acc,
        "attn_grad_norm_q": attn_grad_norm_q,
        "attn_grad_norm_k": attn_grad_norm_k,
        "attn_grad_norm_v": attn_grad_norm_v,
        "attn_grad_present": str(bool(attn_grad_present)),
        "sdpa_input_gradgrad_ok": str(bool(sdpa_input_gradgrad_ok)),
        "rel_diff_probe": rel_diff_probe_val,
        "convergence_delta": convergence_delta,
        "wall_time_total_s": wall_time_total_s,
        "mean_outer_step_time_s": mean_outer_step_time_s,
        "peak_cuda_mem_bytes": peak_cuda_mem_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=["reference", "custom", "triton_fused"])
    parser.add_argument("--mode", type=str, required=True, choices=["FULL", "FO", "FO_STRICT"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-steps", type=int, default=500)
    parser.add_argument("--meta-batch-size", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--autocast", action="store_true", help="Enable torch.autocast during training")
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    try:
        row = run_behavior(
            attention_backend=args.backend,
            mode=args.mode,  # type: ignore[arg-type]
            seed=args.seed,
            outer_steps=args.outer_steps,
            meta_batch_size=args.meta_batch_size,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            outer_lr=args.outer_lr,
            device=torch.device(args.device),
            autocast_enabled=args.autocast,
        )
        status, error = "OK", ""
    except Exception as e:
        row = {
            "backend": args.backend,
            "mode": args.mode,
            "seed": args.seed,
            "outer_steps": args.outer_steps,
            "meta_batch": args.meta_batch_size,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "outer_lr": args.outer_lr,
            "dtype": "NA",
            "compute_dtype": "NA",
            "autocast": int(bool(args.autocast)),
            "final_loss": float("nan"),
            "final_acc": float("nan"),
            "attn_grad_present": "False",
            "attn_grad_norm_q": float("nan"),
            "attn_grad_norm_k": float("nan"),
            "attn_grad_norm_v": float("nan"),
            "sdpa_input_gradgrad_ok": "False",
            "rel_diff_probe": float("nan"),
            "wall_time_total_s": float("nan"),
            "mean_outer_step_time_s": float("nan"),
            "peak_cuda_mem_bytes": 0,
            "convergence_delta": float("nan"),
        }
        status, error = "HARD_FAIL_OTHER", repr(e)

    print(
        f"backend={row['backend']} mode={row['mode']} seed={row['seed']} "
        f"final_loss={row['final_loss']} final_acc={row['final_acc']} "
        f"attn_grad_norm_q={row['attn_grad_norm_q']} "
        f"attn_grad_norm_k={row['attn_grad_norm_k']} "
        f"attn_grad_norm_v={row['attn_grad_norm_v']} "
        f"attn_grad_present={row['attn_grad_present']} "
        f"rel_diff_probe={row.get('rel_diff_probe','NA')} "
        f"convergence_delta={row['convergence_delta']} "
        f"wall_time_total_s={row['wall_time_total_s']} "
        f"mean_outer_step_time_s={row['mean_outer_step_time_s']} "
        f"status={status}"
    )

    if args.csv_out:
        path = Path(args.csv_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "backend",
                    "mode",
                    "inner_steps",
                    "seed",
                    "meta_batch",
                    "outer_lr",
                    "outer_steps",
                    "inner_lr",
                    "dtype",
                    "compute_dtype",
                    "autocast",
                    "final_loss",
                    "final_acc",
                    "attn_grad_present",
                    "attn_grad_norm_q",
                    "attn_grad_norm_k",
                    "attn_grad_norm_v",
                    "sdpa_input_gradgrad_ok",
                    "rel_diff_probe",
                    "wall_time_total_s",
                    "mean_outer_step_time_s",
                    "peak_cuda_mem_bytes",
                    "status",
                    "error",
                    "convergence_delta",
                ],
            )
            if write_header:
                w.writeheader()
            w.writerow({**row, "status": status, "error": error, "convergence_delta": row.get("convergence_delta", float("nan"))})


if __name__ == "__main__":
    main()
