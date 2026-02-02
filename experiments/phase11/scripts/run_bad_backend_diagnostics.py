"""
Phase 11 diagnostics runner for intentionally broken attention wrappers.

This script runs Phase-10 style diagnostics and classifies wrappers as:
- OK: run completes and metrics are produced
- HARD_FAIL: run crashes with an autograd/runtime error
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Callable

import torch

# Allow direct script execution from repo root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase10.scripts import run_maml_backend_compare as p10
from bad_backends import (
    checkpoint_attention,
    checkpoint_no_grad,
    checkpoint_detach_recompute,
    recompute_logits_no_grad_sdpa,
    backward_detach_logits_sim,
    detach_attention_output,
    detach_q_input,
    detach_k_input,
    detach_v_input,
    detach_q_input_strict,
    detach_k_input_strict,
    detach_v_input_strict,
    no_grad_attention,
    once_differentiable_sim,
    stats_detach_logits_sdpa,
    stats_detach_softmax_output_sdpa,
)


Wrapper = Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]


def _vec(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor | None:
    if not any(g is not None for g in grads):
        return None
    return torch.cat([g.flatten() for g in grads if g is not None])


def _attention_second_order_ok(
    sdpa_fn: Callable[..., torch.Tensor],
    *,
    device: torch.device,
    eps: float = 1e-12,
) -> bool:
    """
    Probe gradgrad strictly through SDPA inputs (Q/K/V) to avoid classifier leakage.
    """
    try:
        B, H, T, D = 1, 1, 2, 4
        q = torch.randn(B, H, 1, D, device=device, requires_grad=True)
        k = torch.randn(B, H, T, D, device=device, requires_grad=True)
        v = torch.randn(B, H, T, D, device=device, requires_grad=True)
        out = sdpa_fn(q, k, v)
        loss = out.sum()
        g1 = torch.autograd.grad(
            loss, (q, k, v), retain_graph=True, create_graph=True, allow_unused=True
        )
        g1_scalar = None
        for gi in g1:
            if gi is None:
                continue
            term = gi.pow(2).sum()
            g1_scalar = term if g1_scalar is None else (g1_scalar + term)
        if g1_scalar is None:
            return False
        g2 = torch.autograd.grad(g1_scalar, (q, k, v), allow_unused=True)
        def _nz(g):
            return g is not None and g.detach().abs().sum().item() > eps
        return any(_nz(g) for g in g2)
    except Exception:
        return False


def _run_one_step(
    *,
    model: p10.Phase10TinyAttentionModel,
    inner_steps: int,
    inner_lr: float,
    device: torch.device,
) -> tuple[bool, bool, bool, float, dict[str, float], dict[str, bool]]:
    model.zero_grad(set_to_none=True)
    task = p10.task_sampler(device=device)

    outer_loss_full = p10.meta_loss_on_tasks(
        model=model,
        tasks=[task],
        fo=False,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
    )
    outer_loss_fo = p10.meta_loss_on_tasks(
        model=model,
        tasks=[task],
        fo=True,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
    )

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_names = [n for n, _ in named_params]
    params = [p for _, p in named_params]
    second_order_ok_global = False
    second_order_ok_attn = False
    second_order_ok_head = False
    eps = 1e-12
    attn_param_names = {"q_proj.weight", "k_proj.weight", "v_proj.weight"}
    try:
        g1 = torch.autograd.grad(
            outer_loss_full,
            params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )
        g1_scalar = None
        for gi in g1:
            if gi is None:
                continue
            term = gi.pow(2).sum()
            g1_scalar = term if g1_scalar is None else (g1_scalar + term)
        if g1_scalar is not None:
            g2 = torch.autograd.grad(
                g1_scalar,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            def _nz(g):
                return g is not None and g.detach().abs().sum().item() > eps
            g2_by_name = {name: grad for name, grad in zip(param_names, g2)}
            second_order_ok_global = any(_nz(gj) for gj in g2)
            second_order_ok_attn = any(
                _nz(g2_by_name.get(name)) for name in attn_param_names
            )
            second_order_ok_head = any(
                _nz(grad)
                for name, grad in g2_by_name.items()
                if name.startswith("classifier.")
            )
    except RuntimeError:
        second_order_ok_global = False
        second_order_ok_attn = False
        second_order_ok_head = False

    grads_full = torch.autograd.grad(
        outer_loss_full, params, retain_graph=True, allow_unused=True
    )
    grads_fo = torch.autograd.grad(
        outer_loss_fo, params, retain_graph=True, allow_unused=True
    )
    v_full = _vec(grads_full)
    v_fo = _vec(grads_fo)
    rel_diff = float("nan")
    eps = 1e-9
    if v_full is not None and v_fo is not None and v_full.norm().item() > eps:
        rel_diff = (v_full - v_fo).norm().item() / (v_full.norm().item() + eps)

    # Mirror the Phase-10 FULL path; this is where once_differentiable_sim hard-fails.
    outer_loss_full.backward()

    # Grad norms for attention projections (optional debugging)
    grad_norms = {}
    grad_zeros = {}
    for name, p in model.named_parameters():
        if "proj" in name and p.grad is not None:
            gn = p.grad.detach().abs().sum().item()
            grad_norms[name] = gn
            grad_zeros[name] = gn < 1e-8

    return (
        second_order_ok_global,
        second_order_ok_attn,
        second_order_ok_head,
        rel_diff,
        grad_norms,
        grad_zeros,
    )


def run_backend(
    *,
    wrapper_name: str,
    wrapper: Wrapper | None,
    attention_backend: str,
    seed: int,
    steps: int,
    inner_steps: int,
    inner_lr: float,
    device: torch.device,
) -> dict[str, str]:
    torch.manual_seed(seed)
    base_sdpa = p10.sdpa_custom
    p10.sdpa_custom = base_sdpa if wrapper is None else wrapper(base_sdpa)

    attn_second_order_ok = _attention_second_order_ok(p10.sdpa_custom, device=device)

    model = p10.Phase10TinyAttentionModel().to(device)
    p10.set_attention_backend(model, attention_backend)
    second_order_flags: list[bool] = []
    second_order_attn_flags: list[bool] = []
    second_order_head_flags: list[bool] = []
    rel_diffs: list[float] = []
    grad_norms_list: list[dict[str, float]] = []
    grad_zero_list: list[dict[str, bool]] = []

    try:
        for _ in range(steps):
            (
                second_order_ok,
                second_order_attn_ok,
                second_order_head_ok,
                rel_diff,
                grad_norms,
                grad_zeros,
            ) = _run_one_step(
                model=model,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                device=device,
            )
            second_order_flags.append(second_order_ok)
            second_order_attn_flags.append(second_order_attn_ok)
            second_order_head_flags.append(second_order_head_ok)
            rel_diffs.append(rel_diff)
            grad_norms_list.append(grad_norms)
            grad_zero_list.append(grad_zeros)
        mean_rel = sum(rel_diffs) / len(rel_diffs) if rel_diffs else float("nan")
        # Aggregate grad norms (mean) for q/k/v if present
        def _mean_norm(key_substr: str):
            vals = []
            for d in grad_norms_list:
                for k, v in d.items():
                    if key_substr in k:
                        vals.append(v)
            return sum(vals) / len(vals) if vals else ""

        grad_q = _mean_norm("q_proj.weight")
        grad_k = _mean_norm("k_proj.weight")
        grad_v = _mean_norm("v_proj.weight")

        row = {
            "wrapper": wrapper_name,
            "attention_backend": attention_backend,
            "status": "OK",
            "second_order_path_all": str(all(second_order_flags)),
            "second_order_path_any": str(any(second_order_flags)),
            "second_order_path_attn_all": str(all(second_order_attn_flags)),
            "second_order_path_attn_any": str(any(second_order_attn_flags)),
            "second_order_path_head_all": str(all(second_order_head_flags)),
            "second_order_path_head_any": str(any(second_order_head_flags)),
            "sdpa_input_gradgrad_ok": str(attn_second_order_ok),
            "rel_diff_mean": f"{mean_rel:.6f}",
            "grad_norm_q_proj": grad_q,
            "grad_norm_k_proj": grad_k,
            "grad_norm_v_proj": grad_v,
            "error": "",
        }
    except Exception as exc:
        msg = str(exc)
        if "not have been used in the graph" in msg:
            status = "HARD_FAIL_UNUSED"
        elif "marked with @once_differentiable" in msg:
            status = "HARD_FAIL_ONCE_DIFF"
        else:
            status = "HARD_FAIL_OTHER"
        row = {
            "wrapper": wrapper_name,
            "attention_backend": attention_backend,
            "status": status,
            "second_order_path_all": "",
            "second_order_path_any": "",
            "second_order_path_attn_all": "",
            "second_order_path_attn_any": "",
            "second_order_path_head_all": "",
            "second_order_path_head_any": "",
            "sdpa_input_gradgrad_ok": str(attn_second_order_ok),
            "rel_diff_mean": "",
            "grad_norm_q_proj": "",
            "grad_norm_k_proj": "",
            "grad_norm_v_proj": "",
            "error": msg,
        }
    finally:
        p10.sdpa_custom = base_sdpa

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="all",
        choices=[
            "all",
            "baseline",
            "detach_attention_output",
            "detach_q_input",
            "detach_k_input",
            "detach_v_input",
            "detach_q_input_strict",
            "detach_k_input_strict",
            "detach_v_input_strict",
            "recompute_logits_no_grad_sdpa",
            "backward_detach_logits_sim",
            "checkpoint_attention",
            "checkpoint_no_grad",
            "checkpoint_detach_recompute",
            "no_grad_attention",
            "once_differentiable_sim",
            "stats_detach_logits_sdpa",
            "stats_detach_softmax_output_sdpa",
        ],
        help="Which wrapper to run; 'all' runs the full Phase-11 taxonomy set.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="reference",
        choices=["reference", "custom", "triton_fused"],
        help="Phase-10 attention backend selection.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=5,
                        help="Inner steps (used if --inner-steps-list not set)")
    parser.add_argument(
        "--inner-steps-list",
        type=str,
        default=None,
        help='Comma-separated inner-step values (e.g., "1,5,10,20"). If set, overrides --inner-steps.',
    )
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="experiments/phase11/runs/bad_backend_diagnostics.csv",
    )
    args = parser.parse_args()

    def _parse_int_list(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    wrappers: dict[str, Wrapper | None] = {
        "baseline": None,
        "detach_attention_output": detach_attention_output,
        "detach_q_input": detach_q_input,
        "detach_k_input": detach_k_input,
        "detach_v_input": detach_v_input,
        "detach_q_input_strict": detach_q_input_strict,
        "detach_k_input_strict": detach_k_input_strict,
        "detach_v_input_strict": detach_v_input_strict,
        "recompute_logits_no_grad_sdpa": recompute_logits_no_grad_sdpa,
        "backward_detach_logits_sim": backward_detach_logits_sim,
        "checkpoint_attention": checkpoint_attention,
        "checkpoint_no_grad": checkpoint_no_grad,
        "checkpoint_detach_recompute": checkpoint_detach_recompute,
        "no_grad_attention": no_grad_attention,
        "once_differentiable_sim": once_differentiable_sim,
        "stats_detach_logits_sdpa": stats_detach_logits_sdpa,
        "stats_detach_softmax_output_sdpa": stats_detach_softmax_output_sdpa,
    }
    selected = list(wrappers.keys()) if args.backend == "all" else [args.backend]

    device = torch.device(args.device)
    inner_steps_list = (
        _parse_int_list(args.inner_steps_list)
        if args.inner_steps_list
        else [args.inner_steps]
    )
    rows = []
    for name in selected:
        for inner_steps in inner_steps_list:
            row = run_backend(
                wrapper_name=name,
                wrapper=wrappers[name],
                attention_backend=args.attention_backend,
                seed=args.seed,
                steps=args.steps,
                inner_steps=inner_steps,
                inner_lr=args.inner_lr,
                device=device,
            )
            rows.append(row)
            print(
                f"wrapper={row['wrapper']} attn={row['attention_backend']} k={inner_steps} status={row['status']} "
                f"second_order_path_any={row['second_order_path_any'] or 'NA'} "
                f"second_order_path_attn_any={row.get('second_order_path_attn_any', 'NA') or 'NA'} "
                f"second_order_path_head_any={row.get('second_order_path_head_any', 'NA') or 'NA'} "
                f"sdpa_input_gradgrad_ok={row.get('sdpa_input_gradgrad_ok','NA')} "
                f"rel_diff_mean={row['rel_diff_mean'] or 'NA'}"
            )
            if row["error"]:
                print(f"  error={row['error']}")

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wrapper",
                "attention_backend",
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
