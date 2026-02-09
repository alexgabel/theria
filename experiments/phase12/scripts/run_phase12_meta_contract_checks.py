#!/usr/bin/env python3
"""
Meta-contract checks for Triton second-order behavior.

Checks:
1) Meta-gradient cosine similarity against a reference backend for k={2,5,10}.
2) FULL-vs-FO rel_diff trend agreement against reference backend.

Includes a diffusion-like proxy profile (longer sequence, higher noise) in
addition to the default seqcls profile.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

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


PROFILE_MAP: dict[str, dict[str, float | int]] = {
    "seqcls_default": {
        "seq_len": 32,
        "num_signal_positions": 4,
        "noise_std": 1.0,
        "meta_batch_size": 8,
    },
    "diffusion_proxy": {
        "seq_len": 128,
        "num_signal_positions": 12,
        "noise_std": 1.4,
        "meta_batch_size": 4,
    },
}


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else float("nan")


def _std(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def _flatten_grads(grads: list[torch.Tensor | None]) -> torch.Tensor:
    parts = []
    for g in grads:
        if g is None:
            continue
        parts.append(g.reshape(-1))
    if not parts:
        return torch.zeros(1, device="cpu", dtype=torch.float32)
    return torch.cat(parts).float()


def _meta_grad_vector(
    model: Phase10TinyAttentionModel,
    *,
    tasks: list,
    inner_steps: int,
    inner_lr: float,
    fo: bool,
) -> torch.Tensor:
    outer = meta_loss_on_tasks(
        model=model,
        tasks=tasks,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        fo=fo,
        fo_strict=False,
        return_metrics=False,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        outer, params, create_graph=False, retain_graph=False, allow_unused=True
    )
    return _flatten_grads(list(grads)).detach()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach()
    b = b.detach()
    if a.norm().item() == 0.0 or b.norm().item() == 0.0:
        return float("nan")
    return float(F.cosine_similarity(a, b, dim=0).item())


def _rel_diff(full: torch.Tensor, fo: torch.Tensor, eps: float = 1e-9) -> float:
    denom = full.norm().item()
    if denom <= eps:
        return float("nan")
    return float((full - fo).norm().item() / (denom + eps))


def _build_model_with_state(
    *,
    state_dict: dict[str, torch.Tensor],
    backend: str,
    device: torch.device,
) -> Phase10TinyAttentionModel:
    model = Phase10TinyAttentionModel().to(device)
    model.load_state_dict(deepcopy(state_dict))
    set_attention_backend(model, backend)
    return model


def _trend_sign(values: list[float]) -> int:
    if not values:
        return 0
    delta = values[-1] - values[0]
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-backend", type=str, default="reference")
    parser.add_argument("--candidate-backend", type=str, default="triton_fused_meta")
    parser.add_argument("--profiles", type=str, default="seqcls_default,diffusion_proxy")
    parser.add_argument("--inner-steps", type=str, default="2,5,10")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="experiments/phase12/runs/phase12_meta_contract_checks.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="experiments/phase12/runs/phase12_meta_contract_checks_summary.csv",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    profiles = _parse_str_list(args.profiles)
    inner_steps_list = _parse_int_list(args.inner_steps)
    seeds = _parse_int_list(args.seeds)

    rows: list[dict[str, float | int | str]] = []
    total = len(profiles) * len(inner_steps_list) * len(seeds)
    idx = 0

    for profile_name in profiles:
        if profile_name not in PROFILE_MAP:
            raise ValueError(f"Unknown profile: {profile_name}")
        profile = PROFILE_MAP[profile_name]
        seq_len = int(profile["seq_len"])
        num_signal_positions = int(profile["num_signal_positions"])
        noise_std = float(profile["noise_std"])
        meta_batch_size = int(profile["meta_batch_size"])

        for inner_steps in inner_steps_list:
            for seed in seeds:
                idx += 1
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                base_model = Phase10TinyAttentionModel().to(device)
                init_state = deepcopy(base_model.state_dict())

                tasks = [
                    task_sampler(
                        T=seq_len,
                        D=base_model.cfg.d_model,
                        num_signal_positions=num_signal_positions,
                        noise_std=noise_std,
                        device=device,
                    )
                    for _ in range(meta_batch_size)
                ]

                model_ref_full = _build_model_with_state(
                    state_dict=init_state,
                    backend=args.reference_backend,
                    device=device,
                )
                model_ref_fo = _build_model_with_state(
                    state_dict=init_state,
                    backend=args.reference_backend,
                    device=device,
                )
                model_cand_full = _build_model_with_state(
                    state_dict=init_state,
                    backend=args.candidate_backend,
                    device=device,
                )
                model_cand_fo = _build_model_with_state(
                    state_dict=init_state,
                    backend=args.candidate_backend,
                    device=device,
                )

                reset_triton_meta_bwd_counters()
                g_ref_full = _meta_grad_vector(
                    model_ref_full,
                    tasks=tasks,
                    inner_steps=inner_steps,
                    inner_lr=args.inner_lr,
                    fo=False,
                )
                ref_full_counts = get_triton_meta_bwd_counters(reset=True)
                g_ref_fo = _meta_grad_vector(
                    model_ref_fo,
                    tasks=tasks,
                    inner_steps=inner_steps,
                    inner_lr=args.inner_lr,
                    fo=True,
                )
                ref_fo_counts = get_triton_meta_bwd_counters(reset=True)
                g_cand_full = _meta_grad_vector(
                    model_cand_full,
                    tasks=tasks,
                    inner_steps=inner_steps,
                    inner_lr=args.inner_lr,
                    fo=False,
                )
                cand_full_counts = get_triton_meta_bwd_counters(reset=True)
                g_cand_fo = _meta_grad_vector(
                    model_cand_fo,
                    tasks=tasks,
                    inner_steps=inner_steps,
                    inner_lr=args.inner_lr,
                    fo=True,
                )
                cand_fo_counts = get_triton_meta_bwd_counters(reset=True)

                row = {
                    "profile": profile_name,
                    "seed": seed,
                    "inner_steps": inner_steps,
                    "reference_backend": args.reference_backend,
                    "candidate_backend": args.candidate_backend,
                    "seq_len": seq_len,
                    "num_signal_positions": num_signal_positions,
                    "noise_std": noise_std,
                    "meta_batch_size": meta_batch_size,
                    "cosine_full_ref_vs_candidate": _cosine(g_ref_full, g_cand_full),
                    "cosine_fo_ref_vs_candidate": _cosine(g_ref_fo, g_cand_fo),
                    "rel_diff_ref_full_vs_fo": _rel_diff(g_ref_full, g_ref_fo),
                    "rel_diff_candidate_full_vs_fo": _rel_diff(g_cand_full, g_cand_fo),
                    "ref_full_n_fast_bwd": int(ref_full_counts["n_fast_bwd"]),
                    "ref_full_n_meta_bwd": int(ref_full_counts["n_meta_bwd"]),
                    "ref_fo_n_fast_bwd": int(ref_fo_counts["n_fast_bwd"]),
                    "ref_fo_n_meta_bwd": int(ref_fo_counts["n_meta_bwd"]),
                    "cand_full_n_fast_bwd": int(cand_full_counts["n_fast_bwd"]),
                    "cand_full_n_meta_bwd": int(cand_full_counts["n_meta_bwd"]),
                    "cand_fo_n_fast_bwd": int(cand_fo_counts["n_fast_bwd"]),
                    "cand_fo_n_meta_bwd": int(cand_fo_counts["n_meta_bwd"]),
                }
                rows.append(row)
                print(
                    f"[{idx}/{total}] profile={profile_name} k={inner_steps} seed={seed} "
                    f"cos_full={row['cosine_full_ref_vs_candidate']:.4f} "
                    f"rel_ref={row['rel_diff_ref_full_vs_fo']:.4f} "
                    f"rel_cand={row['rel_diff_candidate_full_vs_fo']:.4f} "
                    f"cand_full(fast={row['cand_full_n_fast_bwd']},meta={row['cand_full_n_meta_bwd']}) "
                    f"cand_fo(fast={row['cand_fo_n_fast_bwd']},meta={row['cand_fo_n_meta_bwd']})"
                )

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "seed",
                "inner_steps",
                "reference_backend",
                "candidate_backend",
                "seq_len",
                "num_signal_positions",
                "noise_std",
                "meta_batch_size",
                "cosine_full_ref_vs_candidate",
                "cosine_fo_ref_vs_candidate",
                "rel_diff_ref_full_vs_fo",
                "rel_diff_candidate_full_vs_fo",
                "ref_full_n_fast_bwd",
                "ref_full_n_meta_bwd",
                "ref_fo_n_fast_bwd",
                "ref_fo_n_meta_bwd",
                "cand_full_n_fast_bwd",
                "cand_full_n_meta_bwd",
                "cand_fo_n_fast_bwd",
                "cand_fo_n_meta_bwd",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    summary_rows: list[dict[str, float | int | str]] = []
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["profile"]), int(row["inner_steps"]))].append(row)

    profile_rel_series: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: {"ref": [], "cand": []}
    )
    for (profile_name, inner_steps), items in sorted(grouped.items()):
        cos_full = _mean([float(x["cosine_full_ref_vs_candidate"]) for x in items])
        cos_fo = _mean([float(x["cosine_fo_ref_vs_candidate"]) for x in items])
        rel_ref = _mean([float(x["rel_diff_ref_full_vs_fo"]) for x in items])
        rel_cand = _mean([float(x["rel_diff_candidate_full_vs_fo"]) for x in items])
        profile_rel_series[profile_name]["ref"].append((inner_steps, rel_ref))
        profile_rel_series[profile_name]["cand"].append((inner_steps, rel_cand))
        summary_rows.append(
            {
                "profile": profile_name,
                "inner_steps": inner_steps,
                "n_runs": len(items),
                "cosine_full_mean": cos_full,
                "cosine_full_std": _std([float(x["cosine_full_ref_vs_candidate"]) for x in items]),
                "cosine_fo_mean": cos_fo,
                "cosine_fo_std": _std([float(x["cosine_fo_ref_vs_candidate"]) for x in items]),
                "rel_diff_ref_mean": rel_ref,
                "rel_diff_ref_std": _std([float(x["rel_diff_ref_full_vs_fo"]) for x in items]),
                "rel_diff_candidate_mean": rel_cand,
                "rel_diff_candidate_std": _std([float(x["rel_diff_candidate_full_vs_fo"]) for x in items]),
                "cand_full_n_fast_bwd_mean": _mean([float(x["cand_full_n_fast_bwd"]) for x in items]),
                "cand_full_n_meta_bwd_mean": _mean([float(x["cand_full_n_meta_bwd"]) for x in items]),
                "cand_fo_n_fast_bwd_mean": _mean([float(x["cand_fo_n_fast_bwd"]) for x in items]),
                "cand_fo_n_meta_bwd_mean": _mean([float(x["cand_fo_n_meta_bwd"]) for x in items]),
                "trend_sign_match": "",
            }
        )

    trend_rows: dict[str, int] = {}
    for profile_name, data in profile_rel_series.items():
        ref_series = [v for _, v in sorted(data["ref"], key=lambda x: x[0])]
        cand_series = [v for _, v in sorted(data["cand"], key=lambda x: x[0])]
        trend_rows[profile_name] = int(_trend_sign(ref_series) == _trend_sign(cand_series))

    for row in summary_rows:
        row["trend_sign_match"] = trend_rows.get(str(row["profile"]), 0)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "inner_steps",
                "n_runs",
                "cosine_full_mean",
                "cosine_full_std",
                "cosine_fo_mean",
                "cosine_fo_std",
                "rel_diff_ref_mean",
                "rel_diff_ref_std",
                "rel_diff_candidate_mean",
                "rel_diff_candidate_std",
                "cand_full_n_fast_bwd_mean",
                "cand_full_n_meta_bwd_mean",
                "cand_fo_n_fast_bwd_mean",
                "cand_fo_n_meta_bwd_mean",
                "trend_sign_match",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
