# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Phase 12 MAML 2nd-Order Demo (Theria)
#
# This notebook is a one-stop demo for colleagues who want to use the current
# Phase-12 stack for MAML with second-order attention gradients.
#
# ## A. Stable delivery
# - Accepted stable baseline:
#   - gate tag: `phase12_meta_gate_phase12_stable_regression_20260309_155142`
#   - frontier tag: `phase12_stable_regression_20260309_155142`
# - Correctness-sensitive recommendation:
#   - `triton_fused_meta_strict FULL`
# - Wall-clock-sensitive recommendation:
#   - `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2`
# - Recommended benchmark and regression commands are eager-only.
#
# ## B. Experimental / blocked work
# - `triton_fused_meta` is still experimental and requires explicit opt-in.
# - CUDA-graph acceleration is blocked in the attention capture path and is not
#   part of the recommended workflow.
# - Even isolated attention-side QK capture currently fails in the repo
#   capture-debug workflow.
# - Do not use `CUDA_GRAPH_STATIC=1` in benchmark or regression commands.
# - Do not treat equal-step parity alone as promotion evidence for
#   `triton_fused_meta`.
#
# ## Contributor note
# - Stable path is accepted on eager execution.
# - CUDA-graph work is isolated R&D.
# - Experimental fused-meta is not the default.

# %%
from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path.cwd()
RUNS_DIR = REPO_ROOT / "experiments/phase12/runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CmdResult:
    returncode: int
    command: str


def run_cmd(cmd: str, *, env: dict[str, str] | None = None, check: bool = True) -> CmdResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print(f"$ {cmd}")
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")
    return CmdResult(returncode=proc.returncode, command=cmd)


def latest_file(pattern: str) -> Path | None:
    paths = [Path(p) for p in glob.glob(pattern)]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def print_top_frontier(summary_path: Path, *, top_n: int = 8) -> None:
    rows = read_csv_rows(summary_path)
    good = []
    for r in rows:
        acc = r.get("final_acc_mean", "nan")
        try:
            acc_f = float(acc)
        except ValueError:
            continue
        if acc_f != acc_f:  # NaN
            continue
        good.append((acc_f, r))
    good.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTop {min(top_n, len(good))} rows from: {summary_path}")
    print("backend | mode | k | final_acc_mean | outer_steps_mean")
    for acc_f, r in good[:top_n]:
        print(
            f"{r['backend']} | {r['mode']} | {r['inner_steps']} | "
            f"{acc_f:.6f} | {r.get('outer_steps_mean', 'NA')}"
        )


def print_failures(per_run_csv: Path, *, max_rows: int = 10) -> None:
    rows = read_csv_rows(per_run_csv)
    bad = [r for r in rows if r.get("status", "OK") != "OK"]
    print(f"\nFailures in {per_run_csv}: {len(bad)}")
    for r in bad[:max_rows]:
        print(
            f"{r.get('backend')} {r.get('mode')} k={r.get('inner_steps')} "
            f"seed={r.get('seed')} status={r.get('status')} error={r.get('error')}"
        )


# %% [markdown]
# ## 1) Environment check
# Run this once from repo root. The notebook assumes `PYTHONPATH=.` and that
# your CUDA env is ready (`conda activate theria-gpu` in shell before launch).
# The canonical status page is `docs/STATUS.md`.
# The canonical workflow page is `experiments/phase12/README.md`.

# %%
print("Repo root:", REPO_ROOT)
print("Python:", sys.executable)


# %% [markdown]
# ## 2) Start here (accepted eager path)
# Copy-paste commands for the current supported modes:
#
# ```bash
# # Correctness-sensitive use
# PYTHONPATH=. python experiments/phase12/scripts/run_phase12_behavior.py \
#   --backend triton_fused_meta_strict \
#   --mode FULL \
#   --device cuda
#
# # Wall-clock-sensitive use
# PYTHONPATH=. python experiments/phase12/scripts/run_phase12_behavior.py \
#   --backend triton_fused_meta_strict \
#   --mode FULL_HYBRID \
#   --meta-every-n-outer 8 \
#   --meta-last-n-inner 2 \
#   --device cuda
# ```
#
# ## 3) Recommended default benchmark flow (stable baseline, eager only)
# This runs:
# 1. Meta-contract gate
# 2. Equal-step + equal-time frontier
#
# Defaults are already pinned to strict baseline:
# - backend: `triton_fused_meta_strict`
# - modes: `FULL,FULL_HYBRID`
# - practical setting: `meta_every_n_outer=8`, `meta_last_n_inner=2`
# - leave `CUDA_GRAPH_STATIC` unset

# %%
RUN_BASELINE = False
baseline_tag = f"phase12_notebook_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if RUN_BASELINE:
    run_cmd(
        "bash experiments/phase12/scripts/run_phase12_maml_gate_and_frontier.sh",
        env={
            "TAG": baseline_tag,
            "DEVICE": "cuda",
            "INNER_STEPS": "2,5,10",
            "SEEDS": "0,1,2,3,4",
            "META_EVERY_N_OUTER": "8",
            "META_LAST_N_INNER": "2",
            "FULL_OUTER_LR": "1e-3",
            "FULL_HYBRID_OUTER_LR": "1e-3",
        },
    )


# %% [markdown]
# ## 4) Inspect latest baseline outputs
# If you ran section 2, this cell finds the latest summaries and prints top rows.

# %%
latest_equal_time = latest_file("experiments/phase12/runs/phase12_maml_seqcls_equal_time_*_summary.csv")
latest_equal_step = latest_file("experiments/phase12/runs/phase12_maml_seqcls_equal_step_*_summary.csv")

if latest_equal_time:
    print_top_frontier(latest_equal_time)
else:
    print("No equal-time summary found yet.")

if latest_equal_step:
    print_top_frontier(latest_equal_step)
else:
    print("No equal-step summary found yet.")


# %% [markdown]
# ## 5) Diffusion-like fused-meta canary
# This is the primary instability canary for the experimental fused meta path.
# It runs the known bad diffusion-like config with:
# - fixed `CUBLAS_WORKSPACE_CONFIG`
# - `THERIA_TRITON_META_ENABLE_FALLBACK=0`
# - fixed failing seeds
# and fails if fused-meta diverges from strict.

# %%
RUN_FUSED_META_CANARY = False
canary_tag = f"phase12_fused_meta_canary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if RUN_FUSED_META_CANARY:
    run_cmd(
        "bash experiments/phase12/scripts/run_phase12_fused_meta_canary.sh",
        env={
            "TAG": canary_tag,
            "DEVICE": "cuda",
        },
    )


# %% [markdown]
# ## 6) Inspect latest canary output
# If the canary fails, the script exits non-zero and the newest CSV points to
# the first divergence step.

# %%
latest_canary = latest_file("experiments/phase12/runs/phase12_fused_meta_canary_*_seed*.csv")
if latest_canary:
    print_failures(latest_canary)
else:
    print("No fused-meta canary CSV found yet.")


# %% [markdown]
# ## 7) Contributor checklist and what to share with colleagues
#
# ### Stable-path contributor checklist
# 1. `python experiments/phase12/scripts/show_phase12_stable_baseline.py`
# 2. `bash experiments/phase12/scripts/run_phase12_stable_frontier_regression.sh`
# 3. Confirm equal-step `FULL` semantics still pass
# 4. Confirm equal-time `FULL_HYBRID` remains on the FO frontier
# 5. Leave `CUDA_GRAPH_STATIC` unset
#
# ### Recommended usage now
# - Correctness-sensitive FULL runs:
#   - backend: `triton_fused_meta_strict`
#   - mode: `FULL`
# - Practical budget-aware runs:
#   - backend: `triton_fused_meta_strict`
#   - mode: `FULL_HYBRID`
#   - `meta_every_n_outer=8`
#   - `meta_last_n_inner=2`
#
# ### Experimental path
# - `triton_fused_meta` requires explicit opt-in:
#   - shell env: `ALLOW_EXPERIMENTAL_BACKENDS=1`
#   - CLI flag: `--allow-experimental-backends`
#
# ### Blocked work
# - CUDA-graph acceleration is not accepted and should stay off the benchmark path.
# - Do not set `CUDA_GRAPH_STATIC=1` in recommended benchmark/regression commands.
# - If CUDA-graph work resumes, start from the isolated capture-debug repro, not
#   from the frontier runners.
#
# ### Reporting checklist
# - Stable: `triton_fused_meta_strict FULL`
# - Practical: `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2`
# - Experimental: `triton_fused_meta`
# - Blocked: CUDA-graph acceleration for the stable practical path
