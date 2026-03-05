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
# ## Deliverables (current state)
# - Stable correctness backend: `triton_fused_meta_strict` in `FULL`.
# - Stable practical backend: `triton_fused_meta_strict` in `FULL_HYBRID`
#   with `meta_every_n_outer=8`.
# - Non-finite failure handling is enabled in frontier runs.
# - Meta-contract gate + frontier scripts are available as default entrypoints.
#
# ## Future work (not solved yet)
# - `triton_fused_meta` is experimental and can be numerically unstable on
#   diffusion-like settings.
# - We still need fused meta-path stabilization before claiming deployment-ready
#   "faster and better" second-order training for diffusion MAML.
# - After stabilization, optimize recompute/meta-backward hotspots using
#   timing counters.

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

# %%
print("Repo root:", REPO_ROOT)
print("Python:", sys.executable)


# %% [markdown]
# ## 2) Recommended default benchmark flow (stable baseline)
# This runs:
# 1. Meta-contract gate
# 2. Equal-step + equal-time frontier
#
# Defaults are already pinned to strict baseline:
# - backend: `triton_fused_meta_strict`
# - modes: `FULL,FULL_HYBRID`
# - practical setting: `meta_every_n_outer=8`

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
            "FULL_OUTER_LR": "1e-3",
            "FULL_HYBRID_OUTER_LR": "1e-3",
        },
    )


# %% [markdown]
# ## 3) Inspect latest baseline outputs
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
# ## 4) Diffusion-like stability isolate (strict vs experimental)
# This is the shortest sanity check to separate:
# - stable strict path (`triton_fused_meta_strict`)
# - experimental fused path (`triton_fused_meta`)
#
# `triton_fused_meta` is blocked by default, so we explicitly opt in.

# %%
RUN_STABILITY_ISOLATE = False
isolate_tag = f"phase12_stability_isolate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
isolate_csv = RUNS_DIR / f"{isolate_tag}.csv"
isolate_summary = RUNS_DIR / f"{isolate_tag}_summary.csv"

if RUN_STABILITY_ISOLATE:
    run_cmd(
        " ".join(
            [
                "PYTHONPATH=. python experiments/phase12/scripts/run_phase12_maml_seqcls_compare.py",
                "--backends triton_fused_meta_strict,triton_fused_meta",
                "--modes FULL",
                "--inner-steps 10",
                "--seeds 0,1",
                "--outer-steps 60",
                "--meta-batch-size 8",
                "--inner-lr 0.4",
                "--outer-lr 1e-3",
                "--seq-len 128",
                "--num-signal-positions 12",
                "--device cuda",
                "--allow-experimental-backends",
                f"--csv-out {isolate_csv}",
                f"--summary-out {isolate_summary}",
            ]
        )
    )


# %% [markdown]
# ## 5) Inspect latest isolate failures
# This prints non-OK rows to quickly locate non-finite failures.

# %%
latest_isolate = latest_file("experiments/phase12/runs/phase12_stability_isolate_*.csv")
if latest_isolate:
    print_failures(latest_isolate)
else:
    print("No stability isolate CSV found yet.")


# %% [markdown]
# ## 6) What to share with colleagues
#
# ### Recommended usage now
# - Correctness-sensitive FULL runs:
#   - backend: `triton_fused_meta_strict`
#   - mode: `FULL`
# - Practical budget-aware runs:
#   - backend: `triton_fused_meta_strict`
#   - mode: `FULL_HYBRID`
#   - `meta_every_n_outer=8`
#
# ### Experimental path
# - `triton_fused_meta` requires explicit opt-in:
#   - shell env: `ALLOW_EXPERIMENTAL_BACKENDS=1`
#   - CLI flag: `--allow-experimental-backends`
#
# ### Future work checklist
# - Stabilize `triton_fused_meta` on diffusion-like configs (remove non-finite failures).
# - Keep meta-contract gate mandatory before/after performance changes.
# - After stability parity with strict path, optimize fused meta recompute/backward hotspots.

