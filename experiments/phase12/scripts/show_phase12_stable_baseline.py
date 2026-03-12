#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("export "):
            continue
        key, value = line[len("export ") :].split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / "experiments/phase12/phase12_stable_baseline.env"
    values = _load_env(env_path)

    gate_tag = values["PHASE12_STABLE_BASELINE_GATE_TAG"]
    frontier_tag = values["PHASE12_STABLE_BASELINE_FRONTIER_TAG"]
    n_outer = values["PHASE12_STABLE_META_EVERY_N_OUTER"]
    last_inner = values["PHASE12_STABLE_META_LAST_N_INNER"]
    tol = values["PHASE12_STABLE_HYBRID_FRONTIER_ACC_TOL"]

    print("Phase 12 stable baseline")
    print(f"- gate tag: {gate_tag}")
    print(f"- frontier tag: {frontier_tag}")
    print(f"- meta_every_n_outer: {n_outer}")
    print(f"- meta_last_n_inner: {last_inner}")
    print(f"- hybrid frontier acc tol: {tol}")
    print("- correctness-sensitive: triton_fused_meta_strict FULL")
    print(
        "- wall-clock-sensitive: triton_fused_meta_strict FULL_HYBRID "
        f"--meta-every-n-outer {n_outer} --meta-last-n-inner {last_inner}"
    )
    print("- benchmark/regression policy: eager only; leave CUDA_GRAPH_STATIC unset")


if __name__ == "__main__":
    main()
