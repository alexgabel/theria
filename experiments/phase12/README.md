# Phase 12 — Current Contract

Phase 12 is now the benchmark and productization layer for second-order MAML
attention backends.

## Frozen product baselines

- Stable correctness baseline:
  - backend: `triton_fused_meta_strict`
  - mode: `FULL`
- Stable practical baseline:
  - backend: `triton_fused_meta_strict`
  - mode: `FULL_HYBRID`
  - knobs: `--meta-every-n-outer 8 --meta-last-n-inner 2`
- Experimental backend only:
  - backend: `triton_fused_meta`

## A. Stable Delivery

Accepted stable baseline (2026-03-09):
- gate tag: `phase12_meta_gate_phase12_stable_regression_20260309_155142`
- frontier tag: `phase12_stable_regression_20260309_155142`

Accepted recommendations:
- correctness-sensitive:
  - `triton_fused_meta_strict FULL`
- wall-clock-sensitive:
  - `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2`

Stable delivery invariants:
- `meta_every_n_outer=8`
- `meta_last_n_inner=2`
- equal-time frontier tolerance: `0.01`
- regression harness: `experiments/phase12/scripts/run_phase12_stable_frontier_regression.sh`

The single source of truth for these defaults is:
- `experiments/phase12/phase12_stable_baseline.env`

Current stable-delivery decision:
- keep this baseline frozen
- do not spend more time on `inner_adapt()` container-overhead micro-optimizations
- keep benchmark and regression usage on the eager path only
- do not use `CUDA_GRAPH_STATIC=1` in recommended commands

`triton_fused` is removed from current frontier recommendations until its
non-finite issue is fixed. It is not a valid deployment frontier backend.

Contributor note:
- the stable path is accepted on eager execution
- CUDA-graph work is isolated R&D
- `triton_fused_meta` is experimental and not the default

Stable delivery exit criteria:
- docs/examples stay aligned with the frozen baseline
- recommended commands stay on the eager path
- no misleading CUDA-graph-ready or universal equal-time-win claims remain

## B. CUDA-Graph R&D

Current status: blocked.

What is blocked:
- static-shape CUDA graph capture on the stable practical workload
- graph-based benchmark or regression runs
- more specifically, even isolated attention-side QK capture fails in the repo
  capture-debug workflow

What this means:
- do not use `CUDA_GRAPH_STATIC=1` in benchmark/regression scripts
- do not make CUDA-graph speed claims
- keep graph work isolated to debug scripts only
- no mainline integration
- no default flags for graph execution
- do not start kernel tuning yet; capture instability makes that premature

If this work resumes later:
- start from the smallest accepted capture-debug repro
- do not restart from the frontier runners

Success criteria:
- isolated capture-debug path passes
- then runner integration passes
- then stable regression passes with graphs enabled

Until then:
- status remains blocked

## C. Experimental Backend Promotion

Current status: not promoted.

Promotion criteria:
- fallback-free
- stable across seeds
- contract gate passes
- equal-time competitive or better

Until those are met:
- `triton_fused_meta` remains explicit opt-in only
- do not promote based on equal-step alone
- it does not replace `reference`
- the stable backend should not be described as a universal equal-time winner

If this track resumes:
- rerun the 2-seed qualification first
- only then run the 5-seed qualification

If it still trails on equal-time:
- keep it experimental
- do not change the stable recommendation

## Mandatory gate policy

Before every performance change:
- run the meta-contract gate
- require cosine thresholds to remain unchanged
- require rel-diff error to remain unchanged
- require no qualitative trend mismatch

The default entrypoint already enforces this ordering:

```bash
bash experiments/phase12/scripts/run_phase12_maml_gate_and_frontier.sh
```

## Current recommended runs (eager path only)

Stable baseline frontier:

```bash
TAG=phase12_frontier_$(date +%Y%m%d_%H%M%S) \
BACKENDS=triton_fused_meta_strict \
MODES=FULL,FULL_HYBRID \
META_EVERY_N_OUTER=8 \
bash experiments/phase12/scripts/run_phase12_maml_gate_and_frontier.sh
```

Leave `CUDA_GRAPH_STATIC` unset for this run.

Practical stable frontier with the frozen hybrid knob:

```bash
TAG=phase12_frontier_hybrid_$(date +%Y%m%d_%H%M%S) \
BACKENDS=triton_fused_meta_strict \
MODES=FULL_HYBRID \
META_EVERY_N_OUTER=8 \
META_LAST_N_INNER=2 \
bash experiments/phase12/scripts/run_phase12_maml_gate_and_frontier.sh
```

Leave `CUDA_GRAPH_STATIC` unset for this run.

Stable practical systems target (default for wall-clock/profiling work):

```bash
bash experiments/phase12/scripts/run_phase12_maml_stable_practical.sh
```

This wrapper intentionally fixes the practical hybrid knob:
- backend: `triton_fused_meta_strict`
- mode: `FULL_HYBRID`
- `meta_every_n_outer=8`
- `meta_last_n_inner=2`

Use this for systems profiling and stable wall-clock comparisons before
attempting to make pure `FULL` cheap.

Stable optimization regression loop after each optimization batch:

```bash
bash experiments/phase12/scripts/run_phase12_stable_frontier_regression.sh
```

This enforces:
- meta-contract gate passes
- equal-step `triton_fused_meta_strict FULL` remains aligned with `reference FULL`
- equal-time `triton_fused_meta_strict FULL_HYBRID` stays on the same-backend FO frontier

## Primary instability canary

The canonical canary is the known bad diffusion-like config, run without
runtime fallback:

```bash
bash experiments/phase12/scripts/run_phase12_fused_meta_canary.sh
```

This script fixes:
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `THERIA_TRITON_META_ENABLE_FALLBACK=0`
- seeds `1 4`
- diffusion-like config: `seq_len=128`, `num_signal_positions=12`, `k=10`
- CSV output paths under `experiments/phase12/runs/`

The canary must pass before `triton_fused_meta` is considered for speed claims.

## Owned instability regression target

Use the fixed bad-seed regression target when making numerical changes to
`triton_fused_meta`:

```bash
PYTHONPATH=. python experiments/phase12/scripts/run_phase12_fused_meta_regression.py \
  --expect-fused-meta fail
```

Before the fix, the known bad seeds must fail/diverge for `triton_fused_meta`.
After the fix, rerun the same target with:

```bash
PYTHONPATH=. python experiments/phase12/scripts/run_phase12_fused_meta_regression.py \
  --expect-fused-meta pass
```

The strict backend must remain clean throughout in both modes.

## Standard debug artifact

The first-divergence tracer now writes both:
- a per-step CSV via `--csv-out`
- a one-row first-divergence summary via `--summary-out`

The summary artifact captures:
- outer step index
- divergence reason
- logits / softmax stats
- dQ / dK / dV stats
- q / k / v projection gradient stats
- failure stage classification:
  - `fast_backward`
  - `recompute`
  - `outer_model_grad`

## Notes

- Use `triton_fused_meta_strict FULL` when correctness is the priority.
- Use `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2`
  when wall-clock matters.
- Use `triton_fused_meta` only with explicit opt-in while stabilization work is
  ongoing.
- Do not use `CUDA_GRAPH_STATIC=1` in recommended benchmark or regression commands.
