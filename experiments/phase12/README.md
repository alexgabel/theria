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

`triton_fused` is removed from current frontier recommendations until its
non-finite issue is fixed. It is not a valid deployment frontier backend.

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

## Current recommended runs

Stable baseline frontier:

```bash
TAG=phase12_frontier_$(date +%Y%m%d_%H%M%S) \
BACKENDS=triton_fused_meta_strict \
MODES=FULL,FULL_HYBRID \
META_EVERY_N_OUTER=8 \
bash experiments/phase12/scripts/run_phase12_maml_gate_and_frontier.sh
```

Practical stable frontier with the frozen hybrid knob:

```bash
TAG=phase12_frontier_hybrid_$(date +%Y%m%d_%H%M%S) \
BACKENDS=triton_fused_meta_strict \
MODES=FULL_HYBRID \
META_EVERY_N_OUTER=8 \
META_LAST_N_INNER=2 \
bash experiments/phase12/scripts/run_phase12_maml_gate_and_frontier.sh
```

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
