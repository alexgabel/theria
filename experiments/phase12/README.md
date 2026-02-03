# Phase 12 — Operator Sensitivity: Signal -> Behavior

Phase 12 asks whether attention-level second-order signal translates into measurable meta-learning behavior.

## Scope (frozen)
- Modes: `FULL`, `FO`, `FO_STRICT`
- Backends: `reference`, `triton_fused`
- Inner steps: `{1,5,10,20}` (long confirmation run uses `{10,20}`)
- Seeds: `{0,1}`
- Meta-batch: `16`
- Inner LR: `0.4`
- Outer optimizer/LR: Adam / `1e-3`

## Repro
Run the full pipeline:

```bash
bash experiments/phase12/scripts/run_phase12_pipeline.sh
```

Long confirmation run (k=10/20, 1000 outer steps):

```bash
RUN_TAG=phase12_long_k10k20 \
OUTER_STEPS=1000 \
INNER_STEPS_LIST="10 20" \
SEEDS="0 1" \
bash experiments/phase12/scripts/run_phase12_pipeline.sh
```

## Canonical outputs (long run)
- Combined runs: `experiments/phase12/runs/phase12_all_phase12_long_k10k20.csv`
- Aggregated summary: `experiments/phase12/runs/phase12_summary_phase12_long_k10k20.csv`
- Risk checks: `experiments/phase12/runs/phase12_risk_checks_phase12_long_k10k20.csv`
- Accuracy plot: `experiments/phase12/figures/phase12_acc_vs_inner_steps_phase12_long_k10k20.png`
- Attention-grad plot: `experiments/phase12/figures/phase12_attn_grad_norm_vs_inner_steps_phase12_long_k10k20.png`

## Results (long run, mean over 2 seeds)
### Reference backend
- `k=10`: FULL acc `0.9385`, FO acc `0.9350` (FULL +0.0035)
- `k=20`: FULL acc `0.9389`, FO acc `0.9408` (FO +0.0020)

### Triton fused backend
- `k=10`: FULL acc `0.9343`, FO acc `0.9329` (FULL +0.0014)
- `k=20`: FULL acc `0.9411`, FO acc `0.9408` (FULL +0.0003)

### FO_STRICT status
- `FO_STRICT` currently hard-fails in this setup (`HARD_FAIL_OTHER`, tensors without `grad_fn`) for both backends.
- It is treated as an unsupported mode in current loop wiring, not a behavioral baseline.

## Interpretation
- FULL and FO are very close in this task regime at high inner depth; differences are small and backend-consistent.
- Triton-backed behavior tracks reference closely, with no large behavioral drift.
- Attention grad norms remain nonzero; FULL does not show a large stable advantage over FO in this specific setup.

## Risk checks
From `phase12_risk_checks_phase12_long_k10k20.csv`:
- Seed determinism check: `PASS` (CPU reference FULL k=5 exact repeat)
- Reference vs Triton forward parity: `WARN`, but tiny drift (`max_abs_err=2.29e-4`, `rel_err=7.15e-4`)

## Close-out notes
Before fully closing Phase 12:
1. Decide FO_STRICT policy (fix to run, or formally exclude from primary comparisons).
2. Keep long-run artifacts above as canonical for citations.
3. Mirror key numbers in `docs/STATUS.md`.
