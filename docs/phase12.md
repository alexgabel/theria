# Phase 12 — Operator Sensitivity: Signal → Behavior

**Theme**  
Which operator-level structures in attention actually matter for meta-learning, and when does second-order signal translate into measurable learning gains?

Phase 12 explicitly bridges Phase 11 (existence of second-order signal) to learning behavior.

---

## Scientific question

Which parts of the attention operator generate useful second-order signal, and under what conditions does that signal improve meta-learning performance? Phase 12 pairs operator-level diagnostics with behavioral outcomes.

---

## Experimental scope (intentionally tight)

Not a hyperparameter sweep; a controlled comparison across meta-gradient regimes.

### Compared modes
- **FULL** — second-order MAML  
- **FO** — first-order MAML  
- **FO_STRICT** — strictly first-order (no accidental curvature)

### Definitions (implementation-level)
- **FULL:** Inner-loop gradients computed with `create_graph=True` in `inner_adapt`, allowing higher-order meta-gradients.  
- **FO (First-Order MAML):** Inner-loop gradients computed with `create_graph=False`; forward operator unchanged, no second-order graph.  
- **FO_STRICT (Strict First-Order):** Same as FO (`create_graph=False`), and adapted parameters are explicitly detached after each inner update (or before the outer loss), ensuring no higher-order signal can flow through reused tensors or operator internals. Implementation mapping: FO = existing Phase-10 behavior; FO_STRICT = FO + `phi = {k: v.detach().clone() for k, v in phi.items()}` before outer loss.

### Attention backends
- `reference`
- `triton_fused`

No backend-specific tuning is performed.

---

## Fixed experimental configuration

Frozen to avoid ambiguity and enable Phase 14 reuse:
- Task: synthetic sequence classification (`theria.tasks.synthetic_seqcls`)
- Meta-batch size: 16 tasks
- Inner steps: {1, 5, 10, 20}
- Inner learning rate: 0.4
- Outer optimizer: Adam (default betas), outer LR: 1e-3
- Outer steps: 500
- Seeds: 2 (fixed, no extra averaging)
- Device: CPU for reference runs; CUDA for Triton runs
- dtype / autocast: fp32, autocast OFF (logged explicitly)

---

## Metrics (signal → behavior)

Phase 12 goes beyond gradient diagnostics.

### Behavioral metrics (primary)
- Post-adaptation top-1 query accuracy, averaged across tasks in the meta-batch and reported as the mean over the final 20 outer steps.
- Outer-loop convergence: query loss vs outer steps (used to compare convergence speed across FULL / FO / FO_STRICT).

### Operator diagnostics (explanatory)
- Attention-local meta-gradient norm (q_proj / k_proj / v_proj) and boolean probe (from Phase 11).
- `rel_diff_full_vs_fo` retained as supporting diagnostic (logged sparsely to avoid doubling compute).

Every behavioral plot is paired with an operator-level signal plot.

---

## Runtime & memory logging

- Timing via `time.perf_counter()`; log total wall time per run and mean time per outer step.
- CUDA only: peak memory via `torch.cuda.max_memory_allocated()` (after `reset_peak_memory_stats()`); omitted on CPU.

---

## Expected outcome (claim-level)

As inner-loop depth increases, second-order signal is converted into measurable accuracy gains only when attention curvature is preserved. Quantitatively, silent-collapse variants drive attention-local second-order signal near zero while `rel_diff` can still grow (≈0.7 at k=20), showing downstream curvature dominance without attention contribution.

Specifically:
- FULL outperforms FO and FO_STRICT at larger inner depths.
- FO and FO_STRICT converge similarly despite different implementations.
- Behavioral gains correlate with attention-local second-order signal.

---

## Phase-12 artifacts

Completion criteria:
1. Accuracy vs inner-steps plot (FULL / FO / FO_STRICT; reference and triton_fused).  
2. Attention-gradient norm vs inner-steps plot (same grid).  
3. Summary table: {mode → attention signal present?, accuracy gain?}.  
4. One-page written summary explaining when and why second-order signal matters.

Artifacts must be paper-grade and reusable.

---

## Relationship to other phases
- Phase 10: Establishes correct higher-order behavior.
- Phase 11: Maps failure modes and safety boundaries.
- Phase 12: Shows when signal becomes learning benefit.
- Phase 13: Questions whether unrolling is the right abstraction.
- Phase 14: Tests robustness and scaling limits.

---

## Non-goals

Phase 12 does not aim to:
- Optimize kernels
- Tune tasks or hyperparameters
- Introduce new Triton features
- Replace FlashAttention

It isolates operator sensitivity, not performance engineering.

---

## Status gating

Phase 12 begins once:
- Reference backend FULL/FO/FO_STRICT grid is complete.
- Task difficulty is verified (FULL > FO at sufficient inner depth).

Only then do Triton runs proceed.

---

## Runner invocation (example)

One run (reference backend, FULL, k=20, seed=0):

```bash
python experiments/phase12/scripts/run_phase12_behavior.py \
  --backend reference \
  --mode FULL \
  --outer-steps 500 \
  --meta-batch-size 16 \
  --inner-steps 20 \
  --inner-lr 0.4 \
  --outer-lr 1e-3 \
  --seed 0 \
  --device cpu \
  --csv-out experiments/phase12/runs/phase12_behavior_reference.csv
```

Repeat for `mode=FO`, `mode=FO_STRICT`, inner_steps in `{1,5,10,20}`, seeds `{0,1}`, and `--backend triton_fused` on CUDA. Each run appends one row to the CSV.
