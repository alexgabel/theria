# Phase 11 — Failure Modes & Meta-Learning Safety Boundaries

**Purpose**  
Classify *meta-learning unsafe* attention implementations via controlled breakage, and distinguish
structural autodiff failures from silent second-order collapse.

Phase 11 introduces intentionally broken attention variants **in experiment code only** (never in
`theria/`) and evaluates them using the Phase-10 MAML diagnostics.

---

## Safety taxonomy (high level)

Phase 11 identifies three distinct classes of failure:

1. **Hard structural failures**  
   Autograd graph is broken; even first-order meta-learning may fail.
2. **First-order-only kernels**  
   First-order gradients work, but higher-order differentiation is explicitly blocked.
3. **Silent curvature collapse**  
   Runs complete successfully, but attention contributes no meaningful second-order signal.

These categories are empirically distinguishable and require different diagnostics.

---

## Intentionally broken backends

The following wrappers simulate real-world kernel patterns:

| Wrapper | Simulates | Expected behavior |
|---|---|---|
| `detach_attention_output` | Detaching SDPA output | Silent collapse of attention curvature |
| `no_grad_attention` | `torch.no_grad()` kernels | Hard autograd failure |
| `once_differentiable_sim` | `@once_differentiable` backward | FO works, FULL fails |
| `stats_detach_logits_sdpa` | Frozen softmax logits | Silent stats-level collapse |
| `stats_detach_softmax_output_sdpa` | Detached softmax output | Strong silent curvature collapse |

---

## Diagnostics runner

Run the Phase-11 taxonomy runner:

```bash
python experiments/phase11/scripts/run_bad_backend_diagnostics.py \
  --backend all \
  --steps 5 \
  --inner-steps 5 \
  --device cpu
```

Output CSV: `experiments/phase11/runs/bad_backend_diagnostics.csv`

## CSV schema

Each row corresponds to one wrapper x attention backend combination.

| Column | Meaning |
|---|---|
| `wrapper` | Name of broken attention variant |
| `attention_backend` | `reference`, `custom`, or `triton_fused` |
| `status` | `OK`, `HARD_FAIL_UNUSED`, `HARD_FAIL_ONCE_DIFF`, `HARD_FAIL_OTHER` |
| `second_order_path_*` | Global second-order path existence |
| `second_order_path_attn_*` | Second-order signal through attention params only (`q_proj/k_proj/v_proj`) |
| `second_order_path_head_*` | Second-order signal through classifier head only |
| `attn_second_order_ok` | Attention-local grad-grad probe on Q/K/V |
| `rel_diff_mean` | Mean FULL-FO meta-gradient discrepancy |
| `error` | Autograd/runtime error (if any) |

## Interpreting results

### Silent failures (most dangerous)

Examples:
- `detach_attention_output`
- `stats_detach_logits_sdpa`
- `stats_detach_softmax_output_sdpa`

Characteristics:
- Run completes (`status=OK`)
- Global second-order path exists
- Attention contributes no second-order signal
- FULL-FO discrepancy is suppressed

Interpretation:

MAML "works", but attention behaves as a first-order feature extractor.

### Hard failures

Example: `no_grad_attention`

Characteristics:
- Autograd error (`HARD_FAIL_UNUSED`)
- Parameters disappear from the computation graph

Interpretation:

Kernel is meta-learning unsafe even for FO in unrolled settings.

### First-order-only kernels

Example: `once_differentiable_sim`

Characteristics:
- FO-MAML runs
- FULL-MAML crashes with `@once_differentiable`

Interpretation:

First-order compatibility != higher-order compatibility.

## Phase-11 safety statement

From Phase 11 evidence:

Attention kernels that detach internal statistics or mark backward as once-differentiable are
meta-learning unsafe for second-order MAML, even when forward and first-order backward are
correct.

This boundary is empirical, reproducible, and backend-agnostic.

---

## Relationship to other phases

- Phase 9: Defines explicit backward and JVP contracts
- Phase 10: Shows these contracts behave correctly under MAML
- Phase 11: Classifies which violations break or silently degrade meta-learning
- Phase 12: Measures behavioral impact (accuracy, adaptation speed)

---

## Non-goals (by design)

Phase 11 does not aim to:
- Fix broken kernels
- Optimize performance
- Implement new Triton kernels

It exists to map the safety boundary, not move it.
