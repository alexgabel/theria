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
   Runs complete successfully, but attention contributes no meaningful second-order signal. This includes explicit detaches and checkpoint recomputation that detaches saved tensors.

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
| `checkpoint_attention` | Correct checkpoint recompute | Safe (keeps higher-order) |
| `checkpoint_no_grad` | Checkpoint recompute under no_grad | Hard autograd failure |
| `checkpoint_detach_recompute` | Checkpoint then detach recompute output | Silent curvature collapse |
| `recompute_logits_no_grad_sdpa` | Logits recomputed under no_grad | Hard autograd failure |
| `backward_detach_logits_sim` | Backward recompute with detached logits | Silent higher-order loss |
| `detach_q/k/v_input` | Detach a single SDPA input (non-strict) | Meta-signal largely preserved |
| `detach_q/k/v_input_strict` | Strict detach of SDPA input | Hard fail (unused grad) |
| `checkpoint_attention` | Correct checkpoint recompute | Safe (keeps higher-order) |
| `checkpoint_no_grad` | Checkpoint recompute under no_grad | Hard autograd failure |
| `checkpoint_detach_recompute` | Checkpoint then detach recompute output | Silent curvature collapse |
| `recompute_logits_no_grad_sdpa` | Logits recomputed under no_grad | Hard autograd failure |
| `backward_detach_logits_sim` | Backward recompute with detached logits | Silent higher-order loss |
| `detach_q/k/v_input` | Detach a single SDPA input (non-strict) | Meta-signal largely preserved |
| `detach_q/k/v_input_strict` | Strict detach of SDPA input | Hard fail (unused grad) |

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
| `sdpa_input_gradgrad_ok` | SDPA-input-local grad-grad probe on Q/K/V |
| `rel_diff_mean` | Mean FULL-FO meta-gradient discrepancy |
| `error` | Autograd/runtime error (if any) |

Note: global second-order paths may exist even when attention-local curvature is entirely suppressed.

## Measured results (reference backend)

Setup:
- `backend=reference`
- `steps=1`, `inner_lr=0.4`, `device=cpu`, `seed=0`
- Inner-depth sweep: `inner_steps in {1, 5, 10, 20}`

### FULL run diagnostics

| Wrapper | k=1 | k=5 | k=10 | k=20 | second_order_path | Status |
|---|---:|---:|---:|---:|---|---|
| `baseline` | 0.1753 | 0.5848 | 0.5045 | 0.6587 | `True` | `OK` |
| `detach_attention_output` | 0.0391 | 0.2047 | 0.4239 | 0.7109 | `True` | `OK` |
| `stats_detach_logits_sdpa` | 0.0829 | 0.3287 | 0.4397 | 0.5414 | `True` | `OK` |
| `stats_detach_softmax_output_sdpa` | 0.0829 | 0.3287 | 0.4397 | 0.5414 | `True` | `OK` |
| `once_differentiable_sim` | - | - | - | - | - | `HARD_FAIL_ONCE_DIFF` |
| `no_grad_attention` | - | - | - | - | - | `HARD_FAIL_UNUSED` |

Values are `rel_diff_full_vs_fo`; hard-fail rows do not produce the metric.
Reminder: for safe kernels, `rel_diff` is expected to grow with inner depth; suppressed or flat `rel_diff` can signal curvature loss.

## Additional findings (checkpoint / recompute / partial detaches)

- `checkpoint_attention` is meta-learning safe on reference/custom/triton_fused (CUDA), preserving attention-local gradgrad; rel_diff similar to baseline.
- `checkpoint_no_grad` always HARD_FAIL_UNUSED (graph broken).
- `checkpoint_detach_recompute` runs but removes attention curvature (silent failure); rel_diff low at k=1, grows by k=20.
- `recompute_logits_no_grad_sdpa` HARD_FAIL_UNUSED; `backward_detach_logits_sim` runs with attention-local probe False (silent higher-order loss, mid rel_diff).
- `stats_detach_logits_sdpa` / `stats_detach_softmax_output_sdpa` run; attention-local probe False; rel_diff lower than baseline but increasing with depth (silent stats collapse).
- `detach_attention_output` mirrors checkpoint_detach_recompute: runs, kills attention curvature, downstream rel_diff can become large at high inner depth.
- Partial detaches (non-strict) do NOT remove attention curvature; strict variants hard-fail (unused grad).
- Quantitatively, silent-collapse variants drive attention-local second-order signal to (near) zero (e.g., grad_norm_q/k/v ≈ 0) while rel_diff can reach ~0.7 at k=20, indicating downstream curvature dominance despite attention curvature removal.

## Checkpointing and meta-learning safety

Gradient checkpointing is meta-learning safe **iff** recomputation preserves second-order connectivity through attention.

Empirically:
- `checkpoint_attention` preserves attention-local grad-grad and matches baseline.
- `checkpoint_no_grad` hard-fails (graph broken).
- `checkpoint_detach_recompute` silently removes attention curvature while leaving global second-order paths intact.

Thus, checkpointing itself is not unsafe; detached recomputation is.

## Backend-agnostic boundary

All failure modes were reproduced across reference, custom, and (where supported) triton_fused attention implementations, indicating these safety boundaries are backend-agnostic.

## Empirical safety theorem (Phase 11)

An attention implementation is second-order meta-learning safe iff:
1. its backward is differentiable (no `@once_differentiable` boundary),
2. recomputation does not occur under `no_grad`,
3. internal attention statistics (logits or softmax outputs) participate in higher-order autodiff,
4. checkpoint recomputation preserves gradient connectivity.

Violating any condition yields either hard autograd failure or silent curvature collapse.

## Key negative result

Silent failures are more dangerous than hard failures. Variants like `detach_attention_output`, `checkpoint_detach_recompute`, and `stats_detach_*` complete successfully, report valid losses, and expose global second-order paths, yet remove all attention-local curvature. These failures are undetectable without targeted second-order probes.
## Interpreting results

### Silent failures (most dangerous)

Examples:
- `detach_attention_output`
- `stats_detach_logits_sdpa`
- `stats_detach_softmax_output_sdpa`
- `checkpoint_detach_recompute`

Characteristics:
- Run completes (`status=OK`)
- Global second-order path exists
- Attention contributes no second-order signal
- FULL-FO discrepancy may be suppressed at low inner depth and reappear downstream at larger depth

Interpretation:

`detach_attention_output` suppresses attention-local second-order signal. Global FULL-FO divergence may reappear at large inner depth due to downstream parameters, making this a silent failure.
In TinyAttention, Q/K/V projections share downstream pathways that preserve higher-order signal unless detachment is graph-breaking (strict), so non-strict partial detaches do not remove curvature.

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
