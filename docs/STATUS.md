# Project Status

**Project:** theria  
**Focus:** Differentiable attention operators with explicit higher-order control  
**Audience:** Researchers working on kernel fusion, higher-order autodiff, and meta-learning (MAML)

## Current phase

Phase 11 — Failure modes & safety boundaries

## High-level summary (for new users)

This project builds a fully explicit SDPA (scaled dot-product attention) operator with:
- Triton-fused forward
- Explicit Triton backward (dQ, dK, dV)
- Explicit frozen-stats JVP
- HVP sanity support via finite differences
- Clear autodiff contracts and failure boundaries

Unlike FlashAttention / PyTorch SDPA, this codebase is designed to be compatible with higher-order differentiation and meta-learning, not just fast first-order training.

The central result is that higher-order autodiff failure in attention is a backward-kernel problem, not a forward-kernel problem — and this repo demonstrates how to fix it. These claims are validated empirically in Phase 10 using controlled MAML experiments and explicit negative controls.

## Phase overview (sealed)

### Phase 0–2: Foundations (sealed)
- SDPA operator contract defined and stabilized
- Reference CPU implementation
- Explicit JVP and HVP semantics defined mathematically
- Numerical validation of higher-order derivatives (CPU)
- Identification of the SDPA higher-order failure boundary

Key insight: SDPA’s higher-order issues are structural, not numerical accidents.

### Phase 3–4: Boundary isolation (sealed)
- Replaced reference attention with diagnostic SDPA kernels
- Located missing derivative edges in fused attention
- Built a custom attention operator with full analytic backward/JVP/HVP
- Established a higher-order boundary contract that fused kernels must satisfy

### Phase 5: Explicit JVP (sealed)
- Analytic JVP implemented
- JVP ≠ VJP contract enforced
- Forward-mode differentiation validated vs finite differences and autograd
- SDPA/FlashAttention forward-mode explicitly marked unsupported

### Phase 6–7: Triton forward (sealed)
- Triton QK forward kernel
- Forward correctness vs reference
- Gradcheck (fp32) and gradgrad existence
- Performance scaffolding (triton_ref vs triton_fast)
- Benchmarks added (performance secondary to correctness)

### Phase 8: Fully fused Triton SDPA forward (sealed)
- Single-kernel Triton SDPA (QK → softmax → PV)
- Numerically stable online softmax (saved row-wise m, l)
- fp16 / bf16 inputs with fp32 accumulation
- Forward parity vs reference
- Backward intentionally deferred

Limitations (by design):
- D ≤ 64, Dv == D
- No mask / causal / dropout
- Contiguous CUDA tensors only

### Phase 9: Explicit Triton backward + JVP/HVP (sealed ✅)

This phase is complete and tagged.

Delivered:
- Explicit Triton backward kernels:
  - sdpa_bwd_dq
  - sdpa_bwd_dk
  - sdpa_bwd_dv
- No autograd fallback in backward
- Reconstructed softmax using saved (m, l) — no full P materialization
- Explicit frozen-stats JVP
- HVP sanity support via finite differences of explicit VJP
- Full parity tests vs reference
- Clear guardrails and failure modes documented

Documentation:
- docs/phase9.md

Key result:

Forward fusion is easy. Backward fusion is where higher-order autodiff breaks — and we fix it explicitly.

### Phase 10: Meta-Learning (MAML) (Complete ✅)

#### Scientific outcome

Phase 10 demonstrates that:
- Higher-order meta-gradients exist and are measurable for explicit and fused SDPA backends.
- Failures in meta-learning arise from missing autodiff structure in backward/JVP, not from fusion itself.
- Explicit backward + frozen-stats JVP is sufficient to preserve second-order signal under MAML.
- The magnitude of second-order effects grows with inner-loop depth, matching MAML theory.

#### Engineering deliverables (completed)
- Backend-agnostic MAML harness (outside `theria/`)
- FO / FULL / FO-STRICT ablations
- Explicit second-order diagnostics
- Stress tests over inner-loop depth
- Optional timing benchmarks (sanity only)
- Kernels unchanged; all experiments live above the operator layer

Relevant tests already present:
- test_maml_smoke.py
- test_maml_inner_decreases_loss.py
- test_maml_full_vs_fo.py

Full report:
- docs/phase10.md

### Phase 11: Failure modes & safety boundaries

**Theme:** When does MAML stop working, and why?  
**Role:** Controlled breakage & taxonomy

#### Core scientific question

At what point does higher-order meta-learning fail, and which failures are:
- optimization-level?
- higher-order gradient-level?
- operator-level (attention-specific)?

#### Concrete experimental program (reconciled)

Break it on purpose backend taxonomy:
- Add experiment-only bad variants (no `theria/` changes):
  - `detach_attention_output`
  - `no_grad_attention`
  - `once_differentiable_sim`
  - `stats_detach` / `logits_detach`
- For each variant:
  - run Phase-10 diagnostics
  - `second_order_path`
  - `rel_diff_full_vs_fo(inner_steps)`
  - FULL vs FO timing
  - record observed behavior

#### Outcome

A meta-learning safety taxonomy:

Pattern | second_order_path | rel_diff | Behavioral effect | Verdict
---|---|---|---|---
detach | ❌ | ≈0 | FO collapse | unsafe
frozen-stats | ⚠️ | ↓ | partial | conditional
explicit backward | ✅ | ↑ | stable | safe

### Phase 12: Operator sensitivity & signal → behavior

**Theme:** Which parts of attention actually matter for meta-learning?

#### Core scientific question

Which operator-level structures generate useful second-order signal?

#### Concrete experimental program

Go beyond gradient metrics:
- Metrics:
  - post-adaptation accuracy
  - outer-loop convergence speed
- Compare:
  - FULL vs FO vs FO_STRICT
- Sweep:
  - `inner_steps` in `{1, 5, 10, 20}`
  - 2 seeds
  - reference vs `triton_fused`

#### Outcome

Plots/tables showing:

"As inner depth increases, second-order signal -> measurable accuracy gains."

This is the bridge from math -> learning behavior.

### Phase 13: Abstraction shift: unrolling vs implicit differentiation

**Theme:** When unrolling becomes the wrong abstraction

#### Core scientific question

Is unrolled MAML the right model of meta-gradients for fused operators?

#### Concrete experimental program

- Compare:
  - unrolled FULL MAML
  - truncated unrolling
  - implicit-style approximations (even crude ones)
- Ask:
  - where does fusion help/hurt?
  - where does autograd boundary actually live?

#### Outcome

Conceptual result:

"Higher-order meta-learning is fundamentally an operator-level problem, not an unrolling-depth problem."

This connects to:
- implicit layers
- equilibrium models
- modern meta-optimization theory

### Phase 14: Robustness, generality & scaling limits

**Theme:** Is Phase 10 a corner case?

#### Core scientific question

How robust is the higher-order story across regimes?

#### Concrete experimental program

Sweep without kernel tuning:
- Shapes: `T` in `{32, 128, 256}`, `D` in `{32, 64}`
- Dtypes: `fp16` / `bf16` / `fp32`
- autocast on/off

Record:
- `second_order_path`
- `rel_diff_full_vs_fo`
- runtime & peak memory

#### Outcome

A robustness + feasibility map:

Regime | FULL works? | Stable? | Practical?
---|---|---|---

This answers engineering reality, not theory.

### Phase 15: Synthesis & paper-level contract

**Theme:** Meta-learning is an operator-level phenomenon

#### Core deliverable

A single, reproducible artifact:

SDPA Meta-Learning Safety Contract

Includes:
1. Failure taxonomy (Phase 11)
2. Signal -> behavior link (Phase 12)
3. Abstraction discussion (Phase 13)
4. Robustness & scaling limits (Phase 14)
5. Timing summary (Phase 10)

#### Outcome

- NeurIPS/ICML-ready narrative
- Thesis chapter writes itself
- Clear positioning:
  - not "better FlashAttention"
  - not "yet another MAML paper"
  - operator-level theory + evidence

## What this project is not (yet)

Intentionally out of scope for now:
- Triton JVP / HVP kernels (possible future work)
- True backward-backward (gradgrad) kernels
- Production-grade FlashAttention replacement
- Masked / causal / dropout support

These are research directions, not missing features.

## Reproducibility & sanity checks

- pytest -q
- pytest -m gpu -q
- CUDA_VISIBLE_DEVICES=0 python scripts/bench_sdpa.py --preset medium --dtype float16

Expected warnings:
- SDPA kernel selection / deprecation warnings
- These are informational, not failures

## Status at a glance

Component | Status
---|---
Fused forward | ✅
Explicit backward | ✅
JVP | ✅ (frozen-stats)
HVP | ✅ (FD sanity)
MAML compatibility | ✅ (validated under explicit contracts)
Performance tuning | Secondary
