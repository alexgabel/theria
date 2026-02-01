Project Status

Project: theria
Focus: Differentiable attention operators with explicit higher-order control
Audience: Researchers working on kernel fusion, higher-order autodiff, and meta-learning (MAML)

Current phase

Phase 10 — Meta-Learning (MAML) Integration & Experiments
(Active development)

High-level summary (for new users)

This project builds a fully explicit SDPA (scaled dot-product attention) operator with:
- Triton-fused forward
- Explicit Triton backward (dQ, dK, dV)
- Explicit frozen-stats JVP
- HVP sanity support via finite differences
- Clear autodiff contracts and failure boundaries

Unlike FlashAttention / PyTorch SDPA, this codebase is designed to be compatible with higher-order differentiation and meta-learning, not just fast first-order training.

The central result is that higher-order autodiff failure in attention is a backward-kernel problem, not a forward-kernel problem — and this repo demonstrates how to fix it.

Phase overview (sealed)

Phase 0–2: Foundations (sealed)
- SDPA operator contract defined and stabilized
- Reference CPU implementation
- Explicit JVP and HVP semantics defined mathematically
- Numerical validation of higher-order derivatives (CPU)
- Identification of the SDPA higher-order failure boundary

Key insight: SDPA’s higher-order issues are structural, not numerical accidents.

Phase 3–4: Boundary isolation (sealed)
- Replaced reference attention with diagnostic SDPA kernels
- Located missing derivative edges in fused attention
- Built a custom attention operator with full analytic backward/JVP/HVP
- Established a higher-order boundary contract that fused kernels must satisfy

Phase 5: Explicit JVP (sealed)
- Analytic JVP implemented
- JVP ≠ VJP contract enforced
- Forward-mode differentiation validated vs finite differences and autograd
- SDPA/FlashAttention forward-mode explicitly marked unsupported

Phase 6–7: Triton forward (sealed)
- Triton QK forward kernel
- Forward correctness vs reference
- Gradcheck (fp32) and gradgrad existence
- Performance scaffolding (triton_ref vs triton_fast)
- Benchmarks added (performance secondary to correctness)

Phase 8: Fully fused Triton SDPA forward (sealed)
- Single-kernel Triton SDPA (QK → softmax → PV)
- Numerically stable online softmax (saved row-wise m, l)
- fp16 / bf16 inputs with fp32 accumulation
- Forward parity vs reference
- Backward intentionally deferred

Limitations (by design):
- D ≤ 64, Dv == D
- No mask / causal / dropout
- Contiguous CUDA tensors only

Phase 9: Explicit Triton backward + JVP/HVP (sealed ✅)

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
- docs/phase9_operator.md

Key result:

Forward fusion is easy. Backward fusion is where higher-order autodiff breaks — and we fix it explicitly.

Current phase (active)

Phase 10: Meta-Learning (MAML)

Scientific goal

Demonstrate and explain:
- Why standard SDPA / FlashAttention break second-order meta-learning
- How explicit backward + frozen-stats JVP restores MAML compatibility
- The practical gap between FO-MAML and SO-MAML in attention-heavy models

Engineering goals
- Clean MAML inner / outer loops using explicit VJP/JVP
- Compare:
  - First-order MAML
  - Second-order MAML (explicit VJP + recomputation)
- Quantify stability vs speed trade-offs
- Keep kernels unchanged — experiments live above the operator layer

Relevant tests already present:
- test_maml_smoke.py
- test_maml_inner_decreases_loss.py
- test_maml_full_vs_fo.py

What this project is not (yet)

Intentionally out of scope for now:
- Triton JVP / HVP kernels (possible future work)
- True backward-backward (gradgrad) kernels
- Production-grade FlashAttention replacement
- Masked / causal / dropout support

These are research directions, not missing features.

Reproducibility & sanity checks
- pytest -q
- pytest -m gpu -q
- CUDA_VISIBLE_DEVICES=0 python scripts/bench_sdpa.py --preset medium --dtype float16

Expected warnings:
- SDPA kernel selection / deprecation warnings
- These are informational, not failures

Status at a glance

Component | Status
---|---
Fused forward | ✅
Explicit backward | ✅
JVP | ✅ (frozen-stats)
HVP | ✅ (FD sanity)
MAML compatibility | 🚧 Phase 10
Performance tuning | Secondary
