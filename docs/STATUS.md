## Project status

Current phase: Phase 8 (In progress: fully fused SDPA forward)

Completed (sealed):
- Operator contract (`sdpa`) defined and stable
- Reference SDPA implementation (CPU)
- Custom `autograd.Function` wiring
- Forward semantic equivalence vs reference
- First-order gradient correctness (`gradcheck`)
- Clean packaging (`pyproject.toml`) and reproducible CPU environment
- Manual smoke test (forward + backward)
- Explicit HVP/JVP semantics for SDPA
- Numerical HVP validation tests (CPU)
- Meta-learning compatibility (MAML-style inner/outer loops)

Key results (Phase 2):
- Established mathematically rigorous higher-order differentiation for SDPA operators
- Demonstrated explicit Hessian-vector product (HVP) and Jacobian-vector product (JVP) semantics
- Validated numerical correctness of higher-order derivatives on CPU reference implementation
- Identified and characterized a clear failure boundary in SDPA attention derivatives, delineating the limits of current differentiation methods

Phase 3 (completed):
- Replaced reference attention with SDPA math and fused kernels for diagnosis
- Precisely located the missing derivative edge in SDPA attention (opaque backward graph / missing JVP)
- Established the higher-order boundary contract that future kernels must satisfy

Phase 4 (completed):
- Custom attention operator with explicit HVP support (double backward + analytic HVP)
- Boundary tests passing for custom attention; fused SDPA/FlashAttention remains expected-fail
- Correctness-first scaffolding in place (autograd-in-backward) for higher-order verification

- Phase 5 (completed):
- Explicit analytic JVP implemented
- JVP matches finite differences and autograd on CPU + GPU
- JVP ≠ VJP contract enforced
- SDPA/FlashAttention forward-mode boundary locked (xfail)
- Custom attention exposes analytic JVP/HVP helpers

Phase 6 (completed):
- Triton QK forward kernel wired with Python backward
- Forward correctness verified vs reference (GPU tests)
- Gradcheck (coarse, fp32) and gradgrad existence verified
- JVP contract preserved against Triton forward (FD + cosine similarity)
- Benchmark script added; performance intentionally not optimized (Phase 7+)
- Not yet: fused backward, fused JVP rule, performance tuning

Phase 7 (in progress):
- Add explicit performance modes: `triton_ref` vs `triton_fast`
- Enable tensor cores (opt-in) and document numerical tradeoffs
- Replace QK kernel with canonical Triton matmul pattern
- Fuse softmax minimally (QK + scale + softmax; keep PV separate)
- Update benchmarks (reference vs triton_ref vs triton_fast; presets)
- Adjust tests: strict for ref/custom/triton_ref; relaxed for triton_fast

Phase 7 exit criteria:
- Triton forward within ~2–4× of cuBLAS
- First-order backward correct
- Double backward exists (may be slow)
- All correctness tests still pass

Phase 8 (in progress):
- Add `triton_fused` backend with a single fused SDPA forward kernel
- Block-wise stable softmax (online m/l accumulation), fp32 accumulators
- Forward tests vs reference (fp16/bf16) with reasonable tolerances
- First-order grads available (fallback backward or explicit skip)
- Benchmarks include fused backend
- Document what is supported and what is not
- Fused v0 contract (forward-only): contiguous Q,K,V shaped (B,H,T,D)/(B,H,M,D)/(B,H,M,Dv) with Dv=D, no mask/causal/dropout, fp16/bf16 inputs, fp32 accumulation, output cast to input dtype; supported head dims initially in {32, 64, 128}.

## Phase 8 — Fused SDPA Forward (COMPLETE)

- Implemented single-kernel Triton SDPA (QK → softmax → PV)
- Numerically stable online softmax (m, l) with edge-case handling
- Forward matches reference within fp16 tolerance
- Performance competitive with PyTorch SDPA reference
- Backward intentionally falls back to reference (Phase 9 target)

Limitations:
- Dv == D
- D ≤ 64
- No mask / causal / dropout

## Phase 9 — Explicit Backward + JVP/HVP in Triton (Planned)

Mission: eliminate autograd fallback; own fused SDPA backward, JVP, and (optionally) HVP with explicit numerical control.

Hard deliverables:
- Explicit backward kernels (dQ, dK, dV) using saved m_i, l_i (no full P materialization)
- TritonFusedSDPAFunction.backward calls Triton kernels (no reference/autograd matmul)
- JVP path without PyTorch autograd; HVP via JVP∘VJP or explicit kernel
- Gradcheck (float64 small shapes) and forward/backward parity vs reference

Planned kernels:
- sdpa_bwd_dv (Pᵀ @ dO), sdpa_bwd_dq/dk (softmax backward with reconstructed P), optional fused dq/dk
- JVP kernel (or backward-like reuse) using saved stats

Tests/coverage:
- Forward vs reference, backward gradcheck, JVP vs autograd.functional.jvp, HVP finite-diff sanity
- Boundary cases: small/large T,M, fp16/bf16, TF32 on/off, asserts for non-contiguous

Documentation to add:
- docs/phase9_backward.md and docs/phase9_jvp.md describing saved data, recomputation, stability, and failure modes if assumptions are broken.

Phase 9 (planned): Explicit Triton backward + JVP/HVP kernels (remove autograd-in-backward).
Phase 10 (planned): Meta-learning integration and higher-order research evaluation.

Key insight (Phase 8): the autodiff boundary is not the forward kernel, it is the backward.
Forward fusion is comparatively easy and performant; backward fusion is where higher-order
gradients break in practice.

Phase 9 (reframed):
- Reclaim autodiff structure inside a fused kernel (not just make it fast).
- Implement explicit fused backward (dQ, dK, dV) using saved (m, l) stats.
- Implement explicit JVP without Python autograd.
- Provide HVP support via explicit kernels (no autograd-in-backward).

Phase 10 (scientific goal):
- Explain why FlashAttention/SDPA break higher-order meta-learning.
- Characterize the autograd boundary and its dependence on kernel fusion.

Not yet implemented (intentional, future phases):
- Triton/CUDA kernels
- Backward-backward kernels (true gradgrad)
- Performance benchmarking and profiling

Notes:
- Phase 0/1 are complete and frozen; any future backend must satisfy the existing tests.
- Phase 2 is complete and verified; correctness is prioritized before GPU acceleration or performance optimization.
- Expected warnings: GPU boundary tests may emit SDPA kernel selection/deprecation warnings (e.g., `sdp_kernel` deprecation, dtype/kernel selection notices). These are expected and do not indicate test failure.
