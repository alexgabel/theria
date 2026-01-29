## Project status

Current phase: Phase 7 (In progress: performance-oriented Triton attention)

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

Phase 8 (planned): Fully fused SDPA forward (QK → softmax → PV) with stable block softmax and minimal intermediates.
Phase 9 (planned): Explicit Triton backward + JVP/HVP kernels (remove autograd-in-backward).
Phase 10 (planned): Meta-learning integration and higher-order research evaluation.

Not yet implemented (intentional, future phases):
- Triton/CUDA kernels
- Backward-backward kernels (true gradgrad)
- Performance benchmarking and profiling

Notes:
- Phase 0/1 are complete and frozen; any future backend must satisfy the existing tests.
- Phase 2 is complete and verified; correctness is prioritized before GPU acceleration or performance optimization.
- Expected warnings: GPU boundary tests may emit SDPA kernel selection/deprecation warnings (e.g., `sdp_kernel` deprecation, dtype/kernel selection notices). These are expected and do not indicate test failure.
