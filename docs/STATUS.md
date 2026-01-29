## Project status

Current phase: Phase 5 (Planned: explicit JVP/backward rules)

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

Phase 5 (planned):
- Replace scaffolding with explicit JVP/backward rules and required saved intermediates
- Formalize JVP math and implementation details for attention operators
- [x] Explicit analytic JVP implemented
- [x] JVP finite-difference tests
- [x] Autograd JVP parity test
- [x] Boundary witness: SDPA JVP failure
- [x] GPU parity check
- [x] JVP ≠ VJP test (shape/asymmetry witness)

Not yet implemented (intentional, future phases):
- Triton/CUDA kernels
- Backward-backward kernels (true gradgrad)
- Performance benchmarking and profiling

Notes:
- Phase 0/1 are complete and frozen; any future backend must satisfy the existing tests.
- Phase 2 is complete and verified; correctness is prioritized before GPU acceleration or performance optimization.
- Expected warnings: GPU boundary tests may emit SDPA kernel selection/deprecation warnings (e.g., `sdp_kernel` deprecation, dtype/kernel selection notices). These are expected and do not indicate test failure.
