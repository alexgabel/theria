## Project status

Current phase: Phase 4 (Custom attention with explicit JVP/HVP contract)

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

Phase 4 goals:
- Introduce a custom attention operator that explicitly exposes JVP/HVP information for (Q, K, V)
- Ensure the new operator passes the `test_attention_boundary_*` suite (grad-of-grad included)
- Prepare for later performance work without sacrificing the established higher-order contract

Not yet implemented (intentional, future phases):
- Triton/CUDA kernels
- Backward-backward kernels (true gradgrad)
- Performance benchmarking and profiling

Notes:
- Phase 0/1 are complete and frozen; any future backend must satisfy the existing tests.
- Phase 2 is complete and verified; current focus is on attention boundary analysis before GPU acceleration or performance optimization.
