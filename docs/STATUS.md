## Project status

Current phase: Phase 3 (Attention boundary analysis)

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

Phase 3 goals:
- Replace reference attention implementation with SDPA-based mathematical formulation and fused kernels
- Analyze and precisely locate the missing derivative edge in SDPA attention to enable robust higher-order differentiation
- Develop strategies to extend differentiation capabilities beyond the identified boundary, improving model expressiveness and training stability

Not yet implemented (intentional, future phases):
- Triton/CUDA kernels
- Backward-backward kernels (true gradgrad)
- Performance benchmarking and profiling

Notes:
- Phase 0/1 are complete and frozen; any future backend must satisfy the existing tests.
- Phase 2 is complete and verified; current focus is on attention boundary analysis before GPU acceleration or performance optimization.