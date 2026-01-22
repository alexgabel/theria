## Project status

Current phase: Phase 2 (Higher-order differentiation: HVP/JVP)

Completed (sealed):
- Operator contract (`sdpa`) defined and stable
- Reference SDPA implementation (CPU)
- Custom `autograd.Function` wiring
- Forward semantic equivalence vs reference
- First-order gradient correctness (`gradcheck`)
- Clean packaging (`pyproject.toml`) and reproducible CPU environment
- Manual smoke test (forward + backward)

In progress (Phase 2 goals):
- Explicit HVP/JVP semantics for SDPA
- Numerical HVP validation tests (CPU)
- Meta-learning compatibility (MAML-style inner/outer loops)

Not yet implemented (intentional, future phases):
- Triton/CUDA kernels
- Backward-backward kernels (true gradgrad)
- Performance benchmarking and profiling

Notes:
- Phase 0/1 are complete and frozen; any future backend must satisfy the existing tests.
- Phase 2 focuses on mathematical correctness of higher-order differentiation before any GPU work.