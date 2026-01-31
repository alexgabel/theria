# Phase 9 — HVP Contract and Sanity

Definition
- HVP is the directional second derivative of the true SDPA map
  \(f(q,k,v) = \sum \text{triton\_sdpa\_fused}(q,k,v)\), where forward stats `(m,l)`
  are **recomputed** for each evaluation.
- For Day 5 sanity we approximate HVP via central finite differences of the explicit
  VJP (dQ/dK/dV Triton kernels), not via autograd-in-backward:
  \[ H(x)[v] \approx \tfrac{\nabla f(x+\epsilon v) - \nabla f(x-\epsilon v)}{2\epsilon}. \]

Utility
- `theria.attention.hvp_utils.hvp_fd_vjp` computes:
    H(x)[v] ≈ (∇f(x+εv) - ∇f(x-εv)) / (2ε)
  where gradients are produced by explicit Triton backward, and `f` is `sum(triton_sdpa_fused(...))`.

Testing
- `tests/test_triton_fused_hvp_sanity.py` compares this FD HVP against an autograd HVP on the CPU reference for tiny shapes (cosine similarity ≥ 0.9).
- Tolerances are loose because second-order derivatives amplify numerical noise.

Scope / Non-goals
- No Triton HVP kernel is implemented in Phase 9.
- Performance is not optimized; this is a correctness sanity check only.
