# Phase 9 — JVP Contract

This project defines an explicit Jacobian–vector product (JVP) for SDPA that **holds the forward statistics `(m, l)` fixed**. The intent matches the fused forward that already materializes and saves these stats.

Contract
- Inputs: `q, k, v, dq, dk, dv, m, l, scale`.
- `m, l` are the row‑wise max and sumexp computed during the original forward pass and are treated as constants in the JVP.
- Output: `dO = J(q,k,v) · (dq,dk,dv)` with `m, l` frozen.

Rationale
- Matches the backward kernels’ saved forward state (checkpointed/flash-style execution).
- Avoids recomputing softmax stats during JVP, which would introduce extra dm/dl terms that do not belong to the saved-state linearization.
- Keeps the JVP composable with VJP/HVP without invoking autograd inside kernels.

Testing implications
- Finite-difference checks must use a **fixed-stats** function: perturb `(q,k,v)` but reuse the original `(m,l)` when forming probabilities.
- A guard test ensures that the frozen-stats JVP differs from a recomputed-stats JVP, confirming the contract is enforced.
