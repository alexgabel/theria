# Phase 9: Explicit SDPA Operator (Forward / Backward / JVP / HVP)

## What this implements
- Fused SDPA forward (Triton): `triton_sdpa_fused`
- Explicit backward (dQ / dK / dV): `sdpa_bwd_dq`, `sdpa_bwd_dk`, `sdpa_bwd_dv`
- Frozen-stats JVP: `sdpa_jvp`
- FD HVP (sanity only): `hvp_fd_vjp` (finite-difference of explicit VJP)

## Operator contracts
- Shapes:
  - Q: (B, H, T, D)
  - K: (B, H, M, D)
  - V: (B, H, M, D)
  - dO: (B, H, T, D)
  - stats m,l: (B, H, T)
- D and Dv must satisfy `D <= 64` and `Dv == D` (v0).
- No mask / dropout / causal support (v0).
- CUDA-only for Triton paths (backward + JVP). Inputs must be contiguous.
- No silent fallback: unsupported shapes/dtypes raise.

## Mathematical definitions (summary)
Let:
- Scores: `S_ij = q_i · k_j^T * scale`, `scale = 1/sqrt(D)`
- Row stats: `m_i = max_j S_ij`, `l_i = sum_j exp(S_ij - m_i)`
- Reconstruction: `P_ij = exp(S_ij - m_i) / l_i`

Backward (frozen stats `m,l`):
- `dV = P^T dO`
- `dP = dO V^T`
- `z_i = sum_j dP_ij P_ij`
- `dS = P ⊙ (dP - z)`
- `dQ = (dS K) * scale`
- `dK = (dS^T Q) * scale`

JVP (frozen stats):
- `dS = (dq K^T + Q dk^T) * scale`
- `dS_c = dS - sum(dS * P, axis=-1, keepdim=True)`
- `dP = P * dS_c`
- `dO = dP V + P dv`

HVP (sanity only):
- Defined for the *true* SDPA function (stats recomputed at perturbed points)
- Approximated via finite-difference of explicit VJP:
  - `H(x)v ≈ (∇f(x+εv) - ∇f(x-εv)) / (2ε)`

For full derivations and background, see:
- `docs/phase9_backward.md` (dQ/dK/dV + JVP details)
- `docs/phase9_hvp.md` (HVP contract + FD definition)
- `docs/theory/*` (softmax JVP / autodiff background)

## What this is NOT
- Not a full autograd replacement.
- Not stable second-order differentiation (HVP is directional sanity only).
- Not feature-complete SDPA (no mask/dropout/causal).
- Not optimized for all GPUs or large head dims (Phase 9 correctness first).

## Why this exists (Phase 10 on‑ramp)
Phase 9 locks the autodiff boundary inside a fused kernel with explicit math.
Phase 10 can now treat this as a stable operator surface for:
- meta-learning experiments,
- higher-order analyses,
- and performance studies without hidden fallback paths.
