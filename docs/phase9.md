# Phase 9: Explicit Frozen-Stats SDPA Operator (FSDPA)

## Overview
Phase 9 defines a frozen-stats SDPA operator whose forward/backward/JVP/HVP contracts are explicit, reproducible, and guarded by assertions. Frozen statistics ensure the backward/JVP paths stay local, memory-bounded, and implementable inside a fused kernel without recomputing the softmax normalization, which is critical for Triton fusion. The goal is to “lock” the autodiff boundary inside a fused kernel so higher-order experiments (Phase 10) can run on a trusted interface that exposes this Frozen-Stats SDPA Operator.

## What this implements
- Forward: `triton_sdpa_fused`
- Backward: `sdpa_bwd_dq`, `sdpa_bwd_dk`, `sdpa_bwd_dv` (reconstructing `P` from saved `m,l`)
- Frozen-stats JVP: `sdpa_jvp`
- HVP sanity: `hvp_fd_vjp` (finite-difference of explicit VJP; no Triton HVP kernel)

## Saved statistics (used in backward/JVP)
- Scores: \(S_{ij} = q_i \cdot k_j^T \cdot \text{scale}\), \(\text{scale} = 1/\sqrt{D}\).
- Row max: \(m_i = \max_j S_{ij}\).
- Row sumexp: \(l_i = \sum_j \exp(S_{ij} - m_i)\).
- Reconstruction: \(P_{ij} = \exp(S_{ij} - m_i)/l_i\).
- \(m,l\) are frozen throughout backward/JVP (they are treated as constants; no gradients computed w.r.t. them).

## Backward formulas (frozen stats)
Let \(O = PV\) and \(G = dO\). Then:
- \(dV = P^T G\).
- \(dP = G V^T\).
- \(z_i = \sum_j dP_{ij} P_{ij}\) (row-wise centering term).
- \(dS = P \odot (dP - z)\) (centered `dP`).
- \(dQ = (dS K) \cdot \text{scale}\).
- \(dK = (dS^T Q) \cdot \text{scale}\).

This is the same math the Triton backward kernels implement blockwise, so their masks/loops are consistent with these formulas.

## Frozen-stats JVP contract
- Inputs: \(q,k,v,dq,dk,dv,m,l,scale\).
- \(m,l\) are the same row stats from the saved forward pass.
- \(dS = (dq \cdot K^T + Q \cdot dk^T) \cdot \text{scale}\).
- \(dS_c = dS - \sum(dS \odot P, \text{axis}=-1, keepdim=True)\).
- \(dP = P \odot dS_c\).
- \(dO = dP \cdot V + P \cdot dv\).
- Implementation accumulates in fp32 and casts back to the input dtype.

**Warning:** This JVP is not `autograd.functional.jvp(triton_sdpa_fused)`; it linearizes the frozen-stats SDPA operator (holds `m,l` fixed). Phase 10 diagnostics leverage this fixed stats surface to check that the backward graph produced by these primitives actually supports higher-order differentiation inside realistic MAML loops.

## HVP sanity contract
- Defined for the *true* SDPA function (stats recomputed at perturbed inputs).
- Approximation: finite-difference of the explicit VJP (central difference).
- Diagnostics treat it as sanity-only; expect noise for small eps.
- No daily-use Triton HVP kernel exists yet.

## Guardrails
- Supported shapes: Q (B,H,T,D), K/V (B,H,M,D), dO (B,H,T,D), stats (B,H,T).
- D and Dv must satisfy \(D \le 64\) and \(Dv = D\) (v0).
- Inputs must be CUDA tensors for Triton kernels and contiguous.
- No mask/dropout/causal support.
- Unsupported shapes/dtypes raise loudly (no fallback).
- Frozen-stats contract applies to JVP: stats are not recomputed.

## Known failure modes
- Non-contiguous inputs, unsupported head dims, or Dv ≠ D.
- NaNs when stats mismatch scores or inputs contain inf/nan; fp16 exponentials fragile.
- HVP FD is noisy for too-small eps.
- Autograd second-order will never match frozen-stats JVP/HVP by design.

## Why this matters (Phase 10 foundation)
Phase 9 locks an explicit autodiff interface so we can run Phase 10 experiments with confidence. Having one file means:
- tests can target a single contract.
- diagnostics know exactly what “frozen stats” JVP/HVP mean.
- Phase 10 can treat `sdpa_bwd_*` and `sdpa_jvp` as stable primitives.
