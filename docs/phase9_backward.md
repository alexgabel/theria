# Phase 9 — Backward / JVP / HVP Contract

## Saved forward statistics
- Scores: \(S_{ij} = q_i \cdot k_j^T \cdot \text{scale}\), scale = 1/\sqrt{D}.
- Row-wise max: \(m_i = \max_j S_{ij}\).
- Row-wise sumexp: \(l_i = \sum_j e^{S_{ij} - m_i}\).
- Reconstruction used throughout backward/JVP:
  \[ P_{ij} = \exp(S_{ij} - m_i) / l_i \]
- In backward/JVP, \(m,l\) are treated as constants (no gradients w.r.t. stats).

## Backward formulas (frozen stats \(m,l\))
Let \(O = P V\) and incoming gradient \(G = dO\).

- \(\mathrm{dV} = P^T G\).
- \(\mathrm{dP} = G V^T\).
- Row-wise \(z_i = \sum_j \mathrm{dP}_{ij} P_{ij}\).
- \(\mathrm{dS} = P \odot (\mathrm{dP} - z)\).
- \(\mathrm{dQ} = (\mathrm{dS} K) \cdot \text{scale}\).
- \(\mathrm{dK} = (\mathrm{dS}^T Q) \cdot \text{scale}\).

## JVP formula (frozen stats)
Given tangents (dq, dk, dv):
- \(dS = (dq \cdot K^T + Q \cdot dk^T) * \text{scale}\).
- Center: \(dS_c = dS - (dS * P)_{row\,sum}\).
- \(dP = P * dS_c\).
- \(dO = dP \cdot V + P \cdot dv\).
JVP assumes \(m,l\) are fixed from the saved forward pass.
Implementation accumulates in fp32 and casts the output back to the input dtype.

**Warning:** This JVP is *not* equal to `autograd.functional.jvp(triton_sdpa_fused)`
because it holds \(m,l\) fixed. It linearizes the frozen-stats operator.

## HVP contract
- Implemented via finite difference of the explicit backward (VJP) using central diff; no Triton HVP kernel.
- Second-order values are for sanity only; tolerate noise.
- HVP uses recomputed forward stats at perturbed points (matches true function), not frozen stats.

**Warning:** Directional sanity only; not a stable Hessian interface.

## Assumptions / guardrails
- Supported shapes: head dims \(D, D_v \le 64\); no multiple-of requirement on T or M (tails masked).
- Inputs must be contiguous.
- No dropout, no masking, no causal support (v0).
- Q/K/V/Dout must be CUDA tensors; stats (m,l) come from fused forward.
- Frozen-stats contract applies to JVP: stats are not recomputed during JVP.
- No fallback: unsupported shapes/dtypes raise; there is no silent reference path.

## What is NOT guaranteed
- No explicit Triton JVP/HVP kernels (JVP is Python; HVP is FD of VJP).
- Numerical parity with autograd JVP/HVP is up to documented tolerances; mixed precision may introduce small drift.

## Known failure modes
- Non-contiguous inputs (guardrails assert and raise).
- Unsupported head dims (D, Dv > 64).
- NaNs can occur if stats are inconsistent with scores or inputs contain inf/nan;
  fp16 exponentials remain fragile for extreme logits.
- HVP finite differences can be noisy for small eps.
- Full autograd second-order will not match frozen-stats JVP/HVP by definition.
