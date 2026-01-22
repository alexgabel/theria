# Operator Contracts in theria

## Goal
Define the minimal mathematical and autodiff contracts required to support
meta-learning (e.g. MAML) with attention-like operators.

This repo does NOT assume full second-order autodiff of backward kernels.
Instead, we require support for:
- First-order gradients (VJP)
- Directional derivatives (JVP / HVP)

## Scaled Dot-Product Attention (SDPA)

Given:
Q ∈ R^{B×H×N×D}
K ∈ R^{B×H×M×D}
V ∈ R^{B×H×M×Dv}

Define:
A = softmax(QKᵀ / √D)
O = AV

### Required outputs
- O

### Required derivatives
- VJP: ∂L/∂Q, ∂L/∂K, ∂L/∂V
- JVP: J_O(Q̇, K̇, V̇)

### Explicit non-requirements
- Full Hessian materialization
- gradgrad through backward kernels

## Meta-learning requirement

For MAML with inner loop step:
θ' = θ − α ∇_θ L_inner(θ)

Outer gradient requires:
∇_θ L_outer(θ') = ∇_θ L_outer − α H_inner · ∇_{θ'} L_outer

Thus, HVP support is sufficient.

## Design invariant

Any optimized kernel must satisfy:
- Numerical equivalence to reference attention
- Correct JVP/HVP behavior
- Stable gradients under gradcheck