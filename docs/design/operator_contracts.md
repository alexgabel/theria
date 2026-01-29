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
$$
Q \in \mathbb{R}^{B\times H\times N\times D}
$$
$$
K \in \mathbb{R}^{B\times H\times M\times D}
$$
$$
V \in \mathbb{R}^{B\times H\times M\times D_v}
$$

Define:
$$
A = \mathrm{softmax}(QK^\top / \sqrt{D})
$$
$$
O = AV
$$

### Required outputs
- $O \in \mathbb{R}^{B\times H\times N\times D_v}$

### Required derivatives
- VJP (standard backward): $\partial L/\partial Q$, $\partial L/\partial K$, $\partial L/\partial V$
- JVP: $J_O(\dot Q, \dot K, \dot V)$ (needed to form HVPs)

### Explicit non-requirements
- Full Hessian materialization
- gradgrad through backward kernels

### Higher-order requirement

- Forward must be numerically correct.
- Backward (VJP) must be correct and differentiable when invoked with `create_graph=True`.
- Kernel must expose a JVP of the attention output with respect to (Q, K, V) or an equivalent directional-derivative rule sufficient to construct HVPs.
- Compliance = passes the `tests/test_attention_boundary_*` suite (e.g., `tests/test_attention_boundary_gpu.py`), which explicitly checks forward, backward, and grad-of-grad behavior.

## Meta-learning requirement

For MAML with inner loop step:
$$
\theta' = \theta - \alpha \nabla_\theta L_{\mathrm{inner}}(\theta)
$$

Outer gradient requires:
$$
\nabla_\theta L_{\mathrm{outer}}(\theta') = \nabla_\theta L_{\mathrm{outer}}
 - \alpha H_{\mathrm{inner}} \cdot \nabla_{\theta'} L_{\mathrm{outer}}
$$

Thus, HVP support is sufficient.

## Design invariant

Any optimized kernel must satisfy:
- Numerical equivalence to reference attention
- Correct JVP/HVP behavior
- Stable gradients under gradcheck
