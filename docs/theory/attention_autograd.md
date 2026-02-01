## Phase 2 Summary / Phase 3 Transition

Phase 2 established explicit Hessian–vector product (HVP) correctness for reference attention implementations, providing a reliable baseline for higher-order differentiation. In contrast, the `SDPAFunction` is now a *known* failure case for Hessian–vector products due to its opaque backward graph and insufficient saved intermediates. This document henceforth serves as the boundary specification for Phase 3, delineating the precise failure modes and guiding subsequent experimental investigation.

## Hessian–Vector Products for SDPA

### Motivation
This section derives Hessian–vector products for scaled dot-product 
attention, identifying the minimal intermediate quantities required 
to support meta-learning without full second-order backpropagation.

### Forward and First-Order Backward
(Standard equations.)

### Softmax Jacobian–Vector Product

Let $S \in \mathbb{R}^{T \times T}$ denote the (row-wise) attention score matrix
$$
S = \frac{QK^\top}{\sqrt{d}}.
$$
The softmax is applied row-wise to $S$:
$$
P = \mathrm{softmax}(S), \quad P_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}.
$$

We are interested not in the full Jacobian of the softmax, but in its
**Jacobian–vector product (JVP)**, i.e. the directional derivative
$$
\dot P = J_{\mathrm{softmax}}(S)\,\dot S,
$$
for a given perturbation $\dot S$.

---

#### Row-wise formulation

Because softmax acts independently on each row, it suffices to consider
one row $s \in \mathbb{R}^T$ with softmax output $p = \mathrm{softmax}(s)$.
Let $\dot s$ be the corresponding perturbation.

The Jacobian of the softmax for a single row is:
$$
J_{ij} = \frac{\partial p_i}{\partial s_j}
       = p_i (\delta_{ij} - p_j).
$$

Applying this Jacobian to $\dot s$ yields the JVP:
$$
\dot p_i = \sum_j J_{ij} \dot s_j
          = p_i \left( \dot s_i - \sum_j p_j \dot s_j \right).
$$

In vector form:
$$
\dot p = p \odot \left( \dot s - \langle \dot s \rangle_p \right),
$$
where $\odot$ denotes elementwise multiplication and
$$
\langle \dot s \rangle_p := \sum_j p_j \dot s_j
$$
is the probability-weighted mean of $\dot s$.

---

#### Batched / matrix form

Applied row-wise to the full score matrix $S$, the softmax JVP becomes:
$$
\dot P = P \odot \left( \dot S - \mathrm{row\_sum}(P \odot \dot S) \right),
$$
where $\mathrm{row\_sum}(\cdot)$ subtracts, from each row, its scalar
probability-weighted mean.

This expression is:
- numerically stable,
- linear in $\dot S$,
- computable without forming any Jacobians explicitly.

---

#### Connection to attention

In scaled dot-product attention, the perturbation $\dot S$ arises from
perturbations in $Q$ and $K$:
$$
\dot S = \frac{1}{\sqrt{d}} \left( \dot Q K^\top + Q \dot K^\top \right).
$$

The softmax JVP is therefore the **core nonlinearity** governing all
second-order effects in attention. Once $\dot P$ is available, Hessian–vector
products with respect to $Q$, $K$, and $V$ reduce to linear algebra involving
$P$, $\dot P$, and the upstream gradient.

This observation underlies the Route B strategy: higher-order information for
attention can be recovered by explicitly implementing the softmax JVP, without
requiring a fully differentiable backward pass.

### Attention HVP Decomposition

Explicit Hessian–vector products with respect to $Q$, $K$, and $V$—
denoted HVP\_Q, HVP\_K, and HVP\_V—are now implemented in `reference_hvp_sdpa.py`
and have been numerically validated against finite difference baselines.
These explicit decompositions concretely realize the theoretical framework
outlined above and serve as a reference for evaluating alternative attention implementations.

## Why SDPAFunction Fails Hessian–Vector Products

Although scaled dot-product attention is mathematically smooth, the current
`SDPAFunction` implementation does **not** support higher-order derivatives.
This failure is structural rather than numerical.

### Opaque backward graph

`SDPAFunction.backward` returns tensors that are **not connected to a
higher-order autograd graph**. Even when first-order gradients are computed with
`create_graph=True`, the backward pass itself is treated as an atomic operation.
As a result, the outputs of the backward pass do not carry `grad_fn` metadata
needed for second-order differentiation.

### Insufficient saved intermediates

The backward pass saves only those intermediates required to compute
first-order gradients. However, Hessian–vector products for attention require
additional quantities—most notably information equivalent to the softmax
Jacobian–vector product (JVP). These are not preserved by standard fused or
custom backward implementations.

### Consequence for autograd

When attempting to compute a Hessian–vector product via double backward
(e.g. differentiating ⟨∇L, v⟩), PyTorch raises an error indicating that the
relevant tensors do not require gradients (e.g., `RuntimeError: element 0 of
tensors does not require grad and does not have a grad_fn`). Autograd cannot
trace Jacobian–vector products *through* the backward pass because the backward
graph is opaque.

### Design contract

This defines the exact failure boundary motivating Route B:

- Forward outputs are correct.
- First-order backward gradients (even with `create_graph=True`) are supported.
- Second-order / grad-of-grad computations fail because the backward pass is
  not itself differentiable—the missing object is the **JVP of the attention
  output with respect to (Q, K, V)** (equivalently, a differentiable backward
  graph).
- Recovering higher-order information requires explicitly providing JVP or HVP
  rules for attention, rather than relying on double backward.

This observation motivates implementing attention with explicit Jacobian–vector
products (e.g. via softmax JVPs) or forward-mode rules, enabling meta-learning
and implicit differentiation without requiring a fully differentiable backward
kernel.

The boundary above is concretely witnessed by `tests/test_attention_boundary_gpu.py`,
which passes forward and first-order backward checks but fails on grad-of-grad.

### Implication for Phase 4

Any custom attention used in Phase 4 must explicitly expose a JVP (or equivalent
directional derivative) of the attention output with respect to `(Q, K, V)`;
without it, grad-of-grad paths remain broken regardless of forward/backward
correctness.

Phase 4 plan: supply the missing derivative explicitly via JVP/HVP rules in a
custom attention operator (e.g., `sdpa_custom`) rather than relying on fused
kernels’ backward graphs.

**Contract statement:** Optimized SDPA/FlashAttention provides a backward that is
not itself differentiable, so $\partial/\partial \theta$ of $\partial L/\partial \theta$
fails. Therefore, meta-learning that requires HVP/JVP through attention must use
(a) explicit attention ops composed of differentiable primitives, or (b) a custom
operator that exposes explicit JVP/HVP (or gradgrad) rules.

**CPU vs GPU boundary:** CPU reference attention (explicit primitives) supports
double backward/HVP, while GPU fused SDPA/FlashAttention paths do not; the GPU
boundary tests (`test_attention_boundary_gpu.py`) capture this distinction.

**Phase 5 pointer:** Phase 5 will formalize the explicit JVP/backward rules
(mathematical derivation + implementation) so the custom operator no longer
relies on autograd-in-backward scaffolding.

## Phase 5 — JVP-First Attention

Phase 5 makes forward-mode differentiation explicit. A **JVP** is a directional
derivative of the forward map (tangent propagation), and it is *not* equivalent
to a **VJP** (reverse-mode gradient). The distinction matters: you can have a
correct VJP while the JVP is missing or silently dropped at a fused kernel
boundary.

For attention, the JVP is defined by the softmax JVP identity:
$$
\dot P = P \odot \left( \dot S - \mathrm{row\_sum}(P \odot \dot S) \right),
$$
and
$$
\dot O = \dot P\,V + P\,\dot V.
$$

**Where PyTorch loses the tangent:** optimized SDPA/FlashAttention kernels expose
only an opaque backward; there is no forward-mode rule, and the kernel boundary
does not propagate tangents. A custom operator can preserve the tangent by
explicitly implementing the JVP using these analytic formulas.

**JVP contract witnesses (tests):** `tests/test_custom_attention_jvp.py`
(finite-difference + autograd parity) and the SDPA boundary xfail in
`tests/test_attention_boundary.py`.

## Phase 3 Goal: Locating the Attention Autograd Boundary

The experimental plan for Phase 3 involves systematically swapping the current
attention implementations along the spectrum:

- from the numerically validated reference attention with explicit HVPs,
- to the `SDPAFunction` mathematical formulation,
- and finally to the fused `SDPA` kernel.

This staged substitution aims to isolate the precise point at which Hessian–vector
products fail. The central hypothesis is that the failure arises from missing
softmax JVP computations or equivalent saved intermediate state within the
backward pass of fused kernels.

The objective at this stage is diagnostic: to precisely characterize the autograd
boundary within the attention computation graph. Fixing the failure or redesigning
backward kernels will follow once the root cause is fully understood.

```
Q,K,V --(matmul/scale)--> scores --(softmax)--> P --(matmul)--> out --(loss)--> L
                             ^                       ^
                             |                       |
                         saved?                  saved?
                             |                       |
                         backward                grad of grad ?
```

Boxes mark where saved tensors or grad_fn metadata must survive for higher-order
derivatives. Phase 3 identifies which arrows go missing in fused/optimized paths.
