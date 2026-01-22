## Hessian–Vector Products for SDPA

### Motivation
This section derives Hessian–vector products for scaled dot-product 
attention, identifying the minimal intermediate quantities required 
to support meta-learning without full second-order backpropagation.

### Forward and First-Order Backward
(Standard equations.)

### Softmax Jacobian–Vector Product

Let $S \in \mathbb{R}^{T \times T}$ denote the (row-wise) attention score matrix
\[
S = \frac{QK^\top}{\sqrt{d}}.
\]
The softmax is applied row-wise to $S$:
\[
P = \mathrm{softmax}(S), \quad P_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}.
\]

We are interested not in the full Jacobian of the softmax, but in its
**Jacobian–vector product (JVP)**, i.e. the directional derivative
\[
\dot P = J_{\mathrm{softmax}}(S)\,\dot S,
\]
for a given perturbation $\dot S$.

---

#### Row-wise formulation

Because softmax acts independently on each row, it suffices to consider
one row $s \in \mathbb{R}^T$ with softmax output $p = \mathrm{softmax}(s)$.
Let $\dot s$ be the corresponding perturbation.

The Jacobian of the softmax for a single row is:
\[
J_{ij} = \frac{\partial p_i}{\partial s_j}
       = p_i (\delta_{ij} - p_j).
\]

Applying this Jacobian to $\dot s$ yields the JVP:
\[
\dot p_i = \sum_j J_{ij} \dot s_j
          = p_i \left( \dot s_i - \sum_j p_j \dot s_j \right).
\]

In vector form:
\[
\dot p = p \odot \left( \dot s - \langle \dot s \rangle_p \right),
\]
where $\odot$ denotes elementwise multiplication and
\[
\langle \dot s \rangle_p := \sum_j p_j \dot s_j
\]
is the probability-weighted mean of $\dot s$.

---

#### Batched / matrix form

Applied row-wise to the full score matrix $S$, the softmax JVP becomes:
\[
\dot P = P \odot \left( \dot S - \mathrm{row\_sum}(P \odot \dot S) \right),
\]
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
\[
\dot S = \frac{1}{\sqrt{d}} \left( \dot Q K^\top + Q \dot K^\top \right).
\]

The softmax JVP is therefore the **core nonlinearity** governing all
second-order effects in attention. Once $\dot P$ is available, Hessian–vector
products with respect to $Q$, $K$, and $V$ reduce to linear algebra involving
$P$, $\dot P$, and the upstream gradient.

This observation underlies the Route B strategy: higher-order information for
attention can be recovered by explicitly implementing the softmax JVP, without
requiring a fully differentiable backward pass.

### Attention HVP Decomposition
- HVP_Q
- HVP_K
- HVP_V

### Implications for Kernel Design
What must be saved / exposed.
Why FlashAttention fails.
Why Triton + JVP is viable.