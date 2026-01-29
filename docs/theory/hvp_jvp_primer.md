# Primer on Jacobian-Vector Products (JVPs) and Hessian-Vector Products (HVPs)

## 1. Motivation

Jacobian-vector products (JVPs) and Hessian-vector products (HVPs) are fundamental tools in meta-learning algorithms such as Model-Agnostic Meta-Learning (MAML). MAML requires differentiation through optimization steps, which involves higher-order derivatives. Direct computation of full Hessians is prohibitively expensive in terms of memory and computation, especially for large models. Instead, JVPs and HVPs allow efficient implicit computation of directional derivatives without materializing large Jacobian or Hessian matrices.

## 2. Definitions

- **Jacobian**: For a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J_f$ is the $m \times n$ matrix of partial derivatives:
  $$
  (J_f)_{ij} = \frac{\partial f_i}{\partial x_j}.
  $$

- **Jacobian-vector product (JVP)**: Given a vector $v \in \mathbb{R}^n$, the JVP is
  $$
  J_f v = \left.\frac{d}{d\epsilon} f(x + \epsilon v) \right|_{\epsilon=0}.
  $$
  This represents the directional derivative of $f$ at $x$ in the direction $v$.

- **Vector-Jacobian product (VJP)**: Given $u \in \mathbb{R}^m$, the VJP is
  $$
  u^\top J_f = \left.\frac{d}{d\epsilon} u^\top f(x + \epsilon v) \right|_{\epsilon=0},
  $$
  commonly used in reverse-mode autodiff.

- **Hessian**: For a scalar-valued function $g: \mathbb{R}^n \to \mathbb{R}$, the Hessian $H_g$ is the $n \times n$ matrix of second derivatives:
  $$
  (H_g)_{ij} = \frac{\partial^2 g}{\partial x_i \partial x_j}.
  $$

- **Hessian-vector product (HVP)**: For $v \in \mathbb{R}^n$,
  $$
  H_g v = \left.\frac{d}{d\epsilon} \nabla g(x + \epsilon v) \right|_{\epsilon=0},
  $$
  i.e., the directional derivative of the gradient of $g$ in direction $v$.

## 3. Autograd Perspectives

- **Reverse-mode autodiff (VJP)** computes vector-Jacobian products efficiently and is the default in PyTorch because it is optimal when the function output dimension is small (e.g., scalar losses).

- **Forward-mode autodiff (JVP)** computes Jacobian-vector products and is efficient when the input dimension is small. It is less commonly used in PyTorch but essential for certain higher-order derivative computations.

- **Computing HVP via double backward**: If the computational graph supports it, HVPs can be computed by applying reverse-mode autodiff twice—first to obtain the gradient, then to differentiate the gradient in a direction. However, this requires the graph to be fully differentiable twice, which is not always the case.

## 4. Simple Worked Example

Consider a scalar function $g(x) = \frac{1}{2} x^\top A x$ where $A$ is a symmetric matrix and $x \in \mathbb{R}^n$.

- Gradient:
  $$
  \nabla g(x) = A x.
  $$

- Hessian:
  $$
  H_g = A.
  $$

- JVP of \( \nabla g \) in direction \( v \):
  $$
  J_{\nabla g} v = A v,
  $$
  which is exactly the Hessian-vector product $H_g v$.

This illustrates that HVPs can be computed as JVPs of gradients.

## 5. Relevance to Theria

In full MAML implementations, HVPs appear naturally when differentiating through inner-loop gradient updates. Efficient and correct computation of HVPs is critical for stable meta-learning.

However, certain operations in Theria, such as SDPA or fused attention kernels, break the assumptions needed for double backward to work, as their backward passes are not fully differentiable. This makes automatic HVP computation via double backward unreliable.

Therefore, Theria requires explicit JVP/HVP rules for these operations to ensure correctness and enable higher-order differentiation.

## 6. Phase-2 Takeaway

- Combining reference attention implementations with explicit HVP rules provides a correctness baseline for meta-learning.

- Phase-3 will focus on systematically locating and repairing missing derivative edges in fused kernels to restore double backward support.

---

*Future work*: Integrating these explicit derivative rules with efficient GPU kernels (e.g., Triton) will be important for scaling meta-learning with complex attention mechanisms.
