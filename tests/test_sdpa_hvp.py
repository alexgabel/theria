import torch
import pytest

def test_sdpa_double_backward_fails():
    """Test that double backward fails for SDPA due to custom backward implementation limitations."""

    from theria.autograd.sdpa_function import SDPAFunction

    torch.manual_seed(0)
    dtype = torch.double

    # Small deterministic inputs
    q = torch.randn(1, 1, 3, 4, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 1, 3, 4, dtype=dtype, requires_grad=True)
    v = torch.randn(1, 1, 3, 4, dtype=dtype, requires_grad=True)

    def loss_fn(q, k, v):
        out = SDPAFunction.apply(q, k, v)
        return out.sum()

    # Direction vectors for HVP
    vq = torch.randn_like(q)
    vk = torch.randn_like(k)
    vv = torch.randn_like(v)

    # Compute loss and first-order gradients
    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)

    # Double backward is expected to fail because SDPAFunction implements a custom backward
    # that does not support higher-order gradients.
    directional_dot = (grads[0] * vq).sum() + (grads[1] * vk).sum() + (grads[2] * vv).sum()
    with pytest.raises(RuntimeError):
        torch.autograd.grad(directional_dot, (q, k, v), retain_graph=True)

def test_sdpa_explicit_hvp_matches_finite_difference():
    """Test explicit Hessian-vector product (HVP) implementation matches finite difference approximation.

    This explicit HVP path is the intended Phase-2 solution for SDPA to enable second-order methods,
    since double backward is not supported.
    """

    from theria.autograd.sdpa_function import SDPAFunction
    from theria.attention.reference_hvp_sdpa import sdpa_hvp  # explicit HVP implementation

    torch.manual_seed(0)
    dtype = torch.double

    # Small deterministic inputs
    q = torch.randn(1, 1, 3, 4, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 1, 3, 4, dtype=dtype, requires_grad=True)
    v = torch.randn(1, 1, 3, 4, dtype=dtype, requires_grad=True)

    # Direction vectors for HVP
    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)

    def loss_fn(q, k, v):
        out = SDPAFunction.apply(q, k, v)
        return out.sum()

    # Compute gradient at original point
    loss = loss_fn(q, k, v)
    grad_q, grad_k, grad_v = torch.autograd.grad(loss, (q, k, v), create_graph=True)

    # Compute explicit HVP using reference implementation
    hvp_q, hvp_k, hvp_v = sdpa_hvp(q, k, v, dq, dk, dv)

    # Finite difference approximation of directional second derivative:
    # Define g(t) = <∇L(q+t*dq, k+t*dk, v+t*dv), (dq, dk, dv)>
    # Approximate g'(0) ≈ (g(eps) - g(-eps)) / (2*eps)
    eps = 1e-4

    def directional_derivative(t):
        q_t = q + t * dq
        k_t = k + t * dk
        v_t = v + t * dv
        loss_t = loss_fn(q_t, k_t, v_t)
        grads_t = torch.autograd.grad(loss_t, (q_t, k_t, v_t), create_graph=False)
        return (grads_t[0] * dq).sum() + (grads_t[1] * dk).sum() + (grads_t[2] * dv).sum()

    fd = (directional_derivative(eps) - directional_derivative(-eps)) / (2 * eps)

    # Inner product of explicit HVP with direction vectors
    hvp_inner = (hvp_q * dq).sum() + (hvp_k * dk).sum() + (hvp_v * dv).sum()

    # Assert explicit HVP inner product matches finite difference approximation
    assert torch.allclose(hvp_inner, fd, rtol=1e-3, atol=1e-4)
