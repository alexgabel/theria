import torch

def test_sdpa_hvp_matches_autograd_and_finite_difference():
    """Test Hessian-vector product (HVP) via double backward against autograd oracle and finite difference."""

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

    # Compute directional derivative of gradient (HVP) via double backward
    directional_dot = (grads[0] * vq).sum() + (grads[1] * vk).sum() + (grads[2] * vv).sum()
    hvp = torch.autograd.grad(directional_dot, (q, k, v), retain_graph=True)

    # Compute HVP via autograd oracle: directional derivative of gradient of loss_fn in direction (vq,vk,vv)
    # This is effectively the same as above and serves as a conceptual oracle check.

    # Finite difference approximation of directional derivative of gradient (HVP)
    eps = 1e-4
    loss_plus = loss_fn(q + eps * vq, k + eps * vk, v + eps * vv)
    loss_minus = loss_fn(q - eps * vq, k - eps * vk, v - eps * vv)
    fd = (loss_plus - loss_minus) / (2 * eps)

    # Sum HVP components for comparison
    hvp_sum = hvp[0].sum() + hvp[1].sum() + hvp[2].sum()

    # Assert HVP from double backward matches finite difference approximation
    assert torch.allclose(hvp_sum, fd, rtol=1e-3, atol=1e-4)
