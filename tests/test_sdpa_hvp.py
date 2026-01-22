import torch

def test_sdpa_hvp_matches_finite_difference():
    """Validate Hessian–vector product (HVP) for SDPA via finite differences.

    This test operates at the operator-contract level and ensures that
    second-order derivatives exist and are numerically correct.
    """
    # NOTE:
    # This test currently fails because SDPAFunction.backward does not
    # construct a differentiable backward graph.
    # This is expected and motivates Route B (explicit JVP/HVP support).
    from theria.autograd.sdpa_function import SDPAFunction

    torch.manual_seed(0)
    dtype = torch.double

    # Small, deterministic tensors
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

    # Compute gradients
    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)

    # Directional derivative of gradient (HVP)
    dot = (grads[0] * vq).sum() + (grads[1] * vk).sum() + (grads[2] * vv).sum()
    hvp = torch.autograd.grad(dot, (q, k, v), retain_graph=False)

    # Finite-difference approximation
    eps = 1e-4
    loss_plus = loss_fn(q + eps * vq, k + eps * vk, v + eps * vv)
    loss_minus = loss_fn(q - eps * vq, k - eps * vk, v - eps * vv)
    fd = (loss_plus - loss_minus) / (2 * eps)

    hvp_sum = hvp[0].sum() + hvp[1].sum() + hvp[2].sum()

    assert torch.allclose(hvp_sum, fd, rtol=1e-3, atol=1e-4)
