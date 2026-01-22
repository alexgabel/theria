import torch
import pytest


def reference_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def test_reference_attention_hvp_against_finite_difference():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64

    q = torch.randn(2, 1, 8, 16, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=False)
    v = torch.randn_like(q, requires_grad=False)
    direction = torch.randn_like(q)

    def loss_fn(q_tensor):
        return reference_attention(q_tensor, k, v).sum()

    _, hvp = torch.autograd.functional.hvp(loss_fn, q, direction)

    eps = 1e-4
    def capture_grad(base_q):
        q_clone = base_q.clone().detach().requires_grad_(True)
        loss = loss_fn(q_clone)
        return torch.autograd.grad(loss, q_clone)[0]

    grad_plus = capture_grad(q + eps * direction)
    grad_minus = capture_grad(q - eps * direction)
    numeric_hvp = (grad_plus - grad_minus) / (2 * eps)

    torch.testing.assert_close(hvp, numeric_hvp, rtol=1e-5, atol=1e-6)


def test_reference_attention_hvp_double_backward():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64

    q = torch.randn(2, 1, 8, 16, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=False)
    v = torch.randn_like(q, requires_grad=False)
    direction = torch.randn_like(q)

    # Define scalar loss function
    loss = reference_attention(q, k, v).sum()

    # Compute first-order gradient with create_graph=True to enable higher-order derivatives
    grad, = torch.autograd.grad(loss, q, create_graph=True)

    # Compute directional derivative <∇L, direction>
    directional_derivative = (grad * direction).sum()

    # Compute Hessian-vector product by differentiating directional derivative w.r.t q
    hvp, = torch.autograd.grad(directional_derivative, q)

    # Compute numeric HVP by finite differences for comparison
    eps = 1e-4
    def capture_grad(base_q):
        q_clone = base_q.clone().detach().requires_grad_(True)
        loss_val = reference_attention(q_clone, k, v).sum()
        return torch.autograd.grad(loss_val, q_clone)[0]

    grad_plus = capture_grad(q + eps * direction)
    grad_minus = capture_grad(q - eps * direction)
    numeric_hvp = (grad_plus - grad_minus) / (2 * eps)

    torch.testing.assert_close(hvp, numeric_hvp, rtol=1e-5, atol=1e-6)
