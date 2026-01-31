import pytest
import torch

from theria.attention.reference import reference_attention
from theria.attention.triton_qk import triton_sdpa_fused
from theria.attention.hvp_utils import hvp_fd_vjp


def _l2(t):
    return t / (t.norm() + 1e-8)


@pytest.mark.gpu
@pytest.mark.phase9
def test_hvp_fd_matches_autograd_directional():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B = H = T = M = D = 8
    dtype = torch.float32

    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)

    dq = _l2(torch.randn_like(q))
    dk = _l2(torch.randn_like(k))
    dv = _l2(torch.randn_like(v))

    # Autograd HVP on reference attention (CPU fp64 for stability)
    q_cpu = q.detach().cpu().double().requires_grad_(True)
    k_cpu = k.detach().cpu().double().requires_grad_(True)
    v_cpu = v.detach().cpu().double().requires_grad_(True)

    def loss_fn(q_, k_, v_):
        out = reference_attention(q_, k_, v_)
        return out.sum()

    # First VJP
    loss = loss_fn(q_cpu, k_cpu, v_cpu)
    grads = torch.autograd.grad(loss, (q_cpu, k_cpu, v_cpu), create_graph=True)

    # Directional second VJP: grads · (dq,dk,dv)
    dot = (grads[0] * dq.cpu()).sum() + (grads[1] * dk.cpu()).sum() + (grads[2] * dv.cpu()).sum()
    hvp_ref = torch.autograd.grad(dot, (q_cpu, k_cpu, v_cpu), retain_graph=False)

    # Explicit finite-difference HVP using Triton backward pipeline
    hvp_q, hvp_k, hvp_v = hvp_fd_vjp(q.detach(), k.detach(), v.detach(), dq, dk, dv, eps=5e-3)

    # Cosine similarity is sufficient for sanity (second-order amplifies noise)
    def _cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)

    # Second-order finite differences are noisy; require coarse agreement only
    assert _cos(hvp_q, hvp_ref[0].to(device)) > 0.25
    assert _cos(hvp_k, hvp_ref[1].to(device)) > 0.25
    assert _cos(hvp_v, hvp_ref[2].to(device)) > 0.25
