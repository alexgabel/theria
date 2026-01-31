import pytest
import torch

from theria.attention.triton_sdpa_backward import sdpa_jvp
from theria.attention.triton_qk import triton_sdpa_fused


@pytest.mark.gpu
def test_jvp_frozen_stats_differs_from_recomputed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, T, M, D = 1, 1, 16, 16, 16
    dtype = torch.float16

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, M, D, device=device, dtype=dtype)
    v = torch.randn(B, H, M, D, device=device, dtype=dtype)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)

    scale = 1.0 / (D ** 0.5)

    # Saved stats (fixed)
    with torch.no_grad():
        _, m_fixed, l_fixed = triton_sdpa_fused(q, k, v, return_stats=True)

    # Frozen-stats JVP
    jvp_frozen = sdpa_jvp(q, k, v, dq, dk, dv, m_fixed, l_fixed, scale).float()

    # Recomputed-stats JVP via finite difference that recomputes softmax stats
    eps = 1e-2
    with torch.no_grad():
        out_plus = triton_sdpa_fused(q + eps * dq, k + eps * dk, v + eps * dv, return_stats=False)
        out_minus = triton_sdpa_fused(q - eps * dq, k - eps * dk, v - eps * dv, return_stats=False)
    jvp_recompute = ((out_plus - out_minus) / (2 * eps)).float()

    # They should not be equal (contract explicitly freezes stats)
    assert not torch.allclose(jvp_frozen, jvp_recompute, rtol=1e-3, atol=1e-3)

