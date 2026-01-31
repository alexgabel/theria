import pytest
import torch

from theria.attention.triton_qk import triton_sdpa_fused
from theria.attention.triton_sdpa_backward import sdpa_jvp


def _l2_normalize(t):
    return t / (t.norm() + 1e-8)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape",
    [
        {"B": 1, "H": 1, "T": 32, "M": 32, "D": 32},
        {"B": 2, "H": 2, "T": 64, "M": 64, "D": 64},
    ],
)
def test_fused_jvp_matches_finite_difference(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, T, M, D = shape["B"], shape["H"], shape["T"], shape["M"], shape["D"]
    dtype = torch.float16

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, M, D, device=device, dtype=dtype)
    v = torch.randn(B, H, M, D, device=device, dtype=dtype)

    dq = _l2_normalize(torch.randn_like(q))
    dk = _l2_normalize(torch.randn_like(k))
    dv = _l2_normalize(torch.randn_like(v))

    scale = 1.0 / (D ** 0.5)

    # Baseline forward to get fixed stats (m, l)
    with torch.no_grad():
        _, m, l = triton_sdpa_fused(q, k, v, return_stats=True)

    # Explicit JVP (no autograd) — by definition uses fixed stats
    jvp_explicit = sdpa_jvp(q, k, v, dq, dk, dv, m, l, scale).float()

    # Autograd JVP of the same frozen-stats function
    def f(q_, k_, v_):
        scores = torch.matmul(q_.float(), k_.float().transpose(-2, -1)) * scale
        p = torch.exp(scores - m.unsqueeze(-1)) / l.unsqueeze(-1)
        return torch.matmul(p, v_.float())

    _, jvp_auto = torch.autograd.functional.jvp(
        f, (q, k, v), (dq, dk, dv), create_graph=False, strict=True
    )

    torch.testing.assert_close(
        jvp_explicit, jvp_auto.float(), rtol=5e-2, atol=5e-2
    )
