import torch
import pytest

from theria.attention.reference import reference_attention
from theria.attention.reference_jvp_sdpa import sdpa_jvp_reference


@pytest.mark.parametrize(
    "shape",
    [
        {"B": 1, "H": 1, "T": 4, "M": 4, "D": 3},
        {"B": 1, "H": 2, "T": 5, "M": 6, "D": 4},
    ],
)
def test_reference_jvp_matches_autograd(shape):
    torch.manual_seed(0)
    B, H, T, M, D = shape["B"], shape["H"], shape["T"], shape["M"], shape["D"]
    device = torch.device("cpu")

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float64)
    k = torch.randn(B, H, M, D, device=device, dtype=torch.float64)
    v = torch.randn(B, H, M, D, device=device, dtype=torch.float64)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)

    scale = 1.0 / (D ** 0.5)

    def f(q_, k_, v_):
        return reference_attention(q_, k_, v_)

    _, jvp_autograd = torch.autograd.functional.jvp(
        f, (q, k, v), (dq, dk, dv), create_graph=False, strict=True
    )

    jvp_ref = sdpa_jvp_reference(q, k, v, dq, dk, dv, scale)

    torch.testing.assert_close(jvp_ref, jvp_autograd, rtol=1e-9, atol=1e-9)
