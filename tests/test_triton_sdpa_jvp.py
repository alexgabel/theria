import pytest
import torch

from theria.attention.custom import sdpa_custom
from theria.attention.triton_sdpa_backward import sdpa_jvp
from theria.attention.reference_jvp_sdpa import sdpa_jvp_reference


@pytest.mark.gpu
@pytest.mark.phase9
@pytest.mark.parametrize(
    "shape",
    [
        {"B": 1, "H": 1, "T": 32, "M": 32, "D": 32},
        {"B": 2, "H": 4, "T": 128, "M": 128, "D": 64},
    ],
)
def test_triton_sdpa_jvp_matches_reference(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, T, M, D = shape["B"], shape["H"], shape["T"], shape["M"], shape["D"]

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, M, D, device=device, dtype=torch.float16)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)

    scale = 1.0 / (D ** 0.5)

    # Forward stats for JVP reconstruction (held fixed during JVP)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    m = scores.max(dim=-1).values
    l = torch.exp(scores - m.unsqueeze(-1)).sum(dim=-1)

    # Reference JVP with *frozen* stats to match sdpa_jvp definition
    out_jvp_ref = sdpa_jvp_reference(q, k, v, dq, dk, dv, scale)

    # Our explicit JVP (no autograd in implementation)
    out_jvp = sdpa_jvp(q, k, v, dq, dk, dv, m, l, scale)

    torch.testing.assert_close(
        out_jvp.to(torch.float32), out_jvp_ref.to(torch.float32), rtol=2e-2, atol=2e-2
    )
