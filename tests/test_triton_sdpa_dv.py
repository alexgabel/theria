import pytest
import torch

from theria.attention.reference_backward_sdpa import sdpa_backward_reference
from theria.attention.triton_sdpa_backward import sdpa_bwd_dv


@pytest.mark.gpu
@pytest.mark.phase9
@pytest.mark.parametrize(
    "shape",
    [
        {"B": 1, "H": 1, "T": 32, "M": 32, "D": 32},
        {"B": 2, "H": 4, "T": 128, "M": 128, "D": 64},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_triton_dv_matches_reference(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, T, M, D = shape["B"], shape["H"], shape["T"], shape["M"], shape["D"]
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, M, D, device=device, dtype=dtype)
    v = torch.randn(B, H, M, D, device=device, dtype=dtype)

    # Forward reference to get m, l
    scale = 1.0 / (D ** 0.5)
    q_fp = q.to(torch.float32)
    k_fp = k.to(torch.float32)
    v_fp = v.to(torch.float32)
    scores = torch.matmul(q_fp, k_fp.transpose(-2, -1)) * scale
    m = scores.max(dim=-1).values
    l = torch.exp(scores - m.unsqueeze(-1)).sum(dim=-1)

    dout = torch.randn(B, H, T, D, device=device, dtype=dtype)
    dout_fp = dout.to(torch.float32)

    # Reference backward
    dq_ref, dk_ref, dv_ref = sdpa_backward_reference(q_fp, k_fp, v_fp, dout_fp, scale)

    # Triton dV
    dv_tri = sdpa_bwd_dv(q_fp, k_fp, dout_fp, m, l, scale)

    # Checks
    assert torch.isfinite(dv_tri).all()
    torch.testing.assert_close(dv_tri.to(torch.float32), dv_ref.to(torch.float32), rtol=2e-2, atol=2e-2)
