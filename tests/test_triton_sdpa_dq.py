import pytest
import torch

from theria.attention.reference_backward import sdpa_backward_reference
from theria.attention.triton_sdpa_backward import sdpa_bwd_dq


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape",
    [
        {"B": 1, "H": 1, "T": 32, "M": 32, "D": 32},
        {"B": 2, "H": 4, "T": 128, "M": 128, "D": 64},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_triton_dq_matches_reference(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, T, M, D = shape["B"], shape["H"], shape["T"], shape["M"], shape["D"]
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, M, D, device=device, dtype=dtype)
    v = torch.randn(B, H, M, D, device=device, dtype=dtype)
    dout = torch.randn(B, H, T, D, device=device, dtype=dtype)

    scale = 1.0 / (D ** 0.5)
    q_fp, k_fp, v_fp, do_fp = [t.to(torch.float32) for t in (q, k, v, dout)]

    # Forward stats
    scores = torch.matmul(q_fp, k_fp.transpose(-2, -1)) * scale
    m = scores.max(dim=-1).values
    l = torch.exp(scores - m.unsqueeze(-1)).sum(dim=-1)

    dq_ref, _, _ = sdpa_backward_reference(q_fp, k_fp, v_fp, do_fp, scale)
    dq_tri = sdpa_bwd_dq(q_fp, k_fp, v_fp, do_fp, m, l, scale)

    assert torch.isfinite(dq_tri).all()
    torch.testing.assert_close(dq_tri.to(torch.float32), dq_ref.to(torch.float32), rtol=2e-2, atol=2e-2)
