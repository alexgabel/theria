import torch
from theria.autograd.sdpa_function import sdpa
from theria.attention.reference import reference_attention

def test_sdpa_matches_reference():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8, requires_grad=True)
    k = torch.randn(1, 1, 4, 8, requires_grad=True)
    v = torch.randn(1, 1, 4, 8, requires_grad=True)

    out1 = sdpa(q, k, v)
    out2 = reference_attention(q, k, v)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_sdpa_gradcheck():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 1, 4, 8, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 1, 4, 8, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(lambda *xs: sdpa(*xs), (q, k, v))
