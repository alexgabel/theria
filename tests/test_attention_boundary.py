# tests/test_attention_boundary.py
import pytest
import torch


def _make_qkv(device, dtype, B=1, H=1, T=4, D=8):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def _hvp_double_backward(loss_fn, q, k, v, dq, dk, dv):
    """
    Returns (hvp_q, hvp_k, hvp_v) via double backward:
      hvp = ∇_{q,k,v} <∇L, (dq,dk,dv)>
    Raises RuntimeError if grad-of-grad is unsupported.
    """
    loss = loss_fn(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
    dot = (grads[0] * dq).sum() + (grads[1] * dk).sum() + (grads[2] * dv).sum()
    hvp = torch.autograd.grad(dot, (q, k, v))
    return hvp


def _reference_attention(q, k, v):
    # (B,H,T,D) -> (B,H,T,D)
    d = q.shape[-1]
    s = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    p = torch.softmax(s, dim=-1)
    return p @ v


def test_boundary_reference_attention_supports_double_backward_cpu():
    device = torch.device("cpu")
    dtype = torch.double

    q, k, v = _make_qkv(device, dtype)
    dq, dk, dv = (torch.randn_like(q), torch.randn_like(k), torch.randn_like(v))

    def loss_fn(q, k, v):
        return _reference_attention(q, k, v).sum()

    hvp_q, hvp_k, hvp_v = _hvp_double_backward(loss_fn, q, k, v, dq, dk, dv)
    assert torch.isfinite(hvp_q).all()
    assert torch.isfinite(hvp_k).all()
    assert torch.isfinite(hvp_v).all()


def test_boundary_sdpa_function_double_backward_fails_cpu():
    """
    This is the *expected* boundary: SDPAFunction backward is opaque,
    so grad-of-grad via double backward fails.
    """
    from theria.autograd.sdpa_function import SDPAFunction

    device = torch.device("cpu")
    dtype = torch.double

    q, k, v = _make_qkv(device, dtype)
    dq, dk, dv = (torch.randn_like(q), torch.randn_like(k), torch.randn_like(v))

    def loss_fn(q, k, v):
        return SDPAFunction.apply(q, k, v).sum()

    with pytest.raises(RuntimeError):
        _hvp_double_backward(loss_fn, q, k, v, dq, dk, dv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDPA backend boundary tests.")
@pytest.mark.parametrize("backend", ["math", "flash", "mem_efficient"])
def test_boundary_torch_sdpa_backend_double_backward_gpu(backend):
    """
    Phase 3 diagnostic: probe PyTorch SDPA backends on GPU.
    We do not hard-assert pass/fail for flash/mem_efficient yet; we record behavior.
    """
    import torch.nn.functional as F

    device = torch.device("cuda")

    # math backend supports fp64; flash usually requires fp16/bf16.
    if backend == "math":
        dtype = torch.double
    else:
        # most flash kernels require fp16/bf16; keep dims small
        dtype = torch.float16

    q, k, v = _make_qkv(device, dtype, B=1, H=1, T=4, D=16)
    dq, dk, dv = (torch.randn_like(q), torch.randn_like(k), torch.randn_like(v))

    def sdpa(q, k, v):
        # torch SDPA expects (B,H,T,D) or (T,B,...) depending; we use (B,H,T,D)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    def loss_fn(q, k, v):
        return sdpa(q, k, v).sum()

    # select backend
    with torch.backends.cuda.sdp_kernel(
        enable_math=(backend == "math"),
        enable_flash=(backend == "flash"),
        enable_mem_efficient=(backend == "mem_efficient"),
    ):
        try:
            hvp = _hvp_double_backward(loss_fn, q, k, v, dq, dk, dv)
            # If it succeeds, require finiteness (still diagnostic)
            for t in hvp:
                assert torch.isfinite(t).all()
        except RuntimeError as e:
            # Diagnostic outcome: record boundary failure explicitly
            # Keep the test informative without making assumptions about PyTorch version.
            pytest.xfail(f"SDPA backend '{backend}' does not support double backward: {e}")