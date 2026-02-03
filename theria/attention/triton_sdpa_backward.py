"""
Triton backward kernels for fused SDPA (Phase 9 scaffolding).

Phase 9 is frozen; Phase 10 may only apply bugfixes here.

This file currently implements sdpa_bwd_dv: dV = P^T @ dO
where P is reconstructed blockwise using saved row-wise m and l.

Constraints (v0):
- Dv == D
- D, Dv <= 64
- Contiguous inputs
"""

import torch
import triton
import triton.language as tl


_ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _assert_backward_contract(q, k, v, dout, m, l, require_v: bool = True):
    """Guardrails for Phase 9 backward/JVP.

    Raises loudly instead of silently falling back.
    """
    # Device / dtype
    if not (q.is_cuda and k.is_cuda and dout.is_cuda and (not require_v or (v is not None and v.is_cuda))):
        raise AssertionError("Phase9 backward requires CUDA tensors")
    dtype_check = (
        ("q", q),
        ("k", k),
        ("v", v) if require_v else (),
        ("dout", dout),
    )
    for item in dtype_check:
        if not item:
            continue
        name, t = item
        if t.dtype not in _ALLOWED_DTYPES:
            raise AssertionError(f"Unsupported dtype for {name}: {t.dtype}")
        # q/k/v are required contiguous in v0; dout may be strided.
        if name != "dout" and not t.is_contiguous():
            raise AssertionError(f"{name} must be contiguous for Phase9 backward")

    # Shapes
    B, H, T, D = q.shape
    if k.ndim != 4 or k.shape[0] != B or k.shape[1] != H or k.shape[3] != D:
        raise AssertionError("k must have shape (B,H,M,D) with D matching q")
    if dout.shape != (B, H, T, D):
        raise AssertionError("dout shape must be (B,H,T,D)")
    if require_v:
        if v is None or v.ndim != 4 or v.shape[0] != B or v.shape[1] != H or v.shape[3] != D:
            raise AssertionError("v must have shape (B,H,M,D) with D matching q")

    # Stats
    if m.shape != (B, H, T) or l.shape != (B, H, T):
        raise AssertionError("m,l must be saved forward stats with shape (B,H,T)")

    # Feature limits
    if D > 64:
        raise AssertionError("Phase9 v0 supports D<=64")

    # No mask/dropout/causal supported in v0; enforced by API (no args)


def _assert_jvp_contract(q, k, v, dq, dk, dv, m, l):
    """Guardrails for Phase 9 JVP (frozen-stats operator)."""
    # Device / dtype
    tensors = (
        ("q", q),
        ("k", k),
        ("v", v),
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
    )
    for name, t in tensors:
        if not t.is_cuda:
            raise AssertionError("Phase9 JVP requires CUDA tensors")
        if t.dtype not in _ALLOWED_DTYPES:
            raise AssertionError(f"Unsupported dtype for {name}: {t.dtype}")
        if not t.is_contiguous():
            raise AssertionError(f"{name} must be contiguous for Phase9 JVP")

    # Shapes
    B, H, T, D = q.shape
    if k.ndim != 4 or k.shape[0] != B or k.shape[1] != H or k.shape[3] != D:
        raise AssertionError("k must have shape (B,H,M,D) with D matching q")
    if v.ndim != 4 or v.shape[0] != B or v.shape[1] != H or v.shape[3] != D:
        raise AssertionError("v must have shape (B,H,M,D) with D matching q")
    if dq.shape != q.shape or dk.shape != k.shape or dv.shape != v.shape:
        raise AssertionError("dq, dk, dv must match q, k, v shapes respectively")
    if m.shape != (B, H, T) or l.shape != (B, H, T):
        raise AssertionError("m,l must be saved forward stats with shape (B,H,T)")

    # Feature limits
    if D > 64:
        raise AssertionError("Phase9 v0 supports D<=64")

    # Stats sanity
    if torch.any(l <= 0):
        raise AssertionError("Phase9 JVP requires l > 0 (forward sumexp stats)")


def _compute_row_delta(q, k, v, dout, m, l, scale):
    """Compute row-wise delta = sum_d(dO * O) for fallback shared backward.

    Used when the caller does not provide precomputed delta. This reconstructs
    frozen-stats probabilities once in PyTorch and is only a fallback path.
    """
    qf = q.float()
    kf = k.float()
    vf = v.float()
    dof = dout.float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    p = torch.exp(scores - m.unsqueeze(-1)) / torch.clamp(l.unsqueeze(-1), min=1e-12)
    out = torch.matmul(p, vf)
    return (dof * out).sum(dim=-1).contiguous()


def sdpa_bwd_shared_staged(q, k, v, dout, m, l, scale, delta=None, block_m: int = 64):
    """Shared staged backward that computes dQ/dK/dV in one query-chunk sweep.

    This path reuses reconstructed probabilities and softmax-adjoint terms per
    query block to avoid separate dq/dk/dv passes. It is a staging baseline for
    lower global-memory traffic before deeper Triton fusion work.
    """
    _assert_backward_contract(q, k, v, dout, m, l, require_v=True)
    if block_m <= 0:
        raise AssertionError("block_m must be positive")

    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    if D != Dk:
        raise AssertionError("D mismatch between q and k")
    if delta is not None and delta.shape != (B, H, T):
        raise AssertionError("delta must have shape (B,H,T)")

    qf = q.float()
    kf = k.float()
    vf = v.float()
    dof = dout.float()
    mf = m.float()
    lf = torch.clamp(l.float(), min=1e-12)

    kT = kf.transpose(-2, -1).contiguous()
    vT = vf.transpose(-2, -1).contiguous()

    dqf = torch.empty_like(qf)
    dkf = torch.zeros_like(kf)
    dvf = torch.zeros_like(vf)

    # Sweep query rows once and update all three gradients from shared terms.
    for m0 in range(0, T, block_m):
        m1 = min(T, m0 + block_m)
        q_blk = qf[:, :, m0:m1, :]
        do_blk = dof[:, :, m0:m1, :]
        m_blk = mf[:, :, m0:m1].unsqueeze(-1)
        l_blk = lf[:, :, m0:m1].unsqueeze(-1)

        scores_blk = torch.matmul(q_blk, kT) * scale
        p_blk = torch.exp(scores_blk - m_blk) / l_blk

        dp_blk = torch.matmul(do_blk, vT)
        if delta is None:
            delta_blk = (dp_blk * p_blk).sum(dim=-1, keepdim=True)
        else:
            delta_blk = delta[:, :, m0:m1].float().unsqueeze(-1)
        ds_blk = p_blk * (dp_blk - delta_blk)

        dqf[:, :, m0:m1, :] = torch.matmul(ds_blk, kf) * scale
        dkf += torch.matmul(ds_blk.transpose(-2, -1), q_blk) * scale
        dvf += torch.matmul(p_blk.transpose(-2, -1), do_blk)

    return dqf.to(q.dtype), dkf.to(k.dtype), dvf.to(v.dtype)


@triton.jit
def _sdpa_bwd_dv_kernel(
    Q, K, DO, DV, M, L,
    Tq, Tk, D, Hdim,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_dob, stride_doh, stride_dot, stride_dod,
    stride_dvbh, stride_dvn, stride_dvk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Each program computes DV for one (bh, key block).
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    # Accumulator for dv block
    acc = tl.zeros((BLOCK_N, BLOCK_DV), dtype=tl.float32)

    # Iterate over query blocks
    q_ptrs_base = Q + pid_bh * stride_qbh
    b = pid_bh // Hdim
    h = pid_bh % Hdim
    do_ptrs_base = DO + b * stride_dob + h * stride_doh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh

    for m0 in range(0, Tq, BLOCK_M):
        # Load q_block: (BLOCK_M, D)
        q_ptrs = q_ptrs_base + (m0 + offs_m) * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(
            q_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # Load k_block for this key tile: (BLOCK_N, D)
        k_ptrs = K + pid_bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(
            k_ptrs,
            mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # scores and P
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale  # (BLOCK_M, BLOCK_N)
        m_idx = m0 + offs_m  # (BM,1)
        m_mask = (m_idx < Tq)
        m_mask_f = m_mask.to(tl.float32)
        m_block = tl.load(m_ptrs_base + m_idx * stride_mt).to(tl.float32)
        l_block = tl.load(l_ptrs_base + m_idx * stride_lt).to(tl.float32)
        # Zero out invalid rows to avoid inf/NaN when subtracting or dividing.
        m_block = m_block * m_mask_f
        l_block = l_block * m_mask_f + (1.0 - m_mask_f)
        mask_m = m_mask_f
        mask_n = (offs_n < Tk).to(tl.float32)[None, :]
        mask_mn_f = mask_m * mask_n
        scores = scores - m_block
        scores = scores * mask_mn_f + (-1.0e9) * (1.0 - mask_mn_f)
        p = tl.exp(scores) * mask_mn_f
        l_block = tl.maximum(l_block, 1e-12)
        p = p / l_block

        # Load dO block: (BLOCK_M, Dv) with Dv==D here
        do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dot + offs_dv[None, :] * stride_dod
        do = tl.load(
            do_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_dv[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # accumulate: (BLOCK_N, BLOCK_DV) += p^T @ do
        acc += tl.dot(tl.trans(p), do, out_dtype=tl.float32, input_precision="ieee")

    # Store dv block
    dv_ptrs = DV + pid_bh * stride_dvbh + offs_n[:, None] * stride_dvn + offs_dv[None, :] * stride_dvk
    tl.store(dv_ptrs, acc, mask=(offs_n[:, None] < Tk) & (offs_dv[None, :] < D))


def sdpa_bwd_dv(q, k, dout, m, l, scale):
    """
    Compute dV = P^T @ dO using saved (m, l) stats from fused forward.

    Args:
        q, k: (B,H,T/D) and (B,H,M,D)
        dout: (B,H,T,Dv) with Dv == D (v0)
        m, l: (B,H,T) row-wise stats from forward
        scale: 1/sqrt(D)

    Returns:
        dv with shape (B,H,M,Dv), same dtype as inputs.
    """
    _assert_backward_contract(q, k, None, dout, m, l, require_v=False)
    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    _, _, Tdo, Dv = dout.shape
    assert D == Dk
    assert T == Tdo
    assert D == Dv, "v0 assumes Dv == D"

    dv = torch.empty((B, H, M, Dv), device=q.device, dtype=q.dtype)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    dv_ = dv.reshape(B * H, M, Dv)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_D = 64
    BLOCK_DV = 64
    grid = (
        B * H,
        triton.cdiv(M, BLOCK_N),
    )

    _sdpa_bwd_dv_kernel[grid](
        q_,
        k_,
        dout,
        dv_,
        m_,
        l_,
        T,
        M,
        D,
        H,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        dv_.stride(0),
        dv_.stride(1),
        dv_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
    )

    return dv


@triton.jit
def _sdpa_bwd_dk_kernel(
    Q, K, V, DO, DK, M, L, DELTA,
    Tq, Tk, D, Hdim,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_dob, stride_doh, stride_dot, stride_dod,
    stride_dkbh, stride_dkn, stride_dkk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
    stride_dbh, stride_dt,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)  # key block index

    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs_base = Q + pid_bh * stride_qbh
    b = pid_bh // Hdim
    h = pid_bh % Hdim
    do_ptrs_base = DO + b * stride_dob + h * stride_doh
    k_ptrs_base = K + pid_bh * stride_kbh
    v_ptrs_base = V + pid_bh * stride_vbh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh
    d_ptrs_base = DELTA + pid_bh * stride_dbh

    # We'll accumulate dk for this key block
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # Loop over query blocks
    for m0 in range(0, Tq, BLOCK_M):
        # load q
        q_ptrs = q_ptrs_base + (m0 + offs_m) * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(
            q_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        delta = tl.load(
            d_ptrs_base + (m0 + offs_m) * stride_dt,
            mask=((m0 + offs_m) < Tq),
            other=0.0,
        ).to(tl.float32)

        # One pass: delta is precomputed/shared across dQ and dK.
        k_ptrs = k_ptrs_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_ptrs_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(
            k_ptrs,
            mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptrs,
            mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
        m_idx = m0 + offs_m
        m_mask = (m_idx < Tq)
        m_mask_f = m_mask.to(tl.float32)
        m_block = tl.load(m_ptrs_base + m_idx * stride_mt).to(tl.float32)
        l_block = tl.load(l_ptrs_base + m_idx * stride_lt).to(tl.float32)
        m_block = m_block * m_mask_f
        l_block = l_block * m_mask_f + (1.0 - m_mask_f)
        m_mask = m_mask_f
        n_mask = (offs_n < Tk).to(tl.float32)[None, :]
        mask_mn = m_mask * n_mask
        scores = scores - m_block
        scores = scores * mask_mn + (-1.0e9) * (1.0 - mask_mn)
        p = tl.exp(scores) * mask_mn
        l_block = tl.maximum(l_block, 1e-12)
        p = p / l_block

        do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dot + offs_d[None, :] * stride_dod
        do = tl.load(
            do_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
        ds = p * (dp - delta)
        dk_acc += tl.dot(tl.trans(ds), q, out_dtype=tl.float32, input_precision="ieee") * scale

    dk_ptrs = DK + pid_bh * stride_dkbh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    tl.store(dk_ptrs, dk_acc, mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D))


def sdpa_bwd_dk(q, k, v, dout, m, l, scale, delta=None):
    """
    Compute dK for SDPA using saved (m, l). Dv == D assumed.
    """
    _assert_backward_contract(q, k, v, dout, m, l, require_v=True)
    B, H, T, D = q.shape
    M = k.shape[2]
    dk = torch.empty_like(k)
    if delta is None:
        delta = _compute_row_delta(q, k, v, dout, m, l, scale)
    if delta.shape != (B, H, T):
        raise AssertionError("delta must have shape (B,H,T)")

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    v_ = v.reshape(B * H, M, D)
    dk_ = dk.reshape(B * H, M, D)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)
    delta_ = delta.reshape(B * H, T).contiguous()

    BLOCK_M = 64 if T >= 256 else 32
    BLOCK_N = 32
    BLOCK_D = 32 if D <= 32 else 64
    grid = (
        B * H,
        triton.cdiv(M, BLOCK_N),
    )

    _sdpa_bwd_dk_kernel[grid](
        q_,
        k_,
        v_,
        dout,
        dk_,
        m_,
        l_,
        delta_,
        T,
        M,
        D,
        H,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        dk_.stride(0),
        dk_.stride(1),
        dk_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        delta_.stride(0),
        delta_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )

    return dk


def sdpa_jvp(q, k, v, dq, dk, dv, m, l, scale):
    """
    Explicit JVP for the **frozen-stats** SDPA operator using saved (m, l).

    Important: this is *not* the JVP of the full recomputed softmax; it
    linearizes the map (q,k,v) -> P(q,k; m,l) @ v with P rebuilt from saved
    stats. Do not compare directly to autograd.functional.jvp(triton_sdpa_fused).

    No autograd inside; CUDA-only in Phase 9. Accumulates in fp32.

    Args:
        q,k,v: (B,H,T,D) / (B,H,M,D)
        dq,dk,dv: same shapes as q,k,v (tangents)
        m,l: (B,H,T) forward row-wise max and sumexp
        scale: 1/sqrt(D)
    Returns:
        dO with shape (B,H,T,D) in q.dtype
    """
    _assert_jvp_contract(q, k, v, dq, dk, dv, m, l)
    B, H, T, D = q.shape

    # fp32 compute
    qf = q.float()
    kf = k.float()
    vf = v.float()
    dqf = dq.float()
    dkf = dk.float()
    dvf = dv.float()

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale  # (B,H,T,M)
    # reconstruct P from saved stats
    P = torch.exp(scores - m.unsqueeze(-1)) / l.unsqueeze(-1)

    dS = (torch.matmul(dqf, kf.transpose(-2, -1)) + torch.matmul(qf, dkf.transpose(-2, -1))) * scale
    dS_centered = dS - (dS * P).sum(dim=-1, keepdim=True)
    dP = P * dS_centered

    dO = torch.matmul(dP, vf) + torch.matmul(P, dvf)
    return dO.to(q.dtype)

@triton.jit
def _sdpa_bwd_dq_kernel(
    Q, K, V, DO, DQ, M, L, DELTA,
    Tq, Tk, D, Hdim,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_dob, stride_doh, stride_dot, stride_dod,
    stride_dqbh, stride_dqm, stride_dqk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
    stride_dbh, stride_dt,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs_base = Q + pid_bh * stride_qbh
    b = pid_bh // Hdim
    h = pid_bh % Hdim
    do_ptrs_base = DO + b * stride_dob + h * stride_doh
    k_ptrs_base = K + pid_bh * stride_kbh
    v_ptrs_base = V + pid_bh * stride_vbh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh
    d_ptrs_base = DELTA + pid_bh * stride_dbh

    # Load q_block once (fp32)
    q_ptrs = q_ptrs_base + offs_m * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(
        q_ptrs,
        mask=(offs_m < Tq) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    delta = tl.load(
        d_ptrs_base + offs_m * stride_dt,
        mask=(offs_m < Tq),
        other=0.0,
    ).to(tl.float32)
    do_ptrs = do_ptrs_base + offs_m * stride_dot + offs_d[None, :] * stride_dod
    do = tl.load(
        do_ptrs,
        mask=(offs_m < Tq) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    # One pass: delta is precomputed/shared across dQ and dK.
    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n0 in range(0, Tk, BLOCK_N):
        k_ptrs = k_ptrs_base + (n0 + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_ptrs_base + (n0 + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk

        k = tl.load(
            k_ptrs,
            mask=((n0 + offs_n)[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptrs,
            mask=((n0 + offs_n)[:, None] < Tk) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
        m_idx = offs_m
        m_mask = (m_idx < Tq)
        m_mask_f = m_mask.to(tl.float32)
        m_block = tl.load(m_ptrs_base + m_idx * stride_mt).to(tl.float32)
        l_block = tl.load(l_ptrs_base + m_idx * stride_lt).to(tl.float32)
        m_block = m_block * m_mask_f
        l_block = l_block * m_mask_f + (1.0 - m_mask_f)
        mask_m = m_mask_f
        mask_n = ((n0 + offs_n) < Tk).to(tl.float32)[None, :]
        mask_mn_f = mask_m * mask_n
        scores = scores - m_block
        scores = scores * mask_mn_f + (-1.0e9) * (1.0 - mask_mn_f)
        p = tl.exp(scores) * mask_mn_f
        p = p / tl.maximum(l_block, 1e-12)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")  # (Mblock, Nblock)
        ds = p * (dp - delta)
        dq_acc += tl.dot(ds, k, out_dtype=tl.float32, input_precision="ieee") * scale

    dq_ptrs = DQ + pid_bh * stride_dqbh + offs_m * stride_dqm + offs_d[None, :] * stride_dqk
    tl.store(dq_ptrs, dq_acc, mask=(offs_m < Tq) & (offs_d[None, :] < D))


def sdpa_bwd_dq(q, k, v, dout, m, l, scale, delta=None):
    """
    Compute dQ for SDPA using saved (m, l). Dv == D assumed.
    """
    _assert_backward_contract(q, k, v, dout, m, l, require_v=True)
    B, H, T, D = q.shape
    M = k.shape[2]
    dq = torch.empty_like(q)
    if delta is None:
        delta = _compute_row_delta(q, k, v, dout, m, l, scale)
    if delta.shape != (B, H, T):
        raise AssertionError("delta must have shape (B,H,T)")

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    v_ = v.reshape(B * H, M, D)
    dq_ = dq.reshape(B * H, T, D)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)
    delta_ = delta.reshape(B * H, T).contiguous()

    BLOCK_M = 64 if T >= 256 else 32
    BLOCK_N = 32
    BLOCK_D = 32 if D <= 32 else 64
    grid = (
        B * H,
        triton.cdiv(T, BLOCK_M),
    )

    _sdpa_bwd_dq_kernel[grid](
        q_,
        k_,
        v_,
        dout,
        dq_,
        m_,
        l_,
        delta_,
        T,
        M,
        D,
        H,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        dq_.stride(0),
        dq_.stride(1),
        dq_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        delta_.stride(0),
        delta_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )

    return dq


__all__ = [
    "sdpa_bwd_dq",
    "sdpa_bwd_dk",
    "sdpa_bwd_dv",
    "sdpa_bwd_shared_staged",
    "sdpa_jvp",
]
