"""
Triton backward kernels for fused SDPA (Phase 9 scaffolding).

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


@triton.jit
def _sdpa_bwd_dv_kernel(
    Q, K, DO, DV, M, L,
    Tq, Tk, D,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_dobh, stride_dom, stride_dok,
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
    do_ptrs_base = DO + pid_bh * stride_dobh
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
        valid_m = (m0 + offs_m) < Tq

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
        do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dom + offs_dv[None, :] * stride_dok
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
    assert q.is_cuda and k.is_cuda and dout.is_cuda
    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    _, _, Tdo, Dv = dout.shape
    assert D == Dk
    assert T == Tdo
    assert D == Dv, "v0 assumes Dv == D"
    assert q.is_contiguous() and k.is_contiguous() and dout.is_contiguous()
    assert m.shape == (B, H, T) and l.shape == (B, H, T)

    dv = torch.empty((B, H, M, Dv), device=q.device, dtype=q.dtype)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    do_ = dout.reshape(B * H, T, Dv)
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
        do_,
        dv_,
        m_,
        l_,
        T,
        M,
        D,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        do_.stride(0),
        do_.stride(1),
        do_.stride(2),
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


__all__ = ["sdpa_bwd_dv"]


@triton.jit
def _sdpa_bwd_dk_kernel(
    Q, K, V, DO, DK, M, L,
    Tq, Tk, D,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_dobh, stride_dom, stride_dok,
    stride_dkbh, stride_dkn, stride_dkk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
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
    do_ptrs_base = DO + pid_bh * stride_dobh
    k_ptrs_base = K + pid_bh * stride_kbh
    v_ptrs_base = V + pid_bh * stride_vbh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh

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
        valid_m = (m0 + offs_m) < Tq

        # pass 1: compute z for these query rows (over all key blocks)
        z_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n0 in range(0, Tk, BLOCK_N):
            k_ptrs = k_ptrs_base + (n0 + tl.arange(0, BLOCK_N))[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = v_ptrs_base + (n0 + tl.arange(0, BLOCK_N))[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(
                k_ptrs,
                mask=((n0 + tl.arange(0, BLOCK_N))[:, None] < Tk) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptrs,
                mask=((n0 + tl.arange(0, BLOCK_N))[:, None] < Tk) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)

            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale
            m_idx = m0 + offs_m  # (BM,1)
            m_mask = (m_idx < Tq)
            m_mask_f = m_mask.to(tl.float32)
            m_block = tl.load(m_ptrs_base + m_idx * stride_mt).to(tl.float32)
            l_block = tl.load(l_ptrs_base + m_idx * stride_lt).to(tl.float32)
            m_block = m_block * m_mask_f
            l_block = l_block * m_mask_f + (1.0 - m_mask_f)
            m_mask = m_mask_f
            n_mask = ((n0 + tl.arange(0, BLOCK_N)) < Tk).to(tl.float32)[None, :]
            mask_mn = m_mask * n_mask
            scores = scores - m_block
            scores = scores * mask_mn + (-1.0e9) * (1.0 - mask_mn)
            p = tl.exp(scores) * mask_mn
            l_block = tl.maximum(l_block, 1e-12)
            p = p / l_block

            do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dom + offs_d[None, :] * stride_dok
            do = tl.load(
                do_ptrs,
                mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)

            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
            z_acc += tl.sum(dp * p, axis=1)

        # pass 2: only the target key block pid_n
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

        do_ptrs = do_ptrs_base + (m0 + offs_m) * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(
            do_ptrs,
            mask=((m0 + offs_m) < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")
        ds = p * (dp - z_acc[:, None])
        dk_acc += tl.dot(tl.trans(ds), q, out_dtype=tl.float32, input_precision="ieee") * scale

    dk_ptrs = DK + pid_bh * stride_dkbh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    tl.store(dk_ptrs, dk_acc, mask=(offs_n[:, None] < Tk) & (offs_d[None, :] < D))


def sdpa_bwd_dk(q, k, v, dout, m, l, scale):
    """
    Compute dK for SDPA using saved (m, l). Dv == D assumed.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and dout.is_cuda
    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    assert D == Dk
    assert v.shape == (B, H, M, D)
    assert dout.shape == (B, H, T, D)
    assert m.shape == (B, H, T) and l.shape == (B, H, T)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and dout.is_contiguous()
    dk = torch.empty_like(k)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    v_ = v.reshape(B * H, M, D)
    do_ = dout.reshape(B * H, T, D)
    dk_ = dk.reshape(B * H, M, D)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = 64
    grid = (
        B * H,
        triton.cdiv(M, BLOCK_N),
    )

    _sdpa_bwd_dk_kernel[grid](
        q_,
        k_,
        v_,
        do_,
        dk_,
        m_,
        l_,
        T,
        M,
        D,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        do_.stride(0),
        do_.stride(1),
        do_.stride(2),
        dk_.stride(0),
        dk_.stride(1),
        dk_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return dk


__all__ = ["sdpa_bwd_dv", "sdpa_bwd_dq", "sdpa_bwd_dk"]


@triton.jit
def _sdpa_bwd_dq_kernel(
    Q, K, V, DO, DQ, M, L,
    Tq, Tk, D,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_dobh, stride_dom, stride_dok,
    stride_dqbh, stride_dqm, stride_dqk,
    stride_mbh, stride_mt,
    stride_lbh, stride_lt,
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
    do_ptrs_base = DO + pid_bh * stride_dobh
    k_ptrs_base = K + pid_bh * stride_kbh
    v_ptrs_base = V + pid_bh * stride_vbh
    m_ptrs_base = M + pid_bh * stride_mbh
    l_ptrs_base = L + pid_bh * stride_lbh

    # Load q_block once (fp32)
    q_ptrs = q_ptrs_base + offs_m * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(
        q_ptrs,
        mask=(offs_m < Tq) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    valid_m = offs_m < Tq

    # Pass 1: accumulate z
    z_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
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

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee") * scale  # (Mblock, Nblock)
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

        do_ptrs = do_ptrs_base + offs_m * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(
            do_ptrs,
            mask=(offs_m < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        # dp = dO @ V^T for this key block (reuse v)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")  # (Mblock, Nblock)
        z_acc += tl.sum(dp * p, axis=1)

    # Pass 2: accumulate dQ
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

        do_ptrs = do_ptrs_base + offs_m * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(
            do_ptrs,
            mask=(offs_m < Tq) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, input_precision="ieee")  # (Mblock, Nblock)
        ds = p * (dp - z_acc[:, None])
        dq_acc += tl.dot(ds, k, out_dtype=tl.float32, input_precision="ieee") * scale

    dq_ptrs = DQ + pid_bh * stride_dqbh + offs_m * stride_dqm + offs_d[None, :] * stride_dqk
    tl.store(dq_ptrs, dq_acc, mask=(offs_m < Tq) & (offs_d[None, :] < D))


def sdpa_bwd_dq(q, k, v, dout, m, l, scale):
    """
    Compute dQ for SDPA using saved (m, l). Dv == D assumed.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and dout.is_cuda
    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    assert D == Dk
    assert dout.shape == (B, H, T, D)
    assert m.shape == (B, H, T) and l.shape == (B, H, T)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and dout.is_contiguous()
    dq = torch.empty_like(q)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    v_ = v.reshape(B * H, M, D)
    do_ = dout.reshape(B * H, T, D)
    dq_ = dq.reshape(B * H, T, D)
    m_ = m.reshape(B * H, T)
    l_ = l.reshape(B * H, T)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = 64
    grid = (
        B * H,
        triton.cdiv(T, BLOCK_M),
    )

    _sdpa_bwd_dq_kernel[grid](
        q_,
        k_,
        v_,
        do_,
        dq_,
        m_,
        l_,
        T,
        M,
        D,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        do_.stride(0),
        do_.stride(1),
        do_.stride(2),
        dq_.stride(0),
        dq_.stride(1),
        dq_.stride(2),
        m_.stride(0),
        m_.stride(1),
        l_.stride(0),
        l_.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return dq
