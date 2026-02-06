"""
Triton implementation of scores = Q @ K^T for SDPA (Phase 6 scaffold).

Forward-only kernel for QK^T; softmax and PV remain in PyTorch.
Backward is implemented in Python using standard matmul formulas to preserve
autograd correctness while keeping the kernel minimal.
"""

import os
from typing import Dict, Tuple, Union
import torch
import triton
import triton.language as tl
from theria.attention.reference import reference_attention
from theria.attention.triton_sdpa_backward import (
    compute_row_delta_triton,
    sdpa_bwd_dq,
    sdpa_bwd_dq_dk_dv_shared,
    sdpa_bwd_dk_dv,
    sdpa_bwd_shared_staged,
)

_TRITON_SDPA_BWD_PROFILE = {
    "calls": 0,
    "use_shared_calls": 0,
    "total_bwd_ms_sum": 0.0,
    "delta_ms_sum": 0.0,
    "dq_ms_sum": 0.0,
    "dk_dv_ms_sum": 0.0,
    "shared_ms_sum": 0.0,
}

_TRITON_SDPA_BWD_BUFFER_CACHE: Dict[Tuple, Dict[str, torch.Tensor]] = {}


def reset_triton_sdpa_bwd_profile() -> None:
    """Reset cumulative backward timing stats for Triton fused SDPA."""
    for key in _TRITON_SDPA_BWD_PROFILE:
        _TRITON_SDPA_BWD_PROFILE[key] = 0 if key.endswith("calls") else 0.0


def get_triton_sdpa_bwd_profile() -> Dict[str, Union[float, int]]:
    """Return cumulative backward timing stats for Triton fused SDPA."""
    return dict(_TRITON_SDPA_BWD_PROFILE)


def _get_bwd_reuse_buffers(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Return reusable backward buffers keyed by shape/device/dtype.
    Enabled via THERIA_SDPA_BWD_REUSE=1 to reduce allocator overhead.
    """
    B, H, T, _ = q.shape
    key = (q.device.type, q.device.index, q.dtype, tuple(q.shape), tuple(k.shape), tuple(v.shape))
    cached = _TRITON_SDPA_BWD_BUFFER_CACHE.get(key)
    if cached is None:
        cached = {
            "delta": torch.empty((B, H, T), device=q.device, dtype=torch.float32),
            "dq": torch.empty_like(q),
            "dk": torch.empty_like(k),
            "dv": torch.empty_like(v),
        }
        # Keep cache bounded in case many shapes are seen.
        if len(_TRITON_SDPA_BWD_BUFFER_CACHE) >= 8:
            _TRITON_SDPA_BWD_BUFFER_CACHE.clear()
        _TRITON_SDPA_BWD_BUFFER_CACHE[key] = cached
    return cached


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _qk_kernel(
    Q,
    K,
    Out,
    M,
    N,
    Kdim,
    stride_qbh,
    stride_qm,
    stride_qk,
    stride_kbh,
    stride_kn,
    stride_kk,
    stride_obh,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # blocks along T
    pid_n = tl.program_id(1)  # blocks along M
    pid_bh = tl.program_id(2)  # flattened batch*head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + pid_bh * stride_kbh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, Kdim, BLOCK_K):
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < Kdim), other=0.0)
        k_tile = tl.load(k_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < Kdim), other=0.0)
        acc += tl.dot(q, k_tile, out_dtype=tl.float32, input_precision="ieee")
        q_ptrs += BLOCK_K * stride_qk
        k_ptrs += BLOCK_K * stride_kk

    out_ptrs = Out + pid_bh * stride_obh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class TritonQKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        """
        q: (B,H,T,D), k: (B,H,M,D) -> scores: (B,H,T,M)
        Uses Triton for the matmul; keeps autograd by providing Python backward.
        """
        assert q.is_cuda and k.is_cuda, "TritonQKFunction requires CUDA tensors"
        B, H, T, D = q.shape
        _, _, M, Dk = k.shape
        assert D == Dk, "Q and K last dimensions must match"

        scores = torch.empty((B, H, T, M), device=q.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(T, META['BLOCK_M']),
            triton.cdiv(M, META['BLOCK_N']),
            B * H,
        )

        # Flatten batch and head into leading dimension for launch simplicity
        q_ = q.reshape(B * H, T, D)
        k_ = k.reshape(B * H, M, D)
        scores_ = scores.reshape(B * H, T, M)

        _qk_kernel[grid](
            q_,
            k_,
            scores_,
            T,
            M,
            D,
            q_.stride(0),
            q_.stride(1),
            q_.stride(2),
            k_.stride(0),
            k_.stride(1),
            k_.stride(2),
            scores_.stride(0),
            scores_.stride(1),
            scores_.stride(2),
        )

        ctx.save_for_backward(q, k)
        return scores

    @staticmethod
    def backward(ctx, grad_out):
        q, k = ctx.saved_tensors
        # grad_out: (B,H,T,M)
        grad_q = torch.matmul(grad_out, k)  # (B,H,T,D)
        grad_k = torch.matmul(grad_out.transpose(-2, -1), q)  # (B,H,M,D)
        return grad_q, grad_k


def triton_qk(q, k):
    """Public wrapper for Triton QK^T (scores) with autograd support."""
    return TritonQKFunction.apply(q, k)


@triton.jit
def _qk_softmax_kernel(
    Q,
    K,
    Out,
    Tq,
    Tk,
    D,
    stride_qbh,
    stride_qm,
    stride_qk,
    stride_kbh,
    stride_kn,
    stride_kk,
    stride_obh,
    stride_om,
    stride_on,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // Tq
    t = pid % Tq

    offs_n = tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)

    q_ptr = Q + bh * stride_qbh + t * stride_qm

    # First pass: compute max
    max_val = tl.full((), -float("inf"), tl.float32)
    for n0 in range(0, Tk, BLOCK_N):
        k_ptrs = K + bh * stride_kbh + (n0 + offs_n)[:, None] * stride_kn + d[None, :] * stride_kk
        k_block = tl.load(k_ptrs, mask=((n0 + offs_n)[:, None] < Tk) & (d[None, :] < D), other=0.0)
        q_vec = tl.load(q_ptr + d * stride_qk, mask=d < D, other=0.0)
        q_vec = q_vec[None, :].to(tl.float32)
        k_block = k_block.to(tl.float32)
        scores = tl.sum(k_block * q_vec, axis=1) * scale
        valid_n = (n0 + offs_n) < Tk
        scores = tl.where(valid_n, scores, -float("inf"))
        max_val = tl.maximum(max_val, tl.max(scores, axis=0))

    # Second pass: compute exp and sum
    sum_exp = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, Tk, BLOCK_N):
        k_ptrs = K + bh * stride_kbh + (n0 + offs_n)[:, None] * stride_kn + d[None, :] * stride_kk
        k_block = tl.load(k_ptrs, mask=((n0 + offs_n)[:, None] < Tk) & (d[None, :] < D), other=0.0)
        q_vec = tl.load(q_ptr + d * stride_qk, mask=d < D, other=0.0)
        q_vec = q_vec[None, :].to(tl.float32)
        k_block = k_block.to(tl.float32)
        scores = tl.sum(k_block * q_vec, axis=1) * scale
        valid_n = (n0 + offs_n) < Tk
        scores = tl.where(valid_n, scores, -float("inf"))
        scores = scores - max_val
        exp_scores = tl.exp(scores)
        sum_exp += tl.sum(exp_scores, axis=0)
        # store softmax block temporarily in Out (we'll normalize later)
        out_ptrs = Out + bh * stride_obh + t * stride_om + (n0 + offs_n) * stride_on
        tl.store(out_ptrs, exp_scores, mask=n0 + offs_n < Tk)

    # Normalize
    inv_sum = 1.0 / sum_exp
    for n0 in range(0, Tk, BLOCK_N):
        out_ptrs = Out + bh * stride_obh + t * stride_om + (n0 + offs_n) * stride_on
        block = tl.load(out_ptrs, mask=n0 + offs_n < Tk, other=0.0)
        block = block * inv_sum
        tl.store(out_ptrs, block, mask=n0 + offs_n < Tk)


def triton_qk_softmax(q, k):
    """
    Fused QK + scaling + softmax (tensor-core friendly); PV remains in PyTorch.
    """
    assert q.is_cuda and k.is_cuda
    assert q.is_contiguous() and k.is_contiguous(), "triton_qk_softmax requires contiguous inputs (v0)"
    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    assert D == Dk

    probs = torch.empty((B, H, T, M), device=q.device, dtype=torch.float32)
    grid = (B * H * T,)
    scale = 1.0 / (D ** 0.5)

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    probs_ = probs.reshape(B * H, T, M)

    BLOCK_N = 128
    BLOCK_D = 128  # head dim ceiling; mask handles smaller D
    assert D <= BLOCK_D, f"triton_qk_softmax supports D <= {BLOCK_D}"
    _qk_softmax_kernel[grid](
        q_,
        k_,
        probs_,
        T,
        M,
        D,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        probs_.stride(0),
        probs_.stride(1),
        probs_.stride(2),
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return probs


@triton.jit
def _fused_sdpa_kernel(
    Q, K, V, Out, MOut, LOut,
    Tq, Tk, D, Dv,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_obh, stride_om, stride_ok,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    # PHASE 9: explicit backward kernels start here (forward saves m_i, l_i for backward/JVP/HVP).
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dv = tl.arange(0, BLOCK_DV)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < Tq) & (offs_k[None, :] < D), other=0.0).to(tl.float32)
    valid_m = offs_m < Tq

    for n0 in range(0, Tk, BLOCK_N):
        k_ptrs = K + pid_bh * stride_kbh + (n0 + offs_n)[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = V + pid_bh * stride_vbh + (n0 + offs_n)[:, None] * stride_vn + offs_dv[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=((n0 + offs_n)[:, None] < Tk) & (offs_k[None, :] < D), other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=((n0 + offs_n)[:, None] < Tk) & (offs_dv[None, :] < Dv), other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_M, BLOCK_N)
        valid_n = (n0 + offs_n) < Tk
        mask_mn = valid_m[:, None] & valid_n[None, :]
        scores = tl.where(mask_mn, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        valid_m_slice = tl.max(valid_m[:, None].to(tl.int32), axis=1) != 0
        m_new = tl.where(valid_m_slice, m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
        l_i = l_new

    # Normalize (guard against zero)
    eps = 1e-6
    l_safe = tl.maximum(l_i, eps)
    acc = acc / l_safe[:, None]
    valid_m_slice = tl.max(valid_m[:, None].to(tl.int32), axis=1) != 0
    acc = tl.where(valid_m_slice[:, None], acc, 0.0)

    out_ptrs = Out + pid_bh * stride_obh + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < Tq) & (offs_dv[None, :] < Dv))
    m_ptrs = MOut + pid_bh * Tq + offs_m
    l_ptrs = LOut + pid_bh * Tq + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < Tq)
    tl.store(l_ptrs, l_i, mask=offs_m < Tq)


def triton_sdpa_fused(q, k, v, return_stats: bool = False):
    """
    Fully fused SDPA forward (QK -> softmax -> PV) in one Triton kernel.
    Assumptions v0: contiguous tensors, Dv == D, no mask/causal/dropout.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    B, H, T, D = q.shape
    _, _, M, Dk = k.shape
    _, _, Mv, Dv = v.shape
    assert D == Dk
    assert M == Mv
    assert Dv == D, "v0 fused path currently requires Dv == D"
    assert D <= 64 and Dv <= 64, "v0 fused kernel assumes D, Dv <= 64 (BLOCK_D/BLOCK_DV)"

    out = torch.empty((B, H, T, Dv), device=q.device, dtype=torch.float32)
    m_stats = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
    l_stats = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64
    BLOCK_DV = 64
    grid = (
        B * H,
        triton.cdiv(T, BLOCK_M),
    )

    q_ = q.reshape(B * H, T, D)
    k_ = k.reshape(B * H, M, D)
    v_ = v.reshape(B * H, M, Dv)
    out_ = out.reshape(B * H, T, Dv)

    _fused_sdpa_kernel[grid](
        q_,
        k_,
        v_,
        out_, m_stats.reshape(-1), l_stats.reshape(-1),
        T,
        M,
        D,
        Dv,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        out_.stride(0),
        out_.stride(1),
        out_.stride(2),
        scale=1.0 / (D ** 0.5),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
    )

    if return_stats:
        return out, m_stats, l_stats
    return out


class TritonFusedSDPAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, m, l = triton_sdpa_fused(q, k, v, return_stats=True)
        # Save out to form row-wise delta once and reuse it across dQ/dK paths.
        ctx.save_for_backward(q, k, v, m, l, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, m, l, out = ctx.saved_tensors
        scale = 1.0 / (q.shape[-1] ** 0.5)
        use_reuse = os.environ.get("THERIA_SDPA_BWD_REUSE", "0") != "0"
        reuse_buffers = _get_bwd_reuse_buffers(q, k, v) if use_reuse else None
        profile_bwd = (
            os.environ.get("THERIA_SDPA_PROFILE_BWD", "0") != "0"
            and grad_out.is_cuda
            and not torch.cuda.is_current_stream_capturing()
        )
        if profile_bwd:
            dev = grad_out.device
            ev_total_start = torch.cuda.Event(enable_timing=True)
            ev_total_end = torch.cuda.Event(enable_timing=True)
            ev_delta_start = torch.cuda.Event(enable_timing=True)
            ev_delta_end = torch.cuda.Event(enable_timing=True)
            ev_total_start.record()
            ev_delta_start.record()
        if reuse_buffers is not None:
            delta = reuse_buffers["delta"]
            compute_row_delta_triton(grad_out, out, delta_out=delta)
        else:
            delta = compute_row_delta_triton(grad_out, out)
        if profile_bwd:
            ev_delta_end.record()

        # Default to legacy Triton-kernel backward path: currently faster in benchmarks.
        shared_mode = os.environ.get("THERIA_SDPA_BWD_SHARED", "0")
        if shared_mode == "2":
            if profile_bwd:
                ev_shared_start = torch.cuda.Event(enable_timing=True)
                ev_shared_end = torch.cuda.Event(enable_timing=True)
                ev_shared_start.record()
            dq, dk, dv = sdpa_bwd_dq_dk_dv_shared(q, k, v, grad_out, m, l, scale, delta=delta)
            if profile_bwd:
                ev_shared_end.record()
        elif shared_mode != "0":
            # Shared staged path: reconstruct softmax terms once per query chunk.
            if profile_bwd:
                ev_shared_start = torch.cuda.Event(enable_timing=True)
                ev_shared_end = torch.cuda.Event(enable_timing=True)
                ev_shared_start.record()
            dq, dk, dv = sdpa_bwd_shared_staged(q, k, v, grad_out, m, l, scale, delta=delta)
            if profile_bwd:
                ev_shared_end.record()
        else:
            # Explicit path with fused dk+dv key-block pass (two kernels total).
            if profile_bwd:
                ev_dq_start = torch.cuda.Event(enable_timing=True)
                ev_dq_end = torch.cuda.Event(enable_timing=True)
                ev_dk_dv_start = torch.cuda.Event(enable_timing=True)
                ev_dk_dv_end = torch.cuda.Event(enable_timing=True)
                ev_dq_start.record()
            dq = sdpa_bwd_dq(
                q, k, v, grad_out, m, l, scale, delta=delta,
                dq_out=(reuse_buffers["dq"] if reuse_buffers is not None else None),
            )
            if profile_bwd:
                ev_dq_end.record()
                ev_dk_dv_start.record()
            dk, dv = sdpa_bwd_dk_dv(
                q, k, v, grad_out, m, l, scale, delta=delta,
                dk_out=(reuse_buffers["dk"] if reuse_buffers is not None else None),
                dv_out=(reuse_buffers["dv"] if reuse_buffers is not None else None),
            )
            if profile_bwd:
                ev_dk_dv_end.record()

        if profile_bwd:
            ev_total_end.record()
            if not torch.cuda.is_current_stream_capturing():
                torch.cuda.synchronize(dev)
            _TRITON_SDPA_BWD_PROFILE["calls"] += 1
            if shared_mode != "0":
                _TRITON_SDPA_BWD_PROFILE["use_shared_calls"] += 1
            _TRITON_SDPA_BWD_PROFILE["total_bwd_ms_sum"] += float(ev_total_start.elapsed_time(ev_total_end))
            _TRITON_SDPA_BWD_PROFILE["delta_ms_sum"] += float(ev_delta_start.elapsed_time(ev_delta_end))
            if shared_mode != "0":
                _TRITON_SDPA_BWD_PROFILE["shared_ms_sum"] += float(ev_shared_start.elapsed_time(ev_shared_end))
            else:
                _TRITON_SDPA_BWD_PROFILE["dq_ms_sum"] += float(ev_dq_start.elapsed_time(ev_dq_end))
                _TRITON_SDPA_BWD_PROFILE["dk_dv_ms_sum"] += float(ev_dk_dv_start.elapsed_time(ev_dk_dv_end))
        return dq, dk, dv


def triton_sdpa_fused_autograd(q, k, v):
    return TritonFusedSDPAFunction.apply(q, k, v)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _qk_kernel_fast(
    Q,
    K,
    Out,
    M,
    N,
    Kdim,
    stride_qbh,
    stride_qm,
    stride_qk,
    stride_kbh,
    stride_kn,
    stride_kk,
    stride_obh,
    stride_om,
    stride_on,
    ACC_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Standard Triton matmul pattern (tensor-core friendly).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k0 in range(0, Kdim, BLOCK_K):
        q_ptrs = (
            Q
            + pid_bh * stride_qbh
            + offs_m[:, None] * stride_qm
            + (k0 + k_range)[None, :] * stride_qk
        )
        k_ptrs = (
            K
            + pid_bh * stride_kbh
            + offs_n[None, :] * stride_kn
            + (k0 + k_range)[:, None] * stride_kk
        )

        q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (k0 + k_range[None, :] < Kdim), other=0.0)
        k_tile = tl.load(
            k_ptrs, mask=(offs_n[None, :] < N) & (k0 + k_range[:, None] < Kdim), other=0.0
        )
        acc += tl.dot(q, k_tile, out_dtype=ACC_DTYPE)
    out_ptrs = Out + pid_bh * stride_obh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class TritonQKFunctionFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        """
        Fast Triton QK^T kernel (tensor-core friendly).
        Uses lower-precision accumulation when legal; expect numerical drift.
        """
        assert q.is_cuda and k.is_cuda, "TritonQKFunctionFast requires CUDA tensors"
        B, H, T, D = q.shape
        _, _, M, Dk = k.shape
        assert D == Dk, "Q and K last dimensions must match"

        if q.dtype == torch.float16:
            acc_dtype = tl.float16
            out_dtype = torch.float16
        elif q.dtype == torch.bfloat16:
            acc_dtype = tl.bfloat16
            out_dtype = torch.bfloat16
        else:
            acc_dtype = tl.float32
            out_dtype = torch.float32

        scores = torch.empty((B, H, T, M), device=q.device, dtype=out_dtype)

        grid = lambda META: (
            triton.cdiv(T, META['BLOCK_M']),
            triton.cdiv(M, META['BLOCK_N']),
            B * H,
        )

        q_ = q.reshape(B * H, T, D)
        k_ = k.reshape(B * H, M, D)
        scores_ = scores.reshape(B * H, T, M)

        _qk_kernel_fast[grid](
            q_,
            k_,
            scores_,
            T,
            M,
            D,
            q_.stride(0),
            q_.stride(1),
            q_.stride(2),
            k_.stride(0),
            k_.stride(1),
            k_.stride(2),
            scores_.stride(0),
            scores_.stride(1),
            scores_.stride(2),
            ACC_DTYPE=acc_dtype,
        )

        ctx.save_for_backward(q, k)
        return scores

    @staticmethod
    def backward(ctx, grad_out):
        q, k = ctx.saved_tensors
        grad_q = torch.matmul(grad_out, k)
        grad_k = torch.matmul(grad_out.transpose(-2, -1), q)
        return grad_q, grad_k


def triton_qk_fast(q, k):
    """Fast Triton QK^T (tensor-core friendly, lower precision)."""
    return TritonQKFunctionFast.apply(q, k)
