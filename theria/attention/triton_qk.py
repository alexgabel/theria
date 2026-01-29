"""
Triton implementation of scores = Q @ K^T for SDPA (Phase 6 scaffold).

Forward-only kernel for QK^T; softmax and PV remain in PyTorch.
Backward is implemented in Python using standard matmul formulas to preserve
autograd correctness while keeping the kernel minimal.
"""

import torch
import triton
import triton.language as tl


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
