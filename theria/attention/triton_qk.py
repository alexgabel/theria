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
