"""
Explicit SDPA JVP (forward-mode) ground truth in pure PyTorch (no autograd).

Used as a math reference for testing Triton JVP / composed kernels.
"""

import torch

from theria.attention.reference_attention import reference_attention


def softmax_jvp(scores: torch.Tensor, dscores: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    JVP of softmax given scores, dscores, and the already-computed softmax p.
    Formula: dP = P * (dscores - sum(dscores * P, axis=-1, keepdim=True))
    """
    dot = (dscores * p).sum(dim=-1, keepdim=True)
    return p * (dscores - dot)


def sdpa_jvp_reference(q, k, v, dq, dk, dv, scale):
    """
    Pure PyTorch reference JVP for SDPA:
        scores = q k^T * scale
        p = softmax(scores)
        out = p v
        dS = (dq k^T + q dk^T) * scale
        dP = softmax_jvp(scores, dS, p)
        dO = dP v + p dv

    Args are torch tensors; works on CPU or CUDA, any dtype; accumulates in fp64 when possible.
    """
    # promote to float64 for accuracy
    qf, kf, vf = q.double(), k.double(), v.double()
    dqf, dkf, dvf = dq.double(), dk.double(), dv.double()

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    p = torch.softmax(scores, dim=-1)

    dS = (torch.matmul(dqf, kf.transpose(-2, -1)) + torch.matmul(qf, dkf.transpose(-2, -1))) * scale
    dP = softmax_jvp(scores, dS, p)

    dO = torch.matmul(dP, vf) + torch.matmul(p, dvf)
    return dO.to(q.dtype)


sdpa_jvp = sdpa_jvp_reference

__all__ = ["sdpa_jvp_reference", "sdpa_jvp"]
__all__ = ["sdpa_jvp_reference"]
