"""
Explicit SDPA backward (reference, no autograd).

Used as the ground truth for Phase 9 Triton backward/JVP/HVP work.
"""

import torch

from theria.attention.reference_attention import reference_attention


def sdpa_backward_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dout: torch.Tensor, scale: float):
    """
    Compute dQ, dK, dV for scaled dot-product attention explicitly (no autograd).

    Args:
        q, k, v: tensors shaped (B, H, T, D), (B, H, M, D), (B, H, M, Dv)
        dout: upstream gradient wrt attention output, shape (B, H, T, Dv)
        scale: scaling factor (typically 1 / sqrt(D))

    Returns:
        dq, dk, dv with the same shapes as q, k, v respectively.
    """
    with torch.no_grad():
        # Detach to ensure no autograd tracking inside this explicit reference.
        q_ = q.detach()
        k_ = k.detach()
        v_ = v.detach()
        dout_ = dout.detach()

        # 1) scores and softmax
        scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale  # (B,H,T,M)
        p = torch.softmax(scores, dim=-1)

        # 3) dV
        dv = torch.matmul(p.transpose(-2, -1), dout_)  # (B,H,M,Dv)

        # 4) dP
        dp = torch.matmul(dout_, v_.transpose(-2, -1))  # (B,H,T,M)

        # 5) z term
        z = (dp * p).sum(dim=-1, keepdim=True)  # (B,H,T,1)

        # 6) dS
        ds = p * (dp - z)  # (B,H,T,M)

        # 7) dQ
        dq = torch.matmul(ds, k_) * scale  # (B,H,T,D)

        # 8) dK
        dk = torch.matmul(ds.transpose(-2, -1), q_) * scale  # (B,H,M,D)

    return dq, dk, dv


__all__ = ["sdpa_backward_reference", "reference_attention"]
