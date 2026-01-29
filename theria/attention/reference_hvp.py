# NOTE: This implementation is for correctness and higher-order autodiff only.
# It is intentionally slow and unfused.
"""
NOTE:
This is an explicit analytic HVP for SDPA under the loss:
    L = sum(P @ V)

It is intended for correctness validation and meta-learning research.
It does NOT cover masking, dropout, or general losses.

# This is a correctness reference, not optimized code:
# - written for clarity and traceability against the math
# - avoids fused ops and in-place tricks
# - meant to stay readable for higher-order differentiation audits
"""
import math
import torch


def sdpa_hvp(q, k, v, vq, vk, vv):
    """
    Explicit Hessian–vector product for
        L = sum(sdpa_reference(q, k, v))
    Returns hvp for (q, k, v) directions.
    """
    d = q.shape[-1]
    scale = 1 / math.sqrt(d)

    # Forward pass: compute scaled attention scores and softmax probabilities
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)

    # Define softmax JVP for directional derivatives through softmax
    def softmax_jvp(delta_scores):
        inner = (delta_scores * probs).sum(dim=-1, keepdim=True)
        return probs * (delta_scores - inner)

    # Compute directional derivative of scores
    scores_dir = (
        torch.matmul(vq, k.transpose(-2, -1))
        + torch.matmul(q, vk.transpose(-2, -1))
    ) * scale
    # Directional derivative of softmax probabilities
    probs_dir = softmax_jvp(scores_dir)

    # Gradient seed w.r.t. probs for L = sum(P @ V) is sum over last dim of V
    grad_probs = v.sum(dim=-1).unsqueeze(-2).expand_as(probs)
    # Directional derivative of gradient w.r.t. probs
    grad_probs_dir = vv.sum(dim=-1).unsqueeze(-2).expand_as(probs)

    # Compute inner terms for gradient w.r.t. scores
    inner = (grad_probs * probs).sum(dim=-1, keepdim=True)
    inner_dir = (grad_probs_dir * probs + grad_probs * probs_dir).sum(dim=-1, keepdim=True)

    # Gradient w.r.t. scores and its directional derivative
    grad_scores = probs * (grad_probs - inner)
    grad_scores_dir = probs_dir * (grad_probs - inner) + probs * (grad_probs_dir - inner_dir)

    # Assemble HVP w.r.t. q and k
    hvp_q = scale * (
        torch.matmul(grad_scores_dir, k) + torch.matmul(grad_scores, vk)
    )
    hvp_k = scale * (
        torch.matmul(grad_scores_dir.transpose(-2, -1), q)
        + torch.matmul(grad_scores.transpose(-2, -1), vq)
    )

    # HVP w.r.t. v is the directional derivative of grad w.r.t v: (δP)^T 1
    hvp_v = probs_dir.sum(dim=-2).unsqueeze(-1).expand_as(v)

    return hvp_q, hvp_k, hvp_v
