import torch
import math

def sdpa_reference(q, k, v):
    """
    Reference scaled dot-product attention.
    q: (B, H, N, D)
    k: (B, H, M, D)
    v: (B, H, M, Dv)
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)