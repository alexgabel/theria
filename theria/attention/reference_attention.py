# theria/attention/reference.py
# NOTE: This implementation is for correctness and higher-order autodiff only.
# It is intentionally slow and unfused.
# TODO(phase5): remove sdpa_reference alias once sdpa_function is fully migrated

import torch
import math

def reference_attention(q, k, v):
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


# --- Backwards-compatible alias (legacy import path) ---
# NOTE: Keep this alias until sdpa_function.py is migrated.
sdpa_reference = reference_attention
