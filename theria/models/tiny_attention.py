# theria/models/tiny_attention.py
from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def sdpa_explicit(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Explicit scaled dot-product attention:
        softmax(q k^T / sqrt(d)) v
    Shapes:
        q: (B, Tq, D)
        k: (B, Tk, D)
        v: (B, Tk, D)
    Returns:
        out: (B, Tq, D)
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # (B, Tq, Tk)
    # Stable softmax
    scores = scores - scores.max(dim=-1, keepdim=True).values
    attn = scores.softmax(dim=-1)  # (B, Tq, Tk)
    return attn @ v  # (B, Tq, D)


@dataclass(frozen=True)
class TinyAttentionConfig:
    d_model: int = 64
    num_classes: int = 5


class TinyAttentionModel(nn.Module):
    """
    Minimal single-head attention classifier.

    Strategy:
    - Use token 0 as a "CLS-like" query that attends to all tokens.
    - Compute attention output for token 0 only, then classify.

    Input:  x (B, T, D)
    Output: logits (B, C)
    """
    def __init__(self, cfg: TinyAttentionConfig):
        super().__init__()
        self.cfg = cfg

        D = cfg.d_model
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)

        self.classifier = nn.Linear(D, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        q = self.q_proj(x[:, :1, :])          # (B, 1, D)  query from token 0
        k = self.k_proj(x)                    # (B, T, D)
        v = self.v_proj(x)                    # (B, T, D)
        h = sdpa_explicit(q, k, v)            # (B, 1, D)
        h = h.squeeze(1)                      # (B, D)
        logits = self.classifier(h)           # (B, C)
        return logits

    def forward_with_params(self, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """
        Functional-call friendly forward method for MAML-style inner loops.
        `params` should be a dict mapping parameter names to tensors.

        Args:
            params: dict of parameters to use instead of self's parameters.
            x: input tensor (B, T, D)

        Returns:
            logits: (B, C)
        """
        # Functional call requires importing torch.func.functional_call externally.
        # Here we just prepare for its usage.
        from torch.func import functional_call

        # Define a helper function to run forward with given params
        def _forward(module, x):
            q = module.q_proj(x[:, :1, :])       # (B, 1, D)
            k = module.k_proj(x)                 # (B, T, D)
            v = module.v_proj(x)                 # (B, T, D)
            h = sdpa_explicit(q, k, v)           # (B, 1, D)
            h = h.squeeze(1)                     # (B, D)
            logits = module.classifier(h)        # (B, C)
            return logits

        return functional_call(self, params, _forward, x)