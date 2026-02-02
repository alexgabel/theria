"""
Phase 11 bad backend variants (experiment-only).
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def detach_attention_output(
    sdpa_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    Return an attention wrapper that detaches the SDPA output.
    """

    def _wrapped(*args, **kwargs) -> torch.Tensor:
        out = sdpa_fn(*args, **kwargs)
        # Keep tensor usage in-graph while zeroing the attention gradient path.
        return out.detach() + (out * 0.0)

    return _wrapped
