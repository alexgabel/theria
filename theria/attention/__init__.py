"""
Attention module.

This package contains reference and experimental attention implementations
used to study higher-order autodiff behavior (HVP/JVP) in meta-learning.

Phase 9 freezes the Triton fused SDPA backend; the surface below is the stable
API users should touch. Everything else is still importable for experimentation,
but not part of the public contract.
"""

from .triton_qk import triton_sdpa_fused
from .triton_sdpa_backward import (
    sdpa_bwd_dq,
    sdpa_bwd_dk,
    sdpa_bwd_dv,
    sdpa_jvp,
)

__all__ = [
    "triton_sdpa_fused",
    "sdpa_bwd_dq",
    "sdpa_bwd_dk",
    "sdpa_bwd_dv",
    "sdpa_jvp",
]
