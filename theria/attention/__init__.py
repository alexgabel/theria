"""
Attention module.

This package contains reference and experimental attention implementations
used to study higher-order autodiff behavior (HVP/JVP) in meta-learning.

Concrete exports were intentionally kept local during Phase 2–3; Phase 4
surfaces a public API for correctness-first custom attention.
"""

from .reference import reference_attention
from .custom import sdpa_custom, sdpa_custom_hvp

__all__ = [
    "reference_attention",
    "sdpa_custom",
    "sdpa_custom_hvp",
]
