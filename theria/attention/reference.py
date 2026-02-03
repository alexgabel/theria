"""
Legacy compatibility shim for Phase 9 reference helpers.

The new reference implementations live in:
  - reference_attention.py
  - reference_backward_sdpa.py
  - reference_jvp_sdpa.py
  - reference_hvp_sdpa.py

This module re-exports them so older imports keep working.
"""

from .reference_attention import reference_attention, sdpa_reference
from .reference_backward_sdpa import sdpa_backward_reference
from .reference_jvp_sdpa import sdpa_jvp_reference
from .reference_hvp_sdpa import sdpa_hvp_reference

__all__ = [
    "reference_attention",
    "sdpa_reference",
    "sdpa_backward_reference",
    "sdpa_jvp_reference",
    "sdpa_hvp_reference",
]
