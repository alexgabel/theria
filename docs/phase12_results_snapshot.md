# Phase 12 results snapshot

This is the compact review-facing snapshot for the accepted stable baseline.

Accepted baseline:
- gate tag: `phase12_meta_gate_phase12_stable_regression_20260309_155142`
- frontier tag: `phase12_stable_regression_20260309_155142`

Canonical pages:
- status: `docs/STATUS.md`
- workflow: `experiments/phase12/README.md`
- colleague summary: `docs/phase12_for_colleagues.md`

| Item | Status | Intended use | Notes |
| --- | --- | --- | --- |
| `triton_fused_meta_strict FULL` | Stable | Accepted stable path | Eager execution only |
| `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2` | Practical | Accepted practical path | Eager execution only |
| `triton_fused_meta` | Experimental | Explicit opt-in research backend | Do not promote on equal-step parity alone |
| CUDA-graph acceleration | Blocked | Not part of the accepted workflow | Do not use `CUDA_GRAPH_STATIC=1` in benchmark/regression commands |

Key cautions:
- No claim that `triton_fused_meta` replaces `reference`.
- No claim that the stable path wins equal-time universally.
- No claim that CUDA-graph acceleration is ready.
