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
| `triton_fused_meta` | Experimental candidate | Several validated diffusion-like shapes where equal-time matters | Promoted candidate, not the default, not a universal replacement |
| CUDA-graph acceleration | Blocked | Not part of the accepted workflow | Do not use `CUDA_GRAPH_STATIC=1` in benchmark/regression commands |

Validation references:
- target workload:
  - `phase12_requalify_diffproxy_s5_20260313_175213`
- held-out shape:
  - `phase12_heldout_diffproxy_s5_20260315_182507`
- larger held-out shape (`256 x 16`):
  - `phase12_heldout_diffproxy_256x16_s5_20260316_105752`

Key cautions:
- No claim that `triton_fused_meta` replaces `reference` universally.
- No claim that `triton_fused_meta` wins every cell.
- No claim that the stable path wins equal-time universally.
- No claim that CUDA-graph acceleration is ready.
