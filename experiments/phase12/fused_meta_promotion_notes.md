# `triton_fused_meta` promotion branch notes

Branch:
- `perf/fused-meta-promotion`

Baseline artifacts for this branch:
- canary tag: `phase12_fused_meta_canary_20260312_115744`
- regression tag: `phase12_fused_meta_regression_20260312_120304`
- profiling CSV:
  - `experiments/phase12/runs/phase12_fused_meta_promotion_profile.csv`
  - `experiments/phase12/runs/phase12_fused_meta_promotion_profile_summary.csv`

Current branch charter:
- target fused-meta equal-time overhead first
- keep fallback disabled during qualification/profiling
- keep parity intact
- do not touch CUDA-graph work
- do not change the accepted stable baseline

Current bounded optimization batch:
- hypothesis: reduce fused-meta `meta_bwd` overhead by simplifying the
  recompute graph used in `triton_fused_meta` higher-order backward
- implementation target: replace the manual `exp / sum / divide` softmax
  reconstruction in the recompute path with a simpler `torch.softmax`-based
  fp32 path after explicit max-shift

Decision rule for this batch:
- if `meta_bwd` premium drops materially, continue on this line
- if it barely moves, stop and switch to the next hypothesis
  (`support_grad_create_graph`)

Second bounded optimization batch:
- hypothesis: remove the now-redundant explicit max-shift in the recompute
  path and align more closely with the strict/reference graph shape
- implementation target: rely on `torch.softmax(scores, dim=-1)` for stable
  normalization and drop the extra `scores.max(...)` reduction node

Decision rule for this batch:
- if `meta_bwd` premium drops materially again, continue on the `meta_bwd`
  line
- if it barely moves, stop `meta_bwd` work and switch to
  `support_grad_create_graph`

Latest successful batch validation:
- canary tag: `phase12_fused_meta_canary_20260312_133621`
- regression tag: `phase12_fused_meta_regression_20260312_134255`
- profiling artifacts:
  - `experiments/phase12/runs/phase12_fused_meta_promotion_profile.csv`
  - `experiments/phase12/runs/phase12_fused_meta_promotion_profile_summary.csv`

Third bounded optimization batch:
- hypothesis: replace the local recompute graph with the compact math-SDPA
  primitive and let PyTorch own the softmax/matmul composition
- implementation target: use
  `torch.nn.functional.scaled_dot_product_attention` on fp32 tensors with the
  math backend forced

Decision rule for this batch:
- if `meta_bwd` premium drops materially again, continue on the `meta_bwd`
  line
- if it barely moves, stop `meta_bwd` work and switch to
  `support_grad_create_graph`

Observed outcome for the third batch:
- canary: pass (`phase12_fused_meta_canary_20260312_143646`)
- regression: pass (`phase12_fused_meta_regression_20260312_161009`)
- profiling: no meaningful gain; `meta_bwd` and `meta_recompute` premiums
  moved back toward the previous worse range
- branch decision: revert the math-SDPA recompute experiment and stop the
  `meta_bwd` line here

Next bounded optimization batch:
- switch focus to `support_grad_create_graph`
- target experimental-path higher-order support-grad overhead only
- keep fallback disabled, parity intact, and qualification out of scope
