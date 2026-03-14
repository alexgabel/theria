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

Current branch outcome:
- status: promotion path active
- conservative defaults unchanged:
  - `triton_fused_meta_strict FULL`
  - `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2`
- `triton_fused_meta` is now a promoted experimental candidate for the target
  diffusion-like workload, especially when equal-time performance matters
- this is not a claim of universal replacement or a win on every cell

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

Current bounded optimization batch:
- hypothesis: unconditional runtime finite scans in the experimental
  recompute/meta-grad path are inflating `support_grad_create_graph` overhead
  during promotion profiling
- implementation target: keep those scans enabled when fallback is active, but
  skip them by default when fallback is explicitly disabled for promotion
  profiling/regression

Decision rule for this batch:
- if `support_grad_create_graph` premium drops materially without stability
  regressions, continue on this line
- if it barely moves, stop and choose a new hypothesis

Observed outcome for the current batch:
- canary: pass (`phase12_fused_meta_canary_20260312_170726`)
- regression: pass (`phase12_fused_meta_regression_20260312_171556`)
- profiling: strong win; `support_grad_create_graph` and `meta_bwd` premiums
  both dropped materially while parity stayed exact
- branch decision: reopen promotion with 2-seed qualification

Latest promotion result:
- 5-seed qualification completed:
  - gate tag: `phase12_meta_gate_phase12_requalify_diffproxy_s5_20260313_084040`
  - frontier tag: `phase12_requalify_diffproxy_s5_20260313_084040`
- outcome:
  - fallback-free
  - stable across seeds
  - exact equal-step parity
  - equal-time still not strong enough for promotion
- branch decision: keep `triton_fused_meta` experimental and continue one more
  bounded optimization cycle focused on the worst equal-time gaps:
  `FULL k=2/5` and `FULL_HYBRID k=2`

Promotion requalification outcome after the final bounded batch:
- 2-seed qualification:
  - gate tag: `phase12_meta_gate_phase12_requalify_diffproxy_s2_20260313_153202`
  - frontier tag: `phase12_requalify_diffproxy_s2_20260313_153202`
- 5-seed qualification:
  - gate tag: `phase12_meta_gate_phase12_requalify_diffproxy_s5_20260313_175213`
  - frontier tag: `phase12_requalify_diffproxy_s5_20260313_175213`
- outcome:
  - fallback-free
  - stable across 5 seeds
  - exact equal-step parity
  - equal-time competitive or better on the target diffusion-like workload
- branch decision: promotion path active
- caveat:
  - this is still not a claim that `triton_fused_meta` wins every cell or is a
    universal replacement
  - `FULL k=2` still trails `triton_fused_meta_strict`
  - run one held-out diffusion-like variant or shape before changing any
    default recommendation globally

Current bounded optimization batch:
- hypothesis: per-call CUDA event synchronization in the experimental-path
  meta/fast backward profiling is inflating the measured equal-time premium in
  official qualification runs, especially on `FULL k=2/5` and
  `FULL_HYBRID k=2`
- implementation target: keep profile counters, but accumulate CUDA event pairs
  lazily and flush them once when counters are read, instead of synchronizing
  on every profiled backward/recompute call

Decision rule for this batch:
- if the worst-cell step-time premiums move materially and stability/parity stay
  clean, consider one more promotion requalification
- if the premiums barely move, stop this line and park promotion again

Observed outcome for the current batch:
- canary: pass (`phase12_fused_meta_canary_20260313_121342`)
- regression: pass (`phase12_fused_meta_regression_20260313_121915`)
- profiling: wrong direction; the worst-cell premiums increased, especially on
  `FULL k=2/5` and `FULL_HYBRID k=2`
- branch decision: revert the lazy-event profiling batch and stop this line

Current bounded optimization batch:
- hypothesis: the experimental recompute/meta-grad path is still adding
  avoidable overhead on no-graph support-grad steps, especially in the short
  inner-step regimes that dominate the remaining equal-time gap
- implementation target: focus on `support_grad_no_graph` only and leave the
  `meta_bwd`/profiling path unchanged
- concrete change: gate the fast-path finite scans on `q/k/v/m/l/grad_out` and
  `dq/dk/dv` the same way the recompute finite scans were gated earlier:
  enabled when fallback is active, skipped by default when fallback is
  explicitly disabled for promotion profiling

Decision rule for this batch:
- if `support_grad_no_graph` premium drops materially without stability or
  parity regressions, consider one more promotion requalification
- if it barely moves, park promotion again
