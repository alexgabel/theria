# Phase 12 for colleagues

This is the shortest useful summary of what is supported today in the Phase 12
MAML stack.

Accepted stable baseline:
- gate tag: `phase12_meta_gate_phase12_stable_regression_20260309_155142`
- frontier tag: `phase12_stable_regression_20260309_155142`

Canonical pages:
- status: [`docs/STATUS.md`](STATUS.md)
- workflow: [`experiments/phase12/README.md`](../experiments/phase12/README.md)
- results snapshot: [`docs/phase12_results_snapshot.md`](phase12_results_snapshot.md)

## Current paths

| Path | Status | Use it for | Notes |
| --- | --- | --- | --- |
| `triton_fused_meta_strict FULL` | Stable | Accepted stable path | Eager execution only |
| `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2` | Practical | Accepted practical path | Eager execution only |
| `triton_fused_meta` | Experimental candidate | Several validated diffusion-like shapes where equal-time matters | Promoted candidate, not the default, not a universal replacement |
| CUDA-graph acceleration | Blocked | Not recommended | Do not use `CUDA_GRAPH_STATIC=1` in benchmark/regression commands |

## Start here

```bash
# Stable use
PYTHONPATH=. python experiments/phase12/scripts/run_phase12_behavior.py \
  --backend triton_fused_meta_strict \
  --mode FULL \
  --device cuda

# Practical use
PYTHONPATH=. python experiments/phase12/scripts/run_phase12_behavior.py \
  --backend triton_fused_meta_strict \
  --mode FULL_HYBRID \
  --meta-every-n-outer 8 \
  --meta-last-n-inner 2 \
  --device cuda

# Small accepted demo flow
bash experiments/phase12/scripts/run_phase12_stable_demo.sh
```

## FAQ

### What should I use?
- Use `triton_fused_meta_strict FULL` for the stable path.
- Use `triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer 8 --meta-last-n-inner 2`
  for the practical path.
- Consider `triton_fused_meta` only as a promoted experimental candidate across
  several validated diffusion-like shapes when equal-time performance matters.

### What should I not use?
- Do not use `CUDA_GRAPH_STATIC=1` in recommended benchmark or regression runs.
- Do not treat `triton_fused_meta` as the default backend.

### What is experimental?
- `triton_fused_meta` is now a promoted experimental candidate backed by 5-seed
  qualification on the target workload and two held-out diffusion-like shapes.
- It is still not the default path.
- It should not be described as a universal replacement or a win on every
  cell.

### Is CUDA-graph acceleration supported?
- Not in the accepted workflow. CUDA-graph capture is blocked in the attention
  math path under the repo capture-debug workflow.

### How do I validate a stable-path change?
- Run:

```bash
bash experiments/phase12/scripts/run_phase12_stable_frontier_regression.sh
```

- This is the accepted stable regression harness. Leave `CUDA_GRAPH_STATIC`
  unset.
