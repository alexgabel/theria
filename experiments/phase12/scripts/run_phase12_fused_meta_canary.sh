#!/usr/bin/env bash
set -euo pipefail

# Primary instability canary for experimental fused meta backward.
# This is the fixed diffusion-like config that previously produced non-finite
# failures. The canary must pass without runtime fallback.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TAG="${TAG:-phase12_fused_meta_canary_$(date +%Y%m%d_%H%M%S)}"
RUNS_DIR="experiments/phase12/runs"
mkdir -p "${RUNS_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export THERIA_TRITON_META_ENABLE_FALLBACK=0

SEEDS="${SEEDS:-1 4}"
OUTER_STEPS="${OUTER_STEPS:-60}"
META_BATCH_SIZE="${META_BATCH_SIZE:-8}"
INNER_STEPS="${INNER_STEPS:-10}"
INNER_LR="${INNER_LR:-0.4}"
OUTER_LR="${OUTER_LR:-1e-3}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_SIGNAL_POSITIONS="${NUM_SIGNAL_POSITIONS:-12}"
DEVICE="${DEVICE:-cuda}"

echo "== Phase 12 fused-meta canary =="
echo "tag=${TAG}"
echo "seeds=${SEEDS}"
echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
echo "THERIA_TRITON_META_ENABLE_FALLBACK=${THERIA_TRITON_META_ENABLE_FALLBACK}"

for seed in ${SEEDS}; do
  out_csv="${RUNS_DIR}/${TAG}_seed${seed}.csv"
  summary_csv="${RUNS_DIR}/${TAG}_seed${seed}_summary.csv"
  echo
  echo "-- seed=${seed} -> ${out_csv}"
  PYTHONPATH=. python experiments/phase12/scripts/run_phase12_fused_meta_instability_diff.py \
    --fused-backend triton_fused_meta \
    --strict-backend triton_fused_meta_strict \
    --mode FULL \
    --seed "${seed}" \
    --outer-steps "${OUTER_STEPS}" \
    --meta-batch-size "${META_BATCH_SIZE}" \
    --inner-steps "${INNER_STEPS}" \
    --inner-lr "${INNER_LR}" \
    --outer-lr "${OUTER_LR}" \
    --seq-len "${SEQ_LEN}" \
    --num-signal-positions "${NUM_SIGNAL_POSITIONS}" \
    --device "${DEVICE}" \
    --fail-on-divergence \
    --csv-out "${out_csv}" \
    --summary-out "${summary_csv}"
done

echo
echo "canary PASSED (tag=${TAG})"
