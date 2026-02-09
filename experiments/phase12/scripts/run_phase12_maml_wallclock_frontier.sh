#!/usr/bin/env bash
set -euo pipefail

# Fixed-wall-clock MAML seqcls frontier:
# reference + triton_fused + triton_full_autograd under FULL/FO.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
RUNS_DIR="experiments/phase12/runs"
mkdir -p "${RUNS_DIR}"

CSV_OUT="${CSV_OUT:-${RUNS_DIR}/phase12_maml_seqcls_wallclock_${TAG}.csv}"
SUMMARY_OUT="${SUMMARY_OUT:-${RUNS_DIR}/phase12_maml_seqcls_wallclock_${TAG}_summary.csv}"

BACKENDS="${BACKENDS:-reference,triton_fused,triton_fused_meta,triton_full_autograd}"
MODES="${MODES:-FULL,FO}"
INNER_STEPS="${INNER_STEPS:-2,5,10}"
SEEDS="${SEEDS:-0,1,2,3,4}"

WALL_CLOCK_BUDGET_S="${WALL_CLOCK_BUDGET_S:-30}"
PILOT_STEPS="${PILOT_STEPS:-30}"
MIN_OUTER_STEPS="${MIN_OUTER_STEPS:-20}"
MAX_OUTER_STEPS="${MAX_OUTER_STEPS:-1000}"

META_BATCH_SIZE="${META_BATCH_SIZE:-16}"
INNER_LR="${INNER_LR:-0.4}"
OUTER_LR="${OUTER_LR:-1e-3}"
SEQ_LEN="${SEQ_LEN:-32}"
NUM_SIGNAL_POSITIONS="${NUM_SIGNAL_POSITIONS:-4}"
DEVICE="${DEVICE:-cuda}"

PYTHONPATH=. python experiments/phase12/scripts/run_phase12_maml_seqcls_compare.py \
  --backends "${BACKENDS}" \
  --modes "${MODES}" \
  --inner-steps "${INNER_STEPS}" \
  --seeds "${SEEDS}" \
  --wall-clock-budget-s "${WALL_CLOCK_BUDGET_S}" \
  --pilot-steps "${PILOT_STEPS}" \
  --min-outer-steps "${MIN_OUTER_STEPS}" \
  --max-outer-steps "${MAX_OUTER_STEPS}" \
  --meta-batch-size "${META_BATCH_SIZE}" \
  --inner-lr "${INNER_LR}" \
  --outer-lr "${OUTER_LR}" \
  --seq-len "${SEQ_LEN}" \
  --num-signal-positions "${NUM_SIGNAL_POSITIONS}" \
  --device "${DEVICE}" \
  --csv-out "${CSV_OUT}" \
  --summary-out "${SUMMARY_OUT}"

echo "wrote ${CSV_OUT}"
echo "wrote ${SUMMARY_OUT}"
