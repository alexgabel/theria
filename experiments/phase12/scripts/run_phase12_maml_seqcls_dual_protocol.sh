#!/usr/bin/env bash
set -euo pipefail

# Runs both equal-step and equal-time comparisons with optional mode-specific LR overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
RUNS_DIR="experiments/phase12/runs"
mkdir -p "${RUNS_DIR}"

BACKENDS="${BACKENDS:-reference,triton_fused,triton_fused_meta,triton_full_autograd}"
MODES="${MODES:-FULL,FO}"
INNER_STEPS="${INNER_STEPS:-2,5,10}"
SEEDS="${SEEDS:-0,1,2,3,4}"

META_BATCH_SIZE="${META_BATCH_SIZE:-16}"
INNER_LR="${INNER_LR:-0.4}"
OUTER_LR="${OUTER_LR:-1e-3}"
SEQ_LEN="${SEQ_LEN:-32}"
NUM_SIGNAL_POSITIONS="${NUM_SIGNAL_POSITIONS:-4}"
DEVICE="${DEVICE:-cuda}"

EQUAL_STEP_OUTER_STEPS="${EQUAL_STEP_OUTER_STEPS:-200}"
WALL_CLOCK_BUDGET_S="${WALL_CLOCK_BUDGET_S:-30}"
PILOT_STEPS="${PILOT_STEPS:-30}"
MIN_OUTER_STEPS="${MIN_OUTER_STEPS:-20}"
MAX_OUTER_STEPS="${MAX_OUTER_STEPS:-1000}"

STEP_CSV="${STEP_CSV:-${RUNS_DIR}/phase12_maml_seqcls_equal_step_${TAG}.csv}"
STEP_SUMMARY="${STEP_SUMMARY:-${RUNS_DIR}/phase12_maml_seqcls_equal_step_${TAG}_summary.csv}"
TIME_CSV="${TIME_CSV:-${RUNS_DIR}/phase12_maml_seqcls_equal_time_${TAG}.csv}"
TIME_SUMMARY="${TIME_SUMMARY:-${RUNS_DIR}/phase12_maml_seqcls_equal_time_${TAG}_summary.csv}"

extra_lr_args=()
[[ -n "${FULL_INNER_LR:-}" ]] && extra_lr_args+=(--full-inner-lr "${FULL_INNER_LR}")
[[ -n "${FULL_OUTER_LR:-}" ]] && extra_lr_args+=(--full-outer-lr "${FULL_OUTER_LR}")
[[ -n "${FO_INNER_LR:-}" ]] && extra_lr_args+=(--fo-inner-lr "${FO_INNER_LR}")
[[ -n "${FO_OUTER_LR:-}" ]] && extra_lr_args+=(--fo-outer-lr "${FO_OUTER_LR}")
[[ -n "${FO_STRICT_INNER_LR:-}" ]] && extra_lr_args+=(--fo-strict-inner-lr "${FO_STRICT_INNER_LR}")
[[ -n "${FO_STRICT_OUTER_LR:-}" ]] && extra_lr_args+=(--fo-strict-outer-lr "${FO_STRICT_OUTER_LR}")
[[ -n "${FULL_FROZEN_INNER_LR:-}" ]] && extra_lr_args+=(--full-frozen-inner-lr "${FULL_FROZEN_INNER_LR}")
[[ -n "${FULL_FROZEN_OUTER_LR:-}" ]] && extra_lr_args+=(--full-frozen-outer-lr "${FULL_FROZEN_OUTER_LR}")
[[ -n "${FULL_HYBRID_INNER_LR:-}" ]] && extra_lr_args+=(--full-hybrid-inner-lr "${FULL_HYBRID_INNER_LR}")
[[ -n "${FULL_HYBRID_OUTER_LR:-}" ]] && extra_lr_args+=(--full-hybrid-outer-lr "${FULL_HYBRID_OUTER_LR}")

extra_mode_args=()
[[ -n "${META_EVERY_N_OUTER:-}" ]] && extra_mode_args+=(--meta-every-n-outer "${META_EVERY_N_OUTER}")
if [[ "${PROFILE_META_BWD:-0}" == "1" ]]; then
  extra_mode_args+=(--profile-meta-bwd)
fi

echo "== Equal-step run =="
PYTHONPATH=. python experiments/phase12/scripts/run_phase12_maml_seqcls_compare.py \
  --backends "${BACKENDS}" \
  --modes "${MODES}" \
  --inner-steps "${INNER_STEPS}" \
  --seeds "${SEEDS}" \
  --outer-steps "${EQUAL_STEP_OUTER_STEPS}" \
  --meta-batch-size "${META_BATCH_SIZE}" \
  --inner-lr "${INNER_LR}" \
  --outer-lr "${OUTER_LR}" \
  --seq-len "${SEQ_LEN}" \
  --num-signal-positions "${NUM_SIGNAL_POSITIONS}" \
  --device "${DEVICE}" \
  --csv-out "${STEP_CSV}" \
  --summary-out "${STEP_SUMMARY}" \
  "${extra_lr_args[@]}" \
  "${extra_mode_args[@]}"

echo "== Equal-time run =="
PYTHONPATH=. python experiments/phase12/scripts/run_phase12_maml_seqcls_compare.py \
  --backends "${BACKENDS}" \
  --modes "${MODES}" \
  --inner-steps "${INNER_STEPS}" \
  --seeds "${SEEDS}" \
  --wall-clock-budget-s "${WALL_CLOCK_BUDGET_S}" \
  --pilot-steps "${PILOT_STEPS}" \
  --min-outer-steps "${MIN_OUTER_STEPS}" \
  --max-outer-steps "${MAX_OUTER_STEPS}" \
  --outer-steps "${EQUAL_STEP_OUTER_STEPS}" \
  --meta-batch-size "${META_BATCH_SIZE}" \
  --inner-lr "${INNER_LR}" \
  --outer-lr "${OUTER_LR}" \
  --seq-len "${SEQ_LEN}" \
  --num-signal-positions "${NUM_SIGNAL_POSITIONS}" \
  --device "${DEVICE}" \
  --csv-out "${TIME_CSV}" \
  --summary-out "${TIME_SUMMARY}" \
  "${extra_lr_args[@]}" \
  "${extra_mode_args[@]}"

echo "wrote ${STEP_CSV}"
echo "wrote ${STEP_SUMMARY}"
echo "wrote ${TIME_CSV}"
echo "wrote ${TIME_SUMMARY}"
