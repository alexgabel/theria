#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source experiments/phase12/phase12_stable_baseline.env

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
OUTER_STEPS="${OUTER_STEPS:-20}"
META_BATCH_SIZE="${META_BATCH_SIZE:-8}"
INNER_STEPS="${INNER_STEPS:-2}"
INNER_LR="${INNER_LR:-0.4}"
OUTER_LR="${OUTER_LR:-1e-3}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_SIGNAL_POSITIONS="${NUM_SIGNAL_POSITIONS:-12}"
TAG="${TAG:-phase12_stable_demo_$(date +%Y%m%d_%H%M%S)}"

RUNS_DIR="experiments/phase12/runs"
FULL_OUT="${RUNS_DIR}/${TAG}_stable_full.csv"
PRACTICAL_OUT="${RUNS_DIR}/${TAG}_practical_hybrid.csv"

echo "== Phase 12 stable demo =="
echo "stable baseline: ${PHASE12_STABLE_BASELINE_FRONTIER_TAG}"
echo "device=${DEVICE} seed=${SEED} outer_steps=${OUTER_STEPS} inner_steps=${INNER_STEPS}"
echo

PYTHONPATH=. "${PYTHON_BIN}" experiments/phase12/scripts/run_phase12_behavior.py \
  --backend triton_fused_meta_strict \
  --mode FULL \
  --seed "${SEED}" \
  --outer-steps "${OUTER_STEPS}" \
  --meta-batch-size "${META_BATCH_SIZE}" \
  --inner-steps "${INNER_STEPS}" \
  --inner-lr "${INNER_LR}" \
  --outer-lr "${OUTER_LR}" \
  --seq-len "${SEQ_LEN}" \
  --num-signal-positions "${NUM_SIGNAL_POSITIONS}" \
  --device "${DEVICE}" \
  --csv-out "${FULL_OUT}"

PYTHONPATH=. "${PYTHON_BIN}" experiments/phase12/scripts/run_phase12_behavior.py \
  --backend triton_fused_meta_strict \
  --mode FULL_HYBRID \
  --seed "${SEED}" \
  --outer-steps "${OUTER_STEPS}" \
  --meta-batch-size "${META_BATCH_SIZE}" \
  --inner-steps "${INNER_STEPS}" \
  --inner-lr "${INNER_LR}" \
  --outer-lr "${OUTER_LR}" \
  --meta-every-n-outer "${PHASE12_STABLE_META_EVERY_N_OUTER}" \
  --meta-last-n-inner "${PHASE12_STABLE_META_LAST_N_INNER}" \
  --seq-len "${SEQ_LEN}" \
  --num-signal-positions "${NUM_SIGNAL_POSITIONS}" \
  --device "${DEVICE}" \
  --csv-out "${PRACTICAL_OUT}"

echo
echo "Wrote:"
echo "- ${FULL_OUT}"
echo "- ${PRACTICAL_OUT}"
echo
echo "Stable path: triton_fused_meta_strict FULL"
echo "Practical path: triton_fused_meta_strict FULL_HYBRID --meta-every-n-outer ${PHASE12_STABLE_META_EVERY_N_OUTER} --meta-last-n-inner ${PHASE12_STABLE_META_LAST_N_INNER}"
