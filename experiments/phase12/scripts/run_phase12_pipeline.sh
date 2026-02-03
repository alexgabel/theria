#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTER_STEPS="${OUTER_STEPS:-500}"
META_BATCH_SIZE="${META_BATCH_SIZE:-16}"
INNER_LR="${INNER_LR:-0.4}"
OUTER_LR="${OUTER_LR:-1e-3}"
SEEDS="${SEEDS:-0 1}"
INNER_STEPS_LIST="${INNER_STEPS_LIST:-1 5 10 20}"

RUN_DIR="experiments/phase12/runs"
FIG_DIR="experiments/phase12/figures"
mkdir -p "${RUN_DIR}" "${FIG_DIR}"

SMOKE_CSV="${RUN_DIR}/phase12_smoke_${RUN_TAG}.csv"
REF_CSV="${RUN_DIR}/phase12_reference_${RUN_TAG}.csv"
TRI_CSV="${RUN_DIR}/phase12_triton_${RUN_TAG}.csv"
RISK_CSV="${RUN_DIR}/phase12_risk_checks_${RUN_TAG}.csv"

echo "== Phase 12: smoke test (CPU, FULL, k=1) =="
python experiments/phase12/scripts/run_phase12_behavior.py \
  --backend reference \
  --mode FULL \
  --seed 0 \
  --outer-steps 20 \
  --meta-batch-size "${META_BATCH_SIZE}" \
  --inner-steps 1 \
  --inner-lr "${INNER_LR}" \
  --outer-lr "${OUTER_LR}" \
  --device cpu \
  --csv-out "${SMOKE_CSV}"

echo "== Phase 12: reference grid (CPU) =="
for mode in FULL FO FO_STRICT; do
  for inner_steps in ${INNER_STEPS_LIST}; do
    for seed in ${SEEDS}; do
      python experiments/phase12/scripts/run_phase12_behavior.py \
        --backend reference \
        --mode "${mode}" \
        --seed "${seed}" \
        --outer-steps "${OUTER_STEPS}" \
        --meta-batch-size "${META_BATCH_SIZE}" \
        --inner-steps "${inner_steps}" \
        --inner-lr "${INNER_LR}" \
        --outer-lr "${OUTER_LR}" \
        --device cpu \
        --csv-out "${REF_CSV}"
    done
  done
done

echo "== Phase 12: triton grid (CUDA) =="
for mode in FULL FO FO_STRICT; do
  for inner_steps in ${INNER_STEPS_LIST}; do
    for seed in ${SEEDS}; do
      python experiments/phase12/scripts/run_phase12_behavior.py \
        --backend triton_fused \
        --mode "${mode}" \
        --seed "${seed}" \
        --outer-steps "${OUTER_STEPS}" \
        --meta-batch-size "${META_BATCH_SIZE}" \
        --inner-steps "${inner_steps}" \
        --inner-lr "${INNER_LR}" \
        --outer-lr "${OUTER_LR}" \
        --device cuda \
        --csv-out "${TRI_CSV}"
    done
  done
done

echo "== Phase 12: summarize results =="
python experiments/phase12/scripts/summarize_phase12_results.py \
  --reference-csv "${REF_CSV}" \
  --triton-csv "${TRI_CSV}" \
  --out-dir "${RUN_DIR}" \
  --fig-dir "${FIG_DIR}" \
  --tag "${RUN_TAG}"

echo "== Phase 12: risk checks =="
python experiments/phase12/scripts/check_phase12_risks.py \
  --determinism-outer-steps 100 \
  --out-csv "${RISK_CSV}" \
  --parity-device cuda

echo "Done."
echo "Smoke CSV:      ${SMOKE_CSV}"
echo "Reference CSV:  ${REF_CSV}"
echo "Triton CSV:     ${TRI_CSV}"
echo "Risk checks:    ${RISK_CSV}"
