#!/usr/bin/env bash
set -euo pipefail

# Stable optimization regression loop:
# 1) meta-contract gate
# 2) stable frontier (reference + triton_fused_meta_strict)
# 3) acceptance checks:
#    - equal-time FULL_HYBRID remains on the FO frontier
#    - equal-step FULL semantics unchanged

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

BASELINE_ENV="${REPO_ROOT}/experiments/phase12/phase12_stable_baseline.env"
if [[ -f "${BASELINE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${BASELINE_ENV}"
fi

TAG="${TAG:-phase12_stable_regression_$(date +%Y%m%d_%H%M%S)}"
GATE_TAG="${GATE_TAG:-phase12_meta_gate_${TAG}}"
FRONTIER_TAG="${FRONTIER_TAG:-${TAG}}"
HYBRID_FRONTIER_ACC_TOL="${HYBRID_FRONTIER_ACC_TOL:-${PHASE12_STABLE_HYBRID_FRONTIER_ACC_TOL:-0.01}}"

echo "== [1/3] Meta-contract gate =="
echo "frozen_baseline gate=${PHASE12_STABLE_BASELINE_GATE_TAG:-unset} frontier=${PHASE12_STABLE_BASELINE_FRONTIER_TAG:-unset} N=${PHASE12_STABLE_META_EVERY_N_OUTER:-8} L=${PHASE12_STABLE_META_LAST_N_INNER:-2} hybrid_tol=${HYBRID_FRONTIER_ACC_TOL}"
TAG="${GATE_TAG}" \
REFERENCE_BACKEND="${REFERENCE_BACKEND:-reference}" \
META_BACKEND="${META_BACKEND:-triton_fused_meta_strict}" \
STRICT_BACKEND="${STRICT_BACKEND:-triton_fused_meta_strict}" \
PROFILES="${PROFILES:-seqcls_default,diffusion_proxy}" \
INNER_STEPS="${INNER_STEPS:-2,5,10}" \
SEEDS="${GATE_SEEDS:-0,1}" \
INNER_LR="${INNER_LR:-0.4}" \
DEVICE="${DEVICE:-cuda}" \
bash experiments/phase12/scripts/run_phase12_meta_contract_gate.sh

echo "== [2/3] Stable frontier =="
TAG="${FRONTIER_TAG}" \
BACKENDS="${BACKENDS:-reference,triton_fused_meta_strict}" \
MODES="${MODES:-FULL,FO,FULL_HYBRID}" \
INNER_STEPS="${INNER_STEPS:-2,5,10}" \
SEEDS="${FRONTIER_SEEDS:-0,1}" \
DEVICE="${DEVICE:-cuda}" \
EQUAL_STEP_OUTER_STEPS="${EQUAL_STEP_OUTER_STEPS:-200}" \
WALL_CLOCK_BUDGET_S="${WALL_CLOCK_BUDGET_S:-30}" \
META_EVERY_N_OUTER="${META_EVERY_N_OUTER:-${PHASE12_STABLE_META_EVERY_N_OUTER:-8}}" \
META_LAST_N_INNER="${META_LAST_N_INNER:-${PHASE12_STABLE_META_LAST_N_INNER:-2}}" \
FULL_OUTER_LR="${FULL_OUTER_LR:-1e-3}" \
FO_OUTER_LR="${FO_OUTER_LR:-1e-3}" \
FULL_HYBRID_OUTER_LR="${FULL_HYBRID_OUTER_LR:-1e-3}" \
META_BATCH_SIZE="${META_BATCH_SIZE:-16}" \
SEQ_LEN="${SEQ_LEN:-32}" \
NUM_SIGNAL_POSITIONS="${NUM_SIGNAL_POSITIONS:-4}" \
INNER_LR="${INNER_LR:-0.4}" \
bash experiments/phase12/scripts/run_phase12_maml_seqcls_dual_protocol.sh

STEP_SUMMARY="experiments/phase12/runs/phase12_maml_seqcls_equal_step_${FRONTIER_TAG}_summary.csv"
TIME_SUMMARY="experiments/phase12/runs/phase12_maml_seqcls_equal_time_${FRONTIER_TAG}_summary.csv"

echo "== [3/3] Acceptance checks =="
PYTHONPATH=. python experiments/phase12/scripts/check_phase12_stable_frontier.py \
  --equal-step-summary "${STEP_SUMMARY}" \
  --equal-time-summary "${TIME_SUMMARY}" \
  --reference-backend "${REFERENCE_BACKEND:-reference}" \
  --stable-backend "${STABLE_BACKEND:-triton_fused_meta_strict}" \
  --hybrid-frontier-acc-tol "${HYBRID_FRONTIER_ACC_TOL}" \
  --fail-on-check

echo "done: gate tag=${GATE_TAG}, frontier tag=${FRONTIER_TAG}"
