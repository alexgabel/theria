#!/usr/bin/env bash
set -euo pipefail

# Default Phase-12 benchmark entrypoint:
# 1) run meta-contract regression gate (must pass),
# 2) run dual-protocol MAML utility frontier.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TAG="${TAG:-phase12_frontier_$(date +%Y%m%d_%H%M%S)}"
GATE_TAG="${GATE_TAG:-phase12_meta_gate_${TAG}}"
FRONTIER_TAG="${FRONTIER_TAG:-${TAG}}"
# If FRONTIER_SEEDS is not provided, reuse SEEDS (if provided) to avoid
# accidentally running a larger frontier grid than the gate grid.
FRONTIER_SEEDS="${FRONTIER_SEEDS:-${SEEDS:-0,1,2,3,4}}"
GATE_SEEDS="${SEEDS:-0,1}"

echo "== [1/2] Meta-contract gate =="
TAG="${GATE_TAG}" \
REFERENCE_BACKEND="${REFERENCE_BACKEND:-reference}" \
META_BACKEND="${META_BACKEND:-triton_fused_meta}" \
STRICT_BACKEND="${STRICT_BACKEND:-triton_fused_meta_strict}" \
PROFILES="${PROFILES:-seqcls_default,diffusion_proxy}" \
INNER_STEPS="${INNER_STEPS:-2,5,10}" \
SEEDS="${GATE_SEEDS}" \
INNER_LR="${INNER_LR:-0.4}" \
DEVICE="${DEVICE:-cuda}" \
MIN_COSINE_FULL_MEAN="${MIN_COSINE_FULL_MEAN:-0.9999}" \
MIN_COSINE_FO_MEAN="${MIN_COSINE_FO_MEAN:-0.9999}" \
MAX_REL_DIFF_ABS_ERROR="${MAX_REL_DIFF_ABS_ERROR:-2e-4}" \
REQUIRE_TREND_SIGN_MATCH="${REQUIRE_TREND_SIGN_MATCH:-1}" \
bash experiments/phase12/scripts/run_phase12_meta_contract_gate.sh

echo "== [2/2] MAML utility frontier (equal-step + equal-time) =="
TAG="${FRONTIER_TAG}" \
BACKENDS="${BACKENDS:-reference,triton_fused,triton_fused_meta}" \
MODES="${MODES:-FULL,FO}" \
INNER_STEPS="${INNER_STEPS:-2,5,10}" \
SEEDS="${FRONTIER_SEEDS}" \
DEVICE="${DEVICE:-cuda}" \
EQUAL_STEP_OUTER_STEPS="${EQUAL_STEP_OUTER_STEPS:-200}" \
WALL_CLOCK_BUDGET_S="${WALL_CLOCK_BUDGET_S:-30}" \
FULL_OUTER_LR="${FULL_OUTER_LR:-1e-3}" \
FO_OUTER_LR="${FO_OUTER_LR:-1e-3}" \
INNER_LR="${INNER_LR:-0.4}" \
OUTER_LR="${OUTER_LR:-1e-3}" \
META_BATCH_SIZE="${META_BATCH_SIZE:-16}" \
SEQ_LEN="${SEQ_LEN:-32}" \
NUM_SIGNAL_POSITIONS="${NUM_SIGNAL_POSITIONS:-4}" \
bash experiments/phase12/scripts/run_phase12_maml_seqcls_dual_protocol.sh

echo "done: gate tag=${GATE_TAG}, frontier tag=${FRONTIER_TAG}"
