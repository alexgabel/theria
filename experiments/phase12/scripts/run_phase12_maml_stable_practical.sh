#!/usr/bin/env bash
set -euo pipefail

# Stable practical systems target for Phase 12:
#   - backend: triton_fused_meta_strict
#   - mode: FULL_HYBRID
#   - meta_every_n_outer=8
#   - meta_last_n_inner=2
#
# This is the default entrypoint for stable wall-clock oriented benchmarking
# and profiling work. It intentionally targets the practical hybrid setting,
# not pure FULL.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

BASELINE_ENV="${REPO_ROOT}/experiments/phase12/phase12_stable_baseline.env"
if [[ -f "${BASELINE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${BASELINE_ENV}"
fi

TAG="${TAG:-phase12_stable_practical_$(date +%Y%m%d_%H%M%S)}"

TAG="${TAG}" \
BACKENDS="${BACKENDS:-triton_fused_meta_strict}" \
MODES="${MODES:-FULL_HYBRID}" \
META_EVERY_N_OUTER="${META_EVERY_N_OUTER:-${PHASE12_STABLE_META_EVERY_N_OUTER:-8}}" \
META_LAST_N_INNER="${META_LAST_N_INNER:-${PHASE12_STABLE_META_LAST_N_INNER:-2}}" \
FULL_HYBRID_OUTER_LR="${FULL_HYBRID_OUTER_LR:-1e-3}" \
bash experiments/phase12/scripts/run_phase12_maml_seqcls_dual_protocol.sh
