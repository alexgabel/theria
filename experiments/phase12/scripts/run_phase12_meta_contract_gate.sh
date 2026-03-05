#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TAG="${TAG:-phase12_meta_gate_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-experiments/phase12/runs}"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-reference}"
META_BACKEND="${META_BACKEND:-triton_fused_meta_strict}"
STRICT_BACKEND="${STRICT_BACKEND:-triton_fused_meta_strict}"
ALLOW_EXPERIMENTAL_BACKENDS="${ALLOW_EXPERIMENTAL_BACKENDS:-0}"
PROFILES="${PROFILES:-seqcls_default,diffusion_proxy}"
INNER_STEPS="${INNER_STEPS:-2,5,10}"
SEEDS="${SEEDS:-0,1}"
INNER_LR="${INNER_LR:-0.4}"
DEVICE="${DEVICE:-cuda}"

# Tight defaults; override via env for exploratory runs.
MIN_COSINE_FULL_MEAN="${MIN_COSINE_FULL_MEAN:-0.9999}"
MIN_COSINE_FO_MEAN="${MIN_COSINE_FO_MEAN:-0.9999}"
MAX_REL_DIFF_ABS_ERROR="${MAX_REL_DIFF_ABS_ERROR:-2e-4}"
REQUIRE_TREND_SIGN_MATCH="${REQUIRE_TREND_SIGN_MATCH:-1}"

args=(
  --reference-backend "${REFERENCE_BACKEND}"
  --meta-backend "${META_BACKEND}"
  --strict-backend "${STRICT_BACKEND}"
  --profiles "${PROFILES}"
  --inner-steps "${INNER_STEPS}"
  --seeds "${SEEDS}"
  --inner-lr "${INNER_LR}"
  --device "${DEVICE}"
  --out-dir "${OUT_DIR}"
  --tag "${TAG}"
  --fail-on-gate
  --min-cosine-full-mean "${MIN_COSINE_FULL_MEAN}"
  --min-cosine-fo-mean "${MIN_COSINE_FO_MEAN}"
  --max-rel-diff-abs-error "${MAX_REL_DIFF_ABS_ERROR}"
)

if [[ "${REQUIRE_TREND_SIGN_MATCH}" == "1" ]]; then
  args+=(--require-trend-sign-match)
fi
if [[ "${ALLOW_EXPERIMENTAL_BACKENDS}" == "1" ]]; then
  args+=(--allow-experimental-backends)
fi

PYTHONPATH=. python experiments/phase12/scripts/run_phase12_meta_contract_dual_compare.py "${args[@]}"

echo "meta-contract gate PASSED (tag=${TAG})"
