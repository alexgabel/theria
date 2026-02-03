#!/usr/bin/env bash
set -euo pipefail

SCRIPT=experiments/phase10/scripts/run_maml_backend_compare.py
INNER_STEPS=5
INNER_LR=0.1

# mode flags: "" = FULL, "--fo" = FO, "--fo-strict" = FO_STR
declare -a MODES=("" "--fo" "--fo-strict")
declare -a BACKENDS=("reference" "custom" "triton_fused")

# Header (TSV)
echo -e "backend\tmode\tsecond_order_path\trel_diff_full_vs_fo"

for backend in "${BACKENDS[@]}"; do
  for mode_flag in "${MODES[@]}"; do

    python "$SCRIPT" \
      --backend "$backend" \
      --inner-steps "$INNER_STEPS" \
      --inner-lr "$INNER_LR" \
      --steps 1 \
      $mode_flag \
    | tail -n 1 \
    | awk '
      {
        backend=""; mode=""; sop=""; diff=""
        for (i=1;i<=NF;i++) {
          if ($i ~ /^backend=/) backend=$i
          if ($i ~ /^mode=/) mode=$i
          if ($i ~ /^second_order_path=/) sop=$i
          if ($i ~ /^rel_diff_full_vs_fo=/) diff=$i
        }
        printf "%s\t%s\t%s\t%s\n", backend, mode, sop, diff
      }
    '

  done
done