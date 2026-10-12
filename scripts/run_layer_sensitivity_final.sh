#!/usr/bin/env bash
set -euo pipefail

PYTHON_RUNNER="${LAYER_SENSITIVITY_PYTHON:-python3}"
ENTRYPOINT="scripts/run_layer_sensitivity.py"

"$PYTHON_RUNNER" "$ENTRYPOINT" prepare-final

for plan_id in all_int4 layer_22_int8; do
  "$PYTHON_RUNNER" "$ENTRYPOINT" run-final-plan \
    --suite determinism --plan-id "$plan_id"
done

"$PYTHON_RUNNER" "$ENTRYPOINT" pending-final --suite determinism
"$PYTHON_RUNNER" "$ENTRYPOINT" audit-determinism

for suite in full_dev high_risk; do
  for plan_id in all_int4 all_int8; do
    "$PYTHON_RUNNER" "$ENTRYPOINT" run-final-plan \
      --suite "$suite" --plan-id "$plan_id"
  done
  for layer_index in $(seq -w 0 27); do
    "$PYTHON_RUNNER" "$ENTRYPOINT" run-final-plan \
      --suite "$suite" --plan-id "layer_${layer_index}_int8"
  done
done

"$PYTHON_RUNNER" "$ENTRYPOINT" pending-final
"$PYTHON_RUNNER" "$ENTRYPOINT" finalize
