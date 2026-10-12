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

for plan_id in all_int4 all_int8 layer_10_int8 layer_22_int8 layer_23_int8; do
  "$PYTHON_RUNNER" "$ENTRYPOINT" run-final-plan \
    --suite full_dev --plan-id "$plan_id"
done

for plan_id in all_int4 layer_10_int8 layer_22_int8 layer_23_int8; do
  "$PYTHON_RUNNER" "$ENTRYPOINT" run-final-plan \
    --suite high_risk --plan-id "$plan_id"
done

"$PYTHON_RUNNER" "$ENTRYPOINT" pending-final
"$PYTHON_RUNNER" "$ENTRYPOINT" finalize
