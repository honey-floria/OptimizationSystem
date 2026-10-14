#!/usr/bin/env bash
set -euo pipefail

PYTHON_RUNNER="${MODULE_SENSITIVITY_PYTHON:-python3}"
ENTRYPOINT="scripts/run_module_sensitivity.py"
WORKFLOW_ARGS=(--workflow combination)

"$PYTHON_RUNNER" "$ENTRYPOINT" "${WORKFLOW_ARGS[@]}" prepare-experiments

plan_ids="$($PYTHON_RUNNER -c '
import json
from pathlib import Path
config = json.loads(Path("configs/module_sensitivity.json").read_text(encoding="utf-8"))
print(" ".join(plan["plan_id"] for plan in config["combination_evaluation"]["plans"]))
')"

for plan_id in $plan_ids; do
  "$PYTHON_RUNNER" "$ENTRYPOINT" "${WORKFLOW_ARGS[@]}" run-plan --plan-id "$plan_id"
done

"$PYTHON_RUNNER" "$ENTRYPOINT" "${WORKFLOW_ARGS[@]}" select-advancement
"$PYTHON_RUNNER" "$ENTRYPOINT" "${WORKFLOW_ARGS[@]}" prepare-final

for suite in determinism full_dev high_risk; do
  for plan_id in $plan_ids; do
    "$PYTHON_RUNNER" "$ENTRYPOINT" "${WORKFLOW_ARGS[@]}" run-final-plan \
      --suite "$suite" --plan-id "$plan_id"
  done
done

"$PYTHON_RUNNER" "$ENTRYPOINT" "${WORKFLOW_ARGS[@]}" pending-final
"$PYTHON_RUNNER" "$ENTRYPOINT" "${WORKFLOW_ARGS[@]}" finalize
