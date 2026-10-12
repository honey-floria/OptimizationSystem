#!/usr/bin/env bash
set -euo pipefail

PYTHON_RUNNER="${MODULE_SENSITIVITY_PYTHON:-python3}"
ENTRYPOINT="scripts/run_module_sensitivity.py"

"$PYTHON_RUNNER" "$ENTRYPOINT" prepare-experiments

plan_ids="$($PYTHON_RUNNER -c '
import json
from pathlib import Path
manifest = json.loads(Path("out/module_sensitivity/pilot/experiment_manifest.json").read_text(encoding="utf-8"))
print(" ".join(plan["plan_id"] for plan in manifest["plans"] if plan.get("runnable")))
')"

for plan_id in $plan_ids; do
  "$PYTHON_RUNNER" "$ENTRYPOINT" run-plan --plan-id "$plan_id"
done

"$PYTHON_RUNNER" "$ENTRYPOINT" pending
