#!/usr/bin/env bash
set -euo pipefail

PYTHON_RUNNER="${MODULE_SENSITIVITY_PYTHON:-python3}"
ENTRYPOINT="scripts/run_module_sensitivity.py"

"$PYTHON_RUNNER" "$ENTRYPOINT" select-advancement
"$PYTHON_RUNNER" "$ENTRYPOINT" prepare-final

for suite in determinism full_dev high_risk; do
  plan_ids="$($PYTHON_RUNNER -c '
import json
import sys
from pathlib import Path
suite = sys.argv[1]
manifest = json.loads(Path("out/module_sensitivity/final_evaluation/experiment_manifest.json").read_text(encoding="utf-8"))
print(" ".join(manifest["suites"][suite]["plan_ids"]))
' "$suite")"
  for plan_id in $plan_ids; do
    "$PYTHON_RUNNER" "$ENTRYPOINT" run-final-plan \
      --suite "$suite" --plan-id "$plan_id"
  done
done

"$PYTHON_RUNNER" "$ENTRYPOINT" pending-final
"$PYTHON_RUNNER" "$ENTRYPOINT" finalize
