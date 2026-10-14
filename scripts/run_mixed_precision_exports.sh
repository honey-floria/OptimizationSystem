#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
DEVICE="${DEVICE:-cuda:0}"

python3 scripts/run_mixed_precision_search.py prepare

for plan_id in \
  all_linear_int4 \
  all_linear_int8 \
  mixed_l10_mlp_int8 \
  mixed_l10_mlp_bf16 \
  mixed_l10_mlp_lm_head_int8 \
  mixed_l10_l23_mlp_lm_head_int8
do
  args=(--plan-id "${plan_id}" --device "${DEVICE}")
  if [[ -n "${MODEL_PATH}" ]]; then
    args+=(--model-path "${MODEL_PATH}")
  fi
  python3 scripts/run_mixed_precision_search.py export "${args[@]}"
done

python3 scripts/run_mixed_precision_search.py pending
