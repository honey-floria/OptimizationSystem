#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
DEVICE="${DEVICE:-cuda:0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${MODEL_PATH}" && ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: MODEL_PATH 不存在或缺少 config.json: ${MODEL_PATH}" >&2
  echo "请把 MODEL_PATH 改成服务器上合并后模型的真实目录。" >&2
  exit 2
fi

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
