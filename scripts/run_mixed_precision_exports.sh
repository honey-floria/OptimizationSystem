#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
DEVICE="${DEVICE:-cuda:0}"
export MIXED_PRECISION_OUTPUT_ROOT="${MIXED_PRECISION_OUTPUT_ROOT:-out/mixed_precision_search}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${MODEL_PATH}" && ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: MODEL_PATH 不存在或缺少 config.json: ${MODEL_PATH}" >&2
  echo "请把 MODEL_PATH 改成服务器上合并后模型的真实目录。" >&2
  exit 2
fi

python3 scripts/run_mixed_precision_search.py prepare

# 核心闭环：INT4 部署基线、INT8 质量上界、主混合精度候选。
# 已存在 manifest 的成功导出会被跳过，因此可安全断点续跑。
for plan_id in \
  all_linear_int4 \
  all_linear_int8 \
  mixed_l10_mlp_lm_head_int8
do
  export_manifest="${MIXED_PRECISION_OUTPUT_ROOT}/exports/${plan_id}/quantized_model_manifest.json"
  if [[ -s "${export_manifest}" ]]; then
    echo "SKIP: ${plan_id} 已完成导出 (${export_manifest})"
    continue
  fi
  args=(--plan-id "${plan_id}" --device "${DEVICE}")
  if [[ -n "${MODEL_PATH}" ]]; then
    args+=(--model-path "${MODEL_PATH}")
  fi
  python3 scripts/run_mixed_precision_search.py export "${args[@]}"
done

python3 scripts/run_mixed_precision_search.py pending
