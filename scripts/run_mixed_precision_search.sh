#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
DEVICE="${DEVICE:-cuda:0}"
PROMPT_FILE="${PROMPT_FILE:-inputs/mixed_precision_benchmark_prompt.txt}"
OUTPUT_ROOT="${MIXED_PRECISION_OUTPUT_ROOT:-out/mixed_precision_search}"
export MIXED_PRECISION_OUTPUT_ROOT="${OUTPUT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${MODEL_PATH}" && ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: MODEL_PATH 不存在或缺少 config.json: ${MODEL_PATH}" >&2
  echo "请把 MODEL_PATH 改成服务器上合并后模型的真实目录。" >&2
  exit 2
fi

plans=(
  all_linear_int4
  all_linear_int8
  mixed_l10_mlp_lm_head_int8
)

python3 scripts/run_mixed_precision_search.py prepare

for plan_id in "${plans[@]}"; do
  export_manifest="${OUTPUT_ROOT}/exports/${plan_id}/quantized_model_manifest.json"
  if [[ ! -f "${export_manifest}" ]]; then
    args=(--plan-id "${plan_id}" --device "${DEVICE}")
    if [[ -n "${MODEL_PATH}" ]]; then
      args+=(--model-path "${MODEL_PATH}")
    fi
    python3 scripts/run_mixed_precision_search.py export "${args[@]}"
  fi

  for suite in full_dev high_risk long_context; do
    evidence="${OUTPUT_ROOT}/evidence/${plan_id}/${suite}.json"
    if [[ ! -f "${evidence}" ]]; then
      python3 scripts/run_mixed_precision_search.py evaluate \
        --plan-id "${plan_id}" --suite "${suite}" --device "${DEVICE}"
    fi
  done

  benchmark_evidence="${OUTPUT_ROOT}/evidence/${plan_id}/benchmark.json"
  if [[ ! -f "${benchmark_evidence}" ]]; then
    python3 scripts/run_mixed_precision_search.py benchmark \
      --plan-id "${plan_id}" --prompt-file "${PROMPT_FILE}" --device "${DEVICE}"
  fi
done

python3 scripts/run_mixed_precision_search.py pending
python3 scripts/run_mixed_precision_search.py finalize
