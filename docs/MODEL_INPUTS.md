# 模型输入校验

## 1. 目标

本工具对应 `docs/TODO.md` 4.1，只验证上游交付的模型资产，不下载、修改或重新打包模型。
校验报告逐项覆盖权重、配置与 Tokenizer、许可证与内部使用范围、版本与哈希、半精度加载、
目标推理框架冒烟推理。

## 2. 准备 Manifest

复制 `inputs/model_manifest.example.json` 为实际 manifest，并填写真实模型信息。所有相对路径都
相对于 manifest 所在目录解析。`weight_files` 必须列出所有权重分片；许可证文件应放在模型目录
内或使用相对于模型目录的路径。

`expected_sha256` 可留空，此时工具会计算并记录每个权重文件的 SHA-256。若上游提供了哈希，
按权重相对路径填写，工具会验证是否一致：

```json
{
  "expected_sha256": {
    "model.safetensors": "真实的 SHA-256"
  }
}
```

只有许可证负责人已经确认后，才能把 `internal_use_approved` 设置为 `true`，并在
`usage_scope` 中写明获批范围。

## 3. 静态检查

```bash
python -m src.input_validation.model_input \
  --manifest inputs/model_manifest.json \
  --report reports/model_input/static.json
```

静态检查不会加载模型，加载和推理两项显示为 `not_run`。命令退出码为 `0` 表示已执行的检查
没有失败；`complete` 仍为 `false`，直到运行时检查通过。

## 4. 加载与推理检查

先在目标硬件环境安装与模型匹配的 PyTorch、Transformers 和厂商运行时，然后执行：

```bash
python -m src.input_validation.model_input \
  --manifest inputs/model_manifest.json \
  --runtime-check \
  --report reports/model_input/runtime.json
```

运行时检查只读取本地模型，并生成一个 token。`runtime.dtype` 仅接受 `float16` 或
`bfloat16`；`runtime.model_class` 支持 `causal_lm` 和 `seq2seq_lm`。只有报告中六项均为
`pass` 且 `complete` 为 `true`，才能勾选 Todo 4.1。
