# 6.4 质量评测

运行 `notebooks/int8_fp8_evaluation_colab.ipynb` 可对真实导出的 INT8 模型执行：

- FinQA 数字字段准确率和解析率；
- 128/512/2048 token 长上下文响应检查；
- 固定字段的 JSON 结构化输出检查；
- NVIDIA FP8 和 Ascend NPU 能力记录。

模型文件从 `banking_llm_project/int8_fp8` 读取，评测证据写入
`OptimizationSystem/out/int8_fp8`。没有 `quantized_model_manifest.json` 时，Notebook 只生成
`not_run` 说明，不会把 FP16 结果当成 INT8 结果。
