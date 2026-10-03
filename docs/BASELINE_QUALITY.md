# FP16 业务质量基线

质量基线使用版本化的 FinQA development 集，共 883 条样本。该集合只用于开发阶段的 FP16
基线和后续压缩方案对比，不使用最终 test 集。

Colab 会以固定 FP16 模型、统一 Prompt 和生成参数进行批量推理，并输出：

```text
out/baseline_quality/
├── baseline_predictions.jsonl
├── baseline_quality_report.json
├── quality_service.jsonl
└── validation.json
```

主指标为 `numeric_accuracy`，同时要求 `parse_rate >= 0.99`。FP16 准确率作为后续 INT8、
INT4 和剪枝方案的相对质量基准；当前阶段不为 FP16 准确率虚构额外绝对门槛。

质量配置可包含独立的 `prompt_suffix` 和 `generation` 覆盖项，用于诊断输出截断或格式问题。
这类实验必须写入新的结果目录，不能覆盖原始 FP16 基线结果。
