# FP16 基线质量诊断（TODO 6.7.1）

来源：`out/baseline_quality_experiment/baseline_quality_report.json`，共 883 条。

## 自动诊断结论

- `answer_extraction_false_negative_candidate`：32
- `calculation_error`：241
- `correct`：8
- `formula_selection_error`：261
- `table_or_context_selection_error`：341

## 题型统计

| 题型 | 样本 | 正确 | 准确率 |
|---|---:|---:|---:|
| average | 74 | 0 | 0.0000 |
| change_or_difference | 126 | 2 | 0.0159 |
| lookup_or_other | 154 | 5 | 0.0325 |
| percentage_or_ratio | 433 | 1 | 0.0023 |
| rate_or_unit | 5 | 0 | 0.0000 |
| total_or_sum | 91 | 0 | 0.0000 |

## 6.7.1 指定分析维度

| 维度 | 样本 | 正确 | 准确率 |
|---|---:|---:|---:|
| single_step | 491 | 3 | 0.0061 |
| multi_step | 255 | 0 | 0.0000 |
| comparison | 46 | 0 | 0.0000 |
| percentage | 433 | 1 | 0.0023 |
| time_series | 337 | 2 | 0.0059 |

## 重要风险

当前评分器取输出中的第一个数字；有 32 条样本在后续数字中出现标准答案，属于潜在误判，必须人工复核。

自动错误分类仅用于分流，不替代人工结论。请在 `manual_review_50.jsonl` 中填写 `manual_review` 字段。
