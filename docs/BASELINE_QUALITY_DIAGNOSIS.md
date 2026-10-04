# FP16 基线质量诊断（6.7.1）

运行以下命令分析冻结的 883 条 FP16 FinQA dev 预测：

```bash
python -m src.evaluation.baseline_quality_diagnosis --project-root .
```

输出文件：

- `out/baseline_quality_diagnosis/diagnosis_report.json`
- `out/baseline_quality_diagnosis/diagnosis_report.md`
- `out/baseline_quality_diagnosis/manual_review_50.jsonl`
- `out/baseline_quality_diagnosis/manual_review_50.json`（普通 JSON 数组，适合 JSON 解析器）
- `out/baseline_quality_diagnosis/validation.json`

工具会验证基础数字归一化、按题型统计准确率，并使用确定性规则将错误分为输出解析、答案
抽取潜在误判、计算、公式选择、表格或上下文选择错误。自动分类只用于建立人工复核队列；
需要在 `manual_review_50.jsonl` 的 `manual_review` 字段中填写最终判断，不能把规则建议直接
视为人工结论。当前 50 条队列已完成复核。完成一条复核时，将 `status` 改为 `completed`，填写布尔值
`prediction_correct`、最终 `error_type` 和可选 `notes`。再次运行命令会保留这些字段并更新
完成数量，不会覆盖已经填写的人工结论。

`.jsonl` 是“一行一个 JSON 对象”的机器格式；如果使用要求整个文件只能是一个 JSON 值的
解析器，请打开 `manual_review_50.json`，它是一个标准 JSON 数组。
