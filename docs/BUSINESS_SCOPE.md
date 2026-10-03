# 多业务场景范围

场景组合配置位于 `configs/business_scope.json`，按照 README 风险分级同时维护：

| 场景 | 风险 | 状态 | 压缩策略 |
|---|---|---|---|
| 内部金融 FAQ | L1 | `planned` | 可尝试 INT4 和较高剪枝率 |
| 内部财报数值问答 | L2 | `active` | INT4/INT8 混合精度，谨慎剪枝 |
| 授信政策与监管制度辅助问答 | L3 | `planned` | 优先 FP16/INT8，不做激进剪枝 |

当前只有 L2 财报数值问答具备 FinQA 评测资产，因此它是主场景。L1/L3 已完成范围规划，
但在各自评测资产通过输入验收前不得标记为 `active`。

核心约束：

- 每个场景独立配置允许输出、人工复核、质量门槛和排除项；
- 风险越高，允许的量化和剪枝策略越保守；
- 所有场景均要求输入依据，高风险回归不允许新增关键错误；
- L2 数值准确率相对 FP16 下降不超过 0.5 个百分点，解析率不低于 99%；
- L3 不允许 INT4 或激进剪枝，所有决策相关输出必须人工复核。

执行验收：

```bash
python -m src.input_validation.business_scope \
  --config configs/business_scope.json \
  --report out/business_scope/validation.json
```

配置发生业务边界、风险等级或门槛变化时，必须重新确认并生成报告。
