# 7.3 金融错误定位报告

状态：`completed_with_declared_coverage_gaps`；基线：`all_linear_int4`；完整 dev 样本：883。

## 当前可评测业务场景

| 场景 | 状态 | 样本数 | 说明 |
|---|---|---:|---|
| `internal_financial_faq` | not_evaluable | 0 | No dataset/prediction binding is available for this planned scenario. |
| `financial_report_numeric_qa` | evaluated | 883 | Frozen full-dev predictions available. |
| `credit_policy_compliance_qa` | not_evaluable | 0 | No dataset/prediction binding is available for this planned scenario. |

## 错误统计

金额、利率和日期/期限允许重叠，不能横向相加。引用错误是结构代理：只验证 evidence 非空且引用形态合法，不能证明语义引用正确。

| 计划 | 金额 | 利率/比例 | 日期/期限 | JSON | 引用结构 | 条款遗漏 | 不当建议 | 应拒答误答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_linear_int4` | 202/338 | 369/567 | 429/709 | 6/883 | 6/883 | N/A | N/A | N/A |
| `all_linear_int8` | 204/338 | 367/567 | 435/709 | 1/883 | 1/883 | N/A | N/A | N/A |
| `mixed_l10_mlp_lm_head_int8` | 207/338 | 371/567 | 435/709 | 4/883 | 4/883 | N/A | N/A | N/A |

## 相对基线的退化

| 计划 | 恢复 | 退化 | 净正确数 | 金额净变化 | 利率/比例净变化 | 日期/期限净变化 |
|---|---:|---:|---:|---:|---:|---:|
| `all_linear_int8` | 41 | 37 | +4 | -2 | +2 | -6 |
| `mixed_l10_mlp_lm_head_int8` | 6 | 11 | -5 | -5 | -2 | -6 |

## 覆盖边界

当前冻结资产只覆盖 L2 内部财报数值问答。条款遗漏、不当金融建议和应拒答误答缺少带对应标注的样本，统计状态为 `not_evaluable`，不是零错误。引用统计仅为结构代理。L1/L3 场景须在数据接入后沿用同一接口重新生成报告。
