# 第 7 章阶段验收报告

状态：`pass`；选择的部署计划：`all_linear_int4`。

## 验收项

| 验收项 | 状态 | 说明 |
|---|---|---|
| `layer_sensitivity_ranking_generated` | pass | 28/28 pilot layers ranked; 3 pre-registered layers completed full-dev confirmation. |
| `module_sensitivity_ranking_generated` | pass | Final ranking contains 7 recovery and 4 special-module candidates. |
| `financial_business_degradation_localized` | pass | Localized negative business-category transitions for: mixed_l10_mlp_lm_head_int8. |
| `deployable_mixed_precision_configuration_generated` | pass | Selected plan all_linear_int4 combines low-bit Linear weights with BF16 special modules and passed final gates. |
| `mixed_precision_business_regression_and_benchmark_completed` | pass | Selected plan has 883 full-dev, 91 high-risk, 89 long-context results and a complete repeated 16-case benchmark. |

## 交付物

- [x] `sensitive_layer_ranking`
- [x] `business_error_classification_report`
- [x] `mixed_precision_configuration`
- [x] `degradation_localization_script`
- [x] `degradation_analysis_report`

## 结论边界

- Layer full-dev ranking covers the pre-registered candidate subset; the 28-layer ranking is pilot evidence.
- Module ranking covers pre-registered promoted candidates, not every module on full-dev.
- The selected all_linear_int4 plan is mixed precision at model scope: Linear INT4 with Embedding, Norm, and KV Cache BF16.
- Financial localization is complete for the active L2 FinQA scenario; L1/L3 and annotation-dependent safety categories remain not_evaluable.
