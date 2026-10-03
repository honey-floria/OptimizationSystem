# 评测资产输入校验

本流程对应 `docs/TODO.md` 4.2。`prepare_assets()` 生成的 `manifest.json` 会记录校准集、
开发评估集、最终评估集和高风险回归集的相对路径、版本、样本数、必需字段与 SHA-256，
并声明数值指标接口及关键错误判定规则。

执行校验：

```bash
python -m src.input_validation.dataset_input \
  --manifest /path/to/datasets/manifest.json \
  --report out/dataset_input/validation.json
```

校验器会实际加载四类资产，并检查：

- 路径和版本是否完整；
- 校准集与评估集是否无问题重叠；
- 高风险回归集是否已从最终测试集移除；
- 每条样本是否符合声明字段；
- 指标和回归判定 callable 是否可导入；
- 样本数量和内容哈希是否与 manifest 一致。

只有报告中九项全部为 `pass` 且 `complete` 为 `true`，才能勾选 Todo 4.2。
