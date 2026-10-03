# FP16/BF16 基线服务

基线服务配置位于 `configs/baseline.json`，当前绑定激活的 L2 财报数值问答场景。服务使用
统一 FinQA Prompt、固定生成参数和随机种子，支持单请求与批量推理，并将每个请求的输入/
输出 token、延迟、吞吐、模型 commit、精度和设备写入 JSONL。

Colab Notebook 会在真实 Qwen2.5-0.5B-Instruct 和 T4 上执行：

1. FP16 模型加载；
2. 单请求推理；
3. 两条样本批量推理；
4. 相同请求确定性重放；
5. 模型与运行环境登记；
6. 七项基线服务验收。

结果目录：

```text
out/baseline/
├── baseline_config.json
├── baseline_evidence.json
├── request_logs.jsonl
└── validation.json
```

只有 `validation.json` 中七项均为 `pass`，才能勾选 Todo 5.2。
