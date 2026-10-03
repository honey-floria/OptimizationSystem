# AWQ 实验（6.2）

6.2 使用与 FP16 基线隔离的 `calibration_finqa_1024`，在 NVIDIA/T4 上运行
AutoAWQ INT4 W4A16。实验矩阵固定为 group size 32、64、128；每个矩阵点单独导出，避免
覆盖结果。

配置位于 `configs/awq.json`，真实执行入口为
`src.quantization.awq.run_awq_plan`。每个计划的结果目录必须包含：

```text
quantization_config.json
quantized_model_manifest.json
quantization_events.jsonl
```

Colab 入口见 `notebooks/awq_experiment_colab.ipynb`。模型仍从
`/content/drive/MyDrive/banking_llm_project/model_input_real` 读取；量化模型和证据只写入
`/content/drive/MyDrive/OptimizationSystem/out/awq`。

AutoAWQ 未安装或某个矩阵点未执行时，验收报告会标记 `not_run`，不会把计划配置误报为真实
量化权重。质量、回归和性能数据需要在真实权重导出后继续运行，不能由配置验收替代。
