# 量化框架接入（6.1）

6.1 建立 AWQ、GPTQ 和 SmoothQuant 的统一配置、导出和日志契约。它不等于已经完成真实量化；
真实权重导出和业务评测分别在 6.2～6.4 完成。

统一配置位于 `configs/quantization.json`，每个方案声明：

- `method`、权重量化 bit、激活量化 bit 和 group size；
- 校准资产角色及其版本/哈希；
- 后端包名和输出目录。

每个真实方案输出目录必须包含：

```text
quantization_config.json
quantized_model_manifest.json
quantization_events.jsonl
```

当前 Colab 单元只采集后端包版本和配置契约。没有安装后端包或没有导出真实权重时，报告会
明确标记 `real_quantization_execution: not_run`，不会伪造量化完成状态。

GPTQ 实验使用仍在活跃维护的 `gptqmodel`；`auto-gptq` 仅作为旧环境兼容版本继续采集。
