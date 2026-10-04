# 统一量化实验 Notebook

推荐使用 `notebooks/quantization_experiments_colab.ipynb` 统一执行 AWQ、GPTQ 和
SmoothQuant/FP8 的导出与环境验收。

质量、回归和 Benchmark 仍建议在各自评测 Notebook 中执行，因为这些步骤耗时较长，便于
单独重跑和排查。统一 Notebook 会自动跳过已有 manifest 的 AWQ/GPTQ 计划，避免重复量化。

模型权重目录：

```text
/content/drive/MyDrive/banking_llm_project/awq
/content/drive/MyDrive/banking_llm_project/gptq
/content/drive/MyDrive/banking_llm_project/int8_fp8
```

分析结果目录：

```text
/content/drive/MyDrive/OptimizationSystem/out/awq
/content/drive/MyDrive/OptimizationSystem/out/gptq
/content/drive/MyDrive/OptimizationSystem/out/int8_fp8
```
