# GPTQ 实验（6.3）

6.3 使用与 AWQ 完全相同的 `calibration_finqa_1024`、INT4 W4A16 和 group size
32/64/128。后端使用 Transformers `GPTQConfig` 与 GPTQModel，避免依赖停止活跃维护的
AutoGPTQ 接口。

完整模型写入 `/content/drive/MyDrive/banking_llm_project/gptq`；配置、manifest、事件日志、
质量、回归和 Benchmark 报告写入 `/content/drive/MyDrive/OptimizationSystem/out/gptq`。

先运行 `notebooks/gptq_experiment_colab.ipynb` 完成真实量化导出，再运行
`notebooks/gptq_quality_benchmark_colab.ipynb`。GPTQ 和 AWQ 的真实速度、显存及并发对比必须
使用同一种 GPU；不同硬件的报告只能各自留档，不能计算直接收益。
