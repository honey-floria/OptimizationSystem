# GPTQ 实验（6.3）

6.3 使用与 AWQ 完全相同的 `calibration_finqa_1024`、INT4 W4A16 和 group size
32/64/128。后端使用 Transformers `GPTQConfig` 与 GPTQModel，避免依赖停止活跃维护的
AutoGPTQ 接口。

完整模型写入 `/content/drive/MyDrive/banking_llm_project/gptq`；配置、manifest、事件日志、
质量、回归和 Benchmark 报告写入 `/content/drive/MyDrive/OptimizationSystem/out/gptq`。

先运行 `notebooks/gptq_experiment_colab.ipynb` 完成真实量化导出，再运行
`notebooks/gptq_quality_benchmark_colab.ipynb`。GPTQ 和 AWQ 的真实速度、显存及并发对比必须
使用同一种 GPU；不同硬件的报告只能各自留档，不能计算直接收益。

GPTQModel 7.5.0 安装时可能把 Transformers 降级到 5.17.0，导致
`AutoModelForCausalLM` 延迟导入失败。Notebook 会在安装 GPTQModel 后重新锁定项目已验证的
Transformers 5.18.0；安装单元结束后必须重启 Colab 运行时。

如果第一次运行已生成 `not_run` 日志，直接重新运行 `notebooks/gptq_experiment_colab.ipynb`。
它会在导入检查失败时立即显示完整根因，并自动跳过已经成功导出的 group size。
