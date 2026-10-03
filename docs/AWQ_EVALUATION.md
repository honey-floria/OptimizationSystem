# AWQ 质量、回归和性能评估

导出验收通过后，运行 `notebooks/awq_quality_benchmark_colab.ipynb`。该 Notebook：

1. 使用与 FP16 基线相同的 FinQA development 集，分别评估 group size 32、64、128；
2. 将候选模型逐样本预测与 FP16 预测比较，统计新增数值回归和关键错误；
3. 使用现有统一 Benchmark 矩阵记录 TTFT、单 token 延迟、吞吐和显存。

完整 AWQ 权重从 `banking_llm_project/awq` 加载，评测报告、预测和 JSONL 日志写入
`OptimizationSystem/out/awq/<plan_id>`。6.2 的质量/回归/性能 TODO 只有在这些真实报告回传并检查后才勾选。
