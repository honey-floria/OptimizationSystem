# 量化结果整理（6.5）

运行以下命令生成跨方案对比表、机器可读报告和验收结果：

```bash
python -m src.quantization.comparison --project-root .
```

输出文件：

- `out/quantization_comparison/comparison_report.json`
- `out/quantization_comparison/comparison_report.md`
- `out/quantization_comparison/validation.json`

报告统一整理 FP16、AWQ INT4、GPTQ INT4、SmoothQuant INT8 和 FP8，记录数值准确率、
解析率、显存、P50 延迟、吞吐、每百万输出 token 的 GPU 小时和 L2/L3 风险标记。缺少真实
实验的字段保持 `not_run` 或 `N/A`，不会以计划配置代替结果。

## 当前限制

- FP16 和 SmoothQuant INT8 已在同一 A100-SXM4-80GB 上运行统一 16 点 Benchmark。
- AWQ/GPTQ 历史结果来自 A100-SXM4-40GB。40GB 与 80GB 型号的显存带宽不同，因此仍不能
  将 INT4 性能直接与当前 FP16 基线比较。
- A100 不支持当前 FP8 实验，FP8 标记为 `unsupported_on_target_a100`，质量和性能记录为
  `N/A`，不阻塞适用方案的完整性检查。
- `configs/quantization_comparison.json` 中 GPU 小时单价尚未批准，因此只计算每百万输出
  token 的 GPU 小时，不生成美元成本。

同硬件有效结果显示，当前 SmoothQuant INT8 相对 FP16：模型显存增加约 5.5%，平均 P50
延迟增加约 296.5%，平均吞吐下降约 74.4%，没有产生性能收益。该 ModelOpt 状态更接近
量化仿真/校准模型，不能作为部署加速收益证据。

完成剩余 6.5 验收前，需要将 AWQ/GPTQ 重跑到 A100 80GB，或将 FP16/INT8 重跑到 A100
40GB，并填写经项目确认的 GPU 小时单价。

同硬件 Benchmark 可直接运行 `notebooks/quantization_comparison_benchmark_colab.ipynb`。
Notebook 会依次生成：

- `out/quantization_comparison/fp16_a100_benchmark_report.json`
- `out/int8_fp8/benchmark_report.json`
- 更新后的 `out/quantization_comparison/comparison_report.json`
- 更新后的 `out/quantization_comparison/validation.json`
