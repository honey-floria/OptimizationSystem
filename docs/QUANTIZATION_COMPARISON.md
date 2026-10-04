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

- FP16 Benchmark 在 Tesla T4 上运行，AWQ/GPTQ 在 A100 上运行，且矩阵分别为 12 和
  16 个测试点，因此不能计算相对显存、延迟和吞吐收益。
- SmoothQuant INT8 尚未运行统一 16 点 Benchmark。
- A100 不支持当前 FP8 实验，FP8 标记为 `unsupported_on_target_a100`，质量和性能记录为
  `N/A`，不阻塞适用方案的完整性检查。
- `configs/quantization_comparison.json` 中 GPU 小时单价尚未批准，因此只计算每百万输出
  token 的 GPU 小时，不生成美元成本。

完成剩余 6.5 验收前，需要在同一 A100 上使用 `configs/quantization_benchmark.json` 重跑
FP16 和 INT8，并填写经项目确认的 GPU 小时单价。FP8 如果不属于目标 A100 的候选方案，需
由项目负责人明确批准以“不适用”关闭，而不能伪造实验结果。

同硬件 Benchmark 可直接运行 `notebooks/quantization_comparison_benchmark_colab.ipynb`。
Notebook 会依次生成：

- `out/quantization_comparison/fp16_a100_benchmark_report.json`
- `out/int8_fp8/benchmark_report.json`
- 更新后的 `out/quantization_comparison/comparison_report.json`
- 更新后的 `out/quantization_comparison/validation.json`
