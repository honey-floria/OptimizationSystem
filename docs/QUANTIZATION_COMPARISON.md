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

## 当前结果与限制

- FP16、SmoothQuant INT8、AWQ 和 GPTQ 已在同一 A100-SXM4-40GB 上运行统一 16 点
  Benchmark，显存、延迟和吞吐变化均可直接比较。
- AWQ 吞吐相对 FP16 下降约 `41.8%`，GPTQ 下降约 `28.0%`～`28.9%`；INT4 模型显存
  变化仅约 `-0.86%`～`+0.30%`，没有形成有效显存或性能收益。
- SmoothQuant INT8 模型显存增加约 `5.83%`，平均 P50 延迟增加约 `299.8%`，平均吞吐
  下降约 `74.8%`。该 ModelOpt 状态更接近量化仿真/校准模型，不能作为部署加速证据。
- A100 不支持当前 FP8 实验，FP8 标记为 `unsupported_on_target_a100`，质量和性能记录为
  `N/A`，不阻塞适用方案的完整性检查。
- `configs/quantization_comparison.json` 中 GPU 小时单价尚未批准，因此只计算每百万输出
  token 的 GPU 小时，不生成美元成本。这是 6.5 唯一剩余项。

补跑可直接运行 `notebooks/quantization_comparison_benchmark_colab.ipynb`。Notebook 会锁定
A100-SXM4-40GB、校验并复用已有六份 AWQ/GPTQ 报告，只重新运行 FP16 和 INT8，然后生成：

- `out/quantization_comparison/fp16_a100_benchmark_report.json`
- `out/int8_fp8/benchmark_report.json`
- 更新后的 `out/quantization_comparison/comparison_report.json`
- 更新后的 `out/quantization_comparison/validation.json`
