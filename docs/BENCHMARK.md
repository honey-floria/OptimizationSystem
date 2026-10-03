# 统一 FP16/BF16 Benchmark

Benchmark 配置位于 `configs/benchmark.json`，对 batch size、输入 token 长度和输出 token
长度执行笛卡尔积测试。每个测试点先预热，再采集多次样本。

报告包含：

- TTFT、单输出 token 延迟和端到端延迟；
- tokens/s、requests/s 和 P50/P95/P99；
- 模型显存、峰值显存、增量峰值显存、实际 KV Cache 张量显存及理论估算值；
- 模型版本、FP16/BF16 配置、GPU、CUDA、PyTorch 和 Transformers 版本。

统一命令：

```bash
python -m src.benchmark.runner run \
  --baseline-config configs/baseline.json \
  --benchmark-config configs/benchmark.json \
  --model-manifest /path/to/model_manifest.json \
  --prompt-file /path/to/prompt.txt \
  --output-dir /path/to/out/benchmark
```

Colab 验收结果目录：

```text
out/benchmark/
├── baseline_fp16_report.json
├── benchmark_config.json
├── benchmark_prompt.txt
└── validation.json
```

只有 `validation.json` 中十一项均为 `pass`，才能勾选 Todo 5.3。
