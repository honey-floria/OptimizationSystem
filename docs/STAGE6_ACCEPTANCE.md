# 第 6 章定制量化阶段验收

运行以下命令汇总 AWQ、GPTQ、INT8 的导出、质量、回归和统一 Benchmark 证据：

```bash
python -m src.quantization.stage6_acceptance --project-root .
```

验收结果写入 `out/stage6/validation.json`。其中“至少一种量化方案可稳定部署”按技术稳定性
验收，即量化模型能够加载并完成 16 点、每点 5 次测量的统一 Benchmark。它不表示模型已经
满足金融生产质量要求；生产结论以 `docs/QUANTIZATION_RECOMMENDATION.md` 为准，目前为
“暂不部署”。

量化模型权重保存在实验 Google Drive 路径，仓库保存模型清单、量化配置、事件日志和真实
运行证据。6.5 的美元单位 token 成本仍等待 A100 40GB 小时单价，不影响技术阶段证据汇总。
