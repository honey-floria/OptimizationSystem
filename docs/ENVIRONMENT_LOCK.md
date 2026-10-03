# 环境锁定

环境锁定对应 Todo 5.4。NVIDIA 与昇腾必须分别以真实机器证据生成锁定文件，不能把待填写模板
当成已验收环境。

当前 NVIDIA 基线锁定范围包括：Python、PyTorch、Transformers、Accelerate、CUDA、cuDNN、
GPU、显存、驱动、Colab 容器版本、完整 `pip freeze` 及其 SHA-256。

Colab 结果目录：

```text
out/environment_lock/
├── nvidia_environment.json
├── ascend_environment.pending.json
├── requirements.freeze.txt
└── validation.json
```

当前阶段预期五项为 `pass`，以下三项保持 `not_run`：

- 真实昇腾环境文件；
- CANN 版本锁定；
- 成功构建且带镜像 digest 的基础容器证据。

在真实昇腾机器完成采集后，用实际锁定文件替换 `ascend_environment.pending.json`。容器构建
必须保存镜像名、不可变 digest 和 `build_status: passed`，不能仅以 Dockerfile 代替构建证据。
