# 硬件环境输入校验

本流程对应 `docs/TODO.md` 4.3。NVIDIA 与昇腾环境必须分别在真实目标机器采集，不能使用
模拟信息代替。环境 manifest 保存在上游资产目录，合并验收报告保存在项目 `out/`。

## NVIDIA 采集

```bash
python -m src.input_validation.hardware_input collect \
  --target nvidia \
  --output /path/to/hardware/nvidia.json
```

## 昇腾采集

```bash
python -m src.input_validation.hardware_input collect \
  --target ascend \
  --output /path/to/hardware/ascend.json
```

自动采集无法识别的昇腾芯片、显存、CANN、驱动、固件或通信信息，需要根据 `npu-smi`、
CANN 安装信息和部署环境补入 manifest，不能猜测版本。

## 合并验收

```bash
python -m src.input_validation.hardware_input validate \
  --nvidia-manifest /path/to/hardware/nvidia.json \
  --ascend-manifest /path/to/hardware/ascend.json \
  --report out/hardware/validation.json
```

只有七项全部为 `pass` 且 `complete` 为 `true`，才能勾选 Todo 4.3。单独完成 Colab
NVIDIA 采集不会完成 4.3，昇腾环境仍是必要门禁。
