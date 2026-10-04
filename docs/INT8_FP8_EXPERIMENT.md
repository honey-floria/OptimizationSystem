# INT8/FP8 实验（6.4）

6.4 使用 NVIDIA ModelOpt 执行 SmoothQuant W8A8，统一记录配置、校准集、导出契约和硬件
能力。Notebook 不会把“后端已安装”误报成“INT8 权重已导出”，也不会把“PyTorch 存在
FP8 dtype”误报成“设备支持 FP8”。FP8 只有在对应设备完成真实 FP8 矩阵运算后才记为
`pass`；未安装后端或未连接对应设备时标记 `not_run`。

运行 `notebooks/int8_fp8_experiment_colab.ipynb`。模型仍从
`banking_llm_project/model_input_real` 读取；完整 INT8 模型写入
`banking_llm_project/int8_fp8`，分析证据写入 `OptimizationSystem/out/int8_fp8`。只有目录中出现
`smoothquant-int8-w8a8/quantized_model_manifest.json` 后，才可以运行 INT8 质量评测 Notebook。

## 执行顺序

1. 在 NVIDIA GPU Colab 中运行 `notebooks/int8_fp8_experiment_colab.ipynb`。
2. 检查 `out/int8_fp8/validation.json` 中 `smoothquant_real_execution` 为 `pass`。
3. 运行 `notebooks/int8_fp8_evaluation_colab.ipynb`。
4. 检查 `out/int8_fp8/evaluation_validation.json` 的三项检查全部为 `pass`。

实验 Notebook 会保存可由 `modelopt.torch.opt.restore` 恢复的 `modelopt_state.pth`、统一量化
配置、manifest 和事件日志。评测 Notebook 使用相同 FinQA dev 样本对比 FP16 与 INT8 数值
准确率，并对最长输入样本和 JSON 输出样本分别执行 FP16/INT8 对照。

## 验收口径

- `运行 SmoothQuant INT8`：必须存在真实 `modelopt_state.pth`，且配置、manifest、事件日志齐全。
- `在 NVIDIA 上验证 FP8 可用性`：必须成功执行真实 FP8 矩阵运算；A100 仅暴露 dtype 不算通过。
- `确认 Ascend 对应数据类型支持情况`：必须在真实 NPU 上执行探测，不接受静态推断。
- 三项质量对比：必须生成 `evaluation.json` 和完整通过的 `evaluation_validation.json`。
