# INT8/FP8 实验（6.4）

6.4 统一记录 SmoothQuant W8A8 配置、校准集、导出契约和硬件能力。Notebook 会检测
SmoothQuant 后端、PyTorch FP8 dtype、CUDA GPU 和昇腾 NPU；未安装后端或未连接对应设备时
标记 `not_run`，不伪造 INT8 权重或 Ascend 证据。

运行 `notebooks/int8_fp8_experiment_colab.ipynb`。模型仍从
`banking_llm_project/model_input_real` 读取；完整 INT8 模型写入
`banking_llm_project/int8_fp8`，分析证据写入 `OptimizationSystem/out/int8_fp8`。
