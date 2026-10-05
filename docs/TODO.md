# 银行金融大模型优化项目 Todo

## 1. 文档说明

本 Todo 文档从项目方案中独立拆出，覆盖 16 周的模型量化、剪枝、质量退化定位、NVIDIA GPU/华为昇腾适配和工程化交付工作。

本项目不负责：

- 原始业务数据采集
- 数据清洗、脱敏和标注
- 知识库构建和向量化
- 业务规则制定
- 金融业务评估集生产

以上内容由数据、业务和评测团队提供。本项目负责校验输入资产是否满足接口要求，并使用这些资产进行模型优化和回归评估。

除本 Todo 外的实施文档已按项目执行顺序合并为
[`docs/PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)；原始专题文档保留用于单独引用。

## 2. 完成定义

项目整体完成需要满足：

- [ ] 至少一个银行金融业务场景完成端到端优化
- [ ] 建立 FP16/BF16 基线并锁定环境
- [ ] 完成至少两种量化方案对比
- [ ] 完成敏感层分析和混合精度配置
- [ ] 完成结构化剪枝实验
- [ ] 完成 NVIDIA GPU 部署和性能测试
- [ ] 完成华为昇腾部署和性能测试
- [ ] 能定位金额、利率、日期、条款等业务错误
- [ ] 输出质量、性能、显存和成本对比报告
- [ ] 支持模型和配置版本管理
- [ ] 支持失败版本回滚
- [ ] 提供可复现的运行命令和部署文档

## 3. 角色与职责

| 角色 | 主要职责 |
|---|---|
| 项目负责人 | 范围、进度、风险和最终交付 |
| 模型优化负责人 | 量化、剪枝、混合精度和退化分析 |
| NVIDIA 负责人 | CUDA、vLLM、TensorRT-LLM 和 GPU 压测 |
| 昇腾负责人 | CANN、MindIE/ATB 和昇腾部署适配 |
| 评测接口人 | 提供指标定义、评估集和回归结果解释 |
| 业务接口人 | 确认金融场景、风险等级和关键错误 |
| 基础设施接口人 | 提供 GPU/昇腾环境和资源配额 |

## 4. 上游输入检查

以下工作只检查输入资产，不生产数据。

### 4.1 模型输入

校验工具使用方式见 [`docs/MODEL_INPUTS.md`](MODEL_INPUTS.md)。真实模型验收使用
`Qwen/Qwen2.5-0.5B-Instruct`，验收依据为
[`out/model_input_real/model_manifest.json`](../out/model_input_real/model_manifest.json) 和
[`out/model_input_real/reports/runtime.json`](../out/model_input_real/reports/runtime.json)。

- [x] 收到基础模型权重
- [x] 收到模型配置和 Tokenizer
- [x] 确认模型许可证和内部使用范围
- [x] 记录模型版本、提交号和权重哈希
- [x] 验证 FP16/BF16 模型能够正常加载
- [x] 验证基础模型在目标推理框架中可以运行

### 4.2 评测资产输入

校验工具使用方式见 [`docs/DATASET_INPUTS.md`](DATASET_INPUTS.md)。验收依据为
[`out/dataset_input/dataset_manifest.json`](../out/dataset_input/dataset_manifest.json) 和
[`out/dataset_input/validation.json`](../out/dataset_input/validation.json)，九项自动检查均已通过。

- [x] 收到 `dataset_manifest.json` 或等价清单
- [x] 收到校准集路径和版本
- [x] 收到评估集路径和版本
- [x] 收到回归集路径和版本
- [x] 确认校准集与评估集相互隔离
- [x] 确认样本格式和字段定义
- [x] 确认业务指标计算脚本或调用接口
- [x] 确认高风险样本和关键错误判定方式
- [x] 记录评测资产版本和校验哈希

### 4.3 硬件环境输入

采集与校验工具使用方式见 [`docs/HARDWARE_INPUTS.md`](HARDWARE_INPUTS.md)。NVIDIA Colab
环境已通过验收，依据为 [`out/hardware/nvidia_colab.json`](../out/hardware/nvidia_colab.json)
和 [`out/hardware/nvidia_validation.json`](../out/hardware/nvidia_validation.json)。昇腾及双平台
合并检查延后到真实昇腾机器执行。

- [x] 确认 NVIDIA GPU 型号、数量和显存
- [x] 确认 CUDA、驱动和容器版本
- [ ] 确认昇腾芯片型号、数量和显存
- [ ] 确认 CANN、驱动和固件版本
- [ ] 确认可用推理框架及版本
- [ ] 确认多卡通信和网络配置
- [ ] 保存环境信息到版本化配置文件

## 5. 第 1～2 周：范围与基线

### 5.1 业务范围

已按 L1/L2/L3 建立多场景组合，当前主场景为 L2 内部财报数值问答。范围配置见
[`configs/business_scope.json`](../configs/business_scope.json)，验收依据见
[`out/business_scope/validation.json`](../out/business_scope/validation.json)。

- [x] 确认首个业务场景，例如信贷政策问答或金融文档抽取
- [x] 确认业务风险等级：L1、L2 或 L3
- [x] 确认允许的模型输出范围
- [x] 确认必须人工复核的高风险输出
- [x] 确认关键质量指标和最低门槛
- [x] 确认不纳入本项目的业务功能

### 5.2 基线服务

基线服务与验收工具已实现，使用方式见 [`docs/BASELINE_SERVICE.md`](BASELINE_SERVICE.md)。
已在 Tesla T4 上完成真实 FP16 推理验收，七项检查结果见
[`out/baseline/validation.json`](../out/baseline/validation.json)。

- [x] 使用 FP16/BF16 建立基线推理服务
- [x] 统一输入模板和生成参数
- [x] 支持单请求推理
- [x] 支持批量推理
- [x] 支持固定随机种子或确定性配置
- [x] 输出请求级性能日志
- [x] 输出模型版本和运行环境信息

### 5.3 Benchmark

统一 Benchmark 与验收工具已实现，使用方式见 [`docs/BENCHMARK.md`](BENCHMARK.md)。
已在 Tesla T4 上完成真实 FP16 Benchmark，十一项检查结果见
[`out/benchmark/validation.json`](../out/benchmark/validation.json)。

- [x] 实现统一 benchmark 命令
- [x] 测量 TTFT
- [x] 测量单 token 延迟
- [x] 测量端到端延迟
- [x] 测量 tokens/s
- [x] 测量 batch throughput
- [x] 测量 P50/P95/P99 延迟
- [x] 测量显存和 KV Cache 显存
- [x] 测量不同 batch size
- [x] 测量不同输入长度和输出长度
- [x] 生成 FP16/BF16 基线报告

### 5.4 环境锁定

环境锁定与验收工具已实现，使用方式见 [`docs/ENVIRONMENT_LOCK.md`](ENVIRONMENT_LOCK.md)。
已完成 NVIDIA/T4 环境锁定，五项检查通过，结果见
[`out/environment_lock/validation.json`](../out/environment_lock/validation.json)。昇腾、CANN 和
真实容器构建按既定计划延后，当前三项保持 `not_run`。

- [x] 创建 NVIDIA 环境文件
- [ ] 创建昇腾环境文件
- [x] 固定 Python、PyTorch 和 Transformers 版本
- [ ] 固定 CUDA/CANN 版本
- [x] 固定推理框架版本
- [ ] 构建基础容器镜像
- [x] 保存 `pip freeze` 或等价依赖清单
- [x] 保存 NVIDIA 硬件和驱动信息（昇腾另行补齐）

### 5.5 阶段验收

NVIDIA 基线阶段验收工具已实现，范围与证据说明见
[`docs/STAGE5_ACCEPTANCE.md`](STAGE5_ACCEPTANCE.md)。完成状态以完整 FinQA dev 质量报告和
`out/stage5/validation.json` 为准，报告拉回前暂不勾选。

NVIDIA/T4 阶段验收已完成，五项检查和五项交付物均通过，结果见
[`out/stage5/validation.json`](../out/stage5/validation.json)。质量实验覆盖 883 条 dev 样本，
解析率为 `1.0`，数值准确率为 `0.0091`；该准确率如实作为后续优化的 FP16 参照值，不代表
金融业务已经达到生产质量门槛。

- [x] FP16/BF16 模型可稳定运行
- [x] Benchmark 可重复执行
- [x] 基线业务质量结果已确认
- [x] 基线性能结果已确认
- [x] 所有输入和环境版本已登记

阶段交付物：

- [x] 基线模型运行说明
- [x] 基线质量报告
- [x] 基线性能报告
- [x] 输入接口文档
- [x] 环境锁定文件

## 6. 第 3～5 周：定制量化

### 6.1 量化框架接入

统一量化配置、导出契约、日志契约和后端版本采集工具已实现，使用方式见
[`docs/QUANTIZATION_FRAMEWORK.md`](QUANTIZATION_FRAMEWORK.md)。真实权重导出留到 6.2～6.4。

- [x] 接入 AWQ 流程
- [ ] 接入 GPTQ 流程
- [ ] 接入 SmoothQuant 或等价 INT8 流程
- [x] 统一量化配置格式
- [x] 统一模型导出格式
- [x] 统一量化日志格式
- [x] 记录量化工具和版本

### 6.2 AWQ 实验

AWQ 实验配置、三种 group size 矩阵和真实导出入口已实现，使用说明见
[`docs/AWQ_EXPERIMENT.md`](AWQ_EXPERIMENT.md)。导出后的质量、回归和性能评估入口见
[`docs/AWQ_EVALUATION.md`](AWQ_EVALUATION.md)；真实评测证据需在 Colab 运行后再勾选。

- [x] 使用上游校准集运行 AWQ
- [x] 测试 INT4 W4A16
- [x] 测试 group size 32
- [x] 测试 group size 64
- [x] 测试 group size 128
- [x] 测试不同量化粒度
- [x] 保存量化权重和配置
- [x] 执行金融业务评估
- [x] 执行回归集测试
- [x] 执行推理性能测试

### 6.3 GPTQ 实验

GPTQ 导出与同硬件 AWQ 对比入口已实现，使用说明见
[`docs/GPTQ_EXPERIMENT.md`](GPTQ_EXPERIMENT.md)。真实结果回传并验收后再勾选以下条目。

- [x] 使用相同校准集运行 GPTQ
- [x] 使用与 AWQ 相同的 bit 数对比
- [x] 使用可比的 group size 对比
- [x] 对比模型质量
- [x] 对比显存占用
- [x] 对比真实推理速度
- [x] 对比高并发性能

GPTQ 三个 group size 已完成真实导出，且均在 A100 上完成统一 Benchmark。质量结果中
g32/g64 通过解析率门槛，g128 的解析率为 0.9887、未通过 0.99 门槛；三组高风险回归均为
0 个新增回归，但 FP16 基线本身在该集合上未答对样本，因此不能将其解释为业务质量通过。

### 6.4 INT8/FP8 实验

SmoothQuant INT8/FP8 配置、硬件能力采集和验收入口已实现，使用说明见
[`docs/INT8_FP8_EXPERIMENT.md`](INT8_FP8_EXPERIMENT.md)。真实 INT8 导出和设备证据回传后再勾选。

已在 NVIDIA A100-SXM4-40GB 上使用 NVIDIA ModelOpt 0.47.0 完成 1024 条校准样本的真实
SmoothQuant W8A8 导出，证据见 [`out/int8_fp8/validation.json`](../out/int8_fp8/validation.json)。
A100 的真实 FP8 矩阵运算探测失败，因此 FP8 在当前 A100 目标环境标记为“不适用”，不再
作为本阶段的部署候选；只有切换 H100/H200 等支持 FP8 的硬件时再执行。Ascend 尚未执行，
当前 NVIDIA-only 范围内延期。INT8
质量对比已完成：883 条 dev 样本的数值准确率为 0.0068，低于 FP16 的 0.0091；三个最长
上下文样本中 FP16 和 INT8 均为 0 个正确；十个结构化输出样本中 FP16 通过 2 个，INT8
通过 0 个。三项检查表示“对比已执行”，不表示 INT8 质量达到部署门槛。

- [x] 运行 SmoothQuant INT8
- [x] 验证权重和激活量化配置
- [ ] 在 NVIDIA 上验证 FP8 可用性（A100 不适用；H100/H200 迁移后再执行）
- [ ] 确认 Ascend 对应数据类型支持情况（当前 NVIDIA-only 范围外，延期）
- [x] 对比数字字段准确率
- [x] 对比长上下文质量
- [x] 对比结构化输出成功率

### 6.5 量化结果整理

统一对比生成器和使用说明见
[`docs/QUANTIZATION_COMPARISON.md`](QUANTIZATION_COMPARISON.md)，当前报告见
[`out/quantization_comparison/comparison_report.md`](../out/quantization_comparison/comparison_report.md)。
对比表、适用方案的质量变化、性能变化和风险标记已完成；FP8 已依据 A100 实测结果标记为
不适用。FP16、INT8、AWQ 和 GPTQ 已在同一 A100 40GB 上完成统一 16 点 Benchmark。INT4
显存变化仅约 `-0.86%`～`+0.30%`，AWQ 吞吐下降约 `41.8%`、GPTQ 吞吐下降约
`28.0%`～`28.9%`；INT8 显存增加约 `5.83%`、P50 延迟增加约 `299.8%`、吞吐下降约
`74.8%`。当前量化产物均未产生部署性能收益。GPU 美元小时单价不在当前实验范围内，
因此暂不计算美元单位 token 成本；已有 GPU-hours/token 结果仍保留。

- [x] 建立 FP16、INT8、INT4、FP8 对比表
- [x] 记录每种方案的质量下降
- [x] 记录每种方案的显存收益
- [x] 记录每种方案的延迟变化
- [x] 记录每种方案的吞吐变化
- [ ] 记录每种方案的单位 token 成本（美元价格暂缓；仅在成本分析阶段补充 GPU 小时单价）
- [x] 标记不适合高风险业务的方案

### 6.6 阶段验收

阶段验收工具和证据口径见 [`docs/STAGE6_ACCEPTANCE.md`](STAGE6_ACCEPTANCE.md)，结果见
[`out/stage6/validation.json`](../out/stage6/validation.json)。AWQ、GPTQ 和 INT8 均完成真实
导出；七个适用方案均完成同硬件、同矩阵、重复测量的 Benchmark，六个 INT4 方案完成
883 条质量评测和 91 条高风险回归。这里的“稳定部署”表示技术上可加载并稳定执行，不代表
金融生产质量通过。推荐初稿见
[`docs/QUANTIZATION_RECOMMENDATION.md`](QUANTIZATION_RECOMMENDATION.md)，当前结论为暂不部署。

- [x] 至少两种量化方案可成功导出
- [x] 至少一种量化方案可稳定部署
- [x] 量化模型可执行完整业务回归
- [x] 量化模型性能数据可复现
- [x] 完成量化方案推荐初稿

阶段交付物：

- [x] AWQ/GPTQ/INT8 量化模型
- [x] 量化配置文件
- [x] 量化运行脚本
- [x] 量化质量对比报告
- [x] 量化性能对比报告

### 6.7 FP16 基线质量修复

当前 Qwen2.5-0.5B FP16 基线在 FinQA dev 上的数值准确率仅为 `0.0091`，高风险回归集
`91/91` 均未答对，存在明显地板效应。在完成本节门禁前，不启动第 7 章的大规模逐层敏感性
实验，避免将“基线本身不会回答”误判为“量化没有引入退化”。
FP16 作为所有质量、延迟、吞吐和显存对比的必要基线保留；本阶段暂不以 FP8 或 Ascend
作为验收前置条件。

#### 6.7.1 评测与错误诊断

诊断工具和人工复核流程见
[`docs/BASELINE_QUALITY_DIAGNOSIS.md`](BASELINE_QUALITY_DIAGNOSIS.md)，自动诊断结果见
[`out/baseline_quality_diagnosis/diagnosis_report.md`](../out/baseline_quality_diagnosis/diagnosis_report.md)。
883 条冻结 FP16 预测已全部分类，并生成 50 条分层人工复核队列。基础数值归一化检查通过；
自动诊断发现 32 条“后续数字包含标准答案”的潜在首数字抽取误判。已逐条复核 50 条，
其中明确区分了评分器假阳性、评分器假阴性、计算错误、公式选择错误、上下文选择错误和
输出解析错误。平均类 74 条和总和类 91 条当前均为 0 条正确。

- [x] 人工复核至少 50 条 FP16 预测、标准答案和评分结果
- [x] 验证数字、百分比、货币单位、负数和千分位归一化逻辑
- [x] 区分检索错误、表格理解错误、公式选择错误、计算错误和输出解析错误
- [x] 统计单步计算、多步计算、比较、比例和时间序列问题的准确率
- [x] 建立可重复的质量修复实验配置和结果目录

#### 6.7.2 Prompt 与确定性计算

工具、配置和 Colab 入口已实现，说明见
[`docs/QUALITY_REPAIR_6_7_2.md`](QUALITY_REPAIR_6_7_2.md)，配置见
[`configs/quality_repair.json`](../configs/quality_repair.json)。四组变体的真实 883 条 dev
评测已执行过一轮；历史完整实验使用 `max_new_tokens=128`，当前小范围试跑进一步提高到
`256` tokens，并支持嵌套 JSON。完整实验结果已记录在总文档中，当前先用固定 80 条样本筛选
Prompt，确认有效后再恢复 883 条完整评测。

- [x] 保留表格行列标题、单位、年份和上下文层级
- [x] 要求模型先输出引用字段和计算公式，再输出最终答案
- [x] 接入受限计算器或公式执行器，避免模型直接完成精确数值运算
- [x] 使用固定 JSON Schema 输出公式、过程、数值和单位
- [x] 对格式错误增加一次确定性修复重试，不修改业务答案
- [x] 建立 zero-shot、structured JSON、few-shot JSON 和 JSON+calculator 四组消融配置
- [x] 完成一轮 883 条 dev 的四方案对比，并保存解析率、结构化成功率、数值准确率和耗时
- [x] 改进结构化解析器，支持 Markdown 包裹、嵌套 evidence、百分比、货币、负数和千分位
- [x] 优先读取 JSON `value` 或 `Final answer`，避免把年份或表格中的第一个数字当成答案
- [x] 对 JSON 输出执行严格字段校验，并在首次格式失败时执行一次确定性重试
- [x] 固定 `do_sample=false`、`num_beams=1`，保证同一模型和 Prompt 下结果可复现
- [x] 增加固定种子 pilot 抽样，记录 80 条样本的原始 dev 索引和索引哈希
- [x] 增加 `structured_json_compact` 紧凑方案，减少 evidence 和 Prompt 长度
- [x] 将 pilot 生成长度提高到 `max_new_tokens=256`，验证 128 token 截断是否为主要问题
- [x] 增加 Colab 运行进度显示，记录当前方案进度和全部方案总进度
- [ ] 完成当前 80 条 pilot 对比，并根据结果决定是否恢复 883 条完整评测

#### 6.7.2.1 数值准确率优化

当前 3B `structured_json` pilot 已达到约 95% 解析率，但数值准确率仍低于 20% 门槛，主要问题从格式失败
转为表格证据选择、公式选择和数值计算错误。后续按以下顺序优化：

- [x] 增加 `evidence_operation` 两阶段方案：模型只输出证据、操作数、操作类型和单位，不直接输出最终数值
- [x] 使用安全 `Decimal` 计算器执行加减乘除、平均、比例和百分比变化
- [x] 记录 evidence、operands、operation、formula 和 calculator 使用情况，支持逐条错误定位
- [x] 增加本地 NVIDIA 4080 CLI，支持固定 80 条 pilot、完整 883 条评测、方案选择和进度输出
- [x] 在 4080 上使用固定 80 条 pilot 对比 `structured_json` 与 `evidence_operation`
- [x] 实现 evidence 中的行、列、年份和操作数回指真实表格单元格的校验
- [x] 为表格行列增加稳定 ID，限制模型只能引用合法单元格
- [x] 实现 evidence—operation—value—unit 一致性校验，失败时只重试证据和操作数
- [x] 在 4080 上复测带稳定 ID 和一致性校验的新方案
- [x] 新增 `cell_ids_operation` 紧凑契约，只输出 cell_ids、operation 和 unit，由程序恢复 operands
- [x] 在 4080 上复测 `structured_json` 与 `cell_ids_operation`
- [ ] 为 cell_id 增加行名、列名和年份语义提示，减少模型按裸索引选错单元格
- [x] 为 cell_id 增加行名、列名和年份语义提示，减少模型按裸索引选错单元格
- [x] 增加 structured_json 回退策略，避免紧凑方案失败时丢失基线答案
- [x] 增加题型路由提示：百分比、比率、平均、总和和变化题使用不同运算规则
- [x] 为 routed cell-id 方案附加合法数值 cell ID 清单，减少引用文字或越界单元格
- [x] 从模型输出的多个 JSON 候选中选择首个通过真实表格校验的候选
- [x] 在固定 80 条 pilot 上运行 `routed_cell_ids_operation`，确认是否超过 structured_json 的 12/80
- [x] 分析 v5 结果：`structured_json` 为 12/80，`cell_ids_operation` 为 11/80，路由方案为 8/80，暂不进入完整 883 条
- [ ] 修正百分比变化、实体名含 average 的题型路由，并重新运行 routed pilot
- [x] 修正百分比变化、实体名含 average 的题型路由，并重新运行 routed pilot
- [x] v6 路由方案达到 12/80，但未超过 structured_json，停止继续堆叠 Prompt 路由
- [ ] 在相同 80 条样本上评估 7B 或其他更强模型，比较是否突破 12/80
- [ ] 下载并验证 `Qwen/Qwen2.5-7B-Instruct` 本地权重
- [ ] 在相同 80 条样本上只运行 7B `structured_json` pilot，记录显存和 OOM 情况
- [x] 完成 7B AWQ INT4 `structured_json` pilot：18/80，数值准确率 22.50%
- [x] 分析 AWQ pilot：准确率超过 20% 门槛，但解析率 96.25% 仍低于 99%，不能宣布质量门禁完成
- [x] 完成 7B GPTQ INT4 `structured_json` pilot：19/80，数值准确率 23.75%
- [x] 对比 AWQ/GPTQ pilot：GPTQ 准确率高 1.25 个百分点、解析率低 2.50 个百分点
- [x] 排查 GPTQ 运行记录中合法 JSON 被标记为未解析的问题，并确认服务器代码版本
- [x] GPTQ pilot 复跑确认：仍为 19/80、解析率 93.75%，5 条未解析可复现
- [x] 核对服务器 commit 与 `quality_repair.py` SHA256，确认与当前仓库一致
- [ ] 同步 ratio/multiple 单位兼容后复跑 GPTQ pilot
- [x] 扩展 `times`/`multiple` 单位白名单、ratio 校验和 4080 Prompt 契约
- [x] 增加 ratio 使用 `times` 单位的单元测试
- [x] 分析 GPTQ v2：解析率 97.50%（78/80），数值准确率 21.25%（17/80），剩余 `mmboe` 和数字多余引号问题
- [x] 增加 `mmboe` 单位契约，并对 `value` 数字后的单个多余引号做保守 JSON 修复
- [x] 为 `mmboe` 和数字多余引号修复增加回归测试
- [x] 在服务器同步代码后复跑 GPTQ v3：解析率 100%（80/80），数值准确率 23.75%（19/80）
- [ ] 对 GPTQ v3 按题型统计错误，优先优化百分比、比率、平均和多步计算的分子/分母选择
- [ ] 重复至少两次 GPTQ pilot，确认 17--19/80 的数值波动范围
- [ ] 在解析率达到 99% 且 pilot 正确数稳定后，再扩展到完整 883 条 dev
- [ ] 增加公式/单位一致性检查：百分比统一 fraction 与 percent 表示，ratio 使用 times，禁止把增长率直接当 ratio
- [ ] 对平均、总和和利息题增加数值证据回指与确定性计算，减少模型自行编造操作数
- [x] 新增 `deterministic_cell_ids_operation`：模型只选择 cell IDs、operation 和 unit
- [x] 增加百分比、增长率、平均和利息题型路由，并由程序执行 Decimal 计算
- [x] 增加确定性平均值和利息计算的回归测试
- [x] 在同一 80 条 GPTQ pilot 上运行 `deterministic_cell_ids_operation`，与 `structured_json` 对照：10/80，解析率 93.75%
- [x] 首次运行 `deterministic_steps_operation`：4/80，解析率 48.75%
- [x] 修复多步方案的“subtract+divide”题型校验误拒绝，并允许显式数值常数参与步骤计算
- [ ] 复跑修复后的确定性方案，确认解析率和数值准确率是否恢复
- [ ] 检查确定性方案对百分比、平均、利息、多步题的分层收益
- [x] 新增候选生成方案：最多生成 3 个 cell-id/operation 候选，由程序逐个校验并确定性计算
- [x] 新增候选语义评分：结合问题年份、行名和列名选择候选
- [ ] 在同一 80 条 GPTQ pilot 上运行 `candidate_cell_ids_operation`
- [x] 新增 FinQA train QLoRA 数据转换、4-bit NF4 加载和 LoRA adapter 保存脚本
- [x] 新增 4080 入口 `--adapter-path` 与 `--load-in-4bit`，支持训练后质量复测
- [ ] 在严格 train/holdout 切分上完成首轮 QLoRA 训练
- [ ] 用同一 80 条 GPTQ/QLoRA pilot 对比数值准确率和解析率
- [ ] 补齐 7B AWQ pilot 的显存峰值和吞吐记录
- [ ] 在同一 7B FP16 或同源未量化模型上复测，分离模型规模收益与 AWQ 量化收益
- [ ] 使用 FinQA train 执行 LoRA/QLoRA 微调前，先固定当前 3B `structured_json` 基线
- [ ] 对百分比、平均、总和和多步计算样本进行困难样本分层统计
- [ ] 对百分比、平均、总和和多步计算样本进行困难样本分层统计
- [ ] 使用 FinQA train 构造“证据、操作数、操作、计算结果、单位”LoRA 监督数据
- [ ] 严格隔离 train/dev/test，禁止使用 dev/test 答案进行 Prompt 或微调
- [ ] pilot 同时改善解析率和正确数后，再使用 3B 在完整 883 条 dev 上复测

#### 6.7.3 模型升级与选择

- [ ] 在相同评测配置下比较 0.5B、1.5B、3B 和 7B 候选模型
- [ ] 确认候选模型的上下文长度能够覆盖最长 FinQA 输入
- [ ] 同时记录准确率、解析率、显存、延迟和吞吐
- [ ] 选择满足质量门槛的最小模型作为新的 FP16 基线
- [ ] 固定模型版本、权重哈希、推理框架和生成参数

#### 6.7.4 金融推理微调

- [ ] 将 FinQA train 转换为“证据字段、公式、计算结果、最终答案”监督格式
- [ ] 按文档或公司划分训练与验证数据，检查并消除数据泄漏
- [ ] 使用 LoRA 或等价参数高效方法执行监督微调
- [x] QLoRA 使用 NF4 4-bit 加载基座模型，降低模型权重显存
- [x] QLoRA 使用 `paged_adamw_8bit` 优化器，降低优化器状态显存峰值
- [ ] 在 4080 上记录 `paged_adamw_8bit` 与普通 `AdamW` 的峰值显存、吞吐和收敛差异
- [ ] 对金额、比例、利率、日期和多步计算样本进行均衡采样
- [ ] 加入当前错误样本进行困难样本训练，但不使用 dev/test 答案训练
- [ ] 保存训练配置、随机种子、数据版本、权重和训练日志
- [ ] 对未微调模型、Prompt 优化模型和微调模型执行统一对比

`paged_adamw_8bit` 是 bitsandbytes 提供的 8-bit 分页版 AdamW 优化器，名称可以拆成三部分理解：

1. `AdamW`：训练 LoRA 参数时使用的优化算法。它维护一阶动量、二阶动量，并通过 decoupled weight decay 抑制过拟合。
2. `8bit`：将优化器内部的动量和方差状态从 FP32 压缩为 8-bit 表示，主要节省优化器状态显存；它不会把模型权重自动变成 8-bit，也不等于 INT8 推理。
3. `paged`：把优化器状态按页管理，在显存紧张时按需换入/换出，降低训练过程中的显存峰值，减少 QLoRA 在 16GB 显存卡上 OOM 的概率。

因此，本项目中的显存压缩是分层的：NF4 负责压缩基座模型权重，LoRA 只训练少量 adapter 参数，`paged_adamw_8bit` 负责压缩并分页管理训练状态。它通常适合 4-bit QLoRA，但可能带来轻微吞吐或收敛差异，所以仍需用同一数据切分、随机种子和训练步数记录显存、速度与验证损失后再做最终选择。

#### 6.7.5 进入第 7 章门禁

- [ ] FinQA dev 数值准确率达到最低 `20%`，并记录后续目标值
- [ ] FinQA dev 解析率不低于 `99%`
- [ ] 91 条高风险回归集中至少有 20 条 FP16 正确样本
- [ ] 结构化输出成功率不低于 `90%`
- [ ] 最长上下文样本质量不低于完整 dev 集对应水平
- [ ] 质量结果至少重复运行两次并确认差异在允许范围内
- [ ] 冻结新的 FP16 基线报告、模型清单和回归集
- [ ] 基于新基线重新运行 AWQ、GPTQ 和 INT8 质量对比

质量修复交付物：

- [ ] FP16 错误分类报告
- [ ] Prompt 与计算器消融报告
- [ ] 候选模型对比报告
- [ ] 微调数据与训练配置
- [ ] 新 FP16 基线质量和性能报告
- [ ] 第 7 章准入验收报告

## 7. 第 6～7 周：退化定位与混合精度

本章以 6.7 冻结的新 FP16 基线为参照；6.7 准入门禁未通过时保持暂停。

### 7.1 逐层敏感性分析

- [ ] 建立逐层替换精度的实验脚本
- [ ] 对每层执行 INT4 与 INT8 对比
- [ ] 记录每层量化误差
- [ ] 记录每层业务质量变化
- [ ] 记录金额、利率、日期类错误变化
- [ ] 记录长上下文和结构化输出变化
- [ ] 生成 layer sensitivity ranking

### 7.2 模块敏感性分析

- [ ] 对 Attention 模块执行单独量化实验
- [ ] 对 MLP 模块执行单独量化实验
- [ ] 对 Embedding 执行精度对比
- [ ] 对 lm_head 执行精度对比
- [ ] 对 LayerNorm 执行精度对比
- [ ] 对 KV Cache 执行精度对比
- [ ] 生成模块级敏感性报告

### 7.3 金融错误定位

- [ ] 建立金额错误统计
- [ ] 建立利率错误统计
- [ ] 建立日期和期限错误统计
- [ ] 建立条款遗漏统计
- [ ] 建立引用错误统计
- [ ] 建立 JSON 格式错误统计
- [ ] 建立不当金融建议统计
- [ ] 建立应拒答问题误答统计
- [ ] 输出按业务场景分组的退化报告

### 7.4 混合精度搜索

- [ ] 生成“全 INT4”候选配置
- [ ] 生成“全 INT8”候选配置
- [ ] 生成“INT4 + 敏感层 INT8”配置
- [ ] 生成“INT4 + 敏感层 FP16”配置
- [ ] 保护 Embedding、LayerNorm 或 lm_head 并重新评估
- [ ] 比较不同敏感层保护数量
- [ ] 根据质量约束筛选候选配置
- [ ] 根据显存和延迟筛选候选配置

### 7.5 阶段验收

- [ ] 生成层级敏感性排名
- [ ] 生成模块级敏感性排名
- [ ] 能定位至少一种金融业务退化原因
- [ ] 生成至少一套可部署混合精度配置
- [ ] 混合精度配置完成业务回归和性能测试

阶段交付物：

- [ ] 敏感层排名
- [ ] 业务错误分类报告
- [ ] 混合精度配置
- [ ] 退化定位自动化脚本
- [ ] 退化分析报告

## 8. 第 8～10 周：剪枝与组合优化

### 8.1 结构化剪枝

- [ ] 确定可剪枝模块和约束
- [ ] 实现 Attention head 剪枝
- [ ] 实现 MLP 中间维度剪枝
- [ ] 实现可选 Transformer 层剪枝
- [ ] 验证剪枝后模型结构合法
- [ ] 测试 5% 剪枝率
- [ ] 测试 10% 剪枝率
- [ ] 测试 15% 剪枝率
- [ ] 测试 20% 剪枝率
- [ ] 记录质量拐点

### 8.2 N:M 稀疏

- [ ] 确认目标 NVIDIA GPU 支持情况
- [ ] 确认推理框架是否支持稀疏 kernel
- [ ] 实现或接入 2:4 稀疏实验
- [ ] 验证模型权重稀疏率
- [ ] 验证稀疏 kernel 是否实际生效
- [ ] 对比稀疏前后真实性能
- [ ] 评估 Ascend 是否存在对应硬件能力

### 8.3 剪枝质量验证

- [ ] 执行完整金融业务评估
- [ ] 执行高风险回归集测试
- [ ] 分析金额、利率、日期字段变化
- [ ] 分析政策和条款类任务变化
- [ ] 分析长上下文变化
- [ ] 分析结构化输出变化
- [ ] 确认剪枝后是否出现关键错误

### 8.4 剪枝和量化组合

- [ ] 测试剪枝 + INT8
- [ ] 测试剪枝 + AWQ INT4
- [ ] 测试剪枝 + 混合精度
- [ ] 测试剪枝 + KV Cache 量化
- [ ] 比较组合方案的质量和性能
- [ ] 计算显存、吞吐和单位 token 成本
- [ ] 绘制质量-成本 Pareto 曲线
- [ ] 选择低风险、中风险、高风险候选配置

### 8.5 阶段验收

- [ ] 至少一种结构化剪枝方案可运行
- [ ] 剪枝后模型可执行完整回归
- [ ] 已验证剪枝是否带来真实硬件收益
- [ ] 已完成剪枝和量化组合实验
- [ ] 已形成候选部署配置

阶段交付物：

- [ ] 剪枝模型
- [ ] 剪枝配置和脚本
- [ ] 稀疏性验证报告
- [ ] 剪枝质量报告
- [ ] 组合优化报告
- [ ] 质量-成本 Pareto 图

## 9. 第 11～12 周：NVIDIA 适配

### 9.1 NVIDIA 基础环境

- [ ] 验证 GPU 型号和显存
- [ ] 验证 CUDA 和驱动版本
- [ ] 构建 NVIDIA 容器
- [ ] 安装 PyTorch 和 Transformers
- [ ] 安装 vLLM
- [ ] 安装 TensorRT-LLM
- [ ] 验证基础 FP16/BF16 推理

### 9.2 vLLM 部署

- [ ] 接入基础模型
- [ ] 接入量化模型
- [ ] 验证 AWQ/GPTQ kernel
- [ ] 验证 continuous batching
- [ ] 验证 tensor parallel
- [ ] 验证 KV Cache 配置
- [ ] 测试不同并发数
- [ ] 测试 P50/P95/P99 延迟
- [ ] 记录显存和吞吐

### 9.3 TensorRT-LLM 部署

- [ ] 转换基础模型
- [ ] 转换量化模型
- [ ] 验证 INT4/INT8/FP8 支持
- [ ] 验证算子融合
- [ ] 验证 Tensor Parallel
- [ ] 验证是否出现算子回退
- [ ] 测试 batch throughput
- [ ] 测试高并发性能
- [ ] 对比 vLLM 和 TensorRT-LLM

### 9.4 NVIDIA 阶段验收

- [ ] 至少一个量化模型可在 vLLM 稳定运行
- [ ] 至少一个候选模型可在 TensorRT-LLM 运行
- [ ] 完成单卡和多卡测试
- [ ] 完成不同并发下的性能测试
- [ ] 完成单百万 token 成本计算
- [ ] 明确最终 NVIDIA 推荐配置

阶段交付物：

- [ ] NVIDIA Dockerfile
- [ ] vLLM 部署脚本
- [ ] TensorRT-LLM 部署脚本
- [ ] NVIDIA benchmark 报告
- [ ] NVIDIA 成本报告

## 10. 第 13～14 周：华为昇腾适配

### 10.1 昇腾环境

- [ ] 确认 Ascend 芯片型号
- [ ] 确认驱动和固件版本
- [ ] 确认 CANN 版本
- [ ] 确认 MindIE、ATB 或对应推理框架
- [ ] 构建昇腾容器或运行环境
- [ ] 验证基础 FP16/BF16 模型运行
- [ ] 保存环境版本信息

### 10.2 模型和权重适配

- [ ] 确认基础模型支持情况
- [ ] 确认 INT8 支持情况
- [ ] 确认 INT4 支持情况
- [ ] 确认 FP8 支持情况
- [ ] 完成模型转换或权重布局转换
- [ ] 验证 Tokenizer 和生成逻辑
- [ ] 验证自定义算子需求
- [ ] 记录不支持的算子

### 10.3 推理性能

- [ ] 验证量化模型能否加载
- [ ] 验证量化 kernel 是否生效
- [ ] 检查算子回退到高精度路径的情况
- [ ] 测量单卡显存
- [ ] 测量单卡延迟
- [ ] 测量多卡吞吐
- [ ] 测量不同 batch size
- [ ] 测量高并发 P95/P99 延迟
- [ ] 执行金融业务质量回归

### 10.4 昇腾阶段验收

- [ ] 至少一个候选模型可在昇腾稳定运行
- [ ] 完成模型转换和部署文档
- [ ] 完成质量和性能对比
- [ ] 明确算子回退和限制
- [ ] 明确昇腾推荐配置
- [ ] 计算昇腾单百万 token 成本

阶段交付物：

- [ ] 昇腾 Dockerfile 或环境说明
- [ ] 模型转换脚本
- [ ] 昇腾部署脚本
- [ ] 昇腾适配层
- [ ] 昇腾 benchmark 报告
- [ ] NVIDIA/昇腾对比报告

## 11. 第 15～16 周：交付与工程化

### 11.1 一键 Pipeline

- [ ] 实现输入 manifest 校验
- [ ] 实现基线运行命令
- [ ] 实现量化运行命令
- [ ] 实现敏感性分析命令
- [ ] 实现剪枝运行命令
- [ ] 实现 NVIDIA benchmark 命令
- [ ] 实现昇腾 benchmark 命令
- [ ] 实现自动报告生成
- [ ] 实现失败任务日志保存

### 11.2 版本管理

- [ ] 为基础模型建立版本号
- [ ] 为量化配置建立版本号
- [ ] 为剪枝配置建立版本号
- [ ] 为推理后端建立版本号
- [ ] 记录模型权重哈希
- [ ] 记录评测资产版本
- [ ] 记录硬件和软件环境
- [ ] 建立模型配置清单

### 11.3 回滚机制

- [ ] 保存上一版可用模型
- [ ] 保存上一版推理配置
- [ ] 提供一键切换模型版本命令
- [ ] 验证回滚后的业务质量
- [ ] 验证回滚后的性能
- [ ] 记录回滚原因和时间

### 11.4 报告和文档

- [ ] 编写项目 README
- [ ] 编写输入接口文档
- [ ] 编写量化使用说明
- [ ] 编写剪枝使用说明
- [ ] 编写 NVIDIA 部署文档
- [ ] 编写昇腾部署文档
- [ ] 编写 benchmark 使用说明
- [ ] 编写问题排查手册
- [ ] 编写模型回滚手册
- [ ] 生成最终技术报告

### 11.5 最终验收

- [ ] 在干净环境中复现主要实验
- [ ] 验证所有脚本路径和参数
- [ ] 验证报告中的指标可追溯
- [ ] 验证高风险回归集通过
- [ ] 验证模型回滚有效
- [ ] 完成项目演示
- [ ] 完成项目交接

最终交付物：

- [ ] 项目源代码
- [ ] NVIDIA 容器和部署脚本
- [ ] 昇腾环境说明和部署脚本
- [ ] 量化模型和剪枝模型
- [ ] 配置文件和版本清单
- [ ] 完整 benchmark 报告
- [ ] 质量退化定位报告
- [ ] 成本分析报告
- [ ] 操作手册和运维文档
- [ ] 最终演示材料

## 12. 推荐验收指标

### 低风险场景

- [ ] 显存降低至少 50%
- [ ] 业务质量下降不超过 1%
- [ ] P95 延迟降低至少 20%
- [ ] 吞吐提升至少 20%

### 中风险场景

- [ ] 显存降低至少 30%
- [ ] 业务质量下降不超过 0.5%
- [ ] 金额、利率、日期字段准确率不低于 FP16 基线
- [ ] 高风险回归集无关键错误

### 高风险场景

- [ ] 优先使用 INT8 或混合精度
- [ ] 关键回归样例无政策、金额、利率和期限错误
- [ ] 模型、配置、环境和评估资产均可追溯
- [ ] 支持完整回滚

## 13. 风险清单

| 风险 | 影响 | 应对措施 |
|---|---|---|
| 校准集不能代表业务分布 | 量化后质量退化 | 反馈上游补充样本，不在优化阶段擅自生产业务数据 |
| INT4 导致数字错误 | 金融业务不可接受 | 保护敏感层，升级 INT8/FP16 |
| 剪枝没有真实加速 | 成本目标无法达成 | 检查稀疏 kernel，优先结构化剪枝 |
| 后端不支持量化格式 | 模型无法部署 | 为 NVIDIA 和昇腾分别维护候选配置 |
| 算子发生高精度回退 | 延迟和吞吐不达标 | 检查 profiler 和算子日志 |
| 多卡通信成为瓶颈 | 吞吐下降 | 测试并行策略和通信配置 |
| 版本不一致导致结果不可复现 | 无法交付 | 锁定容器、驱动、框架和模型版本 |
| 评估指标不一致 | 方案无法比较 | 统一 benchmark 和评测入口 |

## 14. 最终项目运行示例

```bash
python -m banking_optimizer.pipeline \
  --model /models/bank-assistant \
  --business loan_policy \
  --input-manifest inputs/evaluation_manifest.json \
  --hardware nvidia \
  --risk-level high \
  --max-quality-drop 0.5
```

切换昇腾：

```bash
python -m banking_optimizer.pipeline \
  --model /models/bank-assistant \
  --business loan_policy \
  --input-manifest inputs/evaluation_manifest.json \
  --hardware ascend \
  --risk-level high \
  --max-quality-drop 0.5
```

## 15. 项目完成后的核心成果

- [ ] 一套面向金融业务的量化和剪枝实验流程
- [ ] 一套可解释的敏感层和质量退化定位方法
- [ ] 一套 NVIDIA GPU 部署方案
- [ ] 一套华为昇腾部署方案
- [ ] 一套混合精度和风险分级配置
- [ ] 一套性能、成本和质量评估体系
- [ ] 一套可复现、可审计、可回滚的模型交付流程
