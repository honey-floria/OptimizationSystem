# 银行金融大模型优化项目 TODO

> 最后整理：2026-10-12<br>
> 当前主线：7B QLoRA 已完成 6.7.5 质量与制品验收；启动第 7 章退化定位与混合精度分析。

## 0. 项目看板

### 0.1 当前状态

| 阶段 | 状态 | 结论 / 阻塞项 |
|---|---|---|
| 4. 上游输入 | 部分完成 | NVIDIA、模型和评测资产已验收；昇腾环境待提供 |
| 5. 范围与基线 | NVIDIA 已完成 | 0.5B FP16 基线已冻结，但质量仅 `0.91%`，不能作为生产基线 |
| 6.1～6.6 定制量化 | 已完成 | AWQ、GPTQ、INT8 已完成真实导出、质量与同硬件性能对比；当前均无部署收益 |
| 6.7 基线质量修复 | 已完成 | step 724 已通过全部质量门禁；adapter 权重 SHA256 已记录，6.7.5 验收完成 |
| 7. 退化定位与混合精度 | 待启动 | 已满足准入条件；先建立新基线分层结果，再执行敏感性分析 |
| 8. 剪枝与组合优化 | 未开始 | 依赖第 7 章结论 |
| 9. NVIDIA 部署适配 | 基础环境已验证 | GPU、CUDA、PyTorch、Transformers 和 FP16 推理已验证；尚未完成 vLLM/TensorRT-LLM 交付 |
| 10. 昇腾适配 | 阻塞 | 缺少真实昇腾硬件与环境信息 |
| 11. 工程化交付 | 未开始 | 依赖最终候选模型与双平台部署方案 |

### 0.2 下一步执行队列

按优先级执行；未满足上一项验收条件时，不扩展到后续大规模实验。

1. 对 step 724 完整 dev 结果执行百分比、比率、平均、总和和多步计算分层分析。
2. 补齐冻结基线的峰值显存、延迟和吞吐记录。
3. 基于冻结 adapter 重新运行 AWQ、GPTQ 和 INT8 质量对比，并找出主要退化类型。
4. 建立逐层和模块级敏感性分析脚本，先小样本验证，再运行完整 dev。
5. 根据敏感性排名生成并评测混合精度候选配置。
6. 并行修复训练 holdout `eval_loss=null`，供后续训练实验选择 checkpoint。

### 0.3 状态约定

- `[x]`：已有可追溯产物或报告，且该任务本身已经完成。
- `[ ]`：尚未执行、产物不完整，或尚未达到验收条件。
- “延期 / 不适用 / 阻塞”必须在条目中写明原因，不能用 `[x]` 代替完成。
- “代码已实现”与“真实实验已完成”分别记录；只有真实结果回传后才勾选实验项。
- 详细实验过程追加到 [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)，本文件只保留状态、门禁和下一步。

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
[`docs/PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)；已合并的原始专题文档不再单独维护。

## 2. 完成定义

项目整体完成需要满足：

- [ ] 至少一个银行金融业务场景完成端到端优化
- [x] 建立 FP16/BF16 基线并锁定环境（NVIDIA）
- [x] 完成至少两种量化方案对比
- [ ] 完成敏感层分析和混合精度配置
- [ ] 完成结构化剪枝实验
- [ ] 完成 NVIDIA GPU 部署和性能测试
- [ ] 完成华为昇腾部署和性能测试
- [x] 能定位金额、利率、日期、条款等业务错误
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

校验工具使用方式见 [`总文档：模型输入`](PROJECT_DOCUMENTATION.md#2-模型输入)。真实模型验收使用
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

校验工具使用方式见 [`总文档：评测资产输入`](PROJECT_DOCUMENTATION.md#3-评测资产输入)。验收依据为
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

采集与校验工具使用方式见 [`总文档：硬件环境输入`](PROJECT_DOCUMENTATION.md#5-硬件环境输入)。NVIDIA Colab
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

基线服务与验收工具已实现，使用方式见 [`总文档：基线服务`](PROJECT_DOCUMENTATION.md#7-fp16bf16-基线服务)。
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

统一 Benchmark 与验收工具已实现，使用方式见 [`总文档：统一 Benchmark`](PROJECT_DOCUMENTATION.md#9-统一-benchmark)。
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

环境锁定与验收工具已实现，使用方式见 [`总文档：环境锁定`](PROJECT_DOCUMENTATION.md#10-环境锁定)。
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
[`总文档：第 5 章阶段验收`](PROJECT_DOCUMENTATION.md#11-第-5-章阶段验收)。完成状态以完整 FinQA dev 质量报告和
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
[`总文档：量化框架接入`](PROJECT_DOCUMENTATION.md#12-量化框架接入)。真实权重导出留到 6.2～6.4。

- [x] 接入 AWQ 流程
- [x] 接入 GPTQ 流程
- [x] 接入 SmoothQuant 或等价 INT8 流程
- [x] 统一量化配置格式
- [x] 统一模型导出格式
- [x] 统一量化日志格式
- [x] 记录量化工具和版本

### 6.2 AWQ 实验

AWQ 实验配置、三种 group size 矩阵和真实导出入口已实现，使用说明见
[`总文档：AWQ 实验`](PROJECT_DOCUMENTATION.md#13-awq-实验)。导出后的质量、回归和性能评估入口见
[`总文档：AWQ 评估`](PROJECT_DOCUMENTATION.md#14-awq-质量回归与性能评估)；真实评测证据需在 Colab 运行后再勾选。

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
[`总文档：GPTQ 实验`](PROJECT_DOCUMENTATION.md#15-gptq-实验)。真实结果回传并验收后再勾选以下条目。

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
[`总文档：INT8/FP8 实验`](PROJECT_DOCUMENTATION.md#16-int8fp8-实验)。真实 INT8 导出和设备证据回传后再勾选。

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
[`总文档：量化结果整理`](PROJECT_DOCUMENTATION.md#19-量化结果整理)，当前报告见
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

阶段验收工具和证据口径见 [`总文档：第 6 章阶段验收`](PROJECT_DOCUMENTATION.md#20-第-6-章阶段验收)，结果见
[`out/stage6/validation.json`](../out/stage6/validation.json)。AWQ、GPTQ 和 INT8 均完成真实
导出；七个适用方案均完成同硬件、同矩阵、重复测量的 Benchmark，六个 INT4 方案完成
883 条质量评测和 91 条高风险回归。这里的“稳定部署”表示技术上可加载并稳定执行，不代表
金融生产质量通过。推荐初稿见
[`总文档：量化方案推荐`](PROJECT_DOCUMENTATION.md#21-量化方案推荐)，当前结论为暂不部署。

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
[`总文档：FP16 基线质量诊断`](PROJECT_DOCUMENTATION.md#22-fp16-基线质量诊断)，自动诊断结果见
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
[`总文档：Prompt 与确定性计算`](PROJECT_DOCUMENTATION.md#23-prompt-与确定性计算)，配置见
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
- [x] 完成当前 80 条 pilot 对比，并根据结果决定是否恢复 883 条完整评测

#### 6.7.2.1 数值准确率优化

当前 3B `structured_json` pilot 已达到约 95% 解析率，但数值准确率仍低于 20% 门槛，主要问题从格式失败
转为表格证据选择、公式选择和数值计算错误。后续按以下顺序优化：

当前最强结果已更新为 7B QLoRA `structured_json`：step 700 和最终 step 724 在固定 80 条 pilot 中均正确
`37/80`（`46.25%`），解析率和结构化输出成功率均为 `100%`。证据见
[`step 700 报告`](../out/quality_repair_7b_qlora_step700_pilot/comparison_report.json)和
[`最终 adapter 报告`](../out/quality_repair_7b_qlora_v2_final_pilot/comparison_report.json)。
两个 adapter 总分相同，但各有一题相对另一版本由错变对，因此仍需完整 dev 决胜。
该结果仅代表 pilot，尚不能替代完整 883 条 dev、重复性和高风险回归门禁。
以下清单保留已完成实验的追溯记录；未完成的 GPTQ/确定性计算优化作为备选路线，当前优先级低于 QLoRA 完整评测。

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
- [x] 为 cell_id 增加行名、列名和年份语义提示，减少模型按裸索引选错单元格
- [x] 增加 structured_json 回退策略，避免紧凑方案失败时丢失基线答案
- [x] 增加题型路由提示：百分比、比率、平均、总和和变化题使用不同运算规则
- [x] 为 routed cell-id 方案附加合法数值 cell ID 清单，减少引用文字或越界单元格
- [x] 从模型输出的多个 JSON 候选中选择首个通过真实表格校验的候选
- [x] 在固定 80 条 pilot 上运行 `routed_cell_ids_operation`，确认是否超过 structured_json 的 12/80
- [x] 分析 v5 结果：`structured_json` 为 12/80，`cell_ids_operation` 为 11/80，路由方案为 8/80，暂不进入完整 883 条
- [x] 修正百分比变化、实体名含 average 的题型路由，并重新运行 routed pilot
- [x] v6 路由方案达到 12/80，但未超过 structured_json，停止继续堆叠 Prompt 路由
- [x] 在相同 80 条样本上评估 7B，确认结果突破 3B 的 12/80
- [x] 下载并验证 `Qwen/Qwen2.5-7B-Instruct` 本地权重
- [ ] 补齐未量化 7B `structured_json` pilot 的显存峰值和 OOM 记录
- [x] 完成 7B AWQ INT4 `structured_json` pilot：18/80，数值准确率 22.50%
- [x] 分析 AWQ pilot：准确率超过 20% 门槛，但解析率 96.25% 仍低于 99%，不能宣布质量门禁完成
- [x] 完成 7B GPTQ INT4 `structured_json` pilot：19/80，数值准确率 23.75%
- [x] 对比 AWQ/GPTQ pilot：GPTQ 准确率高 1.25 个百分点、解析率低 2.50 个百分点
- [x] 排查 GPTQ 运行记录中合法 JSON 被标记为未解析的问题，并确认服务器代码版本
- [x] GPTQ pilot 复跑确认：仍为 19/80、解析率 93.75%，5 条未解析可复现
- [x] 核对服务器 commit 与 `quality_repair.py` SHA256，确认与当前仓库一致
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
- [x] 复跑修复后的确定性方案：cell-id 为 10/80，多步方案为 3/80，均未超过结构化基线
- [ ] 检查确定性方案对百分比、平均、利息、多步题的分层收益
- [x] 新增候选生成方案：最多生成 3 个 cell-id/operation 候选，由程序逐个校验并确定性计算
- [x] 新增候选语义评分：结合问题年份、行名和列名选择候选
- [x] 在同一 80 条 GPTQ pilot 上运行 `candidate_cell_ids_operation`：9/80，解析率 88.75%
- [x] 新增 FinQA train QLoRA 数据转换、4-bit NF4 加载和 LoRA adapter 保存脚本
- [x] 新增 4080 入口 `--adapter-path` 与 `--load-in-4bit`，支持训练后质量复测
- [x] 在严格 train/holdout 切分上完成首轮 QLoRA 训练（5778 train / 305 eval）
- [x] 用同一 80 条 GPTQ/QLoRA pilot 对比：QLoRA 33/80、解析率 100%，GPTQ 19/80、解析率 100%
- [x] 完成 QLoRA v2 checkpoint 统一 pilot：step 200/300/400/500/600/700 分别为 `30/33/34/34/36/37`，最终 step 724 为 `37/80`，所有版本解析率均为 `100%`
- [x] 根据 checkpoint 曲线停止继续追加训练步数：step 700 到最终 step 724 总分没有提升，后续资源用于完整 dev 决胜而非继续训练
- [ ] 补齐 7B AWQ pilot 的显存峰值和吞吐记录
- [ ] 在同一 7B FP16 或同源未量化模型上复测，分离模型规模收益与 AWQ 量化收益
- [ ] 对百分比、平均、总和和多步计算样本进行困难样本分层统计
- [x] 使用 FinQA train 构造结构化 LoRA 监督数据
- [x] 严格隔离 train/dev/test，禁止使用 dev/test 答案进行 Prompt 或微调
- [x] 使用 7B QLoRA 在完整 883 条 dev 上评测 step 700 与 step 724；用户接受 step 724 的单次完整结果并决定冻结，不再要求第二次完整复跑
- [x] 完成 step 700 与 step 724 的完整 883 条 dev 对比：分别为 `326/883` 和 `328/883`，解析率均为 `99.89%`

#### 6.7.3 模型升级与选择

- [x] 在相同评测配置下比较 0.5B、1.5B、3B 和 7B 候选模型
- [x] 确认候选模型的上下文长度能够覆盖最长 FinQA 输入：最长 10% 为 1521～2225 tokens，准确率不低于完整 dev
- [x] 同时记录准确率、解析率、显存、延迟和吞吐
- [x] 选择满足质量门槛的最小模型作为新的 FP16 基线
- [x] 固定模型版本、权重哈希、推理框架和生成参数；身份与选择结果见 `frozen_adapter.json` 和对应评测 manifest

#### 6.7.4 金融推理微调

- [x] 将 FinQA train 转换为结构化监督格式
- [ ] 按文档或公司划分训练与验证数据，检查并消除数据泄漏
- [x] 使用 QLoRA 执行首轮监督微调
- [x] 根据用户于 2026-10-10 的执行记录，从 `finqa_qlora_7b_v2/checkpoint-100` 恢复训练到全局 `max_steps=724`，使用 `max_length=1024`、梯度累积 32、LoRA `r=16`、`alpha=32`
- [x] 回传并核对 step 724 训练清单以及 step 100～700、724 的 `trainer_state.json` 和 adapter 配置；确认训练到 4 epoch，训练 loss 总体下降且无梯度爆炸
- [x] 使用固定 80 条 pilot 分别评测 step 100～700 和最终 adapter；所有 v2 checkpoint 解析率均为 `100%`
- [x] 确认 step 700 与最终 step 724 为并列候选：总分相同但正确样本不完全相同，不能按训练步数直接选最终版本
- [x] 在完整 883 条 dev 上比较 step 700 与最终 step 724，并按用户决定冻结 step 724；冻结清单为 [`out/finqa_qlora_7b_v2/frozen_adapter.json`](../out/finqa_qlora_7b_v2/frozen_adapter.json)
- [ ] 修复训练 holdout 的 `eval_loss=null`；当前训练状态无法用验证损失自动选择最佳 checkpoint
- [x] QLoRA 使用 NF4 4-bit 加载基座模型，降低模型权重显存
- [x] QLoRA 使用 `paged_adamw_8bit` 优化器，降低优化器状态显存峰值
- [ ] 在 4080 上记录 `paged_adamw_8bit` 与普通 `AdamW` 的峰值显存、吞吐和收敛差异
- [ ] 对金额、比例、利率、日期和多步计算样本进行均衡采样
- [ ] 加入当前错误样本进行困难样本训练，但不使用 dev/test 答案训练
- [x] 保存训练配置、随机种子、数据版本、权重和训练日志；训练清单及 SHA256 已回传并纳入冻结验收
- [x] 在固定 80 条 pilot 上对 GPTQ 基座与 QLoRA 模型执行统一对比

`paged_adamw_8bit` 是 bitsandbytes 提供的 8-bit 分页版 AdamW 优化器，名称可以拆成三部分理解：

1. `AdamW`：训练 LoRA 参数时使用的优化算法。它维护一阶动量、二阶动量，并通过 decoupled weight decay 抑制过拟合。
2. `8bit`：将优化器内部的动量和方差状态从 FP32 压缩为 8-bit 表示，主要节省优化器状态显存；它不会把模型权重自动变成 8-bit，也不等于 INT8 推理。
3. `paged`：把优化器状态按页管理，在显存紧张时按需换入/换出，降低训练过程中的显存峰值，减少 QLoRA 在 16GB 显存卡上 OOM 的概率。

因此，本项目中的显存压缩是分层的：NF4 负责压缩基座模型权重，LoRA 只训练少量 adapter 参数，`paged_adamw_8bit` 负责压缩并分页管理训练状态。它通常适合 4-bit QLoRA，但可能带来轻微吞吐或收敛差异，所以仍需用同一数据切分、随机种子和训练步数记录显存、速度与验证损失后再做最终选择。

#### 6.7.5 进入第 7 章门禁

- [x] FinQA dev 数值准确率达到最低 `20%`：step 724 为 `37.15%`（`328/883`）
- [x] FinQA dev 解析率不低于 `99%`：step 724 为 `99.89%`（`882/883`）
- [x] 91 条 contextual 高风险回归集中至少有 20 条冻结 QLoRA 候选正确样本：NF4 运行时为 `21/91`（`23.08%`）
- [x] 诊断首次 step 724 高风险运行 `0/91` 为无效门禁结果：旧资产仅含 `question/answer`，平均输入约 140 tokens，缺少 FinQA 财报上下文和表格
- [x] 修复高风险资产构建：按标准化问题回连 FinQA test 的 `pre_text/post_text/table/qa`，任何未匹配问题直接失败
- [x] 修复 contextual 资产重建报错：verified 有 3 条 `decimal proportion` 问句，而 FinQA 原题使用 `portion`；将 `portion` 纳入模板词后按内容词唯一匹配，歧义或缺失仍立即失败
- [x] 确认第二次 `0/91` 仍误用了旧资产：数据 SHA256 与输入长度未变化；评测入口已增加无上下文硬失败保护
- [x] 修复 verified 数值型答案的百分比重复换算；修复前后模型原始输出逐条一致，6 条假阴性被纠正，最终为 `21/91`
- [x] 结构化输出成功率不低于 `90%`：step 724 为 `99.89%`
- [x] 最长上下文样本质量不低于完整 dev：最长 10% 样本为 `35/89`（`39.33%`），高于全量 `37.15%`；最长单条样本回答正确
- [x] 完整 dev 重复要求经用户决定豁免：step 724 使用确定性生成，固定 80 条在 pilot 与 full 中逐条完全复现，并直接冻结
- [x] 冻结新的 QLoRA/NF4 基线制品：hydrated `adapter_model.safetensors` 大小为 `161533192` 字节，SHA256 为 `c92a6058f668b0273393e85443b978b3e1620ed0c3a2ef01905e1124bda52fa4`
- [ ] 基于新基线重新运行 AWQ、GPTQ 和 INT8 质量对比

质量修复交付物：

- [x] FP16 错误分类报告
- [x] Prompt 与计算器消融报告
- [ ] 候选模型对比报告
- [x] 微调数据与训练配置
- [ ] 新 FP16 基线质量和性能报告
- [x] 第 7 章准入验收报告

## 7. 第 6～7 周：退化定位与混合精度

本章以 6.7 冻结的 QLoRA/NF4 基线为参照；6.7.5 准入验收已于 2026-10-12 完成。

### 7.1 逐层敏感性分析

目标：在保持模型、数据、Prompt、校准集和生成参数不变的前提下，以“全层 INT4”为起点，每次仅将一个
Transformer block 恢复为 INT8，测量该层带来的量化误差下降和金融业务质量恢复量，为 7.4 的混合精度搜索
提供可复现的层级排名。

#### 7.1.1 实验口径与输入冻结

- [x] 实现 `configs/layer_sensitivity_setup.json`、`src/quantization/layer_sensitivity_setup.py` 和 `scripts/prepare_layer_sensitivity.py`，提供 `merge/freeze/compare-pilot` 三个独立且不覆盖旧证据的命令
- [x] 实现 7.1.1 单元测试，覆盖精度契约、层自动发现、目录哈希、pilot 索引哈希、多标签分类、长上下文选择和配对对比
- [x] 根据用户于 2026-10-12 的服务器核对结果，确认训练服务器 hydrated `models/data/finqa_qlora_7b_v2/adapter_model.safetensors` 大小为 `161533192` 字节，SHA256 为 `c92a6058f668b0273393e85443b978b3e1620ed0c3a2ef01905e1124bda52fa4`，与修正后的 6.7.5 冻结记录一致；`771482db...` 仅为 LFS 指针哈希
- [x] 在服务器生成 `input_freeze.json`，记录 Qwen2.5-7B-Instruct 基座模型、step 724 adapter、tokenizer 和数据集的绝对路径、版本与逐文件 SHA256
- [x] 将冻结的 step 724 LoRA adapter 合并到 FP16 基座模型，导出 7.1 实验母版和 `merged_model_manifest.json`；merged 目录树 SHA256 为 `c59ea48fd633162f9bf6fd0b5fb26b8d3d9c6e6686e2b905e49e44905f5fd946`
- [x] 在固定 80 条 pilot 上核对 merged FP16 与已冻结 QLoRA/NF4；两者均为 `37/80`、解析 `80/80`，5 条恢复、5 条退化，总分相同但逐样本不等价
- [x] 明确 7.1 中 INT4 为 `W4A16 group_size=32`，INT8 为 `W8A16`，激活统一保持 FP16/BF16，并写入可校验配置
- [x] 明确 7.1 不把 AWQ W4A16 与 SmoothQuant W8A8 直接作为逐层对照，配置验证会拒绝丢失该限制的实验契约
- [x] 冻结 1024 条 calibration、80 条 pilot、883 条 FinQA dev、91 条 contextual 高风险集及 89 条最长上下文子集的样本清单与哈希
- [x] 冻结 `structured_json` Prompt、`max_new_tokens=256`、`do_sample=false`、`num_beams=1`、数值容差和评分器版本
- [x] 生成并冻结金额、比例/利率、日期/期限多标签，样本数分别为 `338/567/709`；计数可重叠，不应求和为 883
- [x] 实现从模型实例自动发现 Transformer block 路径与层数，服务器实际发现路径为 `model.layers`、层数为 `28`

#### 7.1.2 工具与实验矩阵

执行口径：Qwen2.5-7B-Instruct 的模型配置声明 `num_hidden_layers=28`，7.1.1 对 merged 实例再次自动发现并
确认 `model.layers` 长度为 28。这里的 28 是 Transformer decoder block 数量；它与同为 28 的
`num_attention_heads` 是两个独立维度，也不是由“7B”参数规模直接换算而来。逐层矩阵必须建立 28 个彼此
独立的 `L-i-I8`：每个计划都从相同的 `ALL-I4` 分配重新开始，仅将第 `i` 层恢复为 INT8，不继承前一个计划
的修改。加上 `ALL-I4/ALL-I8` 共需执行 30 个计划；`REF-FP/REF-NF4` 复用 7.1.1 证据，只注册不重跑，
所以 manifest 总计 32 个条目。少跑任意一个 `L-i-I8`，对应层的敏感度仍为未知。

量化范围：7.1 只量化每个 block 内承担主要参数量和矩阵乘法的 Attention
`q/k/v/o_proj` 与 MLP `gate/up/down_proj` Linear 权重。Embedding、Norm、`lm_head`、KV Cache 和
activation 保持不变，分别避免全局输入/输出干扰、低收益高敏感参数、动态 cache 与激活量化混入单层归因。
INT4 为 W4A16 g32 对称量化，即每个输出行沿输入维度每 32 个权重共享一个 scale；INT8 为 W8A16
per-output-channel 对称量化，即每个输出行共享一个 scale。对称量化固定零点为 0，使用
`scale=max(abs(w))/qmax`；当前 fake-quant 在取整后反量化回 FP16，只用于质量敏感度，不代表真实压缩收益。

- [x] 新增 `configs/layer_sensitivity.json`，定义模型、位宽、group size、数据子集、生成参数、排名规则和输出路径
- [x] 新增 `src/quantization/layer_sensitivity.py` 和 `scripts/run_layer_sensitivity.py`，支持模型身份校验、逐层精度替换、误差采集、评测、断点续跑、配对转移和预览排名汇总
- [x] 新增 `tests/test_layer_sensitivity.py`，覆盖矩阵完整性、单层替换、加权误差、身份与参考组校验、断点续跑、配对转移、失败排除和排名 tie-break
- [x] 注册 `REF-FP`：merged FP16 数学参考，固定 pilot 为 `37/80`、解析 `80/80`
- [x] 注册 `REF-NF4`：6.7.5 冻结 QLoRA/NF4 业务参考，固定 pilot 为 `37/80`、解析 `80/80`
- [x] 生成 `ALL-I4`：所有 Transformer block 使用 W4A16 g32；结果为 `36/80`、解析 `80/80`
- [x] 生成 `ALL-I8`：所有 Transformer block 使用 W8A16；结果为 `36/80`、解析 `79/80`
- [x] 为全部 28 个第 `i` 层生成 `L-i-I8`：仅第 `i` 层使用 W8A16，其余 Transformer block 使用 W4A16 g32；无缺层、重复层或失败计划
- [x] 确认每个 `L-i-I8` 除目标层位宽外的权重、Prompt、生成参数、80 条样本哈希和评分路径完全一致
- [x] 完成后端不支持真实混合位宽时的 symmetric fake-quant 敏感性路径：INT4 使用 W4A16 g32，INT8 使用 per-output-channel W8A16，manifest 明确禁止声称真实存储、显存、延迟或吞吐收益

#### 7.1.3 量化误差采集

- [x] 逐层记录权重 MSE、normalized MSE、cosine similarity、最大绝对误差和裁剪/饱和比例；30 个计划已升级 schema v2，审计 `status=pass`、`valid_plans=30`、`invalid_plans=[]`，模块 cosine 范围为 `[0.9931837, 0.9999594]`
- [x] 按权重元素数对层内 Linear 误差加权汇总，不对大小不同的矩阵做简单平均；28 层均有 7 个 Linear、全模型共 196 个模块
- [x] 保留 `q_proj/k_proj/v_proj/o_proj` 和 `gate_proj/up_proj/down_proj` 明细，但 7.1 排名仅以完整 block 为单位
- [x] 使用固定 calibration 样本采集层输出激活 MSE、normalized MSE 和 cosine similarity；固定前 32 条 calibration、每条最多 512 token 中均匀抽取 32 个位置，30/30 计划通过合同、SHA256、28 层完整性及数值范围审计，layer cosine 范围为 `[0.8915972, 0.9999954]`
- [x] 单独报告量化误差与业务质量的相关性：schema v2 pilot 中误差恢复量对净正确数的 Pearson=-0.3932625、Spearman=-0.4421115，仅作描述，不用权重误差代替业务敏感度

#### 7.1.4 两阶段业务评测

- [x] 在固定 80 条 pilot 上完成 `REF-FP/REF-NF4/ALL-I4/ALL-I8` 和全部 `L-i-I8`；30/30 个可运行计划完成且无 pending
- [x] 对每个层候选记录数值准确率、正确数、解析率、structured output rate、format retry 次数和运行时间；30 个计划总耗时约 84.59 分钟
- [x] 逐样本记录 `ALL-I4` 错→`L-i-I8` 对的 `recovered`、`ALL-I4` 对→`L-i-I8` 错的 `regressed` 及净正确数变化
- [x] 在金额、比例/利率、日期/期限子集上同时报告“正确数/总数”和比例；layer 23 三类分别为 `133/338`、`204/567`、`277/709`，相对 ALL-I4 净增 `6/13/15` 题
- [x] 记录最长 10% 上下文子集和 91 条 contextual 高风险集的恢复数、退化数、解析率和结构化输出变化；layer 23 长上下文净增 1 题且高风险 0 恢复/0 退化，layer 10 两者均净增 1，layer 22 长上下文净退化 1
- [x] 至少重复运行 `ALL-I4` 和一个哨兵层，确认确定性生成可逐条复现；`ALL-I4 + layer-22` 的 80 行复跑均 exact match，预测文件 SHA256 分别为 `4bcd0dac...` 和 `d1451160...`
- [x] 对 pilot 第一梯队 `L-10-I8/L-22-I8/L-23-I8` 执行完整 883 条 FinQA dev；三层分别为 `330/883`、`328/883`、`344/883`，结果是候选子集验证，不表述为 28 层完整排名

#### 7.1.5 排名、产物与验收

- [x] 定义 `quality_recovery_i = accuracy(L-i-I8) - accuracy(ALL-I4)`，并同时保留 `recovered_count/regressed_count/net_correct_gain`；pilot 第一梯队为 layer 10/22/23，均净增 2 题
- [x] 候选排名优先级固定为：`net_correct_gain`、`recovered_count`、高风险恢复量、长上下文恢复量、结构化输出恢复量、normalized quantization error、层号；最终候选顺序为 `23 > 10 > 22`
- [x] 为质量恢复量生成 bootstrap 置信区间，并在大量并列或区间高度重叠时降低排名结论强度；layer 23/10/22 的 95% 区间分别为 `[+0.68,+3.06]`、`[-0.79,+1.25]`、`[-1.25,+1.25]` 个百分点，Top-2 区间重叠，因此总体排名强度为 `low`
- [x] 输出 `experiment_manifest.json`、`layer_metrics.jsonl`、`sample_transitions.jsonl`、`ranking.json`、`ranking.csv`、`report.md` 和 `validation.json`；11/11 合同运行完成、pending 全空、`invalid_runs=[]`，七项最终产物 SHA256 全部复核一致
- [x] 验收时确认所有 Transformer block 均有 INT4/INT8 误差和业务评测记录；30/30 schema v2 计划通过范围、文件哈希与结果引用审计，无失败或缺失运行进入排名
- [x] 验收时确认每个 pilot 排名结论都能回溯到模型 manifest、量化配置、固定样本哈希、逐样本预测和逐层/模块误差记录；原始 schema v1 证据亦保留用于 lineage 核对
- [x] 将三层候选验证排名及推荐保护层集合写入项目总文档；合同推荐为 `[23,10]`，保守解释为 layer 23 确定保护、layer 10 组合候选、layer 22 排除，并明确它不是 28 层完整 layer sensitivity ranking

### 7.2 模块敏感性分析

目标：在 7.1 已冻结的 merged FP16 母版、`ALL-I4`、样本清单、Prompt、生成参数和评分器上，进一步回答
“同一层内 Attention 与 MLP 哪一类更值得保护，以及 Embedding、Norm、`lm_head` 和 KV Cache 是否可以安全
降精度”。Attention/MLP 使用“从 `ALL-I4` 恢复为 INT8 后挽回多少质量”的恢复实验；7.1 中原本保持
FP16 的特殊模块使用“单独降为 INT8/INT4 后损失多少质量”的退化实验。两类分数方向不同，必须分榜报告，
不得直接混成一个排名。

当前进度（2026-10-14）：7.2.1 已完成真实模型验收；untied 模型矩阵的 71 个可运行计划已全部取得终态，
`pending=[]`，其中 69 个 `completed`、KV Cache INT8/INT4 因尚未执行 HQQ runtime capability probe 而为
`unsupported`。28 层 Attention/MLP、Embedding、`lm_head`、block/final Norm 和 FP16 KV Cache pilot 已完成；
7.2.4 已在 full-dev 前预注册 11 个晋级计划（7 个 Attention/MLP 恢复计划和 4 个 Embedding/`lm_head` 计划），
晋级/冻结评测和 7.2.5 双榜/保护策略代码已实现，但完整 dev/high-risk 尚未运行，因此当前结论仍是 pilot
筛选结论，不是最终部署方案。

#### 7.2.1 前置条件与实验契约

- [x] 复用并校验 7.1.1 的模型身份、目录哈希、tokenizer、calibration/pilot/dev/high-risk/长上下文样本哈希、Prompt、生成参数、随机种子和评分器；本地 `prepare` 合同审计已通过，禁止重新抽样形成不可比结果
- [x] 校验 `ALL-I4` 计划、plan SHA256、80 条 pilot 样本哈希、结果与预测哈希完整可追溯；以其逐样本预测作为 Attention/MLP 恢复实验的唯一主对照，不重复生成另一份同名基线
- [x] 自动发现并登记实际模块路径：训练服务器 `inspect-model` 已通过，确认 28 个 block、112 个 Attention Linear、84 个 MLP Linear、57 个 Norm，以及独立的 Embedding 和 `lm_head`；真实路径与 inventory SHA256 已写入 7.2.1 产物
- [x] 冻结模块分组与精度规则：Linear 沿用 7.1 的 W4A16 g32 和 per-output-channel W8A16；Embedding、Norm、`lm_head` 已独立记录量化轴、group size 与共享权重策略，配置不适用时失败而非静默回退
- [x] 检查 Embedding 与 `lm_head` 是否共享权重：训练服务器使用参数对象及底层 storage 双重检测，实际结果为 `embedding_lm_head_tied=false`，因此后续使用四个独立 Embedding/`lm_head` 计划而不是共享权重计划
- [x] 为 KV Cache 单独冻结能力审计契约：记录请求位宽、真实后端声明、模型 `use_cache/past_key_values` 接口和 Transformers QuantizedCache API；禁止把权重 fake-quant 当作 KV Cache 量化，真实 K/V 粒度、scale、layout 在 7.2.3 runtime probe 前保持未验证
- [x] 明确 fake-quant 结果只用于质量敏感性；配置与 validation 在真实低比特存储和 kernel runtime probe 前固定 `deployment_performance_claims_allowed=false`
- [x] 新增 `configs/module_sensitivity.json`、`src/quantization/module_sensitivity.py`、`scripts/run_module_sensitivity.py` 和 `tests/test_module_sensitivity.py`；当前覆盖上游身份、精度契约、模块发现、缺失模块失败、tied weights、Norm、KV Cache 能力检测与 CLI，两项运行矩阵相关覆盖随 7.2.2 实现

#### 7.2.2 Attention 与 MLP 受控恢复矩阵

- [x] 实现 7.2.2/7.2.3 共用实验框架：基础矩阵按实际 tied-weight 状态生成 69（tied）或 71（untied）个计划，其中包含 59 个 Attention/MLP 可运行计划；当前依据 7.1 full-dev 与 7.2 pilot 另预注册 1 个跨层候选 `candidate_l23_all_l10_mlp_int8`，实际 untied 矩阵为 72 个计划、71 个可运行；支持合同哈希、隔离校验、单计划独立进程、断点续跑和 `unsupported` 终态
- [x] 完成 `ATTN-ALL-I8`：所有层仅 Attention 的 `q/k/v/o_proj` 恢复为 INT8，MLP 保持 INT4；80 条 pilot 为 `36/80`，恢复/退化 `4/4`、净增 0，未观察到模块族总体质量收益
- [x] 完成 `MLP-ALL-I8`：所有层仅 MLP 的 `gate/up/down_proj` 恢复为 INT8，Attention 保持 INT4；80 条 pilot 为 `35/80`，恢复/退化 `4/5`、净增 -1，未观察到模块族总体质量收益
- [x] 为全部 28 个 block 分别生成并执行 `L-i-ATTN-I8`：计划覆盖 0～27 层且无缺失/重复，每个计划仅将目标层四个 Attention Linear 恢复为 INT8；12 层净正、9 层持平、7 层净负，pilot 最强为 layer 6 的 `+3`
- [x] 为全部 28 个 block 分别生成并执行 `L-i-MLP-I8`：计划覆盖 0～27 层且无缺失/重复，每个计划仅将目标层三个 MLP Linear 恢复为 INT8；14 层净正、6 层持平、8 层净负，pilot 最强为 layer 26 的 `+3`
- [x] 对每个计划校验只有声明的模块发生变化，并保存生效模块清单、量化配置、父基线引用和计划合同 SHA256；缺层、重复层、越界模块或意外修改均判失败
- [ ] 补充每个实际运行计划的量化前后权重 SHA256；当前已有生效模块清单、合同哈希和逐模块数值误差，但尚未保存逐权重 tensor 的前后哈希
- [x] 对已执行计划逐模块记录参数量、权重 MSE、normalized MSE、cosine similarity、最大绝对误差和裁剪/饱和比例；模块族汇总按权重元素数加权，同时保留 196 个 Linear 明细
- [ ] 使用 7.1 相同的 calibration 子集采集目标模块输出或 block 输出误差，至少记录 activation MSE、normalized MSE、cosine similarity 和异常值比例
- [x] 计算相对 `ALL-I4` 的 `recovered_count`、`regressed_count`、`net_correct_gain` 和准确率变化；`ATTN+MLP-ALL-I8` 审计已完成并与 7.1 `ALL-I8` 的 196 个 Linear、8 类误差/精度字段及预测 SHA256 完全一致
- [x] 根据候选拆分结果完成跨层组合验证：第 23 层 Attention/MLP 为 INT8、第 10 层仅 MLP 为 INT8，其余 Transformer Linear 为 INT4；计划合同只修改声明的 10 个 Linear，80 条 pilot 为 `38/80`、恢复/退化 `2/0`，与两个单项的 `+2` 持平而未叠加，额外保护第 10 层 MLP 约增加 102 MB 理论权重存储但无增量正确题，因此阶段性默认建议仅保护第 23 层，第 10 层 MLP 保留为独立备选

#### 7.2.3 Embedding、Norm、lm_head 与 KV Cache 降精度实验

- [x] 实现特殊模块计划与运行路径：根据实际 tied-weight 状态生成独立或联合 Embedding/`lm_head` 计划，区分 block/final Norm，并为 KV Cache 提供 HQQ INT8/INT4 runtime probe、能力门禁和明确的 `unsupported` 结果；权重特殊模块和 FP16 KV Cache 已完成服务器 pilot
- [x] 实现特殊模块专项证据：记录量化前后范围、NaN/Inf、packed weight/scale/FP16 对照存储；`lm_head` 保存 logits MSE、top-1/top-k 一致率与数值 token 排名变化；KV Cache 增加可运行 FP16 对照及等长 prefill/decode 的 cache bytes、峰值显存、TTFT、TPOT、延迟、吞吐和低比特驻留门禁
- [x] 对 Embedding 分别完成 INT8 和 INT4 单模块降精度 pilot：两者均为 `36/80`、无恢复或退化；INT8/INT4 理论分别节省 519.5/747.1 MiB，完整 dev 前仍不得宣称可部署
- [x] 对 `lm_head` 分别完成 INT8 和 INT4 pilot：INT8 为 `36/80`、INT4 为 `37/80`；两者 top-1 均一致，但 INT4 logits MSE 约为 INT8 的 24 倍、数值 token 平均排名扰动约为 8 倍，因此 INT8 是更稳妥候选
- [x] 完成 block/final Norm 的 FP16 对照、INT8、INT4 pilot，并记录范围、NaN/Inf 和误差；block Norm INT4 降至 `8/80`、净退化 28 题，final Norm INT4 净退化 2 题；INT8 虽基本无损但仅节省约 0.191 MiB/3.5 KiB，默认建议 Norm 保持 FP16
- [x] 完成共享权重适用性判定：真实模型的 Embedding 与 `lm_head` 不共享权重，因此共享权重联合计划不适用，矩阵已生成四个独立 INT8/INT4 计划
- [ ] 在同一权重基线上对 KV Cache 执行 FP16、INT8 和 INT4 对比；分别覆盖 prefill 与逐 token decode，并检查生成长度、batch size 和上下文长度完全一致
- [ ] KV Cache 后端不支持真实 INT8/INT4 时，输出明确的 `unsupported` 能力审计；可增加标记为 sensitivity-only 的量化—反量化 hook，但不得据此声称真实 cache 显存或性能收益
- [ ] 对真实 KV Cache 量化记录 cache bytes/token、峰值显存、TTFT、TPOT、端到端延迟、吞吐及输出质量，并验证请求期间低比特 cache 确实驻留而非立即反量化后长期保存为 FP16

#### 7.2.4 两阶段业务评测与统计

- [x] 实现预注册晋级与最终评测流水线：`select-advancement` 要求完整 pilot 矩阵无 pending，再从显式预注册候选中校验并在 full-dev 前冻结规则/候选和 SHA256；`prepare-final` 冻结 7.1 基线、883/91/89 条样本、业务标签及确定性哨兵，`run-final-plan` 支持断点续跑并逐条校验冻结数据；真实运行完成度由以下实验项单独记录
- [x] 第一阶段已在冻结的 80 条 pilot 上完成全部 Attention/MLP、特殊权重模块和 FP16 KV Cache 计划；71 个可运行计划无 pending，69 个完成、KV INT8/INT4 以明确 `unsupported` 终态退出，未将失败或未验证计划伪装成可排名结果
- [x] pilot 后、任何新 full-dev 结果产生前已预注册 11 个完整 dev 候选：`layer_06_attn_int8`、`layer_26_mlp_int8`、`layer_10_mlp_int8`、`layer_17_mlp_int8`、`layer_25_mlp_int8`、`layer_23_attn_int8`、`layer_23_mlp_int8`，以及 Embedding/`lm_head` 的 INT8/INT4；Norm 默认保留 FP16，KV Cache 等真实低比特驻留 probe 后再决定，后续不得根据 full-dev 结果改候选集合
- [x] 第二阶段已在完整 883 条 FinQA dev、91 条 contextual 高风险集和 89 条最长上下文子集上完成 11 个晋级候选；本轮 KV Cache 没有通过真实 probe，未进入排名
- [x] 每个计划已报告数值准确率、正确数、解析率、structured output rate、format retry、运行时间，以及金额、比例/利率、日期/期限三个冻结多标签子集的“正确数/总数”和比例
- [x] 已保存逐样本 `recovered/regressed/unchanged` 转移；特殊模块使用相对 FP16 特殊组件基线的净变化语义，没有把存储收益当作质量恢复
- [x] 已复跑 Layer 23 Attention、Layer 23 MLP 和 Embedding INT8 三个哨兵，三者逐行预测及 SHA256 均与 pilot 完全一致
- [x] 已使用固定种子 10,000 次配对 bootstrap 生成 95% 置信区间；只有 Layer 10 MLP 区间明确高于 0，其余正向点估计均降级为第二梯队

#### 7.2.5 排名、产物与验收

- [x] 实现最终汇总与验收代码：分离恢复榜和损失—收益榜，生成固定种子配对 bootstrap、逐样本转移、双格式排名、机器可读保护策略、确定性复跑审计和全产物 SHA256；未完成服务器运行时 `finalize` 会拒绝生成最终结论
- [x] 已分别生成 Attention/MLP“质量恢复榜”和特殊模块/KV Cache“质量损失—收益榜”；KV Cache 无真实低比特结果，策略明确保持 FP16
- [x] 已输出 `experiment_manifest.json`、`module_metrics.jsonl`、`sample_transitions.jsonl`、`ranking.json`、`ranking.csv`、`report.md` 和 `validation.json`，汇总项可回溯到冻结合同与逐样本预测
- [x] 已报告 Attention/MLP 模块族与逐层结果、特殊模块建议、KV Cache 支持结论、置信区间和已知限制，并明确当前权重实验为 fake-quant
- [x] 已确认 28 个 Attention 与 28 个 MLP 单层恢复计划无缺失、无重复、无越界，特殊模块覆盖适用精度，失败或 unsupported 计划未进入排名
- [x] 已形成供 7.4 使用的机器可读候选保护策略：Layer 10 MLP 为主候选，`lm_head` INT8，Embedding INT8 为显存备选，Norm/KV Cache FP16，并保留质量门禁、后端约束和禁止组合
- [x] 已将最终模块级结论、推荐配置及 `lm_head` INT4、Norm INT4、未验证 KV Cache 等负结果写入项目总文档，并保留 fake-quant/真实部署边界
- [x] 已实现独立的组合验证工作流和两个前瞻性冻结候选：`candidate_l10_mlp_lm_head_int8` 与 `candidate_l10_mlp_lm_head_embedding_int8`；使用独立输出目录，不覆盖原始 11 计划证据
- [x] 已在服务器运行 `bash scripts/run_module_sensitivity_combinations.sh`，完成两个组合的 pilot、确定性、full-dev、high-risk、长上下文和最终门禁；Embedding FP16 主候选为 337/883（相对 ALL-I4 +9），Embedding INT8 研究候选为 339/883（+11），二者高风险与长上下文均无净退化

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

> 本轮服务器空间预算为约 37 GiB，只闭环三个核心计划：`all_linear_int4`（已导出）、`all_linear_int8` 和
> `mixed_l10_mlp_lm_head_int8`。其余消融候选保留配置但不纳入本轮 `pending`/`finalize` 必需证据。

- [x] 已生成“全 INT4”候选配置：准确语义为全部可量化 Linear INT4，Embedding、Norm 和 KV Cache 保持 BF16
- [x] 已生成“全 INT8”候选配置：全部可量化 Linear INT8，作为质量上界和性能对照
- [x] 已生成“INT4 + 敏感层 INT8”配置：含只保护 Layer 10 MLP、以及 Layer 10 MLP + `lm_head` INT8 的主候选
- [x] 已生成“INT4 + 敏感层 FP16/BF16”配置：其余 Linear INT4，Layer 10 MLP 保持 BF16
- [ ] 在真实 TorchAO 制品上完成 Embedding、Norm 或 `lm_head` 保护重评；代码已实现，Embedding INT8 因当前 Linear 后端不支持而明确标记 `unsupported`，Norm/KV 保持 BF16，待服务器运行
- [ ] 完成不同敏感层保护数量实测；已冻结 0 组、1 组及 Layer 10+23 两组消融候选，待服务器运行
- [ ] 根据质量约束筛选候选配置；已实现 full-dev、高风险、长上下文及解析率自动门禁，待真实制品证据齐全后收口
- [ ] 根据显存和延迟筛选候选配置；已实现相对全 Linear INT4 的峰值显存、P95 延迟和吞吐自动门禁，待服务器 Benchmark
- [x] 已实现 TorchAO 按完整模块名逐 Linear 流式量化、量化后实际 Tensor 子类审计、可移植 state_dict + SHA256、meta-device 加载、质量/性能执行入口、断点续跑、证据登记和最终排名

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

- [x] 验证 GPU 型号和显存
- [x] 验证 CUDA 和驱动版本
- [ ] 构建 NVIDIA 容器
- [x] 安装 PyTorch 和 Transformers
- [ ] 安装 vLLM
- [ ] 安装 TensorRT-LLM
- [x] 验证基础 FP16/BF16 推理

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
- [x] 记录模型权重哈希
- [x] 记录评测资产版本
- [x] 记录硬件和软件环境
- [x] 建立当前冻结模型配置清单

### 11.3 回滚机制

- [ ] 保存上一版可用模型
- [ ] 保存上一版推理配置
- [ ] 提供一键切换模型版本命令
- [ ] 验证回滚后的业务质量
- [ ] 验证回滚后的性能
- [ ] 记录回滚原因和时间

### 11.4 报告和文档

- [x] 编写项目 README
- [x] 编写输入接口文档
- [x] 编写量化使用说明
- [ ] 编写剪枝使用说明
- [ ] 编写 NVIDIA 部署文档
- [ ] 编写昇腾部署文档
- [x] 编写 benchmark 使用说明
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
