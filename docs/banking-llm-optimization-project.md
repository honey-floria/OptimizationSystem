# 面向银行金融场景的模型量化、剪枝与跨硬件推理优化项目

## 1. 项目概述

本项目面向银行及金融机构的私有化大模型部署场景，负责在既定业务数据、模型和评估集的基础上，完成模型量化、结构化剪枝、混合精度配置、质量退化定位以及 NVIDIA GPU 和华为昇腾硬件适配。

项目目标不是单纯追求模型压缩率，而是在金融业务质量约束下，降低显存、推理延迟和线上算力成本，并形成可复现、可审计、可回滚的模型优化流程。

推荐的首个业务场景为银行内部知识与运营助手，例如：

- 信贷政策和产品条款查询
- 反洗钱及合规制度辅助检索
- 客户经理业务助手
- 金融文档字段抽取
- 银行客服回复草稿生成

项目不负责自动授信、自动拒贷、自动交易或其他需要独立金融决策授权的功能。模型只提供辅助结果，最终决策由既有规则系统或人工完成。

## 2. 项目边界

### 2.1 本项目负责

- 建立 FP16/BF16 推理基线
- 接入上游提供的校准集、评估集和回归集
- 对比 AWQ、GPTQ、SmoothQuant、FP8 或 INT8 等方案
- 进行逐层、逐模块敏感性分析
- 设计 INT4、INT8、FP16 混合精度配置
- 实施结构化剪枝和可用的 N:M 稀疏实验
- 定位量化或剪枝导致的业务质量退化
- 适配 NVIDIA GPU 推理后端
- 适配华为昇腾推理后端
- 测量显存、延迟、吞吐、并发和单位 token 成本
- 生成模型、配置、 benchmark 和优化报告
- 提供灰度部署、监控和回滚所需的技术接口

### 2.2 本项目不负责

- 原始业务数据采集
- 客户数据脱敏和隐私处理
- 数据清洗、标注和业务分类
- 知识库构建、文档切分和向量化
- 训练集、校准集、评估集的生产
- 金融业务规则制定
- 业务指标口径定义
- 自动授信、交易或风险决策逻辑
- 生产环境最终审批和合规认证

上游数据或业务团队需要以约定格式提供数据资产。本项目只负责校验输入格式、读取数据并执行实验，不承担数据内容正确性责任。

## 3. 上游输入接口

项目依赖以下输入资产：

| 输入 | 提供方 | 要求 |
|---|---|---|
| 基础模型 | 模型团队 | 模型权重、配置、Tokenizer、许可证 |
| 校准集 | 数据/业务团队 | 已完成脱敏，覆盖目标业务分布 |
| 评估集 | 数据/业务团队 | 与校准集隔离，包含业务指标所需标注或标准答案 |
| 回归集 | 业务/测试团队 | 包含不可退化的高风险样例 |
| 业务指标定义 | 业务/测试团队 | 明确计算方式、阈值和风险等级 |
| 硬件环境 | 基础设施团队 | GPU/昇腾型号、驱动、CANN/CUDA 版本 |

推荐的数据资产目录由上游团队维护，本项目只约定接口，例如：

```text
dataset_manifest.json
calibration.jsonl
evaluation.jsonl
regression.jsonl
```

每条样本至少应包含唯一 ID、输入文本、可选上下文、期望输出或评估所需字段。具体字段由业务团队和评测团队共同确定。

## 4. 金融场景风险分级

不同业务风险等级使用不同的压缩策略。

| 等级 | 典型场景 | 推荐方案 |
|---|---|---|
| L1 低风险 | 内部 FAQ、IT 运维问答 | 可尝试 INT4 和较高剪枝率 |
| L2 中风险 | 客户经理助手、产品条款辅助、文档抽取 | 混合 INT4/INT8，谨慎剪枝 |
| L3 高风险 | 授信政策、监管制度、反洗钱辅助 | 优先 FP16/INT8，通常不做激进剪枝 |

压缩配置必须与业务风险等级绑定，不能只维护一个“全局最优模型”。

## 5. 系统架构

```text
上游提供模型和评测资产
          │
          ▼
输入校验与版本登记
          │
          ▼
FP16/BF16 基线推理
          │
          ├── AWQ / GPTQ / SmoothQuant / FP8
          ├── 逐层、逐模块敏感性分析
          ├── 混合精度搜索
          └── 结构化剪枝 / N:M 稀疏
          │
          ▼
金融业务质量回归
          │
          ▼
NVIDIA / Ascend 后端适配
          │
          ▼
性能、成本、质量和稳定性报告
          │
          ▼
灰度部署与版本回滚
```

## 6. 核心功能设计

### 6.1 基线与统一 Benchmark

建立统一的推理和评测接口，保证不同压缩方案使用相同的输入、生成参数和硬件条件。

需要记录：

- 模型版本和权重哈希
- 推理框架及版本
- 量化配置
- 硬件型号和数量
- CUDA/CANN/驱动版本
- batch size 和并发数
- 输入输出 token 数
- TTFT、单 token 延迟和端到端延迟
- 显存峰值和吞吐
- 业务质量指标

### 6.2 量化方案

至少比较以下配置：

| 方案 | 用途 |
|---|---|
| FP16/BF16 | 质量和性能基线 |
| INT8 SmoothQuant | 高质量、较稳妥的 W8A8 方案 |
| AWQ INT4 | 常见 W4A16 权重量化方案 |
| GPTQ INT4 | AWQ 的离线量化对照方案 |
| FP8 | NVIDIA 新一代硬件上的候选方案 |
| 混合精度 | 对敏感层进行 INT8/FP16 保护 |

推荐优先验证以下配置：

```text
配置 A：全 INT8
配置 B：全 INT4
配置 C：普通层 INT4 + 敏感层 INT8
配置 D：普通层 INT4 + 敏感层 FP16
```

金融场景中，Embedding、LayerNorm、lm_head、输入输出附近层以及业务敏感层不应默认强制压成 INT4。

### 6.3 剪枝方案

优先实现能被实际硬件利用的方案：

- Attention head 结构化剪枝
- MLP 中间维度剪枝
- Transformer 层剪枝
- KV head 剪枝
- NVIDIA 支持条件下的 2:4 N:M 稀疏

建议按 5%、10%、15%、20% 逐步实验，观察质量拐点。非结构化剪枝可以作为对照实验，但不能默认认为零值增加就会降低线上延迟。

### 6.4 敏感性分析

通过逐层或逐模块替换精度，测量恢复的业务质量：

```text
全 INT4 模型
    ↓
仅将第 i 层改为 INT8
    ↓
重新执行金融业务评估
    ↓
计算该层的质量恢复量
```

建议输出：

- Layer sensitivity ranking
- Attention/MLP 敏感性排名
- 各层量化误差
- 激活异常值统计
- 数字字段和结构化输出的退化情况
- 推荐的 INT4/INT8/FP16 分层配置

### 6.5 质量退化定位

退化分析应同时从业务、模块和硬件三个维度进行。

业务维度：

- 政策问答错误
- 金额错误
- 利率错误
- 日期和期限错误
- 条款遗漏
- 引用错误
- JSON 格式错误
- 不当金融建议
- 应拒答问题被错误回答

模型维度：

- Attention 或 MLP
- Embedding
- lm_head
- LayerNorm
- KV Cache
- 特定 Transformer 层

硬件维度：

- 量化 kernel 是否真正生效
- 是否发生 FP16 回退
- 是否存在权重格式转换
- 是否发生算子不支持
- 是否存在多卡通信瓶颈

## 7. 硬件适配方案

### 7.1 NVIDIA GPU

建议分层使用：

```text
研究验证：PyTorch + Transformers
通用部署：vLLM
极致性能：TensorRT-LLM
```

重点验证：

- AWQ/GPTQ INT4 kernel
- INT8 kernel
- FP8
- KV Cache 量化
- Tensor Parallel
- Continuous Batching
- N:M 稀疏 kernel
- 高并发 P95 延迟

### 7.2 华为昇腾

根据实际芯片和软件版本选择 CANN、ACL、ATB、MindIE 或对应的 vLLM-Ascend 适配方案。项目必须单独维护昇腾后端配置，不假设 NVIDIA 的量化权重格式能直接获得相同性能。

需要登记：

- Ascend 芯片型号
- 驱动版本
- CANN 版本
- MindIE/ATB/推理框架版本
- 支持的数据类型
- 支持的算子
- 算子回退情况
- 多卡通信配置

NVIDIA 和昇腾可以使用不同的最佳模型配置，只要业务质量约束相同且结果可审计。

## 8. 评估指标

### 8.1 业务质量

- 政策问答准确率
- 事实性和引用准确率
- 金额、利率、日期、期限 Exact Match
- 金融字段抽取 F1
- JSON 输出成功率
- 长上下文任务准确率
- 知识库外问题拒答率
- 高风险回归集通过率

### 8.2 推理性能

- 显存峰值
- KV Cache 显存
- TTFT
- 单 token 延迟
- 端到端 P50/P95/P99 延迟
- tokens/s
- batch throughput
- 最大稳定并发数
- 多卡扩展效率

### 8.3 成本

至少计算：

```text
每百万 token 成本
单实例月度成本
目标并发下所需卡数
量化前后资源节省比例
```

## 9. 推荐实验矩阵

| 编号 | 配置 | 目的 |
|---|---|---|
| E0 | FP16/BF16 | 建立基线 |
| E1 | INT8 SmoothQuant | 观察稳妥量化效果 |
| E2 | AWQ INT4 | 观察 W4A16 效果 |
| E3 | GPTQ INT4 | 对比离线量化算法 |
| E4 | INT4 + 敏感层 INT8 | 验证混合精度 |
| E5 | 结构化剪枝 5% | 观察轻度剪枝 |
| E6 | 结构化剪枝 10% | 观察实际收益 |
| E7 | 剪枝 + INT8 | 组合优化 |
| E8 | 剪枝 + AWQ INT4 | 高压缩候选 |
| E9 | 混合精度 + 剪枝 | 最终候选方案 |
| E10 | NVIDIA 部署 | GPU 真实性能 |
| E11 | Ascend 部署 | 昇腾真实性能 |

## 10. 项目实施计划

### 第 1～2 周：范围和基线

- 确认银行业务场景和风险等级
- 接收并校验模型、校准集、评估集、回归集
- 搭建 FP16/BF16 推理服务
- 建立统一 benchmark
- 固化硬件和软件版本

交付物：基线报告、输入接口文档、环境锁定文件。

### 第 3～5 周：定制量化

- 接入 AWQ、GPTQ、SmoothQuant
- 对比 INT4、INT8、FP8
- 对比 group size 和量化粒度
- 生成可部署的量化权重
- 执行金融业务回归

交付物：量化模型、量化配置、质量对比报告。

### 第 6～7 周：退化定位和混合精度

- 完成逐层、逐模块敏感性分析
- 分析金额、利率、日期和条款类错误
- 生成 INT4/INT8/FP16 混合配置
- 建立自动化退化报告

交付物：敏感层排名、混合精度配置、退化定位报告。

### 第 8～10 周：剪枝和组合优化

- 实现结构化剪枝
- 测试不同剪枝率
- 验证 N:M 稀疏的硬件收益
- 测试剪枝与量化组合
- 计算质量与成本 Pareto 曲线

交付物：剪枝模型、组合优化报告、候选部署配置。

### 第 11～12 周：NVIDIA 适配

- 接入 vLLM 和 TensorRT-LLM
- 验证量化 kernel
- 完成并发、延迟和吞吐压测
- 计算单百万 token 成本

交付物：NVIDIA 镜像、部署脚本、性能报告。

### 第 13～14 周：昇腾适配

- 接入 CANN/MindIE/对应推理后端
- 完成模型转换或权重适配
- 验证算子和数据类型支持
- 完成昇腾质量与性能压测

交付物：昇腾适配层、部署脚本、跨硬件对比报告。

### 第 15～16 周：交付和工程化

- 一键运行完整优化流程
- 固化模型和配置版本
- 增加模型回滚机制
- 生成自动化报告
- 编写部署和运维文档

交付物：项目代码、容器、报告、操作手册和最终演示。

## 11. 项目目录结构

```text
banking-llm-optimizer/
├── configs/
│   ├── business_scenarios.yaml
│   ├── risk_levels.yaml
│   ├── quantization.yaml
│   ├── pruning.yaml
│   └── hardware/
│       ├── nvidia.yaml
│       └── ascend.yaml
├── inputs/
│   ├── model_manifest.json
│   ├── dataset_manifest.json
│   └── evaluation_manifest.json
├── src/
│   ├── input_validation/
│   ├── quantization/
│   ├── pruning/
│   ├── sensitivity/
│   ├── backends/
│   │   ├── nvidia/
│   │   └── ascend/
│   ├── evaluation/
│   ├── benchmark/
│   ├── cost_model/
│   ├── reporting/
│   └── pipeline.py
├── scripts/
│   ├── run_baseline.sh
│   ├── run_quantization.sh
│   ├── run_sensitivity.sh
│   ├── run_pruning.sh
│   ├── benchmark_nvidia.sh
│   └── benchmark_ascend.sh
├── tests/
├── reports/
├── Dockerfile.nvidia
├── Dockerfile.ascend
└── README.md
```

`inputs/` 中只存放上游资产的 manifest 和路径，不在本项目中保存或处理原始业务数据。

## 12. 验收标准

### 低风险场景候选目标

- 显存占用降低至少 50%
- 业务质量下降不超过 1%
- P95 延迟降低至少 20%
- 吞吐提升至少 20%

### 中风险场景候选目标

- 显存占用降低至少 30%
- 业务质量下降不超过 0.5%
- 金额、利率、日期字段准确率不低于 FP16 基线
- 高风险回归集无关键错误

### 高风险场景候选目标

- 优先采用 INT8 或混合精度
- 关键回归样例不得出现政策、金额、利率和期限错误
- 量化配置、模型版本和评测结果可追溯
- 支持完整回滚

所有指标最终以业务团队和风险管理团队确认的口径为准。

## 13. 最终演示

建议提供统一命令：

```bash
python -m banking_optimizer.pipeline \
  --model /models/bank-assistant \
  --business loan_policy \
  --input-manifest inputs/evaluation_manifest.json \
  --hardware nvidia \
  --risk-level high \
  --max-quality-drop 0.5
```

系统输出：

```text
Recommended deployment profile

Business: loan_policy
Risk level: high
Quantization: INT8
Sensitive layers: FP16
Pruning: disabled
KV Cache: FP16
Backend: TensorRT-LLM

Quality:
- Policy QA accuracy: 98.7%
- Numeric exact match: 99.8%
- Citation accuracy: 97.9%
- Critical regression cases: 100%

Performance:
- Memory reduction: 38%
- P95 latency reduction: 24%
- Throughput improvement: 31%
- Cost per 1M tokens: -35%
```

切换 `--hardware ascend` 后，系统可以输出昇腾平台的独立推荐配置。两个硬件平台使用不同量化格式或不同混合精度策略是允许的，只要满足相同的金融业务质量约束。

## 14. 项目核心价值

本项目的核心不是重新实现 AWQ 或 GPTQ，而是建立一套面向银行业务的优化闭环：

```text
接收标准化模型和评测资产
→ 量化和剪枝实验
→ 敏感层分析
→ 金融质量回归
→ NVIDIA/昇腾适配
→ 成本与性能评估
→ 生成可审计部署配置
```

项目完成后，可以形成一个明确的工程能力组合：

- 了解量化、剪枝和混合精度原理
- 能定位压缩导致的业务质量退化
- 能针对硬件选择推理后端
- 能把离线压缩收益转化为线上成本收益
- 能在金融高风险场景下控制模型变更风险
- 能交付可复现、可评估、可回滚的部署方案
