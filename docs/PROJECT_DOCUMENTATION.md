# OptimizationSystem 项目文档总览

> 本文档按项目执行顺序合并 docs/ 下除 TODO.md 外的 Markdown 文档。原始 Markdown 文件已删除，后续执行记录统一追加于本文档末尾。

## 目录

1. [1. 业务范围](#section-1) — `BUSINESS_SCOPE.md`
2. [2. 模型输入](#section-2) — `MODEL_INPUTS.md`
3. [3. 评测资产输入](#section-3) — `DATASET_INPUTS.md`
4. [4. 数据集与 Colab](#section-4) — `DATASETS.md`
5. [5. 硬件环境输入](#section-5) — `HARDWARE_INPUTS.md`
6. [6. 基线输入接口](#section-6) — `INPUT_INTERFACE.md`
7. [7. FP16/BF16 基线服务](#section-7) — `BASELINE_SERVICE.md`
8. [8. FP16 业务质量基线](#section-8) — `BASELINE_QUALITY.md`
9. [9. 统一 Benchmark](#section-9) — `BENCHMARK.md`
10. [10. 环境锁定](#section-10) — `ENVIRONMENT_LOCK.md`
11. [11. 第 5 章阶段验收](#section-11) — `STAGE5_ACCEPTANCE.md`
12. [12. 量化框架接入](#section-12) — `QUANTIZATION_FRAMEWORK.md`
13. [13. AWQ 实验](#section-13) — `AWQ_EXPERIMENT.md`
14. [14. AWQ 质量、回归与性能评估](#section-14) — `AWQ_EVALUATION.md`
15. [15. GPTQ 实验](#section-15) — `GPTQ_EXPERIMENT.md`
16. [16. INT8/FP8 实验](#section-16) — `INT8_FP8_EXPERIMENT.md`
17. [17. INT8/FP8 质量评测](#section-17) — `INT8_FP8_EVALUATION.md`
18. [18. 统一量化实验 Notebook](#section-18) — `QUANTIZATION_EXPERIMENTS_COLAB.md`
19. [19. 量化结果整理](#section-19) — `QUANTIZATION_COMPARISON.md`
20. [20. 第 6 章阶段验收](#section-20) — `STAGE6_ACCEPTANCE.md`
21. [21. 量化方案推荐](#section-21) — `QUANTIZATION_RECOMMENDATION.md`
22. [22. FP16 基线质量诊断](#section-22) — `BASELINE_QUALITY_DIAGNOSIS.md`
23. [23. Prompt 与确定性计算](#section-23) — `QUALITY_REPAIR_6_7_2.md`

<a id="section-1"></a>

## 1. 业务范围

> 来源：`BUSINESS_SCOPE.md`

### 多业务场景范围

场景组合配置位于 `configs/business_scope.json`，按照 README 风险分级同时维护：

| 场景 | 风险 | 状态 | 压缩策略 |
|---|---|---|---|
| 内部金融 FAQ | L1 | `planned` | 可尝试 INT4 和较高剪枝率 |
| 内部财报数值问答 | L2 | `active` | INT4/INT8 混合精度，谨慎剪枝 |
| 授信政策与监管制度辅助问答 | L3 | `planned` | 优先 FP16/INT8，不做激进剪枝 |

当前只有 L2 财报数值问答具备 FinQA 评测资产，因此它是主场景。L1/L3 已完成范围规划，
但在各自评测资产通过输入验收前不得标记为 `active`。

核心约束：

- 每个场景独立配置允许输出、人工复核、质量门槛和排除项；
- 风险越高，允许的量化和剪枝策略越保守；
- 所有场景均要求输入依据，高风险回归不允许新增关键错误；
- L2 数值准确率相对 FP16 下降不超过 0.5 个百分点，解析率不低于 99%；
- L3 不允许 INT4 或激进剪枝，所有决策相关输出必须人工复核。

执行验收：

```bash
python -m src.input_validation.business_scope \
  --config configs/business_scope.json \
  --report out/business_scope/validation.json
```

配置发生业务边界、风险等级或门槛变化时，必须重新确认并生成报告。

---

<a id="section-2"></a>

## 2. 模型输入

> 来源：`MODEL_INPUTS.md`

### 模型输入校验

#### 1. 目标

本工具对应 `docs/TODO.md` 4.1，只验证上游交付的模型资产，不下载、修改或重新打包模型。
校验报告逐项覆盖权重、配置与 Tokenizer、许可证与内部使用范围、版本与哈希、半精度加载、
目标推理框架冒烟推理。

#### 2. 准备 Manifest

复制 `inputs/model_manifest.example.json` 为实际 manifest，并填写真实模型信息。所有相对路径都
相对于 manifest 所在目录解析。`weight_files` 必须列出所有权重分片；许可证文件应放在模型目录
内或使用相对于模型目录的路径。

`expected_sha256` 可留空，此时工具会计算并记录每个权重文件的 SHA-256。若上游提供了哈希，
按权重相对路径填写，工具会验证是否一致：

```json
{
  "expected_sha256": {
    "model.safetensors": "真实的 SHA-256"
  }
}
```

只有许可证负责人已经确认后，才能把 `internal_use_approved` 设置为 `true`，并在
`usage_scope` 中写明获批范围。

#### 3. 静态检查

```bash
python -m src.input_validation.model_input \
  --manifest inputs/model_manifest.json \
  --report reports/model_input/static.json
```

静态检查不会加载模型，加载和推理两项显示为 `not_run`。命令退出码为 `0` 表示已执行的检查
没有失败；`complete` 仍为 `false`，直到运行时检查通过。

#### 4. 加载与推理检查

先在目标硬件环境安装与模型匹配的 PyTorch、Transformers 和厂商运行时，然后执行：

```bash
python -m src.input_validation.model_input \
  --manifest inputs/model_manifest.json \
  --runtime-check \
  --report reports/model_input/runtime.json
```

运行时检查只读取本地模型，并生成一个 token。`runtime.dtype` 仅接受 `float16` 或
`bfloat16`；`runtime.model_class` 支持 `causal_lm` 和 `seq2seq_lm`。只有报告中六项均为
`pass` 且 `complete` 为 `true`，才能勾选 Todo 4.1。

---

<a id="section-3"></a>

## 3. 评测资产输入

> 来源：`DATASET_INPUTS.md`

### 评测资产输入校验

本流程对应 `docs/TODO.md` 4.2。`prepare_assets()` 生成的 `manifest.json` 会记录校准集、
开发评估集、最终评估集和高风险回归集的相对路径、版本、样本数、必需字段与 SHA-256，
并声明数值指标接口及关键错误判定规则。

执行校验：

```bash
python -m src.input_validation.dataset_input \
  --manifest /path/to/datasets/manifest.json \
  --report out/dataset_input/validation.json
```

校验器会实际加载四类资产，并检查：

- 路径和版本是否完整；
- 校准集与评估集是否无问题重叠；
- 高风险回归集是否已从最终测试集移除；
- 每条样本是否符合声明字段；
- 指标和回归判定 callable 是否可导入；
- 样本数量和内容哈希是否与 manifest 一致。

只有报告中九项全部为 `pass` 且 `complete` 为 `true`，才能勾选 Todo 4.2。

---

<a id="section-4"></a>

## 4. 数据集与 Colab

> 来源：`DATASETS.md`

### 数据集与 Colab 使用说明

#### 1. 用途

本项目使用公开金融问答数据验证量化、剪枝、混合精度和推理后端切换是否造成质量退化。

项目不负责原始数据采集、清洗、脱敏、标注或金融业务规则定义。这里的代码只负责下载、缓存、划分、格式转换和评测输入准备。

#### 2. 推荐数据组合

| 项目用途 | 数据来源 | 默认划分 | 建议规模 |
|---|---|---|---:|
| 量化校准集 | `czyssrs/FinQA` | `train` 随机抽样 | 1,024 |
| 质量调优集 | `czyssrs/FinQA` | `dev` | 883 |
| 最终质量评估集 | `czyssrs/FinQA` | `test` 去除回归重叠 | 约 1,147 |
| 数值高风险回归集 | `Aiera/finqa-verified` | `test` | 91 |

FinQA 包含财务报告文本、表格、问题和数值推理答案，适合验证数字、比例、增长率和表格理解能力。Aiera 的 verified 子集只有 91 条，但每条样本经过人工核验，适合作为第一版固定回归集。

这套数据主要覆盖英文财报数值推理，不等价于真实银行中文业务数据，也不覆盖政策版本、隐私泄露、拒答和工具调用等场景。项目后续接入中文或内部脱敏资产时，应保留相同的输入接口。

#### 3. 数据职责边界

##### 本项目负责

- 使用 Hugging Face `datasets` 下载数据
- 将数据缓存到 Colab Google Drive
- 从 `train` 抽取校准集
- 保持校准集、调优集和最终测试集分离
- 抽样前排除与开发集、最终测试集和高风险回归集重复的问题
- 去除高风险回归集与最终测试集的重复问题
- 生成量化所需 Prompt
- 保存版本、来源和样本数量

##### 本项目不负责

- 修改原始金融答案
- 将公开数据宣称为银行生产数据
- 生产业务标签或专家标准答案
- 对公开财报内容进行合规认证
- 使用真实客户数据

#### 4. Colab 快速开始

打开 [`notebooks/finqa_assets_colab.ipynb`](../notebooks/finqa_assets_colab.ipynb)，在 Google Colab 中运行所有单元格即可。

也可以在 Colab 单元格中直接执行：

```python
!pip -q install -U datasets huggingface_hub pandas pyarrow

from google.colab import drive
drive.mount("/content/drive")

from src.data.finqa_assets import prepare_assets

assets = prepare_assets(
    output_dir="/content/drive/MyDrive/banking_llm_project/datasets",
    calibration_size=1024,
    seed=42,
)

print(assets.summary())
```

如果 Colab 不能导入项目源码，可先运行：

```python
!git clone <你的仓库地址> /content/OptimizationSystem
%cd /content/OptimizationSystem
```

然后再安装依赖并执行 Notebook。

项目直接读取 FinQA 官方仓库的 JSON 文件，不依赖已被 `datasets` 4.x 移除的 Hugging Face
数据集脚本；Aiera verified 子集同样通过 Hub 的 Parquet 导出读取。更新项目源码后如果当前
运行时已经导入过旧版 `finqa_assets.py`，需要重启运行时，再从头执行 Notebook。

#### 5. 数据下载与缓存

`prepare_assets()` 第一次运行会从 Hugging Face 下载数据，之后优先使用 `output_dir/hf_cache` 中的缓存。建议将 `output_dir` 放在 Google Drive，否则 Colab 运行时重置后需要重新下载。

```python
from src.data.finqa_assets import load_assets

assets = load_assets(
    "/content/drive/MyDrive/banking_llm_project/datasets"
)

print(len(assets.calibration))
print(len(assets.quality_dev))
print(len(assets.quality_test))
print(len(assets.high_risk_regression))
```

#### 6. 三套数据的使用规则

```text
calibration
    → AWQ/GPTQ/SmoothQuant 计算量化参数

quality_dev
    → 选择 group size、敏感层和混合精度

quality_test
    → 配置冻结后生成最终质量报告

high_risk_regression
    → 每一个模型版本都执行，统计新增关键错误
```

不要把 `quality_test` 用来反复调参。调参完成后，才使用它生成最终报告。

#### 7. FinQA Prompt 格式

量化校准只输入上下文和问题，不把答案放进 Prompt：

```text
You are a financial analysis assistant.

Financial context:
...

Financial table:
...

Question:
...

Answer:
```

在真正接入 AWQ/GPTQ 时，使用 `assets.calibration_prompts`。

#### 8. 回归判定

Aiera Verified 的答案主要是数字，建议将模型输出解析为数字后再比较，而不是只使用字符串相似度。对关键样本至少记录：

- FP16 是否正确
- 量化模型是否正确
- 剪枝模型是否正确
- 误差绝对值
- 是否为新增退化

最重要的统计项是：

```text
新增退化 = FP16 正确且压缩模型错误
```

#### 9. 来源和许可证

- [FinQA 官方仓库](https://github.com/czyssrs/FinQA)：包含数据说明和 MIT License。
- [Hugging Face dreamerdeo/finqa](https://huggingface.co/datasets/dreamerdeo/finqa)：旧版便捷镜像；因依赖数据集脚本，当前代码不再直接加载。
- [Hugging Face Aiera/finqa-verified](https://huggingface.co/datasets/Aiera/finqa-verified)：91 条人工核验样本，页面标示 MIT License。

运行时建议记录数据集版本、commit/revision、下载时间和样本哈希。即使个人非商用，也不要将数据重新打包进公开仓库或 Docker 镜像；使用前应阅读对应仓库的最新数据卡和许可证。

#### 10. 目录结构

```text
datasets/
├── hf_cache/
├── calibration_finqa_1024/
├── quality_dev_finqa/
├── quality_test_finqa/
├── regression_finqa_verified/
└── manifest.json
```

---

<a id="section-5"></a>

## 5. 硬件环境输入

> 来源：`HARDWARE_INPUTS.md`

### 硬件环境输入校验

本流程对应 `docs/TODO.md` 4.3。NVIDIA 与昇腾环境必须分别在真实目标机器采集，不能使用
模拟信息代替。环境 manifest 保存在上游资产目录，合并验收报告保存在项目 `out/`。

#### NVIDIA 采集

```bash
python -m src.input_validation.hardware_input collect \
  --target nvidia \
  --output /path/to/hardware/nvidia.json
```

#### 昇腾采集

```bash
python -m src.input_validation.hardware_input collect \
  --target ascend \
  --output /path/to/hardware/ascend.json
```

自动采集无法识别的昇腾芯片、显存、CANN、驱动、固件或通信信息，需要根据 `npu-smi`、
CANN 安装信息和部署环境补入 manifest，不能猜测版本。

#### 合并验收

```bash
python -m src.input_validation.hardware_input validate \
  --nvidia-manifest /path/to/hardware/nvidia.json \
  --ascend-manifest /path/to/hardware/ascend.json \
  --report out/hardware/validation.json
```

只有七项全部为 `pass` 且 `complete` 为 `true`，才能勾选 Todo 4.3。单独完成 Colab
NVIDIA 采集不会完成 4.3，昇腾环境仍是必要门禁。

---

<a id="section-6"></a>

## 6. 基线输入接口

> 来源：`INPUT_INTERFACE.md`

### 基线输入接口

当前激活场景是 L2 内部财报数值问答。单个请求使用 `InferenceRequest`，字段如下：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `question` | string | 是 | 财报数值问题 |
| `pre_text` | list[string] | 是 | 表格前文本上下文 |
| `post_text` | list[string] | 是 | 表格后文本上下文 |
| `table` | list[list] | 是 | FinQA 财务表格 |
| `request_id` | string | 否 | 审计和性能日志标识；未提供时自动生成 |

`InferenceRequest.from_finqa_row` 可直接接收 FinQA 行记录，并通过 `finqa-v1` 模板生成统一
Prompt。输入在 tokenizer 阶段按 `configs/baseline.json` 的 `max_input_tokens` 截断。生成参数、
精度、设备和随机种子也全部来自该配置。

输出包含 `request_id`、`success`、`output_text` 和请求级 `metrics`。性能日志不保存完整输入
内容，只记录字符数、token 数、延迟、吞吐、模型版本、精度与设备。

---

<a id="section-7"></a>

## 7. FP16/BF16 基线服务

> 来源：`BASELINE_SERVICE.md`

### FP16/BF16 基线服务

基线服务配置位于 `configs/baseline.json`，当前绑定激活的 L2 财报数值问答场景。服务使用
统一 FinQA Prompt、固定生成参数和随机种子，支持单请求与批量推理，并将每个请求的输入/
输出 token、延迟、吞吐、模型 commit、精度和设备写入 JSONL。

Colab Notebook 会在真实 Qwen2.5-0.5B-Instruct 和 T4 上执行：

1. FP16 模型加载；
2. 单请求推理；
3. 两条样本批量推理；
4. 相同请求确定性重放；
5. 模型与运行环境登记；
6. 七项基线服务验收。

结果目录：

```text
out/baseline/
├── baseline_config.json
├── baseline_evidence.json
├── request_logs.jsonl
└── validation.json
```

只有 `validation.json` 中七项均为 `pass`，才能勾选 Todo 5.2。

---

<a id="section-8"></a>

## 8. FP16 业务质量基线

> 来源：`BASELINE_QUALITY.md`

### FP16 业务质量基线

质量基线使用版本化的 FinQA development 集，共 883 条样本。该集合只用于开发阶段的 FP16
基线和后续压缩方案对比，不使用最终 test 集。

Colab 会以固定 FP16 模型、统一 Prompt 和生成参数进行批量推理，并输出：

```text
out/baseline_quality/
├── baseline_predictions.jsonl
├── baseline_quality_report.json
├── quality_service.jsonl
└── validation.json
```

主指标为 `numeric_accuracy`，同时要求 `parse_rate >= 0.99`。FP16 准确率作为后续 INT8、
INT4 和剪枝方案的相对质量基准；当前阶段不为 FP16 准确率虚构额外绝对门槛。

质量配置可包含独立的 `prompt_suffix` 和 `generation` 覆盖项，用于诊断输出截断或格式问题。
这类实验必须写入新的结果目录，不能覆盖原始 FP16 基线结果。

---

<a id="section-9"></a>

## 9. 统一 Benchmark

> 来源：`BENCHMARK.md`

### 统一 FP16/BF16 Benchmark

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

---

<a id="section-10"></a>

## 10. 环境锁定

> 来源：`ENVIRONMENT_LOCK.md`

### 环境锁定

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

---

<a id="section-11"></a>

## 11. 第 5 章阶段验收

> 来源：`STAGE5_ACCEPTANCE.md`

### 第 5 章 NVIDIA 基线阶段验收

阶段验收范围为当前实际可用的 Tesla T4 / CUDA / Transformers FP16 环境。真实昇腾、CANN
和基础容器构建仍由 Todo 4.3 与 5.4 的未完成门禁控制，不会被 NVIDIA 阶段验收掩盖。

验收汇总以下证据：模型输入、业务范围、数据资产、FP16 服务、完整开发集质量、性能
Benchmark 和 NVIDIA 环境锁定。结果写入 `out/stage5/validation.json`。

---

<a id="section-12"></a>

## 12. 量化框架接入

> 来源：`QUANTIZATION_FRAMEWORK.md`

### 量化框架接入（6.1）

6.1 建立 AWQ、GPTQ 和 SmoothQuant 的统一配置、导出和日志契约。它不等于已经完成真实量化；
真实权重导出和业务评测分别在 6.2～6.4 完成。

统一配置位于 `configs/quantization.json`，每个方案声明：

- `method`、权重量化 bit、激活量化 bit 和 group size；
- 校准资产角色及其版本/哈希；
- 后端包名和输出目录。

每个真实方案输出目录必须包含：

```text
quantization_config.json
quantized_model_manifest.json
quantization_events.jsonl
```

当前 Colab 单元只采集后端包版本和配置契约。没有安装后端包或没有导出真实权重时，报告会
明确标记 `real_quantization_execution: not_run`，不会伪造量化完成状态。

GPTQ 实验使用仍在活跃维护的 `gptqmodel`；`auto-gptq` 仅作为旧环境兼容版本继续采集。

---

<a id="section-13"></a>

## 13. AWQ 实验

> 来源：`AWQ_EXPERIMENT.md`

### AWQ 实验（6.2）

6.2 使用与 FP16 基线隔离的 `calibration_finqa_1024`，在 NVIDIA/T4 上运行
AutoAWQ INT4 W4A16。实验矩阵固定为 group size 32、64、128；每个矩阵点单独导出，避免
覆盖结果。

配置位于 `configs/awq.json`，真实执行入口为
`src.quantization.awq.run_awq_plan`。每个计划的结果目录必须包含：

```text
quantization_config.json
quantized_model_manifest.json
quantization_events.jsonl
```

Colab 入口见 `notebooks/awq_experiment_colab.ipynb`。模型仍从
`/content/drive/MyDrive/banking_llm_project/model_input_real` 读取；完整量化模型写入
`/content/drive/MyDrive/banking_llm_project/awq`，只有配置、manifest 和 JSONL 事件日志写入
`/content/drive/MyDrive/OptimizationSystem/out/awq`。

如果已经完成过一次旧版导出，可先运行 Notebook 中的迁移单元；它会移动已有的权重、tokenizer
和模型配置，并更新 manifest，不需要重新执行量化。

AutoAWQ 未安装或某个矩阵点未执行时，验收报告会标记 `not_run`，不会把计划配置误报为真实
量化权重。质量、回归和性能数据需要在真实权重导出后继续运行，不能由配置验收替代。

---

<a id="section-14"></a>

## 14. AWQ 质量、回归与性能评估

> 来源：`AWQ_EVALUATION.md`

### AWQ 质量、回归和性能评估

导出验收通过后，运行 `notebooks/awq_quality_benchmark_colab.ipynb`。该 Notebook：

1. 使用与 FP16 基线相同的 FinQA development 集，分别评估 group size 32、64、128；
2. 将候选模型逐样本预测与 FP16 预测比较，统计新增数值回归和关键错误；
3. 使用现有统一 Benchmark 矩阵记录 TTFT、单 token 延迟、吞吐和显存。

完整 AWQ 权重从 `banking_llm_project/awq` 加载，评测报告、预测和 JSONL 日志写入
`OptimizationSystem/out/awq/<plan_id>`。6.2 的质量/回归/性能 TODO 只有在这些真实报告回传并检查后才勾选。

---

<a id="section-15"></a>

## 15. GPTQ 实验

> 来源：`GPTQ_EXPERIMENT.md`

### GPTQ 实验（6.3）

6.3 使用与 AWQ 完全相同的 `calibration_finqa_1024`、INT4 W4A16 和 group size
32/64/128。后端使用 Transformers `GPTQConfig` 与 GPTQModel，避免依赖停止活跃维护的
AutoGPTQ 接口。

完整模型写入 `/content/drive/MyDrive/banking_llm_project/gptq`；配置、manifest、事件日志、
质量、回归和 Benchmark 报告写入 `/content/drive/MyDrive/OptimizationSystem/out/gptq`。

先运行 `notebooks/gptq_experiment_colab.ipynb` 完成真实量化导出，再运行
`notebooks/gptq_quality_benchmark_colab.ipynb`。GPTQ 和 AWQ 的真实速度、显存及并发对比必须
使用同一种 GPU；不同硬件的报告只能各自留档，不能计算直接收益。

GPTQModel 7.5.0 安装时可能把 Transformers 降级到 5.17.0，导致
`AutoModelForCausalLM` 延迟导入失败。Notebook 会在安装 GPTQModel 后重新锁定项目已验证的
Transformers 5.18.0；安装单元结束后必须重启 Colab 运行时。

如果第一次运行已生成 `not_run` 日志，直接重新运行 `notebooks/gptq_experiment_colab.ipynb`。
它会在导入检查失败时立即显示完整根因，并自动跳过已经成功导出的 group size。

---

<a id="section-16"></a>

## 16. INT8/FP8 实验

> 来源：`INT8_FP8_EXPERIMENT.md`

### INT8/FP8 实验（6.4）

6.4 使用 NVIDIA ModelOpt 执行 SmoothQuant W8A8，统一记录配置、校准集、导出契约和硬件
能力。Notebook 不会把“后端已安装”误报成“INT8 权重已导出”，也不会把“PyTorch 存在
FP8 dtype”误报成“设备支持 FP8”。FP8 只有在对应设备完成真实 FP8 矩阵运算后才记为
`pass`；未安装后端或未连接对应设备时标记 `not_run`。

运行 `notebooks/int8_fp8_experiment_colab.ipynb`。模型仍从
`banking_llm_project/model_input_real` 读取；完整 INT8 模型写入
`banking_llm_project/int8_fp8`，分析证据写入 `OptimizationSystem/out/int8_fp8`。只有目录中出现
`smoothquant-int8-w8a8/quantized_model_manifest.json` 后，才可以运行 INT8 质量评测 Notebook。

#### 执行顺序

1. 在 NVIDIA GPU Colab 中运行 `notebooks/int8_fp8_experiment_colab.ipynb`。
2. 检查 `out/int8_fp8/validation.json` 中 `smoothquant_real_execution` 为 `pass`。
3. 运行 `notebooks/int8_fp8_evaluation_colab.ipynb`。
4. 检查 `out/int8_fp8/evaluation_validation.json` 的三项检查全部为 `pass`。

实验 Notebook 会保存可由 `modelopt.torch.opt.restore` 恢复的 `modelopt_state.pth`、统一量化
配置、manifest 和事件日志。评测 Notebook 使用相同 FinQA dev 样本对比 FP16 与 INT8 数值
准确率，并对最长输入样本和 JSON 输出样本分别执行 FP16/INT8 对照。

#### 验收口径

- `运行 SmoothQuant INT8`：必须存在真实 `modelopt_state.pth`，且配置、manifest、事件日志齐全。
- `在 NVIDIA 上验证 FP8 可用性`：必须成功执行真实 FP8 矩阵运算；A100 仅暴露 dtype 不算通过。
- `确认 Ascend 对应数据类型支持情况`：必须在真实 NPU 上执行探测，不接受静态推断。
- 三项质量对比：必须生成 `evaluation.json` 和完整通过的 `evaluation_validation.json`。

#### 当前结果

- SmoothQuant W8A8：已在 A100 上完成，ModelOpt 版本为 0.47.0，校准样本数为 1024。
- NVIDIA FP8：A100 上真实 FP8 矩阵运算返回 `NotImplementedError`，保持 `not_run`。
- Ascend FP8：未连接真实 NPU，保持 `not_run`。
- INT8 数值质量：883 条 dev 样本解析率为 0.9604，准确率为 0.0068；FP16 准确率为
  0.0091，绝对下降 0.0023，相对下降约 25%。
- 长上下文：三个最长样本中 FP16 和 INT8 均为 0 个正确，当前模型在该集合上没有有效基线。
- 结构化输出：十个样本中 FP16 通过 2 个，INT8 通过 0 个，存在明确结构化输出退化。
- `evaluation_validation.json` 三项检查全部为 `pass`，代表三类对比已经执行并可追溯，不代表
  INT8 方案满足业务质量门槛。

---

<a id="section-17"></a>

## 17. INT8/FP8 质量评测

> 来源：`INT8_FP8_EVALUATION.md`

### 6.4 质量评测

运行 `notebooks/int8_fp8_evaluation_colab.ipynb` 可对真实导出的 INT8 模型执行：

- FinQA 数字字段准确率和解析率；
- 128/512/2048 token 长上下文响应检查；
- 固定字段的 JSON 结构化输出检查；
- NVIDIA FP8 和 Ascend NPU 能力记录。

模型文件从 `banking_llm_project/int8_fp8` 读取，评测证据写入
`OptimizationSystem/out/int8_fp8`。没有 `quantized_model_manifest.json` 时，Notebook 只生成
`not_run` 说明，不会把 FP16 结果当成 INT8 结果。

---

<a id="section-18"></a>

## 18. 统一量化实验 Notebook

> 来源：`QUANTIZATION_EXPERIMENTS_COLAB.md`

### 统一量化实验 Notebook

推荐使用 `notebooks/quantization_experiments_colab.ipynb` 统一执行 AWQ、GPTQ 和
SmoothQuant/FP8 的导出与环境验收。

质量、回归和 Benchmark 仍建议在各自评测 Notebook 中执行，因为这些步骤耗时较长，便于
单独重跑和排查。统一 Notebook 会自动跳过已有 manifest 的 AWQ/GPTQ 计划，避免重复量化。

模型权重目录：

```text
/content/drive/MyDrive/banking_llm_project/awq
/content/drive/MyDrive/banking_llm_project/gptq
/content/drive/MyDrive/banking_llm_project/int8_fp8
```

分析结果目录：

```text
/content/drive/MyDrive/OptimizationSystem/out/awq
/content/drive/MyDrive/OptimizationSystem/out/gptq
/content/drive/MyDrive/OptimizationSystem/out/int8_fp8
```

---

<a id="section-19"></a>

## 19. 量化结果整理

> 来源：`QUANTIZATION_COMPARISON.md`

### 量化结果整理（6.5）

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

#### 当前结果与限制

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

---

<a id="section-20"></a>

## 20. 第 6 章阶段验收

> 来源：`STAGE6_ACCEPTANCE.md`

### 第 6 章定制量化阶段验收

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

---

<a id="section-21"></a>

## 21. 量化方案推荐

> 来源：`QUANTIZATION_RECOMMENDATION.md`

### 量化方案推荐初稿

#### 结论

当前阶段建议**暂不部署** AWQ、GPTQ 或 SmoothQuant INT8。该结论不是因为量化流程不可用，
而是因为 FP16 基线在 FinQA dev 上的数值准确率仅为 `0.0091`，尚不能代表可用于金融业务的
模型；同时当前量化产物均未取得相对 FP16 的性能收益。

#### 后续研究候选

- 质量优先候选：`awq-int4-w4a16-g32`。解析率为 `1.0`，数值准确率为 `0.0113`，是当前
  质量门禁内结果最好的 INT4 方案，但吞吐相对 FP16 下降约 `41.8%`。
- 性能对照候选：`gptq-int4-w4a16-g64`。在通过解析率门槛的 GPTQ 方案中吞吐最高，但仍比
  FP16 下降约 `28.1%`，数值准确率也低于 FP16。
- 不继续候选：`smoothquant-int8-w8a8`，其解析率低于门槛、结构化输出退化，吞吐下降约
  `74.8%`；`gptq-int4-w4a16-g128` 的解析率低于 `0.99` 门槛。

#### 第 7 章输入

逐层和模块敏感性分析优先使用 AWQ g32，并用 GPTQ g64 作为性能对照。后续只有在业务基线
质量达到生产门槛、量化方案通过完整高风险回归且部署后端产生真实收益时，才重新评估生产
部署推荐。A100 40GB 小时单价尚未批准，因此美元单位 token 成本仍需后补。

---

<a id="section-22"></a>

## 22. FP16 基线质量诊断

> 来源：`BASELINE_QUALITY_DIAGNOSIS.md`

### FP16 基线质量诊断（6.7.1）

运行以下命令分析冻结的 883 条 FP16 FinQA dev 预测：

```bash
python -m src.evaluation.baseline_quality_diagnosis --project-root .
```

输出文件：

- `out/baseline_quality_diagnosis/diagnosis_report.json`
- `out/baseline_quality_diagnosis/diagnosis_report.md`
- `out/baseline_quality_diagnosis/manual_review_50.jsonl`
- `out/baseline_quality_diagnosis/manual_review_50.json`（普通 JSON 数组，适合 JSON 解析器）
- `out/baseline_quality_diagnosis/validation.json`

工具会验证基础数字归一化、按题型统计准确率，并使用确定性规则将错误分为输出解析、答案
抽取潜在误判、计算、公式选择、表格或上下文选择错误。自动分类只用于建立人工复核队列；
需要在 `manual_review_50.jsonl` 的 `manual_review` 字段中填写最终判断，不能把规则建议直接
视为人工结论。当前 50 条队列已完成复核。完成一条复核时，将 `status` 改为 `completed`，填写布尔值
`prediction_correct`、最终 `error_type` 和可选 `notes`。再次运行命令会保留这些字段并更新
完成数量，不会覆盖已经填写的人工结论。

`.jsonl` 是“一行一个 JSON 对象”的机器格式；如果使用要求整个文件只能是一个 JSON 值的
解析器，请打开 `manual_review_50.json`，它是一个标准 JSON 数组。

---

<a id="section-23"></a>

## 23. Prompt 与确定性计算

> 来源：`QUALITY_REPAIR_6_7_2.md`

### 6.7.2 Prompt 与确定性计算

已实现四组可复现实验配置，配置文件为 `configs/quality_repair.json`：

1. `zero_shot_final_answer`：当前数值文本基线。
2. `structured_json`：要求证据、公式、数值和单位的严格 JSON 输出。
3. `few_shot_structured_json`：只使用 calibration/train 样本提供格式示例，禁止使用 dev/test 答案。
4. `structured_json_calculator`：公式由安全 Decimal 计算器复核，不执行 Python `eval`。

JSON 变体在首次输出不符合契约时执行一次确定性格式重试；重试只修复格式，不改变问题、
证据或业务答案。报告会记录 `format_retry_count` 和每条记录的 `format_retry_used`。

工具位于 `src/evaluation/quality_repair.py`，其中 `extract_final_numeric` 优先读取结构化
`value` 或 `Final answer`，避免旧评分器总是抽取文本第一个数字。`parse_structured_output`
会拒绝缺少证据、公式、数值或单位的输出。

真实模型对比仍需在 Colab 使用当前 FP16 模型和 883 条 dev 数据执行。每个 variant 必须
保存原始输出、解析后的数值、JSON 成功率、数值准确率、解析率、结构化输出成功率和性能
指标；few-shot 示例只能来自 calibration/train。

首次 Colab 回传结果使用了旧的 `max_new_tokens=32`，结构化 JSON 大量被截断；同时旧解析器
不支持嵌套 evidence，因此该轮结果不作为最终消融结论。当前配置已将生成长度固定为 `128`
并支持嵌套 JSON；请同步最新代码和配置后重新运行 Notebook。

---

## 后续执行记录

从本节开始，后续 TODO 执行产生的实验结果、分析、修复说明和验收结论统一按时间顺序追加到本总文档末尾；不再为单次执行新建独立 Markdown 文档。

## 6.7.2 实验结果分析（2026-10-04）

本轮使用同一 FP16 模型、883 条 FinQA dev 数据、`max_new_tokens=128`、`do_sample=false` 和
`num_beams=1`，比较四种输出方案。数据集哈希为
`7ba8d507634daa266268414448444421a18868e926632d805949058118b8a6d8`。

| 方案 | 解析率 | 结构化成功率 | 数值准确率 | 正确数 | 重试次数 | 用时 |
|---|---:|---:|---:|---:|---:|---:|
| `zero_shot_final_answer` | 11.89% | 0% | 0.57% | 5/883 | 0 | 17.4 分钟 |
| `structured_json` | **78.71%** | **78.71%** | **1.93%** | **17/883** | 370 | 34.2 分钟 |
| `few_shot_structured_json` | 32.50% | 27.97% | 0.11% | 1/883 | 738 | 63.5 分钟 |
| `structured_json_calculator` | 11.55% | 10.87% | 0.34% | 3/883 | 788 | 49.0 分钟 |

四种方案合计运行约 164.1 分钟。883 条样本先执行四轮基础生成，再产生 1,896 次格式重试，
因此总耗时明显高于普通基线评测。

### 结果解释

1. `structured_json` 是当前最佳方案：相比 FP16 基线的 5 个正确答案提升到 17 个，数值准确率
   从 0.57% 提升到 1.93%，但仍远低于 `minimum_parse_rate=99%` 的门槛，也不能视为业务质量达标。
2. `zero_shot_final_answer` 的准确数与基线相同。基线评测使用“文本中任意数字即可解析”的宽松规则，
   而本方案要求最终答案标记或合法 JSON；模型仍输出解释文本，导致严格解析率从基线 94.68% 降到 11.89%，
   这不是模型能力变好或变坏的直接证据，而是评分入口变严格了。
3. `few_shot_structured_json` 反而最差。两个示例被拼接到每条请求中，0.5B 模型容易复制示例结构或
   示例语义，且额外上下文挤占生成长度；因此重试最多、耗时最长，只有 1 条答案正确。
4. `structured_json_calculator` 没有形成有效收益。只有 3.51% 的样本真正触发计算器，模型大多没有生成
   纯数字算式，仍输出变量名、解释或不完整 JSON；计算器只能校验已经正确提取的公式，不能替代表格定位和推理。
5. 结构化成功不等于答案正确。当前解析器主要验证 JSON 字段和数值可读性，不验证证据是否对应题目、公式是否
   正确或数值是否来自正确表格行，所以 `structured_json` 的 695 条合法结构化输出中只有 17 条数值正确。
6. 多条输出在 128 个生成 token 附近被截断，表现为 JSON 未闭合或只生成了解释开头；格式重试只能修复部分格式，
   不能补回被截断的表格推理。

### 阶段结论

6.7.2 验证了“结构化输出约束”对解析和少量数值正确率有正向作用，但没有解决 0.5B 模型的表格定位、
多步计算和答案可靠性问题。当前应将 `structured_json` 作为后续修复的起点，暂不采用 few-shot 或 calculator
方案作为主方案；下一轮优先缩短输出契约、提高生成长度并增加答案/公式一致性校验，再进行小规模回归后决定是否
重新跑完整 883 条数据。

## 6.7.2 80 条试跑准备（2026-10-04）

为降低完整消融的迭代成本，`configs/quality_repair.json` 已切换为固定 80 条 pilot 配置：

- `evaluation_size=80`、`evaluation_seed=20261004`，运行时从 883 条 dev 中可重复抽样，并在报告中保存原始索引和索引哈希。
- `pilot_variants` 暂时只运行 `zero_shot_final_answer`、现有 `structured_json` 和新增的 `structured_json_compact`。
- `max_new_tokens` 提高到 `256`，用于确认 128 token 截断是否是主要格式失败原因。
- `structured_json_compact` 缩短输出契约，要求单对象、固定四个字段、一条证据字符串和纯数字算式。
- 试跑结果只用于筛选 Prompt，不替代 883 条完整评测，也不改变 6.7.5 的最终门禁。

实现位于 `src/evaluation/quality_repair.py`，会将 pilot 记录写入同一个
`out/quality_repair_6_7_2/comparison_report.json`。Colab 重新同步代码和配置后，直接运行原 Notebook
即可；回传的 80 条结果应继续追加到本文档。

本轮配置同时启用 `show_progress=true` 和 `progress_every_batches=1`。Colab 输出会显示当前方案的
完成数、方案内百分比和所有方案总百分比；完整评测时若不希望逐批输出，可将
`progress_every_batches` 改为 `0`，自动按约 10 个进度节点显示。

## 6.7.2 80 条试跑结果分析（2026-10-04）

本轮使用固定种子 `20261004` 从 883 条 dev 中抽取 80 条，三个方案均使用
`max_new_tokens=256`、`do_sample=false`、`num_beams=1`。

| 方案 | 解析率 | 结构化成功率 | 数值准确率 | 正确数 | 重试次数 | 用时 |
|---|---:|---:|---:|---:|---:|---:|
| `zero_shot_final_answer` | 16.25% | 0% | 2.50% | 2/80 | 0 | 3.0 分钟 |
| `structured_json` | **86.25%** | **86.25%** | 0% | 0/80 | 30 | 4.1 分钟 |
| `structured_json_compact` | 82.50% | 82.50% | **2.50%** | **2/80** | 44 | 7.1 分钟 |

同一批 80 条样本上的原 FP16 基线为 75/80 可解析、0/80 正确，因此 pilot 的正确数仍然很少，
不能用来宣称完整 dev 上的最终提升。

### 结果解释

1. `structured_json` 的格式效果最好：69/80 条输出通过结构化解析，说明 `max_new_tokens=256`
   和一次重试可以把结构化成功率提高到接近 90%，但本轮 69 条中没有一条数值正确，说明格式已经不是
   当前主要瓶颈。
2. `structured_json_compact` 得到 2/80 正确答案，但解析率低于原 `structured_json`（82.50% 对 86.25%），
   重试次数更多（44 对 30），耗时也更长。两条正确答案不足以证明紧凑 Prompt 稳定优于原方案，尤其是样本只有 80 条。
3. `zero_shot_final_answer` 也得到 2/80 正确答案，但只有 13/80 可解析；模型经常输出解释文本，
   严格提取器无法从中找到最终答案标记。这说明自然语言答案偶尔能答对，但不适合作为稳定的服务接口。
4. 输出长度提高到 256 后，结构化方案的主要失败不再是单纯的 JSON 截断，而是模型忽略契约、复制提示词、
   选择错误表格字段或生成错误公式。下一步应优先提升表格证据选择和数值推理能力，而不是继续堆叠格式约束。

### Pilot 结论

本轮不直接选择 `structured_json_compact` 替换原方案。暂时保留原 `structured_json` 作为结构化对照，
将紧凑 Prompt 作为候选；下一步应在相同 80 条上加入更可靠的证据/公式一致性检查，或直接开始 6.7.3
的 1.5B/3B/7B 模型小规模对比。只有候选方案在 pilot 上同时改善解析率和正确数，才值得扩展到完整 883 条评测。

## 6.7.3 模型规模小范围对比准备（2026-10-05）

之前的 0.5B 结果不需要重新运行，继续作为控制组保留。新增
`notebooks/quality_repair_6_7_3_model_scale_colab.ipynb`，在相同的 80 条固定 pilot、相同的
`structured_json`/`structured_json_compact` 配置和相同生成参数下，依次运行
`Qwen2.5-1.5B-Instruct` 与 `Qwen2.5-3B-Instruct`。

两个模型分别写入独立目录：

- `out/quality_repair_6_7_3_model_scale/qwen2.5-1.5b/`
- `out/quality_repair_6_7_3_model_scale/qwen2.5-3b/`

Notebook 会在每个模型结束后释放显存，并输出解析率、结构化成功率、数值准确率和正确数。模型目录默认
使用 `banking_llm_project/model_input_real/qwen2.5-1.5b-instruct` 与
`qwen2.5-3b-instruct`；如果 Google Drive 中目录名不同，只需修改 Notebook 的
`MODEL_CANDIDATES`，不需要重新运行 0.5B。

模型下载说明：新 Notebook 会调用 `huggingface_hub.snapshot_download`，将公开的
`Qwen/Qwen2.5-1.5B-Instruct` 和 `Qwen/Qwen2.5-3B-Instruct` 保存到上述 Google Drive 目录。
目录中已有 `config.json` 时会自动跳过下载；因此第一次运行需要等待下载，之后重跑不会重复下载。

## 4080 本地服务器入口与 Notebook 整理（2026-10-05）

新增本地入口 `scripts/run_quality_repair_4080.py`，不依赖 Google Colab 或 Google Drive，模型和数据均从
服务器本地路径读取。默认配置为单卡安全模式：`batch_size=1`、`max_new_tokens=256`、固定 80 条 pilot、
只运行新的 `evidence_operation` 两阶段方案；需要对照时可传入
`--variants structured_json,evidence_operation`。

服务器环境可先安装 `requirements-server.txt`：

```bash
python3 -m pip install -r requirements-server.txt
```

Pilot 命令示例：

```bash
python3 scripts/run_quality_repair_4080.py \
  --model-path /data/models/Qwen2.5-3B-Instruct \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-dir /data/finqa_assets \
  --output-dir out/quality_repair_3b_4080
```

确认 pilot 有效后，增加 `--full` 执行完整 883 条 dev：

```bash
python3 scripts/run_quality_repair_4080.py \
  --model-path /data/models/Qwen2.5-3B-Instruct \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-dir /data/finqa_assets \
  --output-dir out/quality_repair_3b_4080_full \
  --full
```

如果显存稳定，可以将 `--batch-size` 从 1 调到 2；4080 只有 16GB 显存时不建议一开始使用更大 batch。
入口会保存有效配置、模型 manifest、请求日志、逐条预测和 `comparison_report.json`。

原有 10 个 Colab Notebook 已按 TODO 顺序合并为
`notebooks/optimization_system_all_colab.ipynb`，包含 115 个单元。旧的分散 Notebook 已删除；
本地 4080 运行应优先使用上述 Python 入口。

## 数值准确率低的原因与优化优先级（2026-10-05）

当前低准确率不是单一解析问题，而是“表格定位、公式选择、数值计算、输出格式”连续链路中的多处错误叠加：

1. 模型需要先从长文本和表格中找到正确行列，再判断题目要求的年份或分母，最后完成一到多步计算；
   `0.5B` 模型容量不足，3B 虽有改善但百分比、平均和多步题仍明显薄弱。
2. 结构化 Prompt 只能约束输出形式，不能保证 evidence 对应正确表格单元格，也不能保证 formula 选择正确；
   因此 3B 的 `structured_json` 已达到 95% 解析率，但 pilot 数值准确率仍只有 13.75%。
3. 直接让模型生成精确小数、百分比和负数会放大计算错误；格式重试只能修复 JSON，不能修复业务答案。
4. `structured_json_compact` 在不同模型上不稳定，说明继续压缩格式约束不是主要解决方向。

建议按以下顺序优化：

1. 以 3B + 原始 `structured_json` 作为新 FP16 主基线，先完成完整 883 条评测。
2. 改为两阶段推理：第一阶段只输出带行列/年份的证据和纯数字操作数，第二阶段由安全 Decimal 计算器执行公式；
   模型不再直接承担最终小数运算。
3. 将表格序列化为带唯一 ID 的单元格或行列标记，要求 evidence 只能引用这些 ID，并做 evidence—formula—value
   一致性校验；校验失败进入人工复核或重试。
4. 使用 FinQA train 做 LoRA/监督微调，目标格式固定为“证据、操作数、公式、计算结果、单位”；严禁使用 dev/test
   答案训练，并对百分比、平均、总和和多步样本过采样。
5. 在完整评测前继续用固定 80 条 pilot 做回归；只有解析率和数值正确数同时改善，才扩展到 883 条。
6. 如果 3B 完整评测仍低于 20%，再比较 7B 或 4-bit 7B，而不是继续堆叠 Prompt 变体。

## 6.7.3 模型规模 pilot 结果分析（2026-10-05）

1.5B 和 3B 均使用与 0.5B 完全相同的 80 条样本，样本种子均为 `20261004`，索引哈希均为
`cd2cf6c4ce8c50cbf23c7c007254b060f741a5ec363ee840efd3ea72662b9e61`。

| 模型 | 方案 | 解析率 | 数值准确率 | 正确数 | 重试次数 | 用时 |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | `structured_json` | 86.25% | 0% | 0/80 | 30 | 4.1 分钟 |
| 1.5B | `structured_json` | 91.25% | 11.25% | 9/80 | 56 | 8.8 分钟 |
| 3B | `structured_json` | **95.00%** | **13.75%** | **11/80** | 22 | 7.1 分钟 |
| 1.5B | `structured_json_compact` | **96.25%** | 1.25% | 1/80 | 7 | 2.7 分钟 |
| 3B | `structured_json_compact` | 6.25% | 0% | 0/80 | 79 | 9.6 分钟 |

### 结果解释

1. 原始 `structured_json` 随模型规模明显改善：0.5B 为 0/80，1.5B 为 9/80，3B 为 11/80。说明当前
   主要瓶颈确实包含模型的表格定位和数值推理能力，而不只是输出格式。
2. 3B 是当前最佳候选：解析率达到 95%，数值准确率达到 13.75%，并且比 1.5B 少 34 次格式重试。
   但 80 条结果仍不能代表完整 dev，也尚未达到第 6.7.5 节的 20% 准确率门槛。
3. `structured_json_compact` 不具备跨模型稳定性。它在 1.5B 上解析率较高但只答对 1 条，在 3B 上
   解析率降至 6.25%，79/80 条触发重试，说明更严格、更短的契约反而使 3B 模型大量输出解释或重复提示词。
4. 题型上，3B 在总和类答对 3/6、查找类 4/20、变化类 2/17；百分比/比例类只有 1/30，平均类 1/7。
   百分比、平均和多步计算仍是主要薄弱点，不能只靠增大模型解决。
5. Zero-shot 不是可靠接口：1.5B 仅 1/80 正确且解析率 8.75%，3B 为 0/80 且解析率 0%；后续模型比较应以
   原始 `structured_json` 作为主方案。

### 阶段结论

保留 0.5B 结果作为历史控制组，不需要重跑。下一步优先用 3B 的原始 `structured_json` 在完整 883 条 dev
上运行一次；不再把 `structured_json_compact` 作为主方案。完整评测后再判断 3B 是否达到质量门槛，以及是否需要
对百分比、平均和多步计算样本进行专门 Prompt 或微调。
