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

##### 5.1 `High-risk regression questions have no unique FinQA source context` 报错

执行以下资产构建命令时：

```bash
.venv/bin/python3 -c \
"from src.data.finqa_assets import prepare_assets; prepare_assets('datasets_contextual', calibration_size=1024, seed=42, finqa_revision='main')"
```

如果出现：

```text
ValueError: High-risk regression questions have no unique FinQA source context: [...]
```

这不是 FinQA 下载失败，也不是 Hugging Face 缓存损坏，而是高风险回归样本无法唯一回连到
FinQA `test` 中的原始财报上下文。`Aiera/finqa-verified` 只提供核验后的问题和答案；为了给模型补齐
`pre_text`、`post_text` 和 `table`，`_enrich_regression_rows()` 必须为每条 verified 问题找到唯一的
FinQA test 原题。程序在找不到候选或候选不唯一时主动终止，避免把错误财报或表格接到高风险样本上。

本次失败的 3 条 verified 问题使用 `decimal proportion`，而 FinQA test 对应原题使用 `portion`，例如：

```text
verified: what decimal proportion of total purchase price is related to stock awards?
FinQA:    what portion of total purchase price is related to stock awards?
```

匹配器先尝试标准化后的完整文本匹配，再通过 `_question_content_key()` 去除比例问句模板词进行唯一匹配。
第一次修复已把 `decimal`、`percentage`、`proportion` 和 `ratio` 等词列入
`_QUESTION_TEMPLATE_WORDS`，但漏掉了 FinQA 原题实际使用的 `portion`。因此两边生成的内容键仍然不同：

```text
verified 内容键: total purchase price related stock awards
FinQA 内容键:    portion total purchase price related stock awards
```

服务器上的 `_QUESTION_TEMPLATE_WORDS` 输出若只有 `proportion`、没有 `portion`，就会稳定复现该错误。
修复方式是在 `src/data/finqa_assets.py` 的模板词集合中加入：

```python
"percentage",
"portion",
"proportion",
```

修改后执行：

```bash
.venv/bin/python3 -m unittest tests.test_finqa_assets -v
.venv/bin/python3 -c \
"from src.data.finqa_assets import prepare_assets; prepare_assets('datasets_contextual', calibration_size=1024, seed=42, finqa_revision='main')"
```

如果修改后仍出现完全相同的错误，使用下面的命令确认 Python 实际导入的文件以及运行版本是否包含
`portion`；集群上的 `/userhome/...` 工作副本可能尚未同步本地未提交的修复：

```bash
.venv/bin/python3 -c \
"import src.data.finqa_assets as f; print(f.__file__); print(sorted(f._QUESTION_TEMPLATE_WORDS))"
```

`finqa_revision='main'` 只决定下载哪个 FinQA 数据版本，不会改变本地匹配代码；清理
`datasets_contextual/hf_cache` 也不能修复这一问题。该修复只改变问题到原始上下文的回连方式，
不修改 verified 答案，也不从 FinQA train/dev 引入评测上下文。

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
默认对照 `structured_json` 和新的 `cell_ids_operation` 紧凑方案；也可以通过
`--variants` 选择其他已配置方案。

服务器环境可先安装 `requirements-server.txt`：

```bash
python3 -m pip install -r requirements-server.txt
```

如果服务器还没有模型，可使用 `scripts/download_model_4080.py` 将
`Qwen/Qwen2.5-3B-Instruct` 下载到项目下的默认目录
`models/Qwen2.5-3B-Instruct`：

```bash
python3 scripts/download_model_4080.py
```

下载到其他目录时，将该目录同时传给评测入口的 `--model-path`；评测入口要求该目录直接包含
`config.json` 和模型权重文件。私有仓库可先设置 `HF_TOKEN`，公开的 Qwen 模型不需要 Token。

Pilot 命令示例：

```bash
python3 scripts/run_quality_repair_4080.py \
  --model-path models/Qwen2.5-3B-Instruct \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-dir /data/finqa_assets \
  --output-dir out/quality_repair_3b_4080 \
  --variants structured_json,cell_ids_operation
```

确认 pilot 有效后，增加 `--full` 执行完整 883 条 dev：

```bash
python3 scripts/run_quality_repair_4080.py \
  --model-path models/Qwen2.5-3B-Instruct \
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

## 4080 `evidence_operation` pilot 结果（2026-10-05）

本轮在 NVIDIA 4080 上使用 `Qwen/Qwen2.5-3B-Instruct` FP16、`max_new_tokens=256`、固定种子
`20261004`，从 883 条 dev 中抽取相同的 80 条样本（索引哈希为
`cd2cf6c4ce8c50cbf23c7c007254b060f741a5ec363ee840efd3ea72662b9e61`），对照运行
`structured_json` 和 `evidence_operation`。

| 方案 | 解析率 | 数值准确率 | 正确数 | 结构化成功率 | 计算器使用率 | 重试次数 | 用时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `structured_json` | 93.75% | **15.00%** | **12/80** | 93.75% | 0% | 23 | 306.4 秒 |
| `evidence_operation` | 72.50% | 10.00% | 8/80 | 73.75% | 72.50% | 27 | 311.2 秒 |

本轮不选择 `evidence_operation` 作为主方案。它确实让 58 条样本进入了确定性计算器，但解析率下降
21.25 个百分点，正确数减少 4 条，运行时间基本没有收益。错误主要不是 Decimal 计算，而是模型没有稳定遵守
操作契约：有的输出重复 JSON 和解释文本，有的 `divide` 只给一个操作数，有的百分比题把分子分母反了。
例如一条“2003 年可出售存货占比”样本输出 `[527.7, 384.3]` 和 `divide`，程序得到 `1.3731`，而标准答案
`72.83%` 需要使用 `384.3 / 527.7`；这说明证据顺序和操作方向仍由模型错误决定。

因此当前结论是：

1. 保留 `structured_json` 作为 3B 的当前对照基线，本轮 pilot 的最佳结果为 12/80；
2. 暂不扩展 `evidence_operation` 到完整 883 条，也不把本轮结果解释为质量提升；
3. 下一步优先为表格单元格增加稳定 ID，并限制操作类型和操作数个数；
4. 增加百分比、平均值和除法的操作语义校验，校验失败时只重试证据和操作数；
5. 完成这些约束后，再用同一 80 条样本复测，确认正确数提升后才运行完整 dev。

## 4080 稳定 ID 复测结果（2026-10-05）

第二轮输出位于 `out/quality_repair_3b_4080_v2`。同一批 80 条样本上，原始 `structured_json` 保持
`12/80` 正确、解析率 `93.75%`；带稳定 ID 的 `evidence_operation` 只有 `2/80` 正确、解析率 `6.25%`，
并产生 `75` 条校验失败和 `78` 次重试，用时 `384.7` 秒。

校验失败主要来自完整 evidence-operation 契约过于复杂：模型遗漏 operands 或 unit、引用不存在的 cell ID、
复制了错误的 cell value，或输出 JSON 后继续生成解释。稳定 ID 因此证明了“错误可以被机器拦截”，但没有证明
“模型已经具备稳定生成该契约的能力”。本轮不扩展完整 883 条评测。

下一轮将保留 cell ID 校验和 Decimal 计算，但允许在 evidence 明确、数值数量与操作无歧义时由程序恢复缺失
operands；恢复失败则拒绝计算，不读取解释文本中的数字。目标是在降低输出复杂度的同时保留可追溯性。

## 4080 `cell_ids_operation` v3 结果（2026-10-05）

第三轮结果位于 `out/quality_repair_3b_4080_v3`。`structured_json` 保持 `12/80` 正确、解析率
`93.75%`；紧凑 `cell_ids_operation` 达到 `6/80` 正确、解析率 `71.25%`，校验失败 `23` 条，重试 `37` 次，
用时 `255.5` 秒。紧凑契约相较严格 `evidence_operation` 的 `2/80` 有改善，但仍未超过基线。

当前主要瓶颈已经从重复输出 operands 转为单元格语义选择和操作方向：模型会引用文字或年份单元格、遗漏
operation，或把百分比题的分子和分母顺序反过来。下一步应为 cell ID 附加行名、列名和年份语义提示，并对校验
失败样本启用 `structured_json` 回退；在混合方案超过 `12/80` 前不运行完整 883 条。

## 下一轮代码优化（2026-10-05）

已实现语义化 cell ID 和结构化回退。表格单元格现在会同时显示 `cell_id`、行名、列名和年份，模型仍只需输出
短的 `cell_ids_operation` 契约；如果 cell ID 校验失败、单元格非数字、操作数数量不正确或 JSON 无法解析，程序
会额外请求一次 `structured_json`，并记录 `fallback_used` 和 `validation_error`。这保留了新方案的可追溯校验，
同时避免严格契约失败时直接丢失基线答案。

## 4080 `cell_ids_operation` v4 结果（2026-10-05）

第四轮结果位于 `out/quality_repair_3b_4080_v4`。语义化 cell ID 加回退后，紧凑方案解析率为 `85.00%`、
数值准确率为 `13.75%`、正确数为 `11/80`，校验失败 `21` 条、回退 `21` 次；`structured_json` 仍为
`12/80`、解析率 `93.75%`。新方案已接近基线，但没有超过基线，暂不运行完整 883 条。

紧凑方案直接计算正确 `8` 条，回退额外挽回 `3` 条。剩余主要问题是 operation 缺失、cell ID 仍指向文字或年份
单元格，以及百分比题的分子分母方向错误。下一步应实现题型路由和有限操作方向候选，而不是继续增加 JSON 字段。

## 4080 题型路由优化准备（2026-10-04）

针对 v4 中单元格和运算方向选择错误的问题，新增 `routed_cell_ids_operation` 方案。它
保留 `cell_ids_operation` 的短 JSON 契约，在每条请求中根据题目关键词追加运算规则提示，
并列出当前表格内合法的数值 cell ID。解析器还会逐一校验模型一次输出中的多个 JSON 候选，
优先采用首个通过真实表格校验的候选。

本轮代码和测试已完成，尚未在 4080 上运行。下一步固定同一 80 条 pilot，同时比较：

```bash
python3 scripts/run_quality_repair_4080.py \
  --model-path /userhome/cs5/u3680889/OptimizationSystem/models/Qwen2.5-3B-Instruct \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-dir /userhome/cs5/u3680889/OptimizationSystem/datasets \
  --output-dir out/quality_repair_3b_4080_v5 \
  --variants structured_json,cell_ids_operation,routed_cell_ids_operation
```

门禁仍为超过现有 `structured_json` 的 `12/80` 后才恢复完整 883 条评测。

## 4080 v5 结果分析（2026-10-04）

v5 固定相同的 80 条 pilot 样本，对比结果如下：

| 方案 | 解析率 | 数值准确率 | 正确数 | 校验失败 | 回退 | 用时 |
|---|---:|---:|---:|---:|---:|---:|
| `structured_json` | 93.75% | **15.00%** | **12/80** | 0 | 0 | 300.4 秒 |
| `cell_ids_operation` | 90.00% | 13.75% | 11/80 | 16 | 16 | 306.2 秒 |
| `routed_cell_ids_operation` | 90.00% | 10.00% | 8/80 | 18 | 18 | 386.5 秒 |

多候选 JSON 校验改善了解析率，但没有改善数值正确率；题型路由反而下降，原因是旧路由规则将
`percentage change`、`average price increase` 和 `what percentage increased` 分类错误，且路由方案按条生成导致耗时增加。
这轮不进入完整 883 条评测。已修正路由优先级，CLI 默认恢复为 `structured_json,cell_ids_operation`，
下一轮仅显式加入 `routed_cell_ids_operation` 进行复测。

## 4080 v6 路由复测（2026-10-04）

修正 `percentage change`、`average price increase` 等路由规则后，`routed_cell_ids_operation`
在同一 80 条样本上达到解析率 `88.75%`、数值准确率 `15.00%`、正确数 `12/80`，校验失败和回退各
`19` 条，用时 `381.2` 秒。它追平但没有超过 `structured_json` 的 `12/80`，且两者正确样本重叠较少，
说明 Prompt 路由改变了错误分布，却没有带来稳定总体收益。

因此不运行完整 883 条，也不继续增加关键词规则。下一阶段冻结 3B `structured_json` 作为 FP16 基线，
先比较更大模型；如果模型规模受限，再使用严格隔离的 FinQA train 执行 LoRA/QLoRA 微调。

## 7B 4080 pilot 准备（2026-10-04）

下一轮仅使用固定 80 条样本和 `structured_json`：

```bash
python3 scripts/download_model_4080.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --output-dir /userhome/cs5/u3680889/OptimizationSystem/models/Qwen2.5-7B-Instruct

python3 scripts/run_quality_repair_4080.py \
  --model-path /userhome/cs5/u3680889/OptimizationSystem/models/Qwen2.5-7B-Instruct \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset-dir /userhome/cs5/u3680889/OptimizationSystem/datasets \
  --output-dir out/quality_repair_7b_4080_pilot \
  --variants structured_json --batch-size 1 --max-new-tokens 256
```

4080 若为 16GB，7B FP16 可能因权重、CUDA runtime 和 KV cache 叠加而 OOM。4-bit 结果必须单独标记，不能与
3B FP16 直接作为同一基线比较。

### GPTQ 7B 服务器命令

如果虚拟环境没有 pip，使用 `uv pip` 安装 GPTQ 后端：

```bash
cd /userhome/cs5/u3680889/OptimizationSystem
uv pip install --python /userhome/cs5/u3680889/OptimizationSystem/.venv/bin/python3 \
  "gptqmodel==7.5.0" "transformers==5.18.0" accelerate safetensors
```

下载预量化的 7B GPTQ 模型并运行 pilot：

```bash
/userhome/cs5/u3680889/OptimizationSystem/.venv/bin/python3 scripts/download_model_4080.py \
  --model-id Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --output-dir /userhome/cs5/u3680889/OptimizationSystem/models/Qwen2.5-7B-Instruct-GPTQ-Int4

/userhome/cs5/u3680889/OptimizationSystem/.venv/bin/python3 scripts/run_quality_repair_4080.py \
  --quantization gptq \
  --quantized-model-path /userhome/cs5/u3680889/OptimizationSystem/models/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --model-id Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --dataset-dir /userhome/cs5/u3680889/OptimizationSystem/datasets \
  --output-dir /userhome/cs5/u3680889/OptimizationSystem/out/quality_repair_7b_gptq_pilot \
  --variants structured_json --batch-size 1 --max-new-tokens 256
```

该流程使用已量化权重，不会在 4080 上重新执行 GPTQ 校准；结果需标记为 7B GPTQ INT4。

## 7B AWQ INT4 pilot 结果（2026-10-05）

`out/quality_repair_7b_awq_pilot/comparison_report.json` 显示，预量化
`Qwen2.5-7B-Instruct-AWQ` 在固定 80 条 pilot 上的 `structured_json` 结果为：解析率 `96.25%`、
数值准确率 `22.50%`、正确 `18/80`、格式重试 `7` 次、用时 `535.4` 秒。作为对照，3B FP16
为解析率 `93.75%`、准确率 `15.00%`、正确 `12/80`、重试 `23` 次。

AWQ 7B 的数值准确率首次超过 `20%` 门槛，但解析率仍低于 `99%`，所以质量门禁尚未全部通过。
此外，这个对照同时改变了模型规模和数值格式，不能据此计算 AWQ 的独立量化损失；要得到严格结论，
还需要同源 7B FP16 的 80 条对照。当前 AWQ 输出仍存在多 JSON、`Human:`/`Assistant:` 续写、百分比
分母选择和最大/最低值题套算四则运算等问题。下一步先补显存/吞吐记录，并复测 `18/80` 的可重复性，
再决定是否扩展到完整 883 条。

## 7B GPTQ INT4 pilot 结果（2026-10-05）

`out/quality_repair_7b_gptq_pilot/comparison_report.json` 显示，GPTQModel `7.5.0` 加载的
`Qwen2.5-7B-Instruct-GPTQ-Int4` 在相同 80 条 pilot 上达到解析率 `93.75%`、数值准确率 `23.75%`、
正确 `19/80`、格式重试 `14` 次、总用时 `327.3` 秒。对照：AWQ 为 `18/80`、`22.50%`、`96.25%`、
`535.4` 秒；3B FP16 为 `12/80`、`15.00%`。

GPTQ 当前是准确率最高、pilot 总耗时较低的候选，但解析率仍没有达到 `99%`。质量服务日志的归一化吞吐约为
GPTQ `47.1 token/s`、AWQ `41.6 token/s`，由于生成长度和重试次数不同，只能作为初步信号。GPTQ 还有 5 条未解析记录，
其中索引 `625` 的文本看起来已经是完整 JSON，索引 `722/761` 使用了不在单位白名单中的 `times`；下一步先同步服务器代码并
修复/确认单位解析，再复跑 GPTQ 80 条 pilot。未复跑确认前，不扩展完整 883 条。

## GPTQ pilot 复跑确认（2026-10-05）

复跑仍得到 `19/80`、数值准确率 `23.75%`、解析率 `93.75%`，格式重试 `14` 次，用时约 `328.7` 秒；
5 条未解析索引仍为 `297、377、625、722、761`。这确认 GPTQ 的准确率结果具有可重复性，但解析门禁仍未通过。
下一步先核对服务器 `git rev-parse HEAD` 和 `sha256sum src/evaluation/quality_repair.py`，同步当前解析器并处理
`unit="times"` 及索引 `625` 的合法 JSON 异常，再重新运行同一 80 条 pilot。

版本核对结果：服务器 commit `12508e99abbdc78ecabfca881e0a82a01a3344c6` 和
`quality_repair.py` SHA256 `f39a4834662eb633ae8cbd4bba50cc0f19ed3604ae62010f417993ba997bb949`
与当前仓库一致。进一步检查确认：索引 `625` 是 evidence 数组缺少 `]` 的非法 JSON，索引 `722/761` 是合法
JSON 但使用了白名单外的 `times` 单位，索引 `297/377` 是不完整或混合回答。因此下一步应扩展 ratio/multiple
单位契约并增加测试，再复跑 GPTQ pilot，而不是重新下载或重新量化模型。

随后已完成 ratio 单位契约修复：`times` 和 `multiple` 加入单位白名单，`ratio` 允许空单位、`times` 或
`multiple`，百分比相关 operation 仍要求百分比单位；4080 的结构化 Prompt 也同步说明 ratio 使用 `times`。
新增回归测试覆盖 structured JSON 和 evidence-operation。服务器同步代码后，只需重跑 GPTQ 80 条 pilot，重点观察
原索引 `722/761` 的解析率变化；不需要重新下载或重新量化模型。

## 7B 量化入口实现（2026-10-04）

`scripts/run_quality_repair_4080.py` 现在支持统一参数：

- `--quantization none`：默认 Transformers FP16/BF16 加载；
- `--quantization awq`：调用 `load_awq_service()`；
- `--quantization gptq`：调用 `load_gptq_service()`。

量化模式可以传 `--quantized-model-path`，也可以传已有的
`--quantized-manifest quantized_model_manifest.json`。未传 manifest 时，入口会在输出目录生成最小 manifest，
并将量化方式、模型路径写入结果报告。量化评测仍建议只使用固定 80 条和 `structured_json`，先确认显存、解析率和
数值准确率，再考虑完整评测。

## 7B FinQA QLoRA checkpoint 对比（2026-10-11）

QLoRA v2 从 `checkpoint-100` 恢复到全局 step 724。训练清单确认基座为
`Qwen/Qwen2.5-7B-Instruct`，训练数据为 5778 条、内部 holdout 为 305 条，使用
`max_length=1024`、梯度累积 32、LoRA `r=16`、`alpha=32`、`dropout=0.05`，最终完成 4 epoch。

所有 checkpoint 使用同一个原始 7B 基座、同一 `structured_json` 推理方案，以及固定 seed `20261004`
抽取的同一 80 条 FinQA dev pilot。各报告的样本索引哈希均为
`cd2cf6c4ce8c50cbf23c7c007254b060f741a5ec363ee840efd3ea72662b9e61`，因此结果可以直接比较：

| Adapter | 正确数 | 数值准确率 | 解析率 | 用时 |
|---|---:|---:|---:|---:|
| step 100 | 33/80 | 41.25% | 100% | 170.7 秒 |
| step 200 | 30/80 | 37.50% | 100% | 209.3 秒 |
| step 300 | 33/80 | 41.25% | 100% | 193.1 秒 |
| step 400 | 34/80 | 42.50% | 100% | 185.0 秒 |
| step 500 | 34/80 | 42.50% | 100% | 181.7 秒 |
| step 600 | 36/80 | 45.00% | 100% | 186.0 秒 |
| step 700 | **37/80** | **46.25%** | **100%** | 183.5 秒 |
| 最终 step 724 | **37/80** | **46.25%** | **100%** | 191.5 秒 |

### 数据结论与决策

1. checkpoint 曲线并非单调：step 200 从 `33/80` 降到 `30/80`，step 300 恢复，此后才总体上升。因此训练 loss
   更低或训练步数更多都不能直接等价为任务准确率更高。
2. step 400 与 step 500 同为 `34/80`，step 600 提升到 `36/80`，step 700 提升到 `37/80`；后半程训练仍有收益，
   但 step 700 到 step 724 没有总体增益，当前没有继续追加训练步数的证据。
3. step 700 和 step 724 虽然同为 `37/80`，正确样本集合并不相同。这不是可用“选择更新版本”解决的平局，必须使用
   更大的完整 dev 集比较泛化表现。
4. 所有 v2 checkpoint 的解析率均为 `100%`，所以 checkpoint 选择不受格式门禁影响，主排序指标为完整 dev 数值正确数；
   若正确数相同，再比较错误分层、最长上下文、91 条高风险回归和推理性能。
5. 当前决策是停止继续训练，保留 step 700 和最终 step 724 两个候选，分别运行完整 883 条 dev；胜出版本再完整复跑一次。
   在这些结果回传前，不宣称已选出最佳 adapter，也不进入重新量化阶段。

step 200 在恢复训练后的短暂下降，与学习率从 step 100 的约 `2e-6` 重新升至 step 110 的约
`1.70e-4` 同期出现；step 300 恢复到 `33/80`，之后随训练推进总体上升。step 700 与最终 step 724
并列最佳，且所有版本都保持 `100%` 解析率，说明当前主要瓶颈已经从输出格式转为证据选择、分母选择、
百分比单位、符号方向、金额尺度和多步计算。

step 700 与最终 adapter 并非输出完全相同：两者有 6 条预测值变化，其中 source index `195` 在 step 700
正确、最终版错误，source index `57` 在 step 700 错误、最终版正确，最终形成一进一退的同分结果。因此不能
仅因为 step 724 更晚就默认选择最终 adapter，也不能根据 80 条 pilot 判断两者在完整 dev 上的优劣。

训练状态还暴露出一个独立问题：step 100、200、300、400、500、600、700 和 724 的评估记录中
`eval_loss` 全部为 `null`，`best_metric` 和 `best_model_checkpoint` 也为空。结合当前数据构造逻辑，可能原因是
部分长 Prompt 在 `max_length=1024` 截断后占满序列，使目标 token 全部被标记为 `-100`，从而产生不可用的
holdout loss。训练 loss 从 step 100 的约 `0.180` 总体下降到 step 700 的约 `0.047`，梯度范数未显示爆炸，
但训练 loss 不能替代 holdout 或任务评测，当前 checkpoint 选择必须以独立推理报告为准。该问题应在下一次
训练前修复，不需要为本轮已有 checkpoint 重新训练。

阶段结论：step 700 和最终 step 724 是完整 dev 的两个候选。下一步分别在完整 883 条 FinQA dev 上评测，
选择正确数更高且解析率不低于 `99%` 的版本，再对胜出版本完整复跑一次。完成这一步前，不冻结新的 QLoRA/NF4
基线，也不启动基于新基线的 AWQ、GPTQ 或 INT8 重量化。

## QLoRA step 724 冻结决定（2026-10-11）

完整 883 条 FinQA dev 已分别评测 step 700 和最终 step 724。step 700 正确 `326/883`（`36.92%`），
step 724 正确 `328/883`（`37.15%`）；两者均解析 `882/883`（`99.89%`），结构化输出成功率也均为
`99.89%`。step 724 相对 step 700 有 4 题由错变对、2 题由对变错，净提升 2 题；推理总用时分别为
2320.6 秒和 2337.2 秒，约 `0.7%` 的差异不构成选择障碍。

根据用户决定，不再要求 step 724 进行第二次完整复跑，正式冻结
`models/data/finqa_qlora_7b_v2` 的最终 step 724 adapter。冻结清单写入
`out/finqa_qlora_7b_v2/frozen_adapter.json`，选择报告为
`out/quality_repair_7b_qlora_step724_full/comparison_report.json`，预测文件 SHA256 为
`424485594f5cc4f1fbc61ccf2330fa6a40949de911dff90fdffa24d200697a3a`。step 700 保留为回退候选。

这次冻结表示 adapter 选择已结束，不等于第 6.7.5 节全部通过。当前已通过完整 dev 数值准确率、解析率、
结构化输出和最长上下文四项门禁；仍需运行 91 条高风险回归，并从训练服务器补充 adapter 权重哈希。
4080 评测入口新增 `--dataset-role high_risk_regression`，下一步使用冻结 adapter 对全部 91 条执行统一评测。

### 最长上下文检查

使用 step 724 完整 dev 的 `quality_service.jsonl`，按每道题首次生成请求的 `input_tokens` 降序排列，取最长
10%（89 条，输入长度 1521～2225 tokens）作为长上下文子集。该子集正确 `35/89`（`39.33%`），高于完整
dev 的 `37.15%`；最长单条输入为 2225 tokens，对应 source index `543`，预测 `-1.0`、标准答案 `-1`，
回答正确。因此最长上下文数值质量门禁通过。分析清单见
`out/quality_repair_7b_qlora_step724_full/long_context_analysis.json`。

长上下文子集解析 `88/89`（`98.88%`），唯一未解析项也是完整 dev 中唯一失败的 source index `435`：模型
枚举过多 evidence，输出达到 256-token 上限后截断。它不改变完整 dev 的 `99.89%` 解析门禁结论，但应作为
后续推理契约压缩或输出长度防护的固定回归样本。

### 首次高风险回归运行无效（2026-10-11）

冻结的 step 724 在旧版 `Aiera/finqa-verified` 资产上得到正确 `0/91`、解析 `90/91`。进一步检查确认这不是
可用于判断 adapter 质量的结果：旧资产清单只要求 `question` 和 `answer`，没有 `pre_text`、`post_text` 或
`table`。实际请求平均只有约 140 input tokens（范围 123～171），而完整 FinQA dev 平均约 1121 tokens
（范围 441～2225）。模型收到的只有问题，没有计算所需的财报证据，因此输出中的数字主要是无依据猜测。
旧 FP16 基线在同一资产上也是 `0/91`，进一步说明该门禁存在地板效应。

本次 `0/91` 标记为 `invalid_for_quality_gate`，诊断见
`out/quality_repair_7b_qlora_step724_high_risk/diagnosis.json`，不用于取消 step 724 的冻结状态，也不勾选
高风险门禁。根因已在 `prepare_assets()` 修复：将 verified 问题按标准化文本匹配回原 FinQA test 行，补入
`pre_text/post_text/table/qa`，并保留 verified 的核验答案；任何问题无法匹配时立即失败。服务器重新生成资产后，
必须确认 manifest 的高风险 `required_fields` 已包含上下文字段，再重跑 91 条。

第二次运行仍得到 `0/91`，报告继续记录旧资产 SHA256
`81a6ba140d9806e36d6be5b8f4362a506ff710c853e8f3448889190942a31626`，输入长度仍为 123～171 tokens，
说明服务器实际使用的 `datasets/` 没有被更新。复用同一个输出目录还使 `quality_service.jsonl` 从 92 条追加到
184 条；报告本身是新生成的，但性能日志不能再直接汇总。该次运行同样无效。

评测入口现增加硬保护：选择 `high_risk_regression` 时逐行检查 `pre_text/post_text/table/qa` 及非空上下文，
旧资产会在模型加载和推理前直接抛错。下一次必须先同步修复代码、重建资产并核对 manifest，再使用新的输出目录。

首次重建 `datasets_contextual` 时，91 条 verified 问题中有 3 条无法按完整问题文本匹配。它们使用了
`decimal proportion` 等核验后问法，而 FinQA test 原题使用 `portion` 问句模板。第一次内容词匹配修复
加入了 `proportion/percentage/ratio` 等模板词，却漏掉实际原题中的 `portion`，因此服务器同步该版本后仍然
报出相同的 3 条未匹配问题。最终修复将 `portion` 同样列为模板词，并增加
`proportion` 对 `portion` 的单元测试。资产构建先做完整文本匹配，失败后仅在 FinQA test 内去除这些计算问句
模板词并要求内容词唯一匹配；没有候选或出现多个候选仍立即失败。该修复只改变上下文回连，不改变 verified
核验答案，也不会从 train/dev 引入评测上下文。完整排障说明见第 4 章 5.1 节。

### Contextual 高风险回归最终结果（2026-10-12）

修复上下文回连后，冻结的 step 724 adapter 使用完整 91 条 contextual 高风险回归集运行。有效资产的
SHA256 为 `064f379a0c63ea9cfe49681b1c9a477f8246f9aaf853c85799812bb2feda0693`，输入长度为
396～2590 tokens、平均 1071.84 tokens，确认财报文本和表格已进入模型。运行时实际采用
BitsAndBytes NF4 4-bit 基座加载加 QLoRA adapter，因此该结果不再标记为 FP16，而记录为
`bitsandbytes-nf4`。

第一次 contextual 报告得到 `15/91`，进一步检查发现 `Aiera/finqa-verified` 的 `answer` 列为数值型，
百分比答案采用显示尺度保存，例如参考答案为 `111.97`；旧评分器却依据模型的 `unit=percent` 将预测值
`111.97` 再除以 100，造成 6 条假阴性。评分器现按参考答案的原生类型选择尺度：数值型参考答案直接比较
结构化输出的 `value`，普通 FinQA 中带 `%` 的字符串答案继续使用单位归一化尺度。

修复后使用新输出目录完整复跑，得到正确 `21/91`（`23.08%`）、解析 `91/91`、结构化输出
`91/91`，超过至少 20 条正确的门槛。新旧两次运行的 91 条 `raw_prediction`、公式和 evidence 逐条完全
一致，只有 6 条评分结果从错误改为正确，证明提升来自评分修复而非随机推理波动。正式报告为
`out/quality_repair_7b_qlora_step724_high_risk_contextual_fixed/comparison_report.json`，报告 SHA256 为
`0b37343a1561b8e49dcb81102289ec0ad2f8223915fc15373383520aee160394`，预测 SHA256 为
`064ddb7531bca181bc2c6edceeae0a88e387d06785e4cff3c28822f8337dfd25`。

至此，step 724 已通过完整 dev 准确率、解析率、结构化输出、最长上下文和 contextual 高风险回归五项质量
门禁。训练服务器上的冻结权重文件为 `models/data/finqa_qlora_7b_v2/adapter_model.safetensors`，2026-10-12
对 Git LFS 文件完成 hydration 后，实际权重大小为 `161533192` 字节，Git LFS 对象 ID 与
`sha256sum` 均为 `c92a6058f668b0273393e85443b978b3e1620ed0c3a2ef01905e1124bda52fa4`。此前记录的
`771482dbd581e6909b8d4af3be2854d3dc89aae5c4d78ee0344aa50a6039a260` 是 134 字节 LFS 指针文件的
SHA256，不是实际权重哈希。正确哈希已写入
`out/finqa_qlora_7b_v2/frozen_adapter.json`，6.7.5 质量与制品身份验收全部完成，第 7 章可以启动。完整验收清单
见 `out/finqa_qlora_7b_v2/acceptance_6_7_5.json`。

## 7.1 逐层敏感性分析执行设计（2026-10-12）

### 目标与核心问题

7.1 的目标不是再做一次全模型量化对比，而是通过受控的单层精度替换，回答“当所有 Transformer block
均采用 INT4 时，把第 `i` 层恢复为 INT8 能挽回多少金融业务质量”。实验必须同时观察数值准确率、
金融错误类型、长上下文、结构化输出和量化误差，最终为 7.4 生成可解释的混合精度配置。

### 现有资产审计与前置结论

当前 6.7.5 冻结对象为 Qwen2.5-7B-Instruct 基座上的 step 724 LoRA adapter，验收推理路径是 BitsAndBytes NF4
4-bit 加载基座并挂载 adapter。它在完整 883 条 FinQA dev 上正确 `328/883`（`37.15%`），解析率和结构化
输出成功率均为 `99.89%`；在 91 条 contextual 高风险集上正确 `21/91`。该结果作为已冻结的业务参考，
不应被 7.1 的 merged 模型或量化产物覆盖。

第 6 章保存在 `out/awq` 和 `out/gptq` 的正式量化证据来自旧 Qwen2.5-0.5B 模型，不能与新的 7B QLoRA
基线作为同一实验矩阵。现有 7B AWQ/GPTQ pilot 则加载公开预量化基础模型，没有合并 step 724 adapter，
同样不能代表当前冻结模型。因此 7.1 必须先从冻结 adapter 建立新的 7B merged BF16/FP16 实验母版，
再从该母版产生统一口径的 INT4、INT8 和逐层混合精度候选。

### 精度口径

7.1 中将 INT4 固定为 `W4A16 group_size=32`，将 INT8 固定为 `W8A16`，激活统一保持 FP16/BF16。
Embedding、LayerNorm、lm_head 和 KV Cache 在本节保持不变，它们的独立敏感性属于 7.2。这个定义
使单层实验只改变权重位宽，避免把激活量化、量化算法和特殊模块精度的影响混在一起。

因此不将 AWQ W4A16 与 SmoothQuant W8A8 直接作为逐层 INT4/INT8 对照。如部署后端不支持真实混合位宽，
可先用统一 fake-quant 路径完成敏感性排名，但 manifest 必须清楚标记该结果仅表示质量敏感性，不能用于声称
真实模型大小、显存、延迟或吞吐收益。

### 实验矩阵

| 标识 | 配置 | 用途 |
|---|---|---|
| `REF-FP` | step 724 adapter 合并后的 BF16/FP16 模型 | 权重与激活误差的数学参考 |
| `REF-NF4` | 6.7.5 冻结 QLoRA/NF4 路径 | 已验收的业务质量参考 |
| `ALL-I4` | 全部 Transformer block 为 W4A16 g32 | 逐层恢复实验的起点 |
| `ALL-I8` | 全部 Transformer block 为 W8A16 | INT8 质量上限参考 |
| `L-i-I8` | 第 `i` 层 W8A16，其余 block W4A16 g32 | 测量第 `i` 层的质量恢复量 |

层路径和层数必须从模型配置及实例中自动发现，不在脚本中硬编码。每个 `L-i-I8` 除目标层位宽外，
必须共享相同的模型母版、校准集、量化规则、Prompt、tokenizer、生成参数、数据顺序和评分器。

### 量化误差和业务指标

每层权重记录 MSE、normalized MSE、cosine similarity、最大绝对误差和裁剪/饱和比例。层内的多个 Linear
矩阵按权重元素数加权汇总，不对尺寸差异很大的矩阵做简单平均。同时保留 `q_proj/k_proj/v_proj/o_proj`
和 `gate_proj/up_proj/down_proj` 的明细，便于 7.2 直接复用。固定 calibration 样本还需采集层输出激活的
MSE、normalized MSE 和 cosine similarity。

#### Cosine similarity 的含义与计算口径

权重 cosine similarity 比较量化前权重向量 `x` 与量化后再反量化权重向量 `y` 的方向，计算公式为：

```text
cosine_similarity = dot(x, y) / (norm(x) * norm(y))
```

计算前将一个 Linear 权重矩阵展平成一维向量。结果接近 `1` 表示量化前后方向基本一致，接近 `0` 表示
方向相关性弱，接近 `-1` 表示方向相反。cosine 不直接衡量误差幅度：若 `y = c*x` 且 `c > 0`，即使
权重整体缩放，cosine 仍为 `1`。因此该指标必须与 MSE/NMSE 一起解释，不能单独替代量化误差或业务评测。

数学上 cosine 必须位于 `[-1, 1]`。旧实现直接对超大 FP32 权重向量分别计算点积和两个范数，再用这些
近似结果相除。数百万项的 FP32 归约会累计舍入误差，而且分子的点积与分母的范数沿不同路径累计，误差
不会完全抵消，浮点结果因而可能不再严格满足柯西—施瓦茨不等式。出现大于 `1` 的值表示指标计算失真，
不表示量化后的权重“比完全相同还相似”，也不表示量化权重本身异常。

schema v2 将向量按最多 `1,048,576` 个元素分块，输入保持 FP32，点积、两侧平方和与误差平方和使用
FP64 accumulator 求和，再在 Python 双精度标量中合并各块。cosine 由同一组累计量计算，最后仅将微小的
浮点越界钳制到 `[-1, 1]`；钳制只是边界保险，不能代替高精度累计。两个向量均为零时约定 cosine 为
`1`，仅一侧为零时约定为 `0`。报告写出前还会验证 cosine 范围、MSE/NMSE 非负、饱和率范围以及所有
指标的有限性，非法报告直接失败。

层级和全模型的 `cosine_similarity` 当前定义为“按权重元素数加权的模块 cosine 平均值”，而不是把
所有模块拼成一个向量后重新计算的全局 cosine。两种口径都合法但含义不同，引用本实验结果时必须使用
前一种表述。MSE、NMSE 和饱和率同样按元素数加权，`max_abs_error` 则取模块最大值。

业务评测以 `ALL-I4` 为配对基准，对每个 `L-i-I8` 记录：

- 数值准确率、正确数、解析率和 structured output rate；
- `ALL-I4` 错而单层 INT8 正确的 `recovered_count`；
- `ALL-I4` 正确而单层 INT8 错误的 `regressed_count`；
- `net_correct_gain = recovered_count - regressed_count`；
- 金额、比例/利率、日期/期限子集的正确数和比例；
- 最长 10% 上下文子集和 91 条 contextual 高风险集的恢复、退化、解析及结构化输出变化。

金额、利率和日期标签在运行前生成并冻结，允许一条样本具有多个标签。样本数较少的类别必须报告“正确数/总数”，
不得只报百分比。量化误差和业务质量的相关性单独分析；权重 MSE 不得直接代替业务敏感度排名。

### 执行阶段和机时预估

第一阶段在已有的固定 80 条 pilot 上遍历所有层，用于校验实验口径、发现失败层并获得初步排名。当前 7B
QLoRA 的 80 条 `structured_json` 运行约需 3.2 分钟，所有层 pilot 扫描预计需要数小时。

第二阶段对所有层执行完整 883 条 FinQA dev，才能产生严格意义上的完整 layer sensitivity ranking。当前单次
完整 dev 约需 39 分钟；若模型共 28 层，逐层候选约需 18.2 小时，加入四个参考组和高风险回归后，
整体应预留 20～22 小时的 GPU 机时。层数仍以脚本实际发现结果为准。如只对 pilot 排名的前若干层执行完整 dev，
最终结论必须明确标记为“候选排名”，不能声称已完成所有层的正式排名。

### 排名规则与统计稳定性

定义：

```text
quality_recovery_i = accuracy(L-i-I8) - accuracy(ALL-I4)
```

主排名依次使用 `net_correct_gain`、`recovered_count`、高风险恢复量、长上下文恢复量、结构化输出恢复量、
normalized quantization error 和层号打破并列。对质量恢复量生成 bootstrap 置信区间；如多层区间高度重叠，
报告应将它们标记为同一敏感度梯队，而不是过度解读单个名次。至少重复运行 `ALL-I4` 和一个哨兵层，
检查确定性生成是否能逐条复现。

### 工程产物和验收条件

计划新增：

```text
configs/layer_sensitivity.json
src/quantization/layer_sensitivity.py
tests/test_layer_sensitivity.py

out/layer_sensitivity/
├── experiment_manifest.json
├── baseline_fp/
├── all_int4/
├── all_int8/
├── layer_00_int8/
├── ...
├── layer_metrics.jsonl
├── sample_transitions.jsonl
├── ranking.json
├── ranking.csv
├── report.md
└── validation.json
```

实验脚本需支持模型与数据哈希校验、单层替换、误差采集、断点续跑、已完成项跳过、失败项隔离、
逐样本状态迁移和自动排名。只有当所有 Transformer block 都具有可回溯的 INT4/INT8 误差及完整业务评测记录，
且模型 manifest、配置、数据哈希、预测文件和排名能互相对应时，7.1 才可标记完成。任何失败、缺失或哈希不匹配的
运行不得进入正式排名。

### 7.1.1 前置冻结工具实现（2026-10-12）

已新增 `configs/layer_sensitivity_setup.json`、`src/quantization/layer_sensitivity_setup.py`、
`scripts/prepare_layer_sensitivity.py` 和 `tests/test_layer_sensitivity_setup.py`。工具将 7.1.1 拆成三个不会覆盖 6.7.5
证据的命令：

```bash
# 1. 在 CPU 内存中加载 FP16 基座，合并冻结 adapter，导出模型和完整文件哈希
python3 scripts/prepare_layer_sensitivity.py merge

# 2. 冻结模型、tokenizer、数据、Prompt、评分器、五个样本子集和业务标签
python3 scripts/prepare_layer_sensitivity.py freeze

# 3. 对 merged FP16 执行独立 80 条评测，不传 adapter 也不使用 NF4
python3 scripts/run_quality_repair_4080.py \
  --model-path models/data/Qwen2.5-7B-Instruct-QLoRA-step724-merged-fp16 \
  --model-id Qwen2.5-7B-Instruct-QLoRA-step724-merged-fp16 \
  --dataset-dir datasets_contextual \
  --output-dir out/layer_sensitivity/pilot_merged_fp16 \
  --variants structured_json \
  --batch-size 1 \
  --max-new-tokens 256

# 4. 将 merged FP16 与原有冻结 NF4 pilot 做逐样本配对对比
python3 scripts/prepare_layer_sensitivity.py compare-pilot \
  --reference-report out/quality_repair_7b_qlora_v2_final_pilot/comparison_report.json \
  --candidate-report out/layer_sensitivity/pilot_merged_fp16/comparison_report.json
```

`merge` 在写入前会校验 step 724 adapter 权重 SHA256，目标目录非空时拒绝覆盖；合并后从实际模型
实例自动发现 Transformer block 路径和层数。`freeze` 对基座、adapter 和 tokenizer 逐文件哈希，
冻结 1024 条 calibration、80 条 pilot、883 条 dev、91 条 contextual 高风险和 89 条最长上下文清单，
并对 dev 按问句关键词生成可重叠的金额、比例/利率、日期/期限标签。`compare-pilot` 要求两个报告
的 80 条 source index 完全一致，并输出恢复、退化、稳定正确、稳定错误和原始输出变化数。

当前本地环境没有 7B 基座、FinQA assets 及 PyTorch/Transformers/PEFT，因此真实合并、冻结产物和 pilot
对比需在 4080/训练服务器执行。用户于 2026-10-12 确认训练服务器上
`models/data/finqa_qlora_7b_v2/adapter_model.safetensors` 完成 Git LFS hydration 后大小为 `161533192` 字节，
SHA256 为 `c92a6058f668b0273393e85443b978b3e1620ed0c3a2ef01905e1124bda52fa4`，与修正后的 6.7.5 冻结记录一致，可作为
7.1.1 合并输入。仓库本地副本不参与本次服务器实验，不再视为执行阻塞。`merge` 命令仍会在服务器上
对权重执行 SHA256 硬校验，避免路径误传或权重被后续覆盖。

### 7.1.1 服务器执行结果（2026-10-12）

用户已修复先前由 Git 问题导致的 adapter 文件异常，并确认服务器上
`adapter_model.safetensors` 的 SHA256 为 6.7.5 冻结值
`c92a6058f668b0273393e85443b978b3e1620ed0c3a2ef01905e1124bda52fa4`。原记录的 `771482db...`
是 Git LFS 指针文件哈希，不是 hydrated 权重哈希。`freeze` 命令成功生成输入冻结清单，
状态为 `inputs_frozen`。原始数据资产为：

| 资产 | 样本数 | SHA256 |
|---|---:|---|
| calibration | 1024 | `34af0e443492d4fdc831712c67cd6f12168e923b187b0175817ed9457bd6adfd` |
| FinQA evaluation dev | 883 | `7ba8d507634daa266268414448444421a18868e926632d805949058118b8a6d8` |
| contextual high-risk | 91 | `064f379a0c63ea9cfe49681b1c9a477f8246f9aaf853c85799812bb2feda0693` |

冻结后样本清单的哈希为：

| 子集 | 清单 SHA256 |
|---|---|
| calibration | `dbb4876e0da98ad1aff3a3de0742e2a42db5f120fd20b00e140783347c772092` |
| fixed 80 pilot | `225c221200a11b7281fc7505ca34f77f7d77c362636bba1b4b993abb47a1d7d7` |
| evaluation dev | `d9a4a3818e0ac2612a16c388574e5d9c888a2f1563da115457bcb48097f700eb` |
| contextual high-risk | `441fdc055de9b7b9a1c24872153da9278c102630ce3d235f97d0c70523994c44` |
| longest 10% context | `4b8c2d16ae3b053aed47a0fb502a0f6e44d0e233ad506040813a188377a6b936` |

业务多标签统计为金额 338 条、比例/利率 567 条、日期/期限 709 条。标签可重叠，因此三类计数不应
求和为 883。推理契约已冻结为 `structured_json`、`max_new_tokens=256`、`do_sample=false`、
`num_beams=1`。

adapter 目录树 SHA256 为 `f5ea0841df97ff9a3c1c0c457a905fd61db185de0af6fc3dbe69f40306a95f3b`。该值是包含
adapter 配置和 tokenizer 等全部文件的目录树哈希，不替代上述 `adapter_model.safetensors` 单文件哈希。
合并成功后，merged FP16 模型目录树 SHA256 为
`c59ea48fd633162f9bf6fd0b5fb26b8d3d9c6e6686e2b905e49e44905f5fd946`，脚本自动发现 Transformer block 路径为
`model.layers`，层数为 28。

merged FP16 与冻结 QLoRA/NF4 使用相同的固定 80 条 pilot，索引 SHA256 均为
`cd2cf6c4ce8c50cbf23c7c007254b060f741a5ec363ee840efd3ea72662b9e61`。配对结果为：

| 指标 | QLoRA/NF4 | merged FP16 |
|---|---:|---:|
| 正确数 | 37/80 | 37/80 |
| 数值准确率 | 46.25% | 46.25% |
| 解析率 | 100% | 100% |

两者有 43 条原始输出发生变化，其中 5 条由错变对、5 条由对变错、32 条稳定正确、38 条稳定错误。
因此 merged FP16 通过 7.1.1 的质量与解析门禁，可作为后续量化误差的数学母版；但它与 NF4 仅是总体质量持平，
不是逐样本等价。后续 7.1 必须以 merged FP16 生成 `ALL-I4/ALL-I8/L-i-I8`，同时保留 NF4 作为 6.7.5 业务参考，
不得把两者预测混成同一基线。至此 7.1.1 全部完成，可进入 7.1.2。

### 7.1.2 工具与实验矩阵实现（2026-10-12）

新增 `configs/layer_sensitivity.json`、`src/quantization/layer_sensitivity.py`、
`scripts/run_layer_sensitivity.py` 和 `tests/test_layer_sensitivity.py`。工具首先校验 7.1.1 的 adapter、merged
模型、输入冻结和 Transformer block 身份，然后生成 32 个互不混淆的计划：2 个只读参考组、
`ALL-I4`、`ALL-I8` 和 28 个 `L-i-I8`。其中可运行计划共 30 个，每个计划有独立目录、
`plan.json`、生效配置、量化误差、逐样本预测和 `result.json`。

在当前后端尚未提供可部署的逐层混合打包时，7.1.2 使用明确标记的 symmetric fake-quant：INT4 对
Transformer block 内所有 Linear 权重做 W4A16 g32 对称分组量化—反量化，INT8 做 per-output-channel
W8A16 量化—反量化。反量化后的权重仍以 FP16 运行，因此该路径只用于质量敏感性，不代表真实
模型大小、显存、延迟或吞吐收益。所有量化只作用于 28 个 Transformer block 内的 Linear，Embedding、
LayerNorm、lm_head 和 KV Cache 保持不变。

#### 为什么 7.1 只量化 Transformer block 内的 Linear

Qwen2.5 的 Transformer block 中，绝大多数权重和矩阵乘法集中在二维 Linear 权重矩阵。Attention
包含 `q_proj/k_proj/v_proj/o_proj`，MLP 包含 `gate_proj/up_proj/down_proj`，主要计算形式均为
`y = xW^T`。量化这些矩阵能够覆盖模型主体的大部分参数和计算，也与后续 W4A16/W8A16 部署方案的主要
压缩对象一致。7.1 要比较的是各个完整 Transformer block 对低比特权重量化的相对敏感性，因此必须让
每个单层计划只改变目标 block 内的 Linear，其他全局模块保持一致。

未在本节量化的对象及原因如下：

| 对象 | 7.1 保持不变的原因 |
|---|---|
| Embedding | 词表查表矩阵会影响所有 block 的输入，加入逐层实验会破坏单层归因 |
| RMSNorm/LayerNorm | 参数量很小、压缩收益有限，同时可能具有较高数值敏感性 |
| `lm_head` | 直接控制全局输出 logits，不属于任一 Transformer block；即使实现类型为 Linear 也明确排除 |
| KV Cache | 是推理时动态生成的 activation/cache，不是固定模型权重，需要独立的 cache 量化实验 |
| RoPE、Softmax 等算子 | 没有可按权重位宽量化的持久二维权重 |
| Activation | 本节固定为 FP16/BF16，即 A16，避免把权重量化与激活量化影响混在一起 |

Embedding、Norm、`lm_head` 和 KV Cache 的独立敏感性留给 7.2。这样的范围限制不是认为这些对象永远
不能量化，而是保证 7.1 的 `L-i-I8` 结果可以解释为第 `i` 个 Transformer block 的权重敏感度。

#### 对称量化、INT4 分组和 INT8 输出通道粒度

对称量化以 0 为固定中心，不使用额外的非零 `zero_point`。对一组浮点权重 `w`，先根据该组最大绝对值
计算 scale，再取整和裁剪：

```text
qmax  = 2^(bits - 1) - 1
scale = max(abs(w)) / qmax
q     = clip(round(w / scale), -qmax, qmax)
w_hat = q * scale
```

当前实现中，INT4 使用 `qmax=7`，对称整数范围为 `[-7, 7]`；INT8 使用 `qmax=127`，范围为
`[-127, 127]`。零权重仍映射为整数 0，正负权重共享同一尺度。fake-quant 将 `w_hat` 写回 FP16 权重，
所以真实推理仍由 FP16 tensor 执行，但权重已经带有相应 INT4/INT8 的取整和裁剪误差。

对 Linear 权重 `W[out_features, in_features]`，INT4 的 W4A16 g32 沿每个输出行的输入维度每 32 个权重
划为一组，每组独立计算一个 scale：

```text
W[output 0, input  0..31]  -> scale_0_0
W[output 0, input 32..63]  -> scale_0_1
W[output 1, input  0..31]  -> scale_1_0
...
```

INT4 的可用离散等级较少，若整行只用一个 scale，单个大离群值会放大 scale，使大量较小权重被粗糙
舍入甚至变为 0。g32 把离群值的影响限制在所在的 32 个权重内，通常能显著降低误差，代价是需要保存
更多 scale 元数据。输入维度不能整除 32 时，fake-quant 仅为计算分组临时补零，写回时会去除补齐部分。

INT8 的 W8A16 则采用 per-output-channel 对称量化：`W[o, :]` 对应一个输出神经元/输出通道的全部输入
权重，每个输出行独立使用一个 scale：

```text
scale[o] = max(abs(W[o, :])) / 127
```

这样不同数值范围的输出通道互不干扰。INT8 约有 255 个对称离散等级，分辨率远高于 INT4，通常按输出
通道一个 scale 已能保持很小的权重误差，无需像 INT4 一样每 32 个权重再细分。两种配置的比较应理解为
“低成本 W4A16 g32”与“较高精度 W8A16 per-output-channel”的业务质量差异，而不是只有位宽不同、其他
量化粒度也完全相同的数学对照。

#### 为什么是 28 个 Transformer 层

这里的 `28` 来自模型架构本身，不是脚本根据“7B”参数规模推算出的数字。Qwen2.5-7B-Instruct 的
`config.json` 明确声明：

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "num_hidden_layers": 28
}
```

`num_hidden_layers=28` 表示模型主体按顺序堆叠了 28 个 Transformer decoder block。在 Transformers
加载后的 Qwen2 模型对象中，这组 block 表现为 `model.layers`，有效下标为 `0` 到 `27`；每个 block
包含自注意力、MLP 及相应归一化结构。`num_attention_heads=28` 表示每一层内部有 28 个 query attention
head，虽然数值也恰好为 28，但它与层数是两个独立的架构维度，不能把 attention head 当成 Transformer
层。类似地，“7B”表示模型总参数量的大致等级，也不等于层数；参数量由层数、hidden size、MLP 宽度、
词表和输出头等因素共同决定。

7.1.1 没有只信任配置文件，而是对合并后的实际模型调用 block 自动发现，确认路径为 `model.layers`、
`len(model.layers)=28`，再把该结果和冻结配置写入 manifest。LoRA merge 只把 adapter 增量合并进既有
Linear 权重，不增加或删除 Transformer block，因此合并后的 step 724 模型仍为 28 层。7.1.2 在运行时
再次发现并核对层路径和数量；如果实际模型不再是 28 层，程序会拒绝执行，而不会静默漏跑或多跑。

#### 为什么需要分别运行 28 个单层计划

逐层计划采用单变量隔离设计。`ALL-I4` 先把 28 个 block 全部设为 W4A16 g32；随后每个
`layer_XX_int8` 都重新加载同一个干净的 merged FP16 母版，并从相同的 `ALL-I4` 精度分配出发，只把
一个目标 block 恢复为 W8A16：

```text
layer_00_int8: layer 0 为 INT8，layer 1..27 为 INT4
layer_01_int8: layer 1 为 INT8，其余 27 层为 INT4
...
layer_27_int8: layer 27 为 INT8，layer 0..26 为 INT4
```

这些计划彼此独立、不会累积；`layer_01_int8` 不会继承 `layer_00_int8` 的权重修改。只有遍历全部 28
个 block，才能为每一层取得“从 INT4 恢复到 INT8 后挽回多少业务质量”的独立观测。如果只运行
`ALL-I4` 和 `ALL-I8`，只能判断全模型 INT8 与 INT4 的总体差异，无法定位差异由哪些层造成；少跑一个
`layer_XX_int8`，该层的敏感度就仍然未知。

完整矩阵共有 32 个条目：已有结果注册为 2 个只读参考组 `REF-FP/REF-NF4`，新执行 2 个全局对照
`ALL-I4/ALL-I8`，再执行 28 个单层恢复计划。因此 `plan_count=32`，但真正需要服务器推理的是
`2 + 28 = 30` 个计划。每个可运行计划都使用相同的固定 80 条 pilot、Prompt、tokenizer、生成参数、
评分器和样本顺序。汇总时以 `ALL-I4` 为配对基准，计算每层的 `recovered`、`regressed` 和
`net_correct_gain`，用于筛选后续混合精度方案中应保留为 INT8 的敏感层。

模型在 CPU 上加载和反量化，再整体移入 GPU 评测，避免 16GB GPU 在保留 FP16 模型时叠加大型 FP32
临时量化张量。每次只运行一个计划；已完成且 `result.json` 状态为 `completed` 的计划会被续跑机制跳过。
完成后以 `ALL-I4` 为基准做 source-index 严格对齐的逐样本配对，输出恢复、退化、稳定正确、稳定错误和
原始输出变化数，生成 `matrix_summary.json` 和仅供 pilot 筛选的 `ranking_preview.json`。

服务器执行顺序：

```bash
# 校验 7.1.1 身份并生成 32 个计划
python3 scripts/run_layer_sensitivity.py prepare

# 先执行两个全局对照
python3 scripts/run_layer_sensitivity.py run-plan --plan-id all_int4
python3 scripts/run_layer_sensitivity.py run-plan --plan-id all_int8

# 再遍历 28 个单层 INT8 计划；任一失败时停止，修复后可续跑
for i in $(seq -w 0 27); do
  python3 scripts/run_layer_sensitivity.py run-plan --plan-id "layer_${i}_int8" || break
done

# 检查是否还有未完成计划，然后生成配对汇总
python3 scripts/run_layer_sensitivity.py pending
python3 scripts/run_layer_sensitivity.py summarize
```

`prepare` 完成后应报告 `block_count=28`、`plan_count=32`、`runnable_plan_count=30`，并把已有
merged FP16 `37/80` 注册为 `REF-FP`、把 QLoRA/NF4 `37/80` 注册为 `REF-NF4`。在 30 个可运行计划
全部回传之前，7.1.2 的模型矩阵仍保持未验收状态，不提前勾选服务器执行项。

### 7.1.2 Pilot 矩阵结果与审计（2026-10-12）

服务器已完成并回传全部 30 个可运行计划。`matrix_summary.json` 状态为 `complete`，记录
`completed_runnable_plans=30`、`expected_runnable_plans=30`、`pending=[]`。其中 28 个单层计划覆盖
`layer_00_int8` 到 `layer_27_int8`，没有缺层或重复层。30 个结果均为 `completed`、均包含 80 条预测，
样本索引 SHA256 均为
`cd2cf6c4ce8c50cbf23c7c007254b060f741a5ec363ee840efd3ea72662b9e61`。逐一重新计算预测文件和
`quantization_error.json` 的 SHA256 后未发现不匹配；128 项本地单元测试全部通过。30 个计划总运行时间约
84.59 分钟，平均每个计划 169.17 秒。

全局参考结果如下：

| 计划 | 正确数 | 解析数 | 数值准确率 | structured output rate | weighted NMSE |
|---|---:|---:|---:|---:|---:|
| `REF-FP` | 37/80 | 80/80 | 46.25% | 100% | 不适用 |
| `REF-NF4` | 37/80 | 80/80 | 46.25% | 100% | 不适用 |
| `ALL-I4` | 36/80 | 80/80 | 45.00% | 100% | 0.010137711 |
| `ALL-I8` | 36/80 | 79/80 | 45.00% | 98.75% | 0.000130918 |

INT8 的权重 NMSE 比 INT4 低约 77.44 倍，即降低约 98.71%，但 `ALL-I8` 与 `ALL-I4` 的正确数相同。
两者仍有 43 条原始预测不同，其中 3 条恢复、3 条退化。`ALL-I8` 唯一未解析样本为 source index 656：
模型在合法 JSON 前增加了 `Here is the correct response:`，违反“只输出 JSON 对象”的协议。该结果说明
更小的静态权重误差不会单调转换成更高的生成任务准确率，不能用 MSE/NMSE 代替业务评测。

以 `ALL-I4` 的 36/80 为基准，28 个单层 INT8 计划按净正确数变化可分为：

| 净正确数变化 | 层 | 解释 |
|---:|---|---|
| +2 | 10、22、23 | 38/80，当前 pilot 第一梯队 |
| +1 | 14、5、25、17、21、27、8、26、24 | 37/80，第二梯队 |
| 0 | 20、19、6、11、0、4 | 36/80，与 `ALL-I4` 同分 |
| -1 | 12、7、13、3、1、18、2 | 35/80 |
| -2 | 9 | 34/80 |
| -3 | 15、16 | 33/80 |

前三名的配对变化并不完全相同：layer 10 恢复 3 条、退化 1 条；layer 22 恢复 2 条且无退化；layer 23
恢复 2 条且无退化。因此按现有 tie-break，layer 10 因恢复数更多排名第一；如果策略优先避免新增错误，
layer 22 和 23 是更保守的候选。净增 1 的层中，17、21、27、8、26、24 均为恢复 1 条且无退化，
也比“恢复 2 条、退化 1 条”的 14、5、25 更稳定。

不过，所有 28 个单层计划的恢复只集中在 4 个 source index：31、191、402、611。其中 191 和 402
分别被 11 个层恢复，611 被 8 个层恢复，31 只被 layer 23 恢复；退化集中在另外 8 个样本。这说明 80 条
pilot 的名次由极少数边界样本驱动，单题就会改变 1.25 个百分点，当前差异不足以形成稳定的最终层级排序。
尤其不能把单层结果相加推断多层组合：同时把 10、22、23 恢复为 INT8 未必能获得 +6 个正确数，必须建立
新的组合计划实测。`ALL-I8` 没有优于 `ALL-I4` 也进一步证明层间扰动存在非线性相互作用。

量化误差恢复量与 `net_correct_gain` 的 Pearson 相关系数为 -0.393，Spearman 相关系数为 -0.442；在当前
28 层、80 条 pilot 的离散结果上，它们呈中等负相关而非预期的正相关。该值只能作为“静态权重误差不能
预测业务质量”的描述性证据，不能作显著性或因果结论。业务排名继续以逐样本正确性转移为主，NMSE 仅作
辅助 tie-break。

审计同时发现旧版 cosine 指标存在数值稳定性缺陷：`ALL-I4` 的 196 个 Linear 模块中有 84 个记录的
`cosine_similarity > 1`，最大为 1.009051，违反余弦相似度的数学范围。根因是超大权重矩阵使用 FP32
点积和范数累计造成误差，不是量化结果本身异常。当前排名没有使用 cosine，只使用业务正确数和 NMSE，
因此已生成的 pilot 正确性排名不受影响。计算代码现已修复，但已经落盘的 schema v1 报告不会自动更新，
全部计划重新生成 schema v2 误差记录并通过审计前，cosine 证据仍不得验收。

代码已将误差统计升级为 schema v2：每次最多处理 1,048,576 个元素，输入保持 FP32、点积/平方和/
误差平方和使用 FP64 累计，并在写入前强制校验所有模块、层和全局汇总的 cosine 均在 `[-1,1]`、误差
非负、饱和率在 `[0,1]`。新模型加载参数同时由弃用的 `torch_dtype` 更新为 `dtype`。新增
`recompute-error` 只重算某个计划的 CPU fake-quant 误差，不移动模型到 GPU、不重复 80 条质量推理，并在
覆盖前把原始文件保留为 `quantization_error.schema_v1.json`；它会校验并保留原预测 SHA256，再更新
`result.json` 中的误差 schema、哈希和 weighted NMSE。修复 30 个计划后使用 `audit-errors` 校验 schema、
文件哈希和 cosine 范围：

```bash
python3 scripts/run_layer_sensitivity.py recompute-error --plan-id all_int4
python3 scripts/run_layer_sensitivity.py recompute-error --plan-id all_int8

for i in $(seq -w 0 27); do
  python3 scripts/run_layer_sensitivity.py recompute-error \
    --plan-id "layer_${i}_int8" || break
done

python3 scripts/run_layer_sensitivity.py audit-errors
python3 scripts/run_layer_sensitivity.py summarize
```

只有当 `audit-errors` 返回 `status=pass`、`valid_plans=30`、`invalid_plans=[]`，且
`module_cosine_min/module_cosine_max` 都位于 `[-1,1]` 时，才可勾选 7.1.3 的完整权重误差采集。重新
`summarize` 后需要以 schema v2 NMSE 更新相关系数和同分层的辅助排序；业务预测及 recovered/regressed
不会改变。

服务器完成 schema v2 重算并回传后，最终 `error_audit.json` 为 `status=pass`、`valid_plans=30`、
`invalid_plans=[]`；196 个模块乘 30 个计划的全部新记录中，cosine 范围为
`[0.9931836730501131, 0.9999593877335811]`，没有越界值。30 份原始 schema v1 文件均已保留，旧矩阵
累计 2,648 个模块 cosine 越界，新矩阵为 0。逐计划核对确认旧文件 SHA256 等于 `result.json` 保存的
`original_quantization_error_sha256`，新文件 SHA256 等于当前 `quantization_error_sha256`；预测文件哈希
同时与 `result.json` 和 `comparison_report.json` 一致，说明修复没有改变任何业务预测。

FP64 重算造成的 weighted NMSE 最大绝对变化仅约 `4.03e-10`。`ALL-I4` 从
`0.010137710939269946` 修正为 `0.010137710540054103`，`ALL-I8` 从
`0.0001309179087966406` 修正为 `0.00013091790388376212`。28 层的完整排名顺序保持不变，第一梯队仍为
10、22、23；误差恢复量对净正确数的 Pearson 更新为 `-0.3932625`，Spearman 保持
`-0.4421115`。因此 schema v2 修复恢复了 cosine 证据的数学有效性，没有改变此前的 pilot 业务结论。

综上，7.1.2 的 32 项矩阵定义、30 项服务器执行、身份一致性和 pilot 配对结果可以验收；当前输出仍应称为
`ranking_preview`。建议保护层候选为第一梯队 `{10, 22, 23}`，其中 `{22, 23}` 是零退化的保守子集，
但在完成确定性复跑、候选组合实测、业务子集/高风险评测和完整 883 条 dev 前，不把该集合升级为 7.4 的
最终保护层输入。

### 7.1.3 层输出激活误差采集

#### 目的与原理：这一步在做什么，为什么需要做

7.1.2 已经比较过原始 FP16 权重和量化后再反量化权重之间的误差，但权重只是模型的静态参数，不是模型真正
输出的表示。一次推理中，输入 token 会依次通过 embedding 和 28 个 Transformer block；每个 block 内又有
attention、MLP、残差连接和 LayerNorm。某个权重矩阵的误差可能被后续计算抵消，也可能沿网络逐层累积和
放大，因此“某层权重误差大”不等价于“该层对最终输出或业务答案影响大”。

本实验直接观察量化扰动在网络中的传播。对同一条 calibration Prompt、完全相同的 token IDs 和抽样位置，
分别执行 merged FP16 参考模型和一个量化候选模型，然后配对比较每个 block 输出的 hidden state：

```text
同一条固定 Prompt
        │
        ├── merged FP16 ──→ layer 0 ... layer 27 参考 hidden state
        │
        └── 量化候选模型 ─→ layer 0 ... layer 27 候选 hidden state
                                      │
                                      └── 逐层计算 MSE、NMSE、cosine 和 max abs
```

这里的 30 个候选计划与 7.1.2 完全相同：`ALL-I4`、`ALL-I8`，以及 28 个仅把 layer i 从 INT4 提升为
INT8、其余层保持 INT4 的 `L-i-I8`。逐层记录全部 28 个 block，而不只记录被替换的目标层，是为了同时回答：

- 目标层自身的激活误差降低了多少；
- 该改善到下一层后是保留、放大还是抵消；
- 到 layer 27 时是否仍能观察到影响；
- 某层局部更接近 FP16，是否反而使后续全局误差变差。

因此这一步可以理解为模型的“逐层误差传播探针”：权重误差回答“参数改了多少”，激活误差回答“模型计算过程
实际偏了多少”，业务评测则回答“最终答案是否变对”。三者观察的是不同层面，必须共同使用。

每层记录四类指标。设 `x` 为 FP16 参考激活、`y` 为量化候选激活：

- `MSE = mean((y-x)^2)`：衡量绝对平方误差，但会受到不同层激活尺度影响；
- `NMSE = sum((y-x)^2) / sum(x^2)`：用参考激活能量归一化，更适合跨层比较；
- `cosine = dot(x,y) / (||x||·||y||)`：衡量表示方向是否保持，越接近 1 越好；
- `max abs = max(|y-x|)`：发现少数极端偏差，但不能单独代表普遍质量。

即使激活指标显著改善，也不能直接判定业务更好。数值距离是连续信号，而 FinQA 正确与否是经过生成、解析和
数值评分后的离散结果；微小扰动可能改变一个边界答案，较大扰动也可能不改变最终数字。因此激活指标用于解释
误差传播、定位异常，并在业务结果相同时提供辅助信息，不能替代 `recovered/regressed/net_correct_gain` 排名。

#### 为什么采用固定 32×32 抽样

激活探针固定使用冻结 calibration 的前 32 条数据。每条 Prompt 按 merged 模型 tokenizer 从左侧截断到最多
512 个输入 token，以保留末尾的财务问题和 `Answer:` 提示，并从实际 token 序列中均匀选择最多 32 个位置；
序列长于 32 时首尾位置一定包含在内，短序列
则保留全部位置。这里比较的是 `32 条样本 × 每条最多 32 个 token 位置 × 28 层` 的抽样 block 输出，
不是 1,024 条 calibration 的全 token 激活。采用抽样是为了同时覆盖序列前部、中部和答案前的末端上下文，
并把 Qwen2.5-7B（hidden size 3,584）的 FP16 参考缓存控制在约 196 MiB，而非数 GB。最终报告和 manifest
均明确记录 `scope=sampled_block_output_token_states`，不得把它表述成全量激活评测。

固定样本、token IDs 和位置是为了保证 FP16 与每个候选进行真正的逐元素配对，避免把数据抽样变化误认为量化
误差。使用左截断是因为 FinQA Prompt 的问题与 `Answer:` 位于末尾；均匀位置同时覆盖保留下来的上下文前部、
中部和答案提示附近，而不是只观察最后一个 token。32 条样本乘每条 32 个位置可形成每层 1,024 个 token
state，在代表性、运行时间和存储量之间取得可复现的折中。

#### 为什么分成参考缓存和候选比较两个阶段

7B FP16 模型本身已接近 16GB GPU 的容量边界，不能可靠地把参考模型和候选模型同时放在显存中。实验先单独
运行一次 merged FP16，把选中位置的 28 层输出保存为 CPU/磁盘参考 shard；之后每个候选在独立进程中依次
加载，与同一缓存比较。这样 GPU 中主要始终只有一个模型，进程结束后 CUDA allocator 也会释放。参考执行还
关闭 KV cache 并直接调用 backbone，不执行文本生成和最终词表 `lm_head`，因为本步骤只需要 hidden state。

参考缓存不能只保存每层均值或范数，因为计算 `MSE` 和 cosine 所需的交叉项 `dot(x,y)` 必须由配对的参考与
候选元素得到。保存全部 1,024 条 calibration 的所有 token 又会产生数十 GB 数据，因此才采用可审计的
32×32 hidden-state 探针。约 196 MiB 的 shard 只需留在服务器，最终分析使用 JSON 汇总结果。

#### 为什么使用 FP64 分块累计

每个计划最终比较 102,760,448 个标量。若直接用 FP16/FP32 对如此多的元素累加点积与平方和，舍入误差可能
再次产生理论上不合法的 `cosine > 1`。实现按固定块读取配对元素，并以 FP64 累计 dot product、参考平方和、
候选平方和及误差平方和，最后从全局充分统计量计算 MSE、NMSE 和 cosine。不能先计算每条样本或每层的
cosine 再简单平均，因为不同记录的元素数和能量可能不同；全局累计才能得到严格的元素级结果。

#### 实现与执行方式

实现位于 `src/quantization/layer_sensitivity.py`，配置位于 `configs/layer_sensitivity.json`：

- `prepare-activation-reference` 只加载一次 merged FP16，关闭 KV cache，并直接执行 backbone，避免计算和保存
  词表 logits；28 个 block hook 仅把选中的 token hidden state 复制到 CPU FP16，每条样本独立保存一个 shard。
- reference manifest 绑定 merged tree SHA256、完整 calibration subset SHA256、32 个 source index、逐行 SHA256、
  token IDs SHA256、token 位置 SHA256 和每个缓存 shard SHA256。tokenizer、样本或缓存任一变化都会拒绝比较。
- `run-activation` 每个进程只加载一个候选模型，先应用与 7.1.2 完全相同的 W4A16-g32/W8A16 fake quant，
  再读取 FP16 shard 做逐层配对。参考模型和候选模型不会同时占用 GPU，因此适配 16GB 显存。
- MSE、NMSE 和 cosine 不按样本或层做简单平均。代码对配对元素的 dot product、参考平方和、候选平方和、
  误差平方和使用分块 FP64 累计，再从全局充分统计量计算指标；cosine 强制校验在 `[-1,1]`。
- 每个计划写入 `activation_error.json` 和 `activation_result.json`；`audit-activations` 校验 30 个计划的 schema、
  计划合同、参考 manifest 哈希、结果哈希和数值范围；`summarize-activations` 报告相对 `ALL-I4` 的全层及目标层
  NMSE 恢复量，但不会修改已有的业务排名。

服务器按以下顺序执行。每个 `run-activation` 都是独立进程，前一个模型退出后才加载下一个，既方便断点续跑，
也能释放 CUDA allocator；该流程不会重复 80 条答案生成：

```bash
python3 scripts/run_layer_sensitivity.py prepare-activation-reference

python3 scripts/run_layer_sensitivity.py run-activation --plan-id all_int4
python3 scripts/run_layer_sensitivity.py run-activation --plan-id all_int8

for i in $(seq -w 0 27); do
  python3 scripts/run_layer_sensitivity.py run-activation \
    --plan-id "layer_${i}_int8" || break
done

python3 scripts/run_layer_sensitivity.py pending-activations
python3 scripts/run_layer_sensitivity.py audit-activations
python3 scripts/run_layer_sensitivity.py summarize-activations
```

`prepare-activation-reference` 与已完成计划都具有幂等跳过逻辑。中断后先执行 `pending-activations`，只需补跑
其列出的 plan。验收条件是 `pending=[]`，`activation_audit.json` 为 `status=pass`、`valid_plans=30`、
`invalid_plans=[]`，且 `layer_cosine_min/layer_cosine_max` 均位于 `[-1,1]`。在这些服务器结果回传前，TODO
中的激活误差采集保持未勾选。

回传分析时只需带回 reference 的 `manifest.json`、30 个计划的 `activation_error.json`/
`activation_result.json`，以及矩阵根目录的 `activation_audit.json` 和 `activation_summary.json`。约 196 MiB 的
`sample_*.pt` 参考 shard 可留在服务器；它们只用于服务器端重新审计或补跑，不进入最终 JSON 分析产物。

服务器已完成全部激活探针并回传 JSON 结果。reference manifest 绑定 merged tree
`c59ea48fd633162f9bf6fd0b5fb26b8d3d9c6e6686e2b905e49e44905f5fd946`、完整 calibration subset
`dbb4876e0da98ad1aff3a3de0742e2a42db5f120fd20b00e140783347c772092` 和 32 条探针索引
`23b22df6bf551a4e788327db3b3d3d96e1eca49078d2c6bd0049da2d390eca8b`。实际参考缓存为
205,798,240 bytes。`activation_audit.json` 为 `status=pass`、`valid_plans=30`、`invalid_plans=[]`，全部
840 条逐计划逐层记录的 cosine 位于 `[0.891597151330265, 0.9999954366968903]`。再次独立核对 30 份
`activation_error.json` 后，其 plan 合同、reference manifest SHA256、结果文件 SHA256、28 层数量及每层
3,670,016 个元素均一致；每个计划共比较 28,672 个 token-layer state、102,760,448 个标量。

全局基线对比如下：

| 计划 | aggregate MSE | aggregate NMSE | aggregate cosine | max abs error |
| --- | ---: | ---: | ---: | ---: |
| `ALL-I4` | 3.3234466 | 0.0024791775 | 0.9987940 | 2677 |
| `ALL-I8` | 0.0673994 | 0.0000502776 | 0.9999856 | 278 |

`ALL-I8` 的 aggregate MSE/NMSE 比 `ALL-I4` 低约 49.31 倍，最大绝对误差低约 9.63 倍，说明 W8A16
在隐藏状态层面明显更接近 FP16。但两者在 80 条业务 pilot 上都只有 `36/80`，因此“数值更接近 FP16”并不
自动转化为小样本业务正确数提升。激活误差仍只能作诊断信息。

`ALL-I4` 的误差传播在最后两层明显放大：layer 26 为 NMSE `0.0429121`、cosine `0.9791382`，layer 27
为 NMSE `0.2050575`、cosine `0.8943939`。仅 layer 27 就贡献约 39.80% 的全层误差平方和，layer 26 和
27 合计约 48.28%。中间 layer 3–25 的 NMSE 大多约为 `0.0011–0.0019`。因此只看 aggregate cosine
`0.9987940` 会掩盖最终 block 的明显偏差，后续必须继续保留逐层指标。部分记录存在 2,600 以上的最大绝对
误差，说明少数激活位置可能有很大的极值偏差；当前汇总不能定位具体样本，不能仅凭 max abs 判定普遍退化。

按“相对 `ALL-I4` 的 aggregate activation NMSE 恢复量”排序，前五名为：

| 激活排名 | 层 | aggregate NMSE | 相对恢复 | 相对降幅 | pilot 净正确数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26 | 0.00154550 | 0.00093368 | 37.66% | +1 |
| 2 | 1 | 0.00168437 | 0.00079481 | 32.06% | -1 |
| 3 | 0 | 0.00209173 | 0.00038745 | 15.63% | 0 |
| 4 | 4 | 0.00239036 | 0.00008881 | 3.58% | 0 |
| 5 | 25 | 0.00240368 | 0.00007550 | 3.05% | +1 |

layer 26 从 INT4 升为 INT8 后，其目标层 NMSE 从 `0.0429121` 降至 `0.0131122`，并把后续 layer 27
NMSE 从 `0.2050575` 降至 `0.0413768`，是最明显的下游传播改善。layer 1 和 layer 0 也说明早期层的较小
改动能够影响后续全部层。相反，layer 2 的目标层 NMSE 虽改善 `0.0041129`，全层 aggregate NMSE 却从
`0.00247918` 小幅升至 `0.00248782`，说明单层局部改善经过非线性网络后不保证全局误差单调下降。

已有业务第一梯队与激活排名并不一致：layer 10/22/23 的激活排名分别为 27/9/14，aggregate NMSE 相对
降幅仅约 0.16%/0.75%/0.47%，但三者在 pilot 上都净增 2 题。反过来，激活排名第二的 layer 1 在业务上
净退化 1 题。28 个单层计划中，aggregate activation NMSE 恢复量与业务净正确数的 Pearson 相关系数仅
`0.03725`、Spearman 仅 `0.22148`；目标层 NMSE 恢复量与业务净正确数的 Pearson/Spearman 也只有
`0.12798/0.11427`。aggregate activation 恢复量与静态权重 NMSE 恢复量的 Pearson 为 `0.07037`、
Spearman 为 `-0.19978`。这些结果共同表明静态权重误差、动态激活误差和离散业务正确性是三个不同信号。

因此 7.1.3 激活误差采集可以验收，但不据此改写 7.1.2 的业务候选结论：pilot 第一梯队仍是
`{10,22,23}`，零退化保守子集仍是 `{22,23}`；layer 26 应作为“强激活传播敏感层”保留观察，不能仅凭
激活排名直接加入最终保护层。后续完整 883 条 dev、高风险集及候选组合实测仍以配对业务质量为主，激活指标
用于解释、异常排查和同等业务结果下的辅助选择。

### 7.1.4–7.1.5 剩余业务评测与最终验收实现

激活探针完成后，7.1 尚不能直接给出最终保护层。80 条 pilot 中一题等于 1.25 个百分点，而且已观察到大量
并列和少数边界样本驱动排名。2026-10-12 经用户确认，最终阶段从“28 层全量正式排名”调整为 pilot 第一梯队
layer 10/22/23 的候选验证：三层扩大到完整 883 条 FinQA dev，并补充业务类别、长上下文、独立高风险集及
确定性证据。该选择把剩余机时从约 20 小时压缩到约 2～2.5 小时，但结论只覆盖三个候选，不能声称形成了
28 层完整排名。实现位于 `src/quantization/layer_sensitivity_final.py`，由原有
`scripts/run_layer_sensitivity.py` 统一提供以下命令，`tests/test_layer_sensitivity_final.py` 覆盖分组计数、
配对 bootstrap、排名顺序、保护层过滤和 CLI 合同：

- `prepare-final`：校验并冻结 final contract，不加载模型；
- `pending-final`：按 suite 列出未完成计划；
- `run-final-plan`：运行某个 suite 的单个计划，已完成结果会按预测哈希幂等跳过；
- `audit-determinism`：立即将两项复跑与原 pilot 做逐行及预测 SHA256 严格比较；
- `finalize`：严格审计全部运行，派生子集统计、bootstrap、最终排名、保护层建议和验收文件。

final contract 定义三个 suite：

| Suite | 数据 | 计划 | 用途 |
| --- | ---: | ---: | --- |
| `determinism` | 固定 80 条 pilot | `ALL-I4`、`layer-22` | 与原 pilot 逐行及 SHA256 完全复现 |
| `full_dev` | 完整 883 条 dev | `ALL-I4/ALL-I8 + layer 10/22/23` | 三个候选的业务排名与 bootstrap |
| `high_risk` | 91 条 contextual 高风险集 | `ALL-I4 + layer 10/22/23` | 候选独立恢复/退化和格式稳定性 |

contract SHA256 还覆盖完整 evaluation 参数、W4A16-g32/W8A16 精度合同和 symmetric fake-quant 后端；
`prepare-final` 后若有人修改 dtype、batch size、生成长度、位宽、group size、排除模块或后端，后续命令会拒绝
混用旧 manifest，必须重新明确准备实验。

#### 这次实现具体做了什么

第一步是把“还要跑什么”从人工约定变成机器可校验合同。`prepare-final` 读取已经冻结的 input freeze、
`frozen_subsets.json`、业务标签、7.1.2 matrix manifest 和当前量化配置，重新校验它们的 SHA256，然后把三套
suite、每套允许的 plan ID、bootstrap 参数、排名顺序和保护层准入规则写入新的
`final_evaluation/experiment_manifest.json`。manifest 中的 `contract_sha256` 同时覆盖数据、推理参数、精度和
后端；后续命令每次都会重算合同，防止部分运行偷偷使用了不同 batch、生成长度或量化口径。

第二步是提供统一的单计划执行器 `run-final-plan`。它不会复用上一个进程里已经修改过的模型，而是每次从同一
merged FP16 母版重新加载权重，根据 plan 独立执行 W4A16-g32/W8A16 fake quant，再运行指定 suite。这样
`L-i-I8` 之间只改变目标层，不会发生 layer 0 的修改残留到 layer 1 等交叉污染。执行前还会把实际 Dataset
逐行重新生成 fingerprint，与冻结的 883 条 dev 或 91 条高风险清单比较；数据顺序、问题内容或行数变化都会
直接失败。每次运行保存 effective baseline/quality 配置、逐样本预测、comparison report 和 result 摘要。

第三步是先设置一个低成本的确定性闸门。`determinism` 只重跑原 80 条 pilot 的 `ALL-I4` 和保守哨兵
layer 22，然后 `audit-determinism` 同时比较 80 行完整 JSON 记录和预测文件 SHA256。仅比较最终正确数是不够
的，因为“恢复一题、退化另一题”仍可能保持相同总分；这里要求逐行完全一致。审计失败会写出
`determinism_audit.json` 并以错误退出，一键脚本因此不会继续消耗十几个小时运行完整矩阵。

第四步执行候选质量计划。full dev 包含 `ALL-I4/ALL-I8` 和 layer 10/22/23 共 5 项，其中两个全局参考已经
完成，可以直接按哈希复用；高风险集包含必要的 `ALL-I4` 配对基线和 layer 10/22/23 共 4 项，不再为了全局
上界额外运行 `ALL-I8`。加上两个已经完成的 determinism 计划，新合同共 11 次运行，切换合同时剩余 7 次。
选择这三个候选的冻结依据是 80 条 pilot 中它们均取得 `net_correct_gain=2`。这只能形成候选验证排名，不能
外推为未评估的其余 25 层之间的正式名次。

第五步由 `finalize` 在不再加载模型的情况下完成所有统计。它首先审计合同要求的 11 份 result、comparison report 和
prediction 的引用关系，逐一验证 suite、plan、数据角色、样本数、pilot 索引及文件 SHA256。审计通过后，
以每个 `source_index` 将候选和 `ALL-I4` 配对，记录 `recovered/regressed/stable_correct/stable_wrong`、
解析和结构化输出变化。金额、比例/利率、日期/期限和 89 条长上下文直接从 883 条 full-dev 预测切片派生，
不重复生成；91 条高风险集来自独立推理，因此不会把普通 dev 子集冒充高风险证据。

第六步是统计不确定性，而不仅给出一个点估计。对每个层建立 883 个配对差值：候选恢复记为 `+1`、退化记为
`-1`、结果不变记为 `0`；10,000 次 bootstrap 每次都成对重采样这些差值，得到相对 `ALL-I4` 的准确率恢复
点估计和 95% percentile 区间。成对抽样保留了“同一道题在两个模型下”的相关性，比把两个模型分别抽样更适合
当前对照实验。第一、第二名区间重叠或最高净增并列时，程序自动降低结论强度。

最后，程序先按冻结的七级 tie-break 生成 layer 10/22/23 三层候选排名，再单独应用保护层准入规则。排名靠前不等于自动
保护：候选还必须满足 full-dev 净增为正，且高风险和长上下文均不出现净退化，最多取 3 层。这个二阶段设计
避免“总体多答对几题，但在高风险金融问题上新增错误”的层直接进入 7.4。

#### 各个最终产物分别有什么用

- `experiment_manifest.json`：实验合同、输入身份、suite 和最终产物哈希的总入口；
- `layer_metrics.jsonl`：三个候选层的完整 dev、三类业务标签、长上下文、高风险、格式和 bootstrap 指标；
- `sample_transitions.jsonl`：逐 suite、逐层、逐题的恢复/退化证据，可回溯具体 source index；
- `ranking.json`：机器可读的候选排名、结论强度、分析范围和推荐保护层；
- `ranking.csv`：便于人工筛选、画图或导入表格的扁平排名；
- `report.md`：面向评审的摘要和 Top 10；
- `validation.json`：pending、确定性、层数、数据规模和最终验收状态。

新增的一键脚本 `scripts/run_layer_sensitivity_final.sh` 只是上述命令的安全编排，不另设实验逻辑。每个计划仍由
独立 Python 进程执行，既确保模型从相同母版开始，也能在 16GB GPU 上及时释放 CUDA allocator。脚本和单计划
入口都支持幂等续跑：只有 result、report 和预测哈希能互相对应时才跳过完成项，否则最终审计会拒绝它。

确定性 suite 优先运行；如果 `ALL-I4` 或 layer 22 不能与原始 pilot 逐行一致，应停止后续正式质量运行并
先排查环境。full-dev 预测完成后，金额、比例/利率、日期/期限不再单独执行推理，而是按冻结的多标签文件从
883 条预测派生。三个类别的固定总数是 338、567、709，标签允许重叠。最长 10% 上下文同样从 full-dev
预测按冻结的 89 个 source index 派生。每个分组都同时写出基线和候选的 `正确数/总数`、accuracy、
recovered、regressed、net gain、解析数/率及结构化输出数/率，避免小样本只报告百分比。

候选排名口径在看到 full-dev 结果前固定为：

1. full-dev `net_correct_gain` 降序；
2. full-dev `recovered_count` 降序；
3. 91 条高风险集 `recovered` 降序；
4. 89 条长上下文 `recovered` 降序；
5. structured output 净恢复量降序；
6. schema v2 权重 NMSE 恢复量降序；
7. layer index 升序。

每个层相对 `ALL-I4` 的 full-dev 正确率差使用 10,000 次确定性配对非参数 percentile bootstrap 生成 95%
区间。bootstrap 每次对同一组 source index 成对重采样，不能把基线和候选分别抽样；否则会丢失逐题相关性。
若第一、第二名区间重叠或最高净增存在并列，`ranking_conclusion_strength` 自动降为 `low`，不得把细小名次差
解释成稳定优劣。

保护层建议也在运行前冻结：最多选择排名最前的 3 层，并要求完整 dev 净增大于 0、高风险净增不小于 0、
长上下文净增不小于 0。不满足条件的层即使静态量化误差或激活误差突出，也不会自动进入 7.4 的保护集合。

`finalize` 运行前会检查候选合同要求的 11 次运行全部存在，并逐一核对 suite、plan、comparison report SHA256、预测
SHA256、数据角色、样本数及 determinism pilot 索引哈希。任一错配都会终止，失败结果不会进入排名。成功后
生成以下 7 个正式产物：

```text
out/layer_sensitivity/final_evaluation/
├── experiment_manifest.json
├── layer_metrics.jsonl
├── sample_transitions.jsonl
├── ranking.json
├── ranking.csv
├── report.md
└── validation.json
```

服务器建议按“先确定性、再完整 dev、最后高风险”的顺序执行：

```bash
python3 scripts/run_layer_sensitivity.py prepare-final

python3 scripts/run_layer_sensitivity.py run-final-plan \
  --suite determinism --plan-id all_int4
python3 scripts/run_layer_sensitivity.py run-final-plan \
  --suite determinism --plan-id layer_22_int8

python3 scripts/run_layer_sensitivity.py pending-final --suite determinism
python3 scripts/run_layer_sensitivity.py audit-determinism

for plan in all_int4 all_int8 layer_10_int8 layer_22_int8 layer_23_int8; do
  python3 scripts/run_layer_sensitivity.py run-final-plan \
    --suite full_dev --plan-id "$plan" || break
done

for plan in all_int4 layer_10_int8 layer_22_int8 layer_23_int8; do
  python3 scripts/run_layer_sensitivity.py run-final-plan \
    --suite high_risk --plan-id "$plan" || break
done

python3 scripts/run_layer_sensitivity.py pending-final
python3 scripts/run_layer_sensitivity.py finalize
```

也可以直接运行等价的一键脚本；它使用相同的幂等跳过逻辑，意外中断后重新执行即可从未完成计划继续：

```bash
bash scripts/run_layer_sensitivity_final.sh
```

如服务器需要指定虚拟环境解释器，可使用独立环境变量，而不修改脚本：

```bash
LAYER_SENSITIVITY_PYTHON=.venv/bin/python \
  bash scripts/run_layer_sensitivity_final.sh
```

每次计划使用独立 Python 进程，因此 7B 模型退出后会释放 CUDA allocator；脚本支持安全重跑，已完成计划按
结果和预测哈希跳过。最终只有在 `pending-final` 的三个列表全空、`validation.json status=pass`、10/22/23
三层全部进入 candidate ranking、两项 determinism 均 exact match 后，才能勾选候选验证相关验收项，并把
`recommended_protected_layers` 作为 7.4 的候选层级输入。`ranking.json status=candidate_subset_final`、
`full_layer_ranking_claim_allowed=false` 必须保留，不能把该结果改写成 28 层完整排名。

### 7.1.4–7.1.5 候选验证结果与结论（2026-10-12）

服务器已完成候选合同并回传全部产物。`experiment_manifest.json status=completed`，三个 suite 的 pending
列表均为空，`validation.json status=pass`、`invalid_runs=[]`。合同覆盖 determinism 2 项、full-dev 5 项和
high-risk 4 项，共 11 项；切换候选合同后实际剩余的 7 项推理累计约 74.7 分钟。旧的未完成
`full_dev/layer_00_int8` 目录没有 `result.json`，不在新合同允许的 plan ID 中，也没有进入审计或排名。

两项确定性复跑均逐行完全一致：`ALL-I4` 的原始与复跑预测 SHA256 均为
`4bcd0dac2343e532c14b7838d43d6d7dbe3cd7786542a41ef3490e7d6456c11b`，layer 22 均为
`d1451160bf50691777e7b76625545394459e717355017bc5710a7a74981d7b56`。最终七项汇总产物的实际 SHA256
也全部与 manifest 中的 `final_artifact_sha256` 一致，因此以下结论具有完整的合同、预测和文件哈希追溯链。

完整 dev 的候选结果如下。置信区间是相对 `ALL-I4` 的 10,000 次配对 bootstrap 95% percentile 区间，
单位为准确率百分点：

| 方案 | 正确数/883 | 准确率 | 恢复 | 退化 | 净增 | 95% 区间 | 高风险净增 | 长上下文净增 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALL-I4` | 328/883 | 37.15% | — | — | — | — | 基线 18/91 | 基线 38/89 |
| `ALL-I8` | 338/883 | 38.28% | — | — | +10 | 未计算层级区间 | 未运行 | — |
| `layer-23-I8` | **344/883** | **38.96%** | 22 | 6 | **+16** | **[+0.68, +3.06]** | 0 | +1 |
| `layer-10-I8` | 330/883 | 37.37% | 12 | 10 | +2 | [-0.79, +1.25] | +1 | +1 |
| `layer-22-I8` | 328/883 | 37.15% | 15 | 15 | 0 | [-1.25, +1.25] | +1 | -1 |

layer 23 是唯一 bootstrap 区间整体高于 0 的候选：相对 `ALL-I4` 提升 16 题、1.81 个百分点，且 22 次
恢复只伴随 6 次退化。它在三个允许重叠的业务标签子集上也全部正向：金额为 `133/338`，相对基线净增 6；
比例/利率为 `204/567`，净增 13；日期/期限为 `277/709`，净增 15。89 条长上下文恢复 1、退化 0；91 条
高风险样本保持 18/91，恢复和退化均为 0。解析和结构化输出由 877/883 提升到 878/883，但格式重试由 12
增加到 13。该层的业务收益覆盖多个分组，并非只由 pilot 中的四个边界样本造成。

layer 10 满足预注册准入条件，但证据较弱。它在 full-dev 只净增 2 题，12 次恢复与 10 次退化接近抵消，
bootstrap 区间跨 0；高风险和长上下文各净增 1，但解析/结构化输出从 877 降到 876，结构化转移为恢复 1、
丢失 2。因此它可以保留为第二组合候选，不能单独表述为稳定有效的保护层。

layer 22 不进入保护集合。它在 full-dev 恢复和退化各 15 题、净收益为 0，比例/利率与日期/期限子集均净退化
2 题，长上下文恢复 1、退化 2。虽然高风险集净增 1，但它同时违反“full-dev 必须正收益”和“长上下文不得
退化”两个门槛。

自动合同按照预注册规则输出 `recommended_protected_layers=[23,10]`。工程上的保守解释进一步区分为：

- layer 23：确定保护层，进入后续混合精度组合；
- layer 10：可选组合候选，仅在 `23+10` 实测优于只保护 23 时保留；
- layer 22：排除，不进入后续保护组合。

`ranking_conclusion_strength=low` 的直接原因是 layer 23 与 layer 10 各自相对基线的 bootstrap 区间存在
重叠。这不否定 layer 23 相对 `ALL-I4` 的正收益，因为 layer 23 自身区间下界仍高于 0；它限制的是“三层之间
精确名次具有很高稳定性”的表述。另一个限制是当前只验证了 pilot 选出的 3/28 层，高风险基线本身也只有
18/91，因此不能据此声称获得了全模型完整排名或生产级金融安全性。

`layer-23-I8` 的正确数高于 `ALL-I8`，不表示单层 INT8 在数值意义上比全 INT8 更精确。fake-quant 扰动会
改变离散生成边界，局部混合精度可能在特定任务上得到更高正确数；与此同时 `ALL-I8` 解析 881/883、仅 5 次
格式重试，格式稳定性优于 layer 23 的 878/883 和 13 次重试。7.1 结果只用于质量敏感性与候选选择，不允许
推导真实显存、延迟或吞吐收益。

### 7.1 收口后续路线

7.1 的候选范围已经完成验收，下一步按 TODO 顺序进入 7.2 模块敏感性分析。首先实现 7.2.1 的实验契约和
模块自动发现，不立即启动大规模服务器评测：复用 7.1 冻结身份与 `ALL-I4` 预测，登记 28 层 Attention
`q/k/v/o_proj`、MLP `gate/up/down_proj`、Embedding、RMSNorm、`lm_head` 和 KV Cache，检查 Embedding 与
`lm_head` 是否 tied，并把权重 fake-quant 与真实 KV Cache 量化能力严格分开。

7.2.1 合同、配置和单元测试通过后，再运行 7.2.2 的固定 80 条 pilot。先用 `ATTN-ALL-I8` 与
`MLP-ALL-I8` 判断总体上是哪类子模块贡献恢复，再执行逐层 Attention/MLP 隔离计划；完整 dev 晋级规则必须在
看到结果前冻结。7.2 和 7.3 完成后，7.4 至少比较 `ALL-I4`、只保护 layer 23、同时保护 layer 23+10 三种
层级组合，并继续测试模块级保护方案。不得假设 layer 23 与 layer 10 的单层收益可以线性相加。

## 7.2 模块敏感性分析实现设计（2026-10-12）

### 7.2 与 7.1 的区别

7.1 把一个完整 Transformer block 当作最小实验单位。它能回答“第几层值得从 INT4 恢复到 INT8”，但不能
解释收益究竟来自该层的 Attention 还是 MLP，也没有触碰 Embedding、Norm、`lm_head` 和 KV Cache。

7.2 将问题进一步拆开，但仍保持 7.1 的受控变量原则：模型母版、`ALL-I4`、80 条 pilot、Prompt、生成参数、
评分器和逐样本顺序全部复用，只有计划中声明的模块允许改变。这样，候选和基线的差异才可以归因到目标模块，
而不是数据重新抽样、Prompt 改动或另一种量化算法。

7.2 包含两种方向相反的实验：

| 实验类型 | 起点 | 唯一变量 | 主要问题 |
| --- | --- | --- | --- |
| Attention/MLP 恢复实验 | Transformer Linear 全部为 `ALL-I4` | 目标模块从 INT4 恢复为 INT8 | 提高哪个模块的精度最能挽回质量 |
| 特殊模块退化实验 | `ALL-I4`，特殊模块仍为 FP16 | Embedding、Norm、`lm_head` 或 KV Cache 单独降精度 | 哪些特殊模块可以安全量化，哪些必须保护 |

因此两类实验不能混成一个分数榜。恢复实验希望 `net_correct_gain` 越大越好；退化实验首先要求
`net_correct_change` 不低于质量门禁，然后才能讨论参数、显存或速度收益。

### 7.2.2 Attention 与 MLP 受控恢复矩阵

#### 模块边界

每个 Qwen2.5 decoder block 被拆成两组二维 Linear 权重：

- Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`；
- MLP：`gate_proj`、`up_proj`、`down_proj`。

Attention 负责构造 query/key/value、建立 token 间的信息交互并输出注意力结果；MLP 在每个 token 位置执行
高维非线性变换，通常占 block 中更大的参数比例。两组模块承担的功能和参数规模不同，量化误差对业务输出的
影响也不一定相同，所以不能只用整个 block 的平均误差推断它们的敏感性。

#### 为什么从 ALL-I4 恢复到 INT8

如果分别从 FP16 出发，只量化 Attention 或只量化 MLP，那么两个候选面对的其余权重环境不同于最终计划使用的
INT4 模型，结论不一定能指导混合精度部署。当前实现统一从 `ALL-I4` 开始：所有 block 的七个 Linear 都先按
W4A16 g32 量化，然后只把计划指定的模块改为 W8A16。候选与基线之间只存在一组精度差异。

恢复量定义为：

```text
quality_recovery(plan) = accuracy(plan) - accuracy(ALL-I4)
net_correct_gain       = recovered_count - regressed_count
```

其中 `recovered` 表示同一道题在 `ALL-I4` 下错误、候选下正确；`regressed` 表示 `ALL-I4` 下正确、候选下
错误。只比较总正确数会掩盖“恢复很多题、同时又破坏很多题”的候选，因此实现同时保存完整逐题转移。

#### 计划矩阵

实现生成以下 59 个 Attention/MLP 可运行计划：

| 数量 | 计划 | 含义 |
| ---: | --- | --- |
| 1 | `attn_all_int8` | 28 层 Attention 全部为 INT8，MLP 保持 INT4 |
| 1 | `mlp_all_int8` | 28 层 MLP 全部为 INT8，Attention 保持 INT4 |
| 28 | `layer_00_attn_int8`～`layer_27_attn_int8` | 仅目标层四个 Attention Linear 恢复 INT8 |
| 28 | `layer_00_mlp_int8`～`layer_27_mlp_int8` | 仅目标层三个 MLP Linear 恢复 INT8 |
| 1 | `attn_mlp_all_int8_audit` | Attention 与 MLP 全部恢复 INT8，只用于复现 7.1 `ALL-I8` 的实现审计 |

全局计划用于回答“总体上保护 Attention 还是 MLP 更有效”；逐层计划用于回答“哪一层的哪一类子模块最敏感”。
两者都需要，因为全局收益不是 28 个单层收益的简单相加：量化扰动经过多层传播后会发生放大、抵消和生成边界
跳变。

`attn_mlp_all_int8_audit` 不用于排名。它会把 196 个 Transformer Linear 的 bits、group size、numel、MSE、
NMSE、cosine、最大误差和饱和率与 7.1 的 `ALL-I8` 逐模块精确比较。任何字段不一致都说明 7.2 的模块选择或量化
实现偏离 7.1，程序会立即失败，后续排名不可信。

#### 量化数学与误差指标

Linear INT4 沿每个输出行的输入维度按 32 个权重分组，每组独立计算 scale；INT8 按输出通道计算 scale。
两者都使用对称量化，零点固定为 0：

```text
qmax  = 2^(bits - 1) - 1
scale = max(abs(w)) / qmax
q     = clip(round(w / scale), -qmax, qmax)
w_hat = q * scale
```

当前权重路径把 `w_hat` 写回 FP16 tensor，是量化—反量化 fake-quant。它真实注入了舍入和裁剪误差，适合比较
质量敏感性，但 tensor 最终仍按 FP16 存储和计算，所以不能用来声称模型文件、显存或延迟已经下降。

每个 Linear 保存参数量、MSE、normalized MSE、cosine similarity、最大绝对误差和饱和率；层级和模块族汇总
按 `numel` 加权，避免小矩阵和大矩阵被不合理地赋予相同权重。业务侧则在固定 80 条 pilot 上保存准确率、
解析率、结构化输出率、格式重试和逐样本恢复/退化。

#### 隔离和续跑保障

每个计划在写入磁盘前都会生成完整的生效精度映射。逐层 Attention 计划必须恰好改变四个 Linear，逐层 MLP
计划必须恰好改变三个 Linear；缺层、重复层、越界层和额外模块变化都会被拒绝。`plan.json` 包含父
`ALL-I4` 引用、精度合同、生效映射和合同 SHA256，执行时重新计算哈希，防止准备后被手工修改。

每个计划由独立 Python 进程重新加载同一个 merged FP16 母版，再施加该计划的完整量化分配，不继承上一个
计划已经修改的权重。`result.json status=completed` 或经过能力审计的 `status=unsupported` 都属于终态；
中断后重新执行会跳过终态计划，只继续缺失或失败项。

#### 为什么第 10 层与第 23 层进入候选

本节的层号沿用代码的零基索引：`model.layers.10` 是按自然语言计数的第 11 个 decoder block，
`model.layers.23` 是第 24 个；模型共有 28 个 block。两个 block 的结构与其他层相同，都按
“RMSNorm → Self-Attention → 残差 → RMSNorm → MLP → 残差”处理隐藏状态，不存在预先写死的“金融计算层”。
Attention 的 `q/k/v/o_proj` 负责选择并汇总上下文 token 信息；MLP 的 `gate/up/down_proj` 对每个 token 的
表示执行门控非线性变换。对 FinQA 而言，可以把前者理解为关联问题、年份、表格字段和值，把后者理解为转换
已经汇集的特征，但具体语义不是架构能够直接保证的，必须由业务评测验证。

按深度位置，第 10 层位于模型中部约 39%，此时隐藏状态通常已经包含基础语义，并继续形成问题、表格证据和
运算关系的中间表示；第 23 层位于后段约 86%，后面只剩四个 block，其输出更接近最终 logits，因此量化扰动
留给后续层修正的空间更少。这里的“中部组合、后部收敛”只是基于层位置的机制解释，不是证明某一层固定负责
某一种金融运算。

选择两层的直接依据来自 7.1 的配对业务结果，而不是权重误差大小：

| 层 | full-dev 正确数 | 恢复/退化 | 净增 | 95% 配对 bootstrap 区间 | 结论强度 |
| ---: | ---: | ---: | ---: | ---: | --- |
| ALL-I4 | 328/883 | — | — | — | 对照 |
| 23 | 344/883 | 22/6 | +16 | `[+0.68%, +3.06%]` | 较强，区间未跨 0 |
| 10 | 330/883 | 12/10 | +2 | `[-0.79%, +1.25%]` | 较弱，区间跨 0 |

第 23 层的收益覆盖金额、比例/利率、日期/期限三类业务标签，分别净增 6、13、15 题；89 条最长上下文净增
1 题，91 条高风险集持平。7.2 的 80 条子模块拆分进一步显示：第 23 层 Attention 单独恢复 source index
`611`，MLP 单独恢复 `31`，整层恢复时恰好恢复 `31/611`，三者均无退化。这说明 pilot 中两类模块提供了互补
收益，与 full-dev 的整层正收益方向一致，因此质量优先时暂时同时保护第 23 层 Attention 与 MLP。

第 10 层则需要更保守地解释。80 条 pilot 中，Attention 恢复 `191/611`、退化 `440`，净增 1；MLP 恢复
`191/402`、无退化，净增 2；整层恢复 `191/402/611`、退化 `440`，仍只净增 2。加入 Attention 没有在总正确
数上超过 MLP 单独恢复，因此当前候选只保护第 10 层 MLP。第 10 层 full-dev 的置信区间跨 0，不能把它描述为
已经确认的重要层，只能称为需要组合验证的次级候选。

权重和激活误差也说明不能从数值误差直接推断业务重要性。第 23 层的 activation recovery 只排第 14，第 10 层
排第 27；它们并不是激活误差恢复最大的层。量化误差只有在改变年份选择、表格行、运算符、单位或数值 token
的决策边界时才会改变最终正确性。因此本项目的保护顺序以配对业务恢复和风险子集非退化为主，MSE/NMSE、
cosine 等只作为机制证据和复核指标。

根据这些已观察结果，配置中预注册 `candidate_l23_all_l10_mlp_int8`：第 23 层 Attention 与 MLP、以及
第 10 层 MLP 使用 INT8，其余 Transformer Linear 使用 INT4。该计划只修改 10 个 Linear，并通过模块隔离
测试；因为不同层的扰动可能放大或抵消，禁止把三个单项 pilot 收益直接相加。对应原始证据保存在
`out/layer_sensitivity/final_evaluation/ranking.json` 和 `out/module_sensitivity/pilot/<plan_id>/result.json`。

#### 跨层组合验证结果

组合计划已在相同的冻结 80 条 pilot 上完成，得到 38/80 正确、解析率与结构化输出率均为 100%，相对
`ALL-I4` 恢复 2 题、退化 0 题、净增 2 题。它没有达到两个单项结果简单相加所暗示的 40/80：

| 配置 | 正确数 | 恢复/退化 | 净增 |
| --- | ---: | ---: | ---: |
| `ALL-I4` | 36/80 | — | — |
| 第 23 层 Attention + MLP INT8 | 38/80 | 2/0 | +2 |
| 第 10 层 MLP INT8 | 38/80 | 2/0 | +2 |
| 第 23 层全部 + 第 10 层 MLP INT8 | 38/80 | 2/0 | +2 |

逐题对照显示组合不是单项恢复集合的并集。第 23 层单独恢复 source index `31/611`，第 10 层 MLP 单独恢复
`191/402`，组合只保留 `191/611`。其中 `31` 从基线的 `86.5%` 经第 23 层单项变为正确的 `87.0%`，组合后
变为 `86.7%`；`402` 从基线的 `6.5` 经第 10 层 MLP 单项变为正确的 `103.1`，组合后变为 `648.5`。
这些是“单项恢复没有保留”，不是相对 `ALL-I4` 的退化；组合仍未破坏任何基线正确题。

原因是第 10 层产生的隐藏状态还要经过第 11～22 层后进入第 23 层。两处权重同时改变后，第 23 层面对的输入
已不同于单项实验，扰动可能相互放大、抵消或跨越新的 token 决策边界。组合的 80 条输出中，62 条与
`ALL-I4` 原始输出完全一致、63 条与第 23 层单项一致、66 条与第 10 层 MLP 单项一致，另有 8 条与三个对照
都不同，确认它形成了新的生成轨迹，而不是机械选择某个单项结果。

组合把全模型权重 NMSE 从 `0.0101377` 降至 `0.0094809`，约下降 6.48%，但业务正确数没有超过任一单项。
它把约 4.37 亿个权重元素从 INT4 恢复为 INT8，理论权重存储比 `ALL-I4` 增加约 218 MB，其中相对“仅第
23 层”额外保护第 10 层 MLP 约增加 102 MB，却没有带来 pilot 增量正确题。因此当前质量—存储权衡下，组合
方案被第 23 层单项支配，不作为默认推荐。

还需注意，当前主指标按最终数值评分。组合恢复的数值虽然匹配参考答案，但部分生成 `formula` 不能严格推出
对应 `value`，所以 `+2` 只证明最终数值命中，不等价于证据和计算链完全正确。后续晋级应同时检查公式一致性，
不能把该 pilot 结果描述为金融推理能力已经稳定提高。

阶段性建议调整为：质量与工程效率优先时，只保护第 23 层 Attention 和 MLP 为 INT8；第 10 层 MLP 保留为
独立备选，但因其 full-dev 证据尚缺失且组合没有增量，暂不并入默认策略。第 23 层拥有 883 条 full-dev 的
`+16` 证据，而组合是否值得进入 full-dev 只在需要排除 80 条小样本偶然性时再评估。

### 7.2.3 Embedding、Norm、lm_head 与 KV Cache

#### 为什么这些对象单独测试

这些对象在 7.1 中被刻意保持为 FP16：

- Embedding 是所有 token 的输入查表矩阵，一次扰动会影响后续全部 block；
- `lm_head` 直接把最终 hidden state 映射为词表 logits，细小误差可能改变下一个 token 的 argmax；
- RMSNorm 参数量很小，但它控制每层激活缩放，压缩收益有限且可能数值敏感；
- KV Cache 是推理过程中动态生成的 key/value activation，不是静态模型权重，不能使用权重 fake-quant 代替。

因此特殊模块实验保持 Transformer Linear 为相同的 `ALL-I4`，再把一个特殊对象从 FP16 单独降为 INT8 或
INT4。其正确解释是“降精度造成多少损失”，不是“恢复了多少质量”。结果使用
`degraded_from_fp16_component_baseline`、`recovered_from_fp16_component_baseline` 和
`net_correct_change`，避免与 7.2.2 的恢复分数混淆。

#### Embedding 与 lm_head

Embedding INT4 沿 embedding dimension 使用 g32 对称分组，INT8 按 embedding row；`lm_head` INT4 沿输入
维度使用 g32，INT8 按输出通道。程序不假设二者一定独立，而是读取真实模型参数对象和底层 storage：

- 未共享时，生成 `embedding_int8/int4` 与 `lm_head_int8/int4` 四个独立计划；
- 共享时，只生成 `shared_embedding_lm_head_int8/int4` 两个联合计划，禁止把同一矩阵重复量化后宣称两个独立
  因果结论。

这也是基础矩阵计划数存在两种情况的原因：tied weights 时共 69 个计划、68 个可运行；untied weights 时共
71 个计划、70 个可运行。唯一不可运行项是复用已有证据的 `all_int4_ref`；`kv_cache_fp16_ref` 现在会真实
执行，作为 KV Cache INT8/INT4 的时延、显存和 cache bytes 对照。基础
拆分结果完成后，可以在配置中追加预注册的 `candidate_recovery_plans`，但必须记录选择依据并继续执行相同的
模块隔离校验，不能用追加候选改变已经观察过的单模块计划。

当前已追加一个候选计划，因此在 Embedding 与 `lm_head` 不共享权重的真实模型上，实际矩阵为 72 个计划、
71 个可运行。候选的选择依据、模块集合和解释限制见上一节。

当前实现会记录 Embedding/`lm_head` 的量化前后范围、NaN/Inf、误差和理论低比特存储（packed weight 与 FP16
scale 分开计数）。`lm_head` 还在固定 calibration prompt 上保存 logits MSE、top-1 一致率、top-k 重合率，
以及参考最高数值 token 的排名变化。该 probe 使用同一份经过 `ALL-I4` 处理的 decoder hidden state，只替换
原始/量化 `lm_head` 权重，因此测到的是输出头本身造成的 logits 扰动；它仍是 fake-quant 机制证据，不代表
部署 kernel 的实际吞吐或显存。

##### `lm_head` 专项探针的含义与实现

业务评测只告诉我们最终答案是否正确，不能解释错误是在中间推理阶段产生，还是最终 hidden state 映射到词表
时被 `lm_head` 的量化误差改变。`lm_head` 专项探针就是补充这一层机制证据：观察完全相同的最后一层 hidden
state 经过原始 FP16 输出头和 fake-quant INT8/INT4 输出头后，下一 token 的 logits 排序发生多大变化。

`lm_head_int8` 和 `lm_head_int4` 计划按以下步骤执行：

1. 模型刚加载时，先把原始 FP16 `lm_head.weight` 复制到 CPU，避免量化后丢失参考；
2. 按计划把 Transformer Linear 置为相同的 `ALL-I4`，并只把 `lm_head` 量化成目标位宽；
3. 从冻结 calibration 集顺序取前 4 条，使用统一 FinQA prompt，并追加严格 JSON 输出提示；输入最多截断为
   1024 token；
4. 只运行 Qwen decoder body（`model.model`），取得最后一个输入位置的 hidden state；同一 hidden state 同时乘以
   原始和量化后的 `lm_head` 权重，从而隔离输出头本身的影响；
5. 词表矩阵在 CPU 上按 4096 行分块做矩阵—向量乘法，避免同时构造两份巨大的 GPU logits/权重副本。

每条样本记录四类指标：

- `logits_mse`：原始与量化 logits 的均方误差；
- `top1_consistent`：两个输出头的 argmax token 是否相同；
- `topk_overlap`：默认 top-5 token 集合的重合比例；
- `reference_top_numeric_rank_change`：先从 tokenizer 词表识别数字、金额、百分比等数值形 token，找到原始输出中
  分数最高的数值 token，再观察它在量化输出中的排名变化。

汇总项为 mean logits MSE、top-1 一致率、平均 top-k 重合率和数值 token 平均绝对排名变化，逐样本明细与汇总
写入 `<plan_id>/lm_head_probe.json`。这里的“数值 token”是词表层面的诊断对象，不保证就是该题最终正确答案；
探针也只看给定 prompt 后的下一 token，不替代后续 80 条完整自回归业务评测。其额外耗时主要来自保存约一份
输出头权重和在 CPU 上计算两组完整词表 logits。

#### Block Norm 与最终 Norm

Qwen2.5 使用 RMSNorm，因此文档和产物统一称为 Norm，不把它误标成 LayerNorm。实现把 Norm 分为：

- 28 层内的 `input_layernorm` 与 `post_attention_layernorm`，对应 `block_norm_int8/int4`；
- 模型输出前的最终 Norm，对应 `final_norm_int8/int4`。

Norm 权重是一维向量，不适用 Linear 的 per-output-channel/g32 规则。当前使用对称 per-tensor INT8/INT4，零点
固定为 0，并单独实现一维量化路径。两组 Norm 分开测试，是为了避免“所有 Norm 一起变化”后无法判断风险来自
层内缩放还是最终输出缩放。运行产物会记录权重范围与量化前后 NaN/Inf；考虑
到 Norm 参数量很小，即使质量无损，也必须证明存在实际工程收益才值得降低精度。

#### KV Cache 为什么必须走真实后端

KV Cache 保存每一层历史 token 的 key/value。上下文增长时，其容量大致随层数、序列长度、batch、KV head
数量、head dimension 和每元素字节数线性增长。把 cache 从 FP16 降到 INT8/INT4 理论上能够降低动态显存，
但还会引入每步量化/反量化成本；短上下文下甚至可能变慢。因此 KV Cache 不能使用“改一份静态权重再测质量”
的方式模拟。

当前实现使用 Transformers `cache_implementation="quantized"`，并通过 HQQ 分别请求 `nbits=8` 和 `nbits=4`；
HQQ 的 key/value 推荐量化轴固定为 1。依据是
[Transformers KV Cache 官方说明](https://huggingface.co/docs/transformers/main/kv_cache)。运行分为两道门禁：

1. `inspect-model` 检查模型 forward 是否支持 `use_cache`、`past_key_values`，以及当前 Transformers 是否暴露
   QuantizedCache API；
2. `probe-kv-cache` 在真实模型和设备上执行短生成，检查 HQQ 依赖、INT8/INT4 参数及后端实际可运行性。

只有 probe 成功的位宽才允许进入 80 条业务评测。缺少 HQQ、版本不支持或运行失败时，计划写入
`status=unsupported` 和原始错误，不静默回退为 FP16，也不阻塞其他权重计划。短 probe 成功后，正式计划会在
相同输入长度和生成长度下分别运行 FP16、INT8、INT4，记录 cache class、cache tensor bytes、峰值显存、
TTFT、TPOT、端到端延迟和吞吐。只有返回的 cache 对象可识别为真实量化 cache、能够观察到 cache tensor 且
通过低比特驻留门禁时，才设置 `performance_claims_allowed=true`；否则不得声称真实显存或速度收益。

##### KV Cache 专项探针的含义与实现

KV Cache 探针分为“能力探测”和“正式 benchmark”两层。前者回答“后端能不能运行”，后者回答“这次运行是否
真的保留了低比特 cache，以及时间和显存表现如何”。两者都不能用静态权重 fake-quant 代替。

能力探测由 `probe-kv-cache` 执行。程序先检查 HQQ 依赖和 Transformers QuantizedCache API，再使用固定短文本
生成 8 个 token，分别传入 INT8/INT4 的 `cache_implementation="quantized"`、`backend="hqq"` 和 `nbits`。
调用成功只会把该位宽加入 `runtime_verified_precisions`，允许它进入 pilot；此时仍明确记录
`low_bit_cache_residency_verified=false` 和 `performance_claims_allowed=false`。缺依赖、参数不兼容或运行异常会
写成 `unsupported`，绝不静默回退到 FP16。

正式 benchmark 在 `kv_cache_fp16_ref`、`kv_cache_int8` 和 `kv_cache_int4` 计划内执行：

1. 从 calibration 集选择渲染后最长的一条 prompt，并统一截断到 1024 token；三个计划使用相同 batch、输入上限
   和生成长度；
2. 先生成 2 token 预热；再独立生成 1 token 测近似 TTFT，最后独立生成固定 32 token 测端到端时间；
3. CUDA 运行前清空缓存统计、重置 peak memory，并在计时前后同步设备；记录首 token/完整生成峰值显存、端到端
   延迟、吞吐，以及 `(32-token 总时间 - TTFT) / 31` 得到的近似 TPOT；
4. 使用 `return_dict_in_generate=true` 取得返回的 `past_key_values`，递归遍历其中的 tensor；按底层指针去重后记录
   cache tensor 总字节数、tensor 数、各 dtype 的字节分布，以及除以输入加生成 token 数后的
   `cache_bytes_per_token`；
5. 只有请求位宽为 4/8、cache 类名明确包含 Quantized/HQQ、能够观察到 cache tensor，并且内部确有整数存储时，
   才设置 `low_bit_residency_verified=true`。这可以防止“接受了量化参数，却长期只保存 FP16 cache”的假阳性。

结果写入 `<plan_id>/kv_cache_benchmark.json`，同时嵌入 `result.json`。其中 peak memory 是该次生成期间 CUDA
allocator 的整体峰值，并非纯 cache 独占值；TTFT 和 TPOT 来自两次独立的 `generate` 调用，也属于工程近似。
因此最终必须将 INT8/INT4 与相同输入的 FP16 对照一起解释，并结合 `cache_dtype_bytes`、输出质量和多次运行波动，
不能只凭一次延迟或类名宣称部署收益。正式计划仍会额外运行冻结的 80 条完整业务评测，用于判断 cache 量化是否
改变最终答案。

简而言之，`lm_head` 探针定位“输出词表决策是否被权重量化扰动”，KV Cache 探针验证“动态历史状态是否真的
以低位宽驻留并带来可测收益”。它们都是对业务准确率的机制补充，而不是另一个准确率评分器。

### 7.2 完整 pilot 结果分析（2026-10-13）

71 个可运行计划已经全部取得终态，`pending=[]`：69 个计划完成，KV Cache INT8/INT4 因真实 HQQ capability
probe 尚未执行而以 `unsupported` 结束。全部有时长的 69 个计划累计约 195.3 分钟，中位数约 164.1 秒；
block Norm INT4 因大量格式/生成扰动耗时约 444.6 秒，是明显离群项。以下结论仅针对固定 80 条 pilot，完整
dev/high-risk 尚未运行。

#### Attention 与 MLP 全局及逐层结果

全局模块族恢复没有收益：`ATTN-ALL-I8` 为 36/80、恢复/退化 4/4、净变化 0；`MLP-ALL-I8` 为 35/80、
恢复/退化 4/5、净变化 -1；Attention+MLP 全 INT8 审计也只有 36/80、净变化 0。因此不能使用“整个模块族统一
升到 INT8”的粗粒度策略，收益集中在少数层。

逐层 Attention 中，12 层净正、9 层持平、7 层净负，28 个计划净变化合计为 +5；逐层 MLP 中，14 层净正、
6 层持平、8 层净负，合计为 +9。这个汇总显示 MLP 有略强的恢复倾向，但各计划复用同一批样本，合计数不能
解释为一个可直接部署的组合收益。

pilot 中较强的候选为：

| 候选 | 正确数 | 恢复/退化 | 净变化 |
| --- | ---: | ---: | ---: |
| layer 6 Attention INT8 | 39/80 | 3/0 | +3 |
| layer 26 MLP INT8 | 39/80 | 3/0 | +3 |
| layer 10 MLP INT8 | 38/80 | 2/0 | +2 |
| layer 17 MLP INT8 | 38/80 | 2/0 | +2 |
| layer 25 MLP INT8 | 38/80 | 2/0 | +2 |
| layer 23 Attention INT8 | 37/80 | 1/0 | +1 |
| layer 23 MLP INT8 | 37/80 | 1/0 | +1 |

这些恢复高度重叠：layer 6 Attention 和 layer 26 MLP 都只恢复 source index `191/402/611`；layer 10、17、25
分别恢复这三题中的两个。加上 layer 23 的结果，七个候选的恢复并集也只有 `31/191/402/611` 四题。这再次
证明单项收益不能相加，多个位置可能只是把同一条生成轨迹推过相同的决策边界。

固定种子 10,000 次配对 bootstrap 的 pilot 区间也不支持强排名：`+3` 候选的 95% 区间约为 `[0,+8.75]`
个百分点，`+2` 约为 `[0,+6.25]`，`+1` 约为 `[0,+3.75]`，下界都没有严格大于 0。因此当前只能形成晋级
梯队，不能宣称 layer 6 一定优于 layer 26，或任一 `+1` 层已经稳定有效。

#### Embedding 结果

Embedding INT8 与 INT4 都保持 36/80，恢复/退化均为 0；原始输出分别有 4/7 条变化，但没有改变最终数值
正确性。INT8 理论上从 1039.5 MiB 降至约 520.0 MiB，节省 519.5 MiB（约 50%）；INT4 降至约
292.4 MiB，节省 747.1 MiB（约 71.9%）。两者均无 NaN/Inf。

因此 Embedding INT4 是当前特殊模块中收益最大的候选，INT8 可作为保守对照。由于 80 条样本不能排除低频
token、长上下文或特定金融词汇退化，仍需 full-dev/high-risk 后才能写入部署策略。

#### `lm_head` 结果

`lm_head` INT8 为 36/80、净变化 0，INT4 为 37/80、恢复/退化 2/1、净变化 +1。专项 probe 中两者的四条
样本 top-1 一致率均为 100%，但机制指标差异明显：

| 指标 | INT8 | INT4 |
| --- | ---: | ---: |
| mean logits MSE | 0.00723 | 0.17405 |
| mean top-5 overlap | 1.00 | 0.90 |
| 数值 token 平均绝对排名变化 | 49.0 | 394.25 |
| 理论存储节省 | 519.5 MiB | 747.1 MiB |

INT4 logits MSE 约为 INT8 的 24 倍，数值 token 排名扰动约为 8 倍，个别样本排名移动 700～872 位；其 pilot
`+1` 的配对区间跨 0，不能解释为 INT4 提升了质量。当前建议把 INT8 作为稳健候选，INT4 只作为高压缩率的
探索候选进入完整评测。

#### Norm 结果

block Norm INT8 为 37/80、恢复/退化 1/0，但全部 56 个 block Norm 量化后理论上仅节省约 0.191 MiB；如此小
的收益不足以抵消额外 kernel/数值风险。block Norm INT4 则降至 8/80，恢复/退化 2/30，79/80 条原始输出发生
变化；其特殊权重 normalized MSE 约 0.0341、最大绝对误差 0.787，说明 per-tensor INT4 无法安全表示这些
RMSNorm scale 的范围。

final Norm INT8 保持 36/80，但只节省约 3.5 KiB；INT4 为 34/80、净退化 2 题，且只节省约 5.2 KiB。
综合质量和收益，block/final Norm 均应默认保持 FP16；block Norm INT4 作为明确负结果禁止进入后续候选。

#### KV Cache 结果与限制

FP16 KV Cache 对照完成：1024 个输入 token、32 个生成 token 下，返回 `DynamicCache`，观测 cache tensor
约 60,497,920 bytes，即约 57,290 bytes/token；TTFT 约 152.4 ms、近似 TPOT 23.67 ms、吞吐 36.1 token/s，
CUDA allocator 峰值约 14.48 GiB。cache 的 56 个 tensor 全部为 FP16，与 28 层 key/value 对应。

INT8/INT4 当前结果不能解释为“HQQ 后端确定不支持”。`kv_cache_capability.json` 仍显示
`runtime_probe_completed=false`，两个计划只是因为真实后端没有先完成验证而写入 `unsupported`。因此当前
部署策略只能保留 FP16，且不得声称 INT8/INT4 的显存、延迟或吞吐收益。若要补测，必须先执行
`probe-kv-cache`；probe 成功后还需显式重跑两个已有 `unsupported` 计划，断点续跑不能直接复用它们。

#### full-dev 候选规模与建议

原始宽松规则会让 26 个独立 Attention/MLP 计划和 6 个特殊模块计划晋级，共 32 个计划；按照 7.1 实测每个候选
full-dev 加 high-risk 约 23.6 分钟计算，预计需要约 12.7～14 小时。由于其中大量候选只有 `+1`，为控制昂贵
评测的规模，并避免把 pilot 小样本波动全部带入第二阶段，已在任何 7.2 full-dev 结果产生前将晋级集合显式冻结为
以下 11 个计划：

| 类别 | 计划 | pilot 净变化 | 纳入理由 |
|---|---|---:|---|
| Attention 恢复 | `layer_06_attn_int8` | +3 | Attention 单层恢复计划中最强 |
| MLP 恢复 | `layer_26_mlp_int8` | +3 | MLP 单层恢复计划中最强 |
| MLP 恢复 | `layer_10_mlp_int8` | +2 | pilot 强候选，且已有跨层组合实验依据 |
| MLP 恢复 | `layer_17_mlp_int8` | +2 | pilot 强候选 |
| MLP 恢复 | `layer_25_mlp_int8` | +2 | pilot 强候选 |
| Attention 恢复 | `layer_23_attn_int8` | +1 | pilot 正向，且已有 7.1 full-dev 层级证据 |
| MLP 恢复 | `layer_23_mlp_int8` | +1 | pilot 正向，且已有 7.1 full-dev 层级证据 |
| 特殊模块退化 | `embedding_int8` | 0 | pilot 无质量退化且理论存储节省为正 |
| 特殊模块退化 | `embedding_int4` | 0 | pilot 无质量退化且理论存储节省为正 |
| 特殊模块退化 | `lm_head_int8` | 0 | pilot 无质量退化，数值扰动小于 INT4 |
| 特殊模块退化 | `lm_head_int4` | +1 | pilot 未退化且理论存储节省为正 |

该预注册修订把预计运行时间降至约 4.5 小时。Norm 因收益极小或 INT4 退化而默认保留 FP16；KV Cache INT8/INT4
尚未通过真实低比特驻留 probe，不进入本轮 full-dev；其余仅 `+1` 且缺少额外机制证据的单层计划也不进入本轮。
配置使用 `explicit_pilot_shortlist` 和明确的计划 ID 白名单，后续不得根据 full-dev 结果回改候选集合。

### 7.2.4 两阶段评测与统计实现

完整 pilot 矩阵完成后，`select-advancement` 严格读取预注册规则，而不是根据 full-dev 结果临时挑选计划；任何
可运行计划仍为 pending 时都会拒绝晋级。本轮恢复候选只从上述 7 个显式 Attention/MLP 白名单计划中校验，要求
80 条 pilot 至少净增 1 题；特殊模块只校验上述 4 个显式候选，要求质量不下降且理论存储为正。
晋级决策、排除原因和规则写入 `advancement.json` 并计算内容哈希。只有这一步完成，才能执行
`prepare-final`。

最终评测准备会冻结晋级计划、7.1 `ALL-I4` full-dev/high-risk 基线、883 条 dev、91 条 contextual 高风险集、
89 条最长上下文索引、业务标签以及 80 条确定性复跑样本。每次加载最终计划时都会重新校验文件 SHA256；完整
数据集还会逐条比较问题和原始记录哈希，防止只校验数量却误用了重排或修改后的数据。

每个晋级计划运行 883 条 full-dev 和 91 条高风险集；最长上下文与金额、比例/利率、日期/期限指标由冻结索引
从完整 dev 的逐样本预测中配对计算。另固定复跑第 23 层 Attention、第 23 层 MLP 和 Embedding INT8 三个
哨兵的冻结 80 条 pilot，并要求逐行预测完全一致。统计汇总保存 recovered/regressed/unchanged 转移、解析率、结构化输出率、format retry、业务子集
正确数，以及固定种子 10,000 次配对 bootstrap 的 95% 区间。

### 7.2.5 排名、保护策略与验收实现

最终汇总明确分成两张榜：Attention/MLP 恢复榜按 full-dev 净恢复、恢复数、高风险和长上下文表现排序；特殊
模块/KV Cache 榜先看质量门禁，再比较实际或理论收益。两类分数不混排。保护策略只接纳 full-dev、高风险、
长上下文均通过门禁的候选；同一特殊组件若多个位宽通过，只选择榜中最优位宽。KV Cache 若没有验证低比特
驻留则保持 FP16。

最终产物包括 `experiment_manifest.json`、`module_metrics.jsonl`、`sample_transitions.jsonl`、`ranking.json`、
`ranking.csv`、`protection_policy.json`、`report.md` 和 `validation.json`。代码会在汇总前检查计划状态、样本数、
预测文件哈希和确定性复跑，失败或缺失运行不能进入排名。报告状态使用 `candidate_subset_final`，明确它只覆盖
预注册晋级子集，不冒充 28 层完整模块排名；权重结果仍标注为 fake-quant，部署性能需要真实 kernel 验证。

#### 7.2.5 完整评测结论（2026-10-15）

11 个预注册候选的 full-dev、high-risk 和三个确定性哨兵已经全部完成，`validation.json` 状态为 `pass`，无
pending 或 invalid run。三个哨兵的逐行预测与 pilot 完全一致。本轮推理累计约 4 小时 30 分钟，其中 11 个
full-dev 约 3 小时 46 分钟、11 个 high-risk 约 35 分钟、三个确定性复跑约 9 分钟。`ALL-I4` 基线为
`328/883 = 37.146%`。

恢复候选的完整结果如下。准确率变化和置信区间单位均为百分点；业务列分别是金额、比例/利率、日期/期限子集的
净正确题变化，三个标签集合可以重叠，不能横向相加。

| 排名 | 计划 | full-dev | 净变化 | 准确率变化 | 配对 bootstrap 95% CI | high-risk | 长上下文 | 金额/比例/日期 | 结论 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `layer_10_mlp_int8` | 338/883 | **+10** | **+1.133** | **[+0.227, +2.039]** | 0 | 0 | +6 / +6 / +9 | 唯一统计上明确改善的主候选 |
| 2 | `layer_23_mlp_int8` | 332/883 | +4 | +0.453 | [-0.793, +1.586] | 0 | 0 | +4 / +6 / +9 | 点估计正向，但区间跨 0 |
| 3 | `layer_26_mlp_int8` | 332/883 | +4 | +0.453 | [-0.340, +1.359] | 0 | 0 | +1 / +2 / +2 | 点估计正向，但区间跨 0 |
| 4 | `layer_06_attn_int8` | 332/883 | +4 | +0.453 | [-0.227, +1.246] | 0 | **-1** | +6 / +5 / +3 | 长上下文门禁失败 |
| 5 | `layer_23_attn_int8` | 329/883 | +1 | +0.113 | [-0.793, +1.019] | 0 | +1 | 0 / -1 / +1 | 收益弱，仅列第二梯队 |
| 6 | `layer_25_mlp_int8` | 328/883 | 0 | 0 | [-1.133, +1.133] | +1 | +1 | +3 / +2 / +3 | full-dev 无净收益 |
| 7 | `layer_17_mlp_int8` | 325/883 | **-3** | -0.340 | [-1.359, +0.680] | 0 | **-1** | +3 / -2 / -1 | pilot 正向未能复现，淘汰 |

`layer_10_mlp_int8` 不只是总分最高：它在三个业务子集均为正，高风险和长上下文均不退化，而且置信区间下界仍
大于 0，因此是唯一足以进入组合验证的 Transformer 子模块。Layer 23 MLP、Layer 26 MLP 和 Layer 23
Attention 虽被点估计门禁写入自动候选策略，但统计证据不足，不应与 Layer 10 同等级解释。Layer 6 Attention
说明“full-dev 总体正向”仍可能掩盖长上下文回归；Layer 17 MLP 则说明 80 条 pilot 的 `+2` 可能是抽样波动。

特殊模块结果如下。存储节省来自 fake-quant 编码的理论 packed weight 与 scale 大小，不代表当前后端已经获得同等
显存、延迟或吞吐收益。

| 排名 | 计划 | full-dev | 净变化 | 95% CI（百分点） | high-risk | 长上下文 | 理论节省 | 金额/比例/日期 | 结论 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `lm_head_int8` | 331/883 | +3 | [-0.227, +0.906] | +1 | 0 | 519.5 MiB | +4 / +4 / +2 | 最稳妥特殊模块候选 |
| 2 | `embedding_int8` | 331/883 | +3 | [-0.227, +0.906] | 0 | **-1** | 519.5 MiB | +3 / +4 / +3 | 显存优先备选，需复核长上下文 |
| 3 | `embedding_int4` | 328/883 | 0 | [-0.680, +0.566] | +1 | **-1** | 747.1 MiB | 0 / 0 / +1 | 节省更多但稳定性不足 |
| 4 | `lm_head_int4` | 312/883 | **-16** | **[-3.284, -0.453]** | +1 | **-3** | 747.1 MiB | -1 / -15 / -9 | 显著退化，明确禁止 |

`lm_head_int4` 是最重要的负结果：pilot 曾净增 1 题，但 full-dev 净退化 16 题，且置信区间完全低于 0；其中
比例/利率题净退化 15 题，说明小样本 top-1 一致或 pilot 正向不能替代完整生成评测。`lm_head_int8` 的完整集、
高风险和长上下文方向最稳，是特殊模块默认选择。Embedding INT8 虽总体未退化，但因为长上下文少 1 题，只在
显存收益确有必要时进入组合候选。Norm 的 INT8 收益过小、INT4 已出现明显质量风险，KV Cache 尚未验证真实低比特
驻留，因此两者继续保持 FP16。

自动生成的 `protection_policy.json` 基于预注册点估计门禁，列出了 Layer 10/23/26 MLP、Layer 23 Attention、
`lm_head` INT8 和 Embedding INT8。该文件是“待验证候选集合”，不是把这些模块同时设置为 INT8 的部署证明：除
Layer 10 MLP 外，其余恢复候选置信区间均跨 0；并且所有结果来自单组件实验，组合后的误差可能抵消或放大。此前
Layer 23 与 Layer 10 的 pilot 组合没有表现出单项收益相加，也直接证明了不能用加法预测组合准确率。

这些结果是单组件 fake-quant 敏感性证据，不能假定收益能够相加。为验证交互效应，已前瞻性冻结两个独立的组合
候选，并使用 `combination` 工作流写入单独目录，绝不覆盖原 11 计划的冻结产物：

| 组合候选 | Layer 10 MLP | `lm_head` | Embedding | 用途 |
|---|---|---|---|---|
| `candidate_l10_mlp_lm_head_int8` | INT8 | INT8 | FP16 | 质量优先主候选 |
| `candidate_l10_mlp_lm_head_embedding_int8` | INT8 | INT8 | INT8 | 显存优先备选 |

两个组合都必须运行固定 80 条确定性/pilot、883 条 full-dev 和 91 条 high-risk，并从 full-dev 中检查 89 条
长上下文。组合晋级不再利用同一 pilot 做优胜者筛选；`pilot_min_net_correct_gain=-80` 表示两项只要运行成功就都
进入完整评测，从而避免组合验证中的二次挑选偏差。最终仍要求 full-dev 正增、高风险和长上下文不退化。

### 当前实现文件、执行顺序与产物

实现文件如下：

```text
configs/module_sensitivity.json
src/quantization/module_sensitivity.py
src/quantization/module_sensitivity_experiments.py
src/quantization/module_sensitivity_final.py
scripts/run_module_sensitivity.py
scripts/run_module_sensitivity_pilot.sh
scripts/run_module_sensitivity_final.sh
scripts/run_module_sensitivity_combinations.sh
tests/test_module_sensitivity.py
```

服务器执行顺序：

```bash
# 7.2.1：真实模型模块发现和能力登记
python3 scripts/run_module_sensitivity.py inspect-model

# 7.2.3：真实 KV Cache 后端短探测；不支持时会留下 unsupported 证据
python3 scripts/run_module_sensitivity.py probe-kv-cache

# 7.2.2/7.2.3：生成矩阵并逐计划执行固定 80 条 pilot
bash scripts/run_module_sensitivity_pilot.sh

# 查看仍缺少结果的可运行计划
python3 scripts/run_module_sensitivity.py pending

# 7.2.4：pilot 完成后冻结晋级决策与最终评测合同
python3 scripts/run_module_sensitivity.py select-advancement
python3 scripts/run_module_sensitivity.py prepare-final

# 7.2.4/7.2.5：运行确定性、full-dev、高风险评测并生成排名
bash scripts/run_module_sensitivity_final.sh

# 7.2.5 后续：在独立目录验证两个最终组合候选，支持断点续跑
bash scripts/run_module_sensitivity_combinations.sh

# 单独检查或收口
python3 scripts/run_module_sensitivity.py pending-final
python3 scripts/run_module_sensitivity.py finalize
```

如服务器模型路径与 7.1 manifest 中记录的路径不同，可在 `inspect-model`、`probe-kv-cache` 和单独的
`run-plan` 命令中使用 `--model-path`。一键脚本默认使用 manifest 路径；需要改路径时应先修正服务器上的冻结
manifest 或逐计划调用入口，不能让不同计划加载不同母版。

准备完成后的目录结构为：

```text
out/module_sensitivity/
├── setup/
│   ├── contract.json
│   ├── module_inventory.json
│   ├── kv_cache_capability.json
│   └── validation.json
├── pilot/
│   ├── experiment_manifest.json
│   ├── <plan_id>/plan.json
│   ├── <plan_id>/quantization_error.json
│   ├── <plan_id>/comparison_report.json
│   ├── <plan_id>/structured_json/predictions.jsonl
│   └── <plan_id>/result.json
└── final_evaluation/
    ├── advancement.json
    ├── experiment_manifest.json
    ├── full_dev|high_risk|determinism/<plan_id>/...
    ├── module_metrics.jsonl
    ├── sample_transitions.jsonl
    ├── ranking.json / ranking.csv
    ├── protection_policy.json
    ├── report.md
    └── validation.json
```

截至当前，合同、矩阵生成、隔离 fake-quant、单计划业务评测、逐样本转移、`ALL-I8` 一致性审计、断点续跑和
`unsupported` 门禁均已实现；71 个可运行 pilot 计划和 11 个晋级候选的完整 dev/high-risk/确定性评测均已完成，
配对 bootstrap、双榜、保护策略和最终验收产物已生成。两个后续组合候选也已完成：Embedding FP16 主候选
`candidate_l10_mlp_lm_head_int8` 为 337/883（相对 ALL-I4 +9），Embedding INT8 研究候选为 339/883（+11）；
二者高风险和长上下文均无净退化。它们仍是 fake-quant 质量证据，不能替代真实部署后端 Benchmark。

## 7.4 混合精度搜索实现

### 为什么选择 TorchAO，而不是直接使用 HQQ `dynamic_config`

7.4 必须准确表达“只把第 10 层 MLP 升到 INT8”。Transformers 的 HQQ 集成会把
`model.layers.10.mlp.gate_proj` 归一化为 `mlp.gate_proj`，因此动态配置适合按模块类型统一设定位宽，却不能可靠地区分
第 10 层和其他层。当前实现先按完整模块名生成每个 `nn.Linear` 的位宽分配，再逐模块调用 TorchAO `quantize_`。
导出后还会逐模块检查权重是否真的变成 TorchAO 量化 Tensor 子类；未替换成功会立即失败，而不是仅凭配置文件宣称已量化。

Embedding 是 `nn.Embedding`，不属于当前 Linear weight-only 后端的覆盖范围。因此 7.2 中质量点估计更高的
`Layer 10 MLP INT8 + lm_head INT8 + Embedding INT8` 被保留为研究候选并标记 `unsupported`；默认可部署主候选改为：

```text
其余 Linear W4A16-g32
Layer 10 MLP W8A16
lm_head W8A16
Embedding / LayerNorm / KV Cache BF16
```

这会放弃 Embedding INT8 约 519.5 MiB 的理论额外节省，但避免把 fake-quant 后立即反量化的结果冒充真实低比特驻留。

### 冻结候选矩阵

| 计划 | 用途 | 默认 Linear | 保护项 | 后端状态 |
| --- | --- | --- | --- | --- |
| `all_linear_int4` | 部署基线 | INT4 | 无 | 可运行 |
| `all_linear_int8` | 质量上界 | INT8 | 无 | 可运行 |
| `mixed_l10_mlp_int8` | 单敏感组消融 | INT4 | Layer 10 MLP INT8 | 可运行 |
| `mixed_l10_mlp_bf16` | 高精度保护对照 | INT4 | Layer 10 MLP BF16 | 可运行 |
| `mixed_l10_mlp_lm_head_int8` | 主部署候选 | INT4 | Layer 10 MLP、`lm_head` INT8 | 可运行 |
| `mixed_l10_l23_mlp_lm_head_int8` | 保护数量消融 | INT4 | Layer 10/23 MLP、`lm_head` INT8 | 可运行，但非默认候选 |
| `mixed_l10_mlp_lm_head_embedding_int8` | 7.2 最优点估计复核 | INT4 | 再加 Embedding INT8 | 当前 `unsupported` |

Layer 23 MLP 在 7.2 完整评测中没有通过统计门禁；保留它只是为了满足“保护 1 组还是 2 组”的前瞻性消融，不能因
其结果碰巧较好而回写 7.2 的敏感层结论。

### 真实制品与筛选门禁

每个可运行候选使用 TorchAO 内置的 `tile_packed_to_4d` INT4 格式，避免 `plain` 格式对可选 MSLK 包的强制依赖，
并保存模型配置、Tokenizer、TorchAO 量化 state_dict、完整模块—位宽清单、文件大小、SHA256 和后端
版本。加载时先在 meta device 上创建空模型，再使用 `assign=True` 装入量化 Tensor，避免先构造一份完整高精度权重；
SHA256 不一致时拒绝加载。

导出阶段同样不能先把完整 7B BF16 模型放到 16 GiB GPU 后再量化，否则 INT4 padding/packing 的临时张量会造成
OOM。实现会把高精度母版留在 CPU，并对单个 Linear 调用 TorchAO `quantize_(..., device=cuda)`，逐模块流式搬运和量化；全部 packed
权重建立后，才把保持 BF16 的 Embedding、Norm 和敏感层搬入 GPU。脚本同时启用 expandable segments 以降低缓存碎片。

质量门禁要求：full-dev 解析率至少 99%，并且相对真实 `all_linear_int4`，full-dev、高风险集和冻结长上下文集的正确数
净变化都不得小于 0。性能门禁要求：峰值显存和最差 P95 延迟不超过全 INT4 的 1.15 倍，平均吞吐不低于其 0.85 倍。
只有四套证据都齐全的候选才进入最终排名；排序优先 full-dev 净增，其次峰值显存和 P95 延迟。

### 文件与服务器执行

```text
configs/mixed_precision_search.json
src/quantization/mixed_precision.py
scripts/run_mixed_precision_search.py
scripts/run_mixed_precision_search.sh
scripts/run_mixed_precision_exports.sh
inputs/mixed_precision_benchmark_prompt.txt
tests/test_mixed_precision.py
```

完整流程支持按证据文件断点续跑：

```bash
# requirements-server.txt 已加入 torchao；先安装与服务器 PyTorch/CUDA 匹配的版本
MODEL_PATH=/models/data/Qwen2.5-7B-Instruct-QLoRA-step724-merged-fp16 \
  bash scripts/run_mixed_precision_search.sh

# 查看缺失证据或在证据齐全后重新收口
python3 scripts/run_mixed_precision_search.py pending
python3 scripts/run_mixed_precision_search.py finalize
```

若只想先验证导出合同，可运行 `bash scripts/run_mixed_precision_exports.sh`。单项也可使用 `export`、`evaluate`、
`benchmark`、`record-evidence` 子命令；外部已有报告可登记而无需重跑。输出统一位于
`out/mixed_precision_search/`，其中 `exports/` 是真实量化制品，`runs/` 是原始执行结果，`evidence/` 是冻结筛选证据，
最终推荐写入 `ranking.json`。在 `finalize` 成功且 `deployment_claims_allowed=true` 之前，不把任何候选称为已验证部署配置。
