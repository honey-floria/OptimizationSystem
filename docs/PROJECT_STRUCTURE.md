# 项目文件结构说明

本文档以树形方式说明 `OptimizationSystem` 中主要目录和文件的用途。树中只展开稳定、需要理解或经常使用的内容；缓存、模型分片和大量逐条预测文件不逐一列出。

## 1. 总体结构

```text
OptimizationSystem/
├── README.md                         # 项目目标、范围、架构和总体方案
├── main.py                           # PyCharm 自动生成的示例文件；当前不是项目入口
├── requirements-colab.txt            # Google Colab 实验依赖
├── requirements-server.txt           # 本地 NVIDIA 服务器训练/评测依赖
├── .gitignore                        # Git 忽略规则；模型、数据、虚拟环境等不入库
│
├── configs/                          # 所有可复现实验的输入配置
├── inputs/                           # 输入清单示例，不存放真实大模型权重
├── scripts/                          # 面向用户的训练、下载和服务器评测入口
├── src/                              # 核心 Python 实现
├── tests/                            # 与 src/、scripts/ 对应的自动化测试
├── notebooks/                        # Colab 统一执行 Notebook
├── docs/                             # 项目说明、TODO 和实施记录
├── out/                              # 实验报告、日志、验证结果和预测输出
│
├── datasets/                         # 本地数据集；被 Git 忽略，可能仅在运行机器存在
├── models/                           # 下载模型、量化权重和 LoRA adapter；被 Git 忽略
├── .venv/                            # 本地 Python 虚拟环境；自动生成且被 Git 忽略
├── .idea/                            # PyCharm 配置；本地开发环境文件
└── .git/                             # Git 仓库元数据，不应手动修改
```

## 2. 配置文件

`configs/` 决定实验“如何运行”。代码提供能力，配置固定参数和复现口径。

```text
configs/
├── business_scope.json               # L1/L2/L3 业务场景、风险等级和质量门槛
├── baseline.json                     # FP16/BF16 服务、Prompt、随机种子和生成参数
├── baseline_quality.json             # FinQA 基线质量评测配置
├── baseline_quality_diagnosis.json   # FP16 错误诊断和人工抽样配置
├── benchmark.json                    # 基线性能测试矩阵、预热与重复次数
│
├── quantization.json                 # 统一量化计划和导出/日志契约
├── quantization_benchmark.json       # 量化模型的统一性能测试矩阵
├── quantization_comparison.json      # 跨方案质量门槛、处置结论和 GPU 成本参数
├── awq.json                          # AWQ INT4 位宽、group size 和校准参数
├── gptq.json                         # GPTQ INT4 位宽、group size 和校准参数
├── int8_fp8.json                     # SmoothQuant INT8、FP8 能力和校准参数
│
├── quality_repair.json               # Colab 质量修复实验及多种 Prompt 方案
├── quality_repair_4080.json          # 4080/服务器质量修复与 LoRA 评测配置
└── environment/
    ├── nvidia_colab_t4.json          # 已采集的 NVIDIA Colab/T4 环境模板
    └── ascend.pending.json           # 等待真实昇腾机器补齐的环境模板
```

修改配置后运行实验时，应把最终生效配置保存到对应的 `out/` 目录，不能只保留命令行参数。

## 3. 用户执行入口

```text
scripts/
├── download_model_4080.py            # 从 Hugging Face 下载模型到本地 models/data/
├── train_finqa_qlora.py              # 使用 FinQA train 训练 7B QLoRA adapter
└── run_quality_repair_4080.py         # 在 NVIDIA 上运行 FP16、LoRA、AWQ 或 GPTQ 质量评测

notebooks/
└── optimization_system_all_colab.ipynb
                                       # 按 TODO 顺序合并的 Colab 全流程 Notebook

inputs/
└── model_manifest.example.json        # 上游模型资产清单示例
```

三个脚本的关系：

```text
download_model_4080.py
        │
        ▼
models/data/Qwen2.5-7B-Instruct
        │
        ├── train_finqa_qlora.py ──► models/data/<adapter>/
        │                                  │
        └──────────────────────────────────┤
                                           ▼
                              run_quality_repair_4080.py
                                           │
                                           ▼
                                  out/quality_repair_*/
```

`main.py` 目前仍是 IDE 示例代码，不参与上述流程，不应把它当作正式启动入口。

## 4. 核心源码

```text
src/
├── baseline/
│   ├── __init__.py                   # 导出基线服务公共接口
│   └── service.py                    # 确定性 FP16/BF16 推理、请求日志和证据校验
│
├── benchmark/
│   ├── __init__.py                   # Benchmark 包定义
│   └── runner.py                     # TTFT、延迟、吞吐、显存和矩阵测试
│
├── data/
│   ├── __init__.py                   # 导出 FinQA 数据接口
│   └── finqa_assets.py               # 下载/读取 FinQA，构建校准、dev 和高风险回归资产
│
├── environment_lock/
│   ├── __init__.py                   # 环境锁定包定义
│   └── manager.py                    # 采集依赖、硬件和运行环境，生成可复现清单
│
├── input_validation/
│   ├── __init__.py                   # 输入校验包定义
│   ├── business_scope.py             # 校验业务范围和风险等级配置
│   ├── dataset_input.py              # 校验数据 manifest、隔离关系、字段和哈希
│   ├── hardware_input.py             # 采集并校验 NVIDIA/Ascend 环境
│   └── model_input.py                # 校验模型、Tokenizer、许可证、哈希及加载能力
│
├── evaluation/
│   ├── __init__.py                   # 导出通用评测接口
│   ├── finqa_metrics.py              # 数值解析、容差判定和回归分类
│   ├── baseline_quality.py           # 运行并验证 FP16 FinQA 质量基线
│   ├── baseline_quality_diagnosis.py # 对错误类型分类并生成人工复核队列
│   ├── quality_repair.py             # Prompt、JSON 解析、确定性计算和 LoRA 评测核心逻辑
│   └── stage5_acceptance.py          # 第 5 章 NVIDIA 基线阶段验收
│
└── quantization/
    ├── __init__.py                   # 导出统一量化接口
    ├── framework.py                  # 量化配置、后端适配器、导出和日志契约
    ├── awq.py                        # AWQ INT4 导出、加载、评测和证据校验
    ├── gptq.py                       # GPTQ INT4 导出、加载、评测和证据校验
    ├── int8_fp8.py                   # SmoothQuant INT8、FP8 能力探测和质量对比
    ├── comparison.py                 # 汇总 FP16/INT8/AWQ/GPTQ 的质量与性能报告
    └── stage6_acceptance.py          # 第 6 章定制量化阶段验收
```

核心调用关系：

```text
input_validation ──► baseline ──► benchmark
         │                │             │
         └──────► evaluation ◄──────────┘
                          │
                          ▼
                    quantization
                          │
                          ▼
              comparison / stage acceptance
```

## 5. 测试目录

```text
tests/
├── test_model_input.py               # 模型输入校验测试
├── test_dataset_input.py             # 数据资产校验测试
├── test_hardware_input.py            # 硬件环境校验测试
├── test_business_scope.py            # 业务范围配置测试
├── test_environment_lock.py          # 环境锁定测试
│
├── test_baseline_service.py          # 基线推理服务测试
├── test_benchmark_runner.py          # Benchmark 统计和验证测试
├── test_baseline_quality.py          # 基线质量流程测试
├── test_baseline_quality_diagnosis.py# 错误诊断测试
├── test_finqa_metrics.py             # 数值解析与正确性判定测试
├── test_quality_repair.py            # Prompt、JSON、计算器和候选方案测试
├── test_quality_repair_4080_cli.py    # 服务器评测命令行参数测试
├── test_finqa_qlora.py               # QLoRA 数据转换和训练参数测试
│
├── test_quantization_framework.py    # 统一量化契约测试
├── test_awq.py                       # AWQ 流程测试
├── test_gptq.py                      # GPTQ 流程测试
├── test_int8_fp8.py                  # INT8/FP8 流程测试
├── test_quantization_comparison.py   # 跨方案报告测试
├── test_stage5_acceptance.py         # 第 5 章验收测试
└── test_stage6_acceptance.py         # 第 6 章验收测试
```

`tests/__pycache__/` 和其他 `__pycache__/` 是 Python 自动生成的字节码缓存，可删除，不应提交或手动维护。

## 6. 文档目录

```text
docs/
├── TODO.md                           # 当前状态看板、任务门禁和下一步执行队列
├── PROJECT_DOCUMENTATION.md          # 按执行顺序合并的实施说明与历史实验记录
├── PROJECT_STRUCTURE.md              # 本文档：项目文件和目录导航
└── FP16 基线质量修复.md              # FP16 低质量问题的专题分析和修复方案
```

阅读顺序建议：

```text
README.md
   ├── 想知道现在做到哪里 ──► docs/TODO.md
   ├── 想知道某个实验怎么做 ─► docs/PROJECT_DOCUMENTATION.md
   └── 想知道文件放在哪里 ───► docs/PROJECT_STRUCTURE.md
```

## 7. 实验输出目录

`out/` 保存小型、可追溯的实验结果。模型权重通常放在被 Git 忽略的 `models/`，不应混入 `out/`。

```text
out/
├── model_input_real/                 # 真实模型静态检查、加载检查和 manifest
├── dataset_input/                    # 数据 manifest 和九项输入验收结果
├── hardware/                         # NVIDIA/Ascend 硬件采集与验证结果
├── environment_lock/                 # 环境清单、依赖冻结文件和验收结果
├── business_scope/                   # 业务范围配置验证结果
│
├── baseline/                         # FP16/BF16 基线服务配置、请求日志和验收
├── baseline_quality/                 # 正式 FP16 质量预测和报告
├── baseline_quality_experiment/      # 历史/实验性基线质量结果
├── baseline_quality_diagnosis/       # 错误分类报告和 50 条人工复核记录
├── benchmark/                        # FP16 性能报告和验证结果
├── stage5/                           # 第 5 章基线阶段验收结果
│
├── awq/                              # AWQ 各 group size 的模型清单、质量和性能证据
├── gptq/                             # GPTQ 各 group size 的模型清单、质量和性能证据
├── int8_fp8/                         # SmoothQuant INT8、FP8 探测和质量/性能报告
├── quantization_framework/           # 统一量化契约验收结果
├── quantization_comparison/          # FP16/INT8/INT4 汇总比较报告
├── stage6/                           # 第 6 章量化阶段验收结果
│
├── quality_repair_6_7_2/             # 0.5B Prompt/计算器完整实验与 pilot
├── quality_repair_6_7_3_model_scale/ # 1.5B/3B 模型规模对比
├── quality_repair_3b_4080*/          # 3B 在 4080 上的历次方案迭代
├── quality_repair_7b_awq_pilot/      # 7B AWQ 固定 80 条 pilot
├── quality_repair_7b_gptq_pilot*/    # 7B GPTQ 及解析修复复测
├── quality_repair_7b_gptq_*_pilot*/  # 确定性计算和多候选历史实验
├── quality_repair_7b_qlora_*_pilot/  # 7B QLoRA checkpoint/最终 adapter 质量评测
│
├── quality_repair_7b_qlora_step724_full/
│                                       # 冻结候选的完整 dev 评测
├── quality_repair_7b_qlora_step724_high_risk/
│                                       # 首次无效运行的诊断记录（仅保留 diagnosis.json）
├── quality_repair_7b_qlora_step724_high_risk_contextual_fixed/
│                                       # 修复后的 contextual 高风险评测
└── finqa_qlora_7b_v2/                # 当前 QLoRA 正式训练与验收清单
```

常见输出文件含义：

```text
comparison_report.json                # 某次实验的核心指标汇总
validation.json                       # 自动验收结果；通常包含 pass/fail/not_run
model_manifest.json                   # 基座模型身份、路径和哈希
quantized_model_manifest.json         # 量化模型身份、格式和后端信息
training_manifest.json                # 训练数据、参数、step 和 adapter 信息
*.effective.json                      # 合并配置文件与命令行覆盖后的实际运行配置
predictions.jsonl                     # 逐样本预测、解析值和正确性
quality_service.jsonl                 # 请求级质量评测服务日志
request_logs.jsonl                    # 请求级性能和运行信息
```

带 `_v2`、`_v3` 或 `pilot` 的目录是历史迭代证据，不代表当前推荐版本。当前结论以 `docs/TODO.md` 的项目看板和执行队列为准。

## 8. 本地大文件和自动生成内容

```text
models/                               # 原始模型、合并模型、量化权重、LoRA adapter
datasets/                             # FinQA 原始数据和准备后的评测资产
.venv/                                # Python 包和解释器
__pycache__/                          # Python 字节码缓存
.pytest_cache/                        # pytest 缓存
.idea/                                # IDE 工作区设置
```

这些目录不属于源码主体。通常不应提交到 Git，也不应在文档中依赖某台机器的绝对路径。模型和数据应通过 manifest、哈希与相对逻辑名称登记。

## 9. 当前主线对应文件

当前工作是完成 7B QLoRA 质量门禁，然后基于新基线重新量化。相关文件链路如下：

```text
scripts/train_finqa_qlora.py
        │
        ├── configs/quality_repair_4080.json
        ├── datasets/raw_finqa/train.json
        ├── models/data/Qwen2.5-7B-Instruct/
        └── models/data/finqa_qlora_7b_v2/
                            │
                            ▼
scripts/run_quality_repair_4080.py
        │
        ├── src/evaluation/quality_repair.py
        ├── src/evaluation/finqa_metrics.py
        └── out/quality_repair_7b_qlora_v2_pilot/
                            │
                            ▼
                  docs/TODO.md 的 6.7.5 门禁
                            │
                            ▼
              合并 LoRA → 冻结 FP16/BF16 新基线
                            │
                            ▼
        src/quantization/{awq,gptq,int8_fp8}.py
                            │
                            ▼
             out/quantization_comparison/
```

