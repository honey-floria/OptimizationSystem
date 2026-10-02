# 数据集与 Colab 使用说明

## 1. 用途

本项目使用公开金融问答数据验证量化、剪枝、混合精度和推理后端切换是否造成质量退化。

项目不负责原始数据采集、清洗、脱敏、标注或金融业务规则定义。这里的代码只负责下载、缓存、划分、格式转换和评测输入准备。

## 2. 推荐数据组合

| 项目用途 | 数据来源 | 默认划分 | 建议规模 |
|---|---|---|---:|
| 量化校准集 | `dreamerdeo/finqa` | `train` 随机抽样 | 1,024 |
| 质量调优集 | `dreamerdeo/finqa` | `validation` | 883 |
| 最终质量评估集 | `dreamerdeo/finqa` | `test` 去除回归重叠 | 约 1,147 |
| 数值高风险回归集 | `Aiera/finqa-verified` | `test` | 91 |

FinQA 包含财务报告文本、表格、问题和数值推理答案，适合验证数字、比例、增长率和表格理解能力。Aiera 的 verified 子集只有 91 条，但每条样本经过人工核验，适合作为第一版固定回归集。

这套数据主要覆盖英文财报数值推理，不等价于真实银行中文业务数据，也不覆盖政策版本、隐私泄露、拒答和工具调用等场景。项目后续接入中文或内部脱敏资产时，应保留相同的输入接口。

## 3. 数据职责边界

### 本项目负责

- 使用 Hugging Face `datasets` 下载数据
- 将数据缓存到 Colab Google Drive
- 从 `train` 抽取校准集
- 保持校准集、调优集和最终测试集分离
- 去除高风险回归集与最终测试集的重复问题
- 生成量化所需 Prompt
- 保存版本、来源和样本数量

### 本项目不负责

- 修改原始金融答案
- 将公开数据宣称为银行生产数据
- 生产业务标签或专家标准答案
- 对公开财报内容进行合规认证
- 使用真实客户数据

## 4. Colab 快速开始

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

## 5. 数据下载与缓存

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

## 6. 三套数据的使用规则

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

## 7. FinQA Prompt 格式

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

## 8. 回归判定

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

## 9. 来源和许可证

- [FinQA 官方仓库](https://github.com/czyssrs/FinQA)：包含数据说明和 MIT License。
- [Hugging Face dreamerdeo/finqa](https://huggingface.co/datasets/dreamerdeo/finqa)：Colab 中使用的便捷镜像，包含 `train`、`validation`、`test`。
- [Hugging Face Aiera/finqa-verified](https://huggingface.co/datasets/Aiera/finqa-verified)：91 条人工核验样本，页面标示 MIT License。

运行时建议记录数据集版本、commit/revision、下载时间和样本哈希。即使个人非商用，也不要将数据重新打包进公开仓库或 Docker 镜像；使用前应阅读对应仓库的最新数据卡和许可证。

## 10. 目录结构

```text
datasets/
├── hf_cache/
├── calibration_finqa_1024/
├── quality_dev_finqa/
├── quality_test_finqa/
├── regression_finqa_verified/
└── manifest.json
```
