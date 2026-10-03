# 基线输入接口

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
