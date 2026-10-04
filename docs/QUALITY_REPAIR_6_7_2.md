# 6.7.2 Prompt 与确定性计算

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
