"""Diagnose FP16 FinQA errors for Todo 6.7.1."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.evaluation.finqa_metrics import NUMBER_PATTERN, parse_numeric_answer
from src.input_validation.model_input import NOT_RUN, PASS, CheckResult


ARITHMETIC_EXPRESSION = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:[+\-*/]|÷|×)\s*[-+]?\d"
)
FORMULA_WORDS = re.compile(
    r"\b(?:calculate|formula|divide|dividing|multiply|multiplying|subtract|adding)\b",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid quality diagnosis input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid quality predictions {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def numeric_candidates(value: Any) -> list[Decimal]:
    text = str(value)
    candidates = []
    for match in NUMBER_PATTERN.finditer(text):
        try:
            number = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            continue
        if "%" in text[match.end() : match.end() + 2]:
            number /= Decimal("100")
        candidates.append(number)
    return candidates


def classify_question(question: str) -> str:
    text = question.lower()
    if any(term in text for term in ("percentage", "percent", "ratio", "proportion")):
        return "percentage_or_ratio"
    if "average" in text:
        return "average"
    if any(term in text for term in ("change", "increase", "decrease", "difference", "growth")):
        return "change_or_difference"
    if any(term in text for term in ("total", "combined", "sum of")):
        return "total_or_sum"
    if any(term in text for term in ("per ", "each ", "how many times")):
        return "rate_or_unit"
    return "lookup_or_other"


def analysis_dimensions(question: str) -> list[str]:
    text = question.lower()
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    percentage = any(
        term in text for term in ("percentage", "percent", "ratio", "proportion")
    )
    comparison = any(
        term in text for term in ("compare", "compared", "versus", "difference")
    )
    time_series = len(set(years)) >= 2 or any(
        term in text for term in ("period ended", "over the period", "from 20")
    )
    multi_step = any(
        term in text
        for term in (
            "percentage change",
            "growth rate",
            "average",
            "ratio",
            "proportion",
            "cumulative",
        )
    )
    calculation_cue = percentage or comparison or time_series or any(
        term in text
        for term in ("change", "increase", "decrease", "average", "total", "sum")
    )
    dimensions = ["multi_step" if multi_step else "single_step"] if calculation_cue else []
    if comparison:
        dimensions.append("comparison")
    if percentage:
        dimensions.append("percentage")
    if time_series:
        dimensions.append("time_series")
    return dimensions


def classify_error(row: dict[str, Any], tolerance: Decimal) -> str:
    if row.get("correct") is True:
        return "correct"
    prediction = str(row.get("prediction", ""))
    reference = parse_numeric_answer(row.get("reference"))
    candidates = numeric_candidates(prediction)
    if not candidates:
        return "output_parse_error"
    if reference is not None and any(
        abs(candidate - reference) <= tolerance for candidate in candidates[1:]
    ):
        return "answer_extraction_false_negative_candidate"
    if ARITHMETIC_EXPRESSION.search(prediction):
        return "calculation_error"
    if FORMULA_WORDS.search(prediction):
        return "formula_selection_error"
    return "table_or_context_selection_error"


def _normalization_audit() -> list[dict[str, Any]]:
    cases = [
        ("plain_decimal", "127.40", Decimal("127.40")),
        ("percentage", "93.5%", Decimal("0.935")),
        ("currency_and_thousands", "$1,234.50", Decimal("1234.50")),
        ("negative", "-688", Decimal("-688")),
        ("positive_sign", "+18.6", Decimal("18.6")),
    ]
    return [
        {
            "name": name,
            "input": value,
            "expected": str(expected),
            "actual": str(actual) if actual is not None else None,
            "passed": actual == expected,
        }
        for name, value, expected in cases
        for actual in [parse_numeric_answer(value)]
    ]


def _select_review_rows(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["suggested_error_type"], row["question_type"])].append(row)
    keys = sorted(groups)
    selected = []
    position = 0
    while len(selected) < sample_size and keys:
        key = keys[position % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop(0))
        else:
            keys.remove(key)
            if not keys:
                break
            position -= 1
        position += 1
    return selected


def merge_manual_reviews(report: dict[str, Any], review_path: Path) -> None:
    if not review_path.is_file():
        return
    existing = {
        row.get("index"): row.get("manual_review")
        for row in _load_jsonl(review_path)
        if isinstance(row.get("manual_review"), dict)
    }
    completed = 0
    for row in report["manual_review_rows"]:
        manual_review = existing.get(row.get("index"))
        if manual_review:
            row["manual_review"] = manual_review
        decision = row["manual_review"]
        if (
            decision.get("status") == "completed"
            and isinstance(decision.get("prediction_correct"), bool)
            and isinstance(decision.get("error_type"), str)
            and decision["error_type"].strip()
        ):
            completed += 1
    report["coverage"]["manual_review_completed"] = completed


def build_diagnosis(project_root: str | Path, config_file: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = _load_json(config_path)
    source_report_path = root / config["source_report"]
    source_report = _load_json(source_report_path)
    predictions_path = source_report_path.parent / source_report["predictions"]["file"]
    source_rows = _load_jsonl(predictions_path)
    tolerance = Decimal(str(config["numeric_tolerance"]))

    diagnosed_rows = []
    for source_row in source_rows:
        candidates = numeric_candidates(source_row.get("prediction"))
        diagnosed_rows.append(
            {
                **source_row,
                "question_type": classify_question(str(source_row.get("question", ""))),
                "analysis_dimensions": analysis_dimensions(
                    str(source_row.get("question", ""))
                ),
                "suggested_error_type": classify_error(source_row, tolerance),
                "numeric_candidates": [str(value) for value in candidates],
                "parsed_first_number": (
                    str(parse_numeric_answer(source_row.get("prediction")))
                    if parse_numeric_answer(source_row.get("prediction")) is not None
                    else None
                ),
                "normalized_reference": (
                    str(parse_numeric_answer(source_row.get("reference")))
                    if parse_numeric_answer(source_row.get("reference")) is not None
                    else None
                ),
            }
        )

    sample_size = int(config["manual_review_sample_size"])
    review_rows = []
    for row in _select_review_rows(diagnosed_rows, sample_size):
        review_rows.append(
            {
                **row,
                "manual_review": {
                    "status": "pending",
                    "prediction_correct": None,
                    "error_type": None,
                    "notes": "",
                },
            }
        )

    error_counts = Counter(row["suggested_error_type"] for row in diagnosed_rows)
    question_counts = Counter(row["question_type"] for row in diagnosed_rows)
    question_correct = Counter(
        row["question_type"] for row in diagnosed_rows if row.get("correct") is True
    )
    dimension_counts = Counter(
        dimension
        for row in diagnosed_rows
        for dimension in row["analysis_dimensions"]
    )
    dimension_correct = Counter(
        dimension
        for row in diagnosed_rows
        if row.get("correct") is True
        for dimension in row["analysis_dimensions"]
    )
    normalization = _normalization_audit()
    return {
        "schema_version": 1,
        "diagnosis_name": "fp16_finqa_quality_repair_6_7_1",
        "classification_version": config["classification_version"],
        "source": {
            "report": str(source_report_path.relative_to(root)),
            "predictions": str(predictions_path.relative_to(root)),
            "metrics": source_report.get("metrics", {}),
        },
        "coverage": {
            "source_rows": len(source_rows),
            "diagnosed_rows": len(diagnosed_rows),
            "manual_review_sample_size": len(review_rows),
            "manual_review_completed": 0,
        },
        "normalization_audit": normalization,
        "error_distribution": dict(sorted(error_counts.items())),
        "question_type_statistics": {
            question_type: {
                "total": total,
                "correct": question_correct[question_type],
                "numeric_accuracy": question_correct[question_type] / total,
            }
            for question_type, total in sorted(question_counts.items())
        },
        "required_dimension_statistics": {
            dimension: {
                "total": dimension_counts[dimension],
                "correct": dimension_correct[dimension],
                "numeric_accuracy": (
                    dimension_correct[dimension] / dimension_counts[dimension]
                    if dimension_counts[dimension]
                    else 0.0
                ),
            }
            for dimension in (
                "single_step",
                "multi_step",
                "comparison",
                "percentage",
                "time_series",
            )
        },
        "findings": {
            "first_number_bias_candidates": error_counts[
                "answer_extraction_false_negative_candidate"
            ],
            "classification_is_heuristic": True,
            "manual_review_required": True,
        },
        "diagnosed_rows": diagnosed_rows,
        "manual_review_rows": review_rows,
    }


def validate_diagnosis(report: dict[str, Any]) -> dict[str, Any]:
    coverage = report.get("coverage", {})
    normalization = report.get("normalization_audit", [])
    source_rows = coverage.get("source_rows", 0)
    checks = [
        CheckResult(
            "full_fp16_predictions_diagnosed",
            PASS if source_rows == 883 and coverage.get("diagnosed_rows") == source_rows else "fail",
            "All 883 frozen FP16 development predictions are classified.",
        ),
        CheckResult(
            "normalization_logic_audited",
            PASS if normalization and all(item.get("passed") for item in normalization) else "fail",
            "Decimal, percentage, currency, signed number, and thousands-separator cases are verified.",
        ),
        CheckResult(
            "error_taxonomy_recorded",
            PASS if sum(report.get("error_distribution", {}).values()) == source_rows else "fail",
            "Every prediction has a deterministic suggested error category.",
        ),
        CheckResult(
            "question_type_statistics_recorded",
            PASS
            if sum(
                item["total"]
                for item in report.get("question_type_statistics", {}).values()
            )
            == source_rows
            and set(report.get("required_dimension_statistics", {}))
            == {"single_step", "multi_step", "comparison", "percentage", "time_series"}
            else "fail",
            "Question types plus single-step, multi-step, comparison, percentage, and time-series dimensions are recorded.",
        ),
        CheckResult(
            "manual_review_queue_created",
            PASS if coverage.get("manual_review_sample_size", 0) >= 50 else "fail",
            "A stratified queue of at least 50 predictions is ready for human review.",
        ),
        CheckResult(
            "manual_review_completed",
            PASS if coverage.get("manual_review_completed", 0) >= 50 else NOT_RUN,
            "Human decisions must be filled in for at least 50 queued predictions.",
        ),
    ]
    return {
        "complete": all(check.status == PASS for check in checks),
        "checks": [check.__dict__ for check in checks],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FP16 基线质量诊断（TODO 6.7.1）",
        "",
        f"来源：`{report['source']['report']}`，共 {report['coverage']['source_rows']} 条。",
        "",
        "## 自动诊断结论",
        "",
    ]
    for name, count in report["error_distribution"].items():
        lines.append(f"- `{name}`：{count}")
    lines.extend(["", "## 题型统计", "", "| 题型 | 样本 | 正确 | 准确率 |", "|---|---:|---:|---:|"])
    for name, item in report["question_type_statistics"].items():
        lines.append(
            f"| {name} | {item['total']} | {item['correct']} | {item['numeric_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 6.7.1 指定分析维度",
            "",
            "| 维度 | 样本 | 正确 | 准确率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in report["required_dimension_statistics"].items():
        lines.append(
            f"| {name} | {item['total']} | {item['correct']} | {item['numeric_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 重要风险",
            "",
            f"当前评分器取输出中的第一个数字；有 {report['findings']['first_number_bias_candidates']} 条样本在后续数字中出现标准答案，属于潜在误判，必须人工复核。",
            "",
            "自动错误分类仅用于分流，不替代人工结论。请在 `manual_review_50.jsonl` 中填写 `manual_review` 字段。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/baseline_quality_diagnosis.json")
    parser.add_argument("--output-dir", default="out/baseline_quality_diagnosis")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    report = build_diagnosis(root, args.config)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "manual_review_50.jsonl"
    merge_manual_reviews(report, review_path)
    (output_dir / "diagnosis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "diagnosis_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    with review_path.open("w", encoding="utf-8") as stream:
        for row in report["manual_review_rows"]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "manual_review_50.json").write_text(
        json.dumps(report["manual_review_rows"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validation = validate_diagnosis(report)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
