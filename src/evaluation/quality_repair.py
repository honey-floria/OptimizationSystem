"""Prompt and deterministic-calculation utilities for Todo 6.7.2."""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.finqa_metrics import NUMBER_PATTERN, parse_numeric_answer
from src.evaluation.finqa_metrics import evaluate_numeric_answers
from src.baseline.service import InferenceRequest


FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|final\s+numeric\s+answer|answer|result)\s*[:=]\s*"
    r"([^\n\r]+)",
    re.IGNORECASE,
)
ALLOWED_UNITS = {
    "",
    "%",
    "percent",
    "million",
    "millions",
    "billion",
    "billions",
    "thousand",
    "thousands",
    "dollars",
    "shares",
}


@dataclass(frozen=True)
class PromptVariant:
    name: str
    prompt_suffix: str
    output_mode: str
    calculator_enabled: bool = False
    format_retry: bool = False


class CalculationError(ValueError):
    """Raised when a formula contains unsupported syntax."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> Decimal:
    """Evaluate only numeric arithmetic using Decimal, never Python eval."""

    normalized = (
        expression.replace(",", "")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"Invalid arithmetic expression: {expression}") from exc

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Pow) and right != right.to_integral_value():
                raise CalculationError("Exponent must be an integer")
            try:
                return _BINARY_OPERATORS[type(node.op)](left, right)
            except (ArithmeticError, InvalidOperation) as exc:
                raise CalculationError("Arithmetic operation failed") from exc
        raise CalculationError("Only numeric arithmetic is allowed")

    return visit(tree)


def _json_candidates(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_structured_output(text: str) -> dict[str, Any] | None:
    """Parse the required evidence/formula/value/unit response contract."""

    for payload in _json_candidates(text):
        value = payload.get("value")
        evidence = payload.get("evidence")
        formula = payload.get("formula")
        unit = payload.get("unit", "")
        if (
            value is not None
            and isinstance(evidence, list)
            and isinstance(formula, str)
            and isinstance(unit, str)
            and unit.lower() in ALLOWED_UNITS
        ):
            parsed = parse_numeric_answer(f"{value}{unit}")
            if parsed is not None:
                return {
                    "evidence": [
                        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                        for item in evidence
                    ],
                    "formula": formula,
                    "value": str(value),
                    "unit": unit,
                    "normalized_value": str(parsed),
                    "structured": True,
                }
    return None


def extract_final_numeric(text: str) -> dict[str, Any]:
    """Prefer structured/final-answer values over the first number in prose."""

    structured = parse_structured_output(text)
    if structured is not None:
        return structured
    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if matches:
        value_text = matches[-1].group(1)
        parsed = parse_numeric_answer(value_text)
        if parsed is not None:
            return {
                "value": value_text.strip(),
                "normalized_value": str(parsed),
                "structured": False,
                "extraction": "final_answer_marker",
            }
    return {
        "value": None,
        "normalized_value": None,
        "structured": False,
        "extraction": "unparseable_final_answer",
    }


def repair_with_calculator(parsed: dict[str, Any]) -> dict[str, Any]:
    """Recalculate a plain arithmetic formula when it is safe to do so."""

    formula = parsed.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        return parsed
    try:
        calculated = safe_calculate(formula)
    except CalculationError:
        return parsed
    return {**parsed, "normalized_value": str(calculated), "calculator_used": True}


def build_prompt_variants(config: dict[str, Any]) -> list[PromptVariant]:
    variants = []
    for item in config.get("variants", []):
        variants.append(
            PromptVariant(
                name=str(item["name"]),
                prompt_suffix=str(item["prompt_suffix"]),
                output_mode=str(item["output_mode"]),
                calculator_enabled=bool(item.get("calculator_enabled", False)),
                format_retry=bool(item.get("format_retry", False)),
            )
        )
    if not variants:
        raise ValueError("quality repair config must define at least one variant")
    return variants


def build_few_shot_suffix(examples: list[dict[str, Any]], limit: int = 2) -> str:
    """Build examples from calibration/train data only, never from dev/test."""

    blocks = []
    for row in examples[:limit]:
        question = row.get("question") or row.get("qa", {}).get("question")
        answer = row.get("answer") or row.get("qa", {}).get("answer")
        if question is None or answer is None:
            continue
        blocks.append(
            f"Example question: {question}\nExample final answer: {answer}"
        )
    return "\n\n".join(blocks)


def quality_repair_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["evidence", "formula", "value", "unit"],
        "properties": {
            "evidence": {"type": "array", "items": {"type": "string"}},
            "formula": {"type": "string"},
            "value": {"type": "number"},
            "unit": {"type": "string"},
        },
        "additionalProperties": False,
    }


def _reference_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer is None and isinstance(row.get("qa"), dict):
        answer = row["qa"].get("answer")
    if answer is None:
        raise ValueError("Quality repair row has no reference answer")
    return str(answer)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_quality_repair_experiment(
    service: Any,
    dataset: Any,
    dataset_manifest: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
    calibration: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all configured prompt variants with deterministic post-processing."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    variants = build_prompt_variants(config)
    calibration_rows = list(calibration or [])
    batch_size = int(config.get("batch_size", 4))
    if batch_size <= 0:
        raise ValueError("quality repair batch_size must be positive")
    results = []
    original_generation = service.config.get("generation")
    if isinstance(config.get("generation"), dict):
        service.config["generation"] = config["generation"]
    for variant in variants:
        started = time.perf_counter()
        variant_dir = output_path / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = variant_dir / "predictions.jsonl"
        predictions = []
        references = []
        structured_count = 0
        calculator_count = 0
        format_retry_count = 0
        prompt_suffix = variant.prompt_suffix
        if variant.name == "few_shot_structured_json":
            few_shot = build_few_shot_suffix(calibration_rows)
            prompt_suffix = f"{few_shot}\n\n{prompt_suffix}" if few_shot else prompt_suffix
        with predictions_path.open("w", encoding="utf-8") as stream:
            for start in range(0, len(dataset), batch_size):
                rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
                requests = [
                    InferenceRequest.from_finqa_row(
                        row, request_id=f"quality-repair-{variant.name}-{start + offset + 1:06d}"
                    )
                    for offset, row in enumerate(rows)
                ]
                generated = service.generate_batch(requests, prompt_suffix=prompt_suffix)
                for row, request, result in zip(rows, requests, generated):
                    raw_output = result["output_text"]
                    parsed = parse_structured_output(raw_output)
                    retry_used = False
                    if parsed is None and variant.format_retry and variant.output_mode == "json":
                        retry = service.generate_one(
                            request,
                            prompt_suffix=(
                                f"{prompt_suffix}\nYour previous response did not match the required JSON schema. "
                                "Return one valid JSON object only, with no markdown or explanation."
                            ),
                        )
                        raw_output = retry["output_text"]
                        parsed = parse_structured_output(raw_output)
                        format_retry_count += 1
                        retry_used = True
                    if parsed is None:
                        parsed = extract_final_numeric(raw_output)
                    if parsed.get("structured"):
                        structured_count += 1
                    if variant.calculator_enabled:
                        repaired = repair_with_calculator(parsed)
                        if repaired.get("calculator_used"):
                            calculator_count += 1
                        parsed = repaired
                    extracted = parsed.get("normalized_value")
                    reference = _reference_answer(row)
                    metrics = evaluate_numeric_answers(
                        [extracted if extracted is not None else ""],
                        [reference],
                        tolerance=float(config.get("tolerance", 1e-4)),
                    )
                    record = {
                        "index": len(predictions),
                        "request_id": request.request_id,
                        "question": request.question,
                        "raw_prediction": raw_output,
                        "extracted_value": extracted,
                        "reference": reference,
                        "parsed": metrics["parsed"] == 1,
                        "correct": metrics["correct"] == 1,
                        "structured_output": parsed.get("structured", False),
                        "calculator_used": parsed.get("calculator_used", False),
                        "format_retry_used": retry_used,
                        "formula": parsed.get("formula"),
                        "evidence": parsed.get("evidence"),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    predictions.append(extracted if extracted is not None else "")
                    references.append(reference)
        metrics = evaluate_numeric_answers(
            predictions,
            references,
            tolerance=float(config.get("tolerance", 1e-4)),
        )
        results.append(
            {
                "variant": variant.name,
                "output_mode": variant.output_mode,
                "calculator_enabled": variant.calculator_enabled,
                "prompt_suffix": prompt_suffix,
                "metrics": metrics,
                "structured_output_rate": structured_count / len(dataset) if len(dataset) else 0.0,
                "calculator_use_rate": calculator_count / len(dataset) if len(dataset) else 0.0,
                "format_retry_count": format_retry_count,
                "duration_seconds": time.perf_counter() - started,
                "predictions_file": str(predictions_path.relative_to(output_path)),
                "predictions_sha256": _sha256(predictions_path),
            }
        )
    service.config["generation"] = original_generation
    report = {
        "schema_version": 1,
        "experiment_name": config.get("experiment_name"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "role": config.get("dataset_role"),
            "size": len(dataset),
            "version": dataset_manifest.get("assets", {}).get(config.get("dataset_role"), {}).get("version"),
            "sha256": dataset_manifest.get("assets", {}).get(config.get("dataset_role"), {}).get("sha256"),
        },
        "variants": results,
        "schema": quality_repair_schema(),
        "generation": config.get("generation", original_generation),
        "note": "Few-shot examples must come from calibration/train, never evaluation dev/test.",
    }
    (output_path / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
