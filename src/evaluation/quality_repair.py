"""Prompt and deterministic-calculation utilities for Todo 6.7.2."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import random
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
ALLOWED_OPERATIONS = {
    "add",
    "sum",
    "subtract",
    "difference",
    "absolute_difference",
    "multiply",
    "divide",
    "average",
    "ratio",
    "percent",
    "percentage",
    "percent_change",
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


def parse_evidence_operation_output(text: str) -> dict[str, Any] | None:
    """Parse evidence and operands without trusting a model-generated final value."""

    for payload in _json_candidates(text):
        evidence = payload.get("evidence")
        operands = payload.get("operands")
        operation = str(payload.get("operation", "")).strip().lower()
        unit = payload.get("unit", "")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not isinstance(operands, list)
            or not operands
            or operation not in ALLOWED_OPERATIONS
            or not isinstance(unit, str)
            or unit.lower() not in ALLOWED_UNITS
        ):
            continue
        numeric_operands = []
        for operand in operands:
            parsed_operand = parse_numeric_answer(operand)
            if parsed_operand is None:
                break
            numeric_operands.append(parsed_operand)
        else:
            return {
                "evidence": [
                    item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                    for item in evidence
                ],
                "operands": [str(value) for value in numeric_operands],
                "operation": operation,
                "unit": unit,
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


def repair_with_evidence_operation(parsed: dict[str, Any]) -> dict[str, Any]:
    """Compute an extracted operation deterministically from numeric operands."""

    operands = [Decimal(value) for value in parsed.get("operands", [])]
    operation = parsed.get("operation")
    if not operands or not isinstance(operation, str):
        return parsed
    try:
        if operation in {"add", "sum"}:
            result = sum(operands, Decimal("0"))
            formula = " + ".join(str(value) for value in operands)
        elif operation in {"subtract", "difference"}:
            if len(operands) != 2:
                return parsed
            result = operands[0] - operands[1]
            formula = f"{operands[0]} - {operands[1]}"
        elif operation == "absolute_difference":
            if len(operands) != 2:
                return parsed
            result = abs(operands[0] - operands[1])
            formula = f"abs({operands[0]} - {operands[1]})"
        elif operation == "multiply":
            result = operands[0]
            for operand in operands[1:]:
                result *= operand
            formula = " * ".join(str(value) for value in operands)
        elif operation == "divide":
            if len(operands) != 2:
                return parsed
            result = operands[0] / operands[1]
            formula = f"{operands[0]} / {operands[1]}"
        elif operation == "average":
            result = sum(operands, Decimal("0")) / Decimal(len(operands))
            formula = f"({ ' + '.join(str(value) for value in operands) }) / {len(operands)}"
        elif operation in {"ratio", "percent", "percentage"}:
            if len(operands) != 2:
                return parsed
            result = operands[0] / operands[1]
            formula = f"{operands[0]} / {operands[1]}"
        elif operation == "percent_change":
            if len(operands) != 2:
                return parsed
            result = (operands[0] - operands[1]) / operands[1]
            formula = f"({operands[0]} - {operands[1]}) / {operands[1]}"
        else:
            return parsed
    except (ArithmeticError, InvalidOperation):
        return parsed
    return {
        **parsed,
        "formula": formula,
        "normalized_value": str(result),
        "calculator_used": True,
    }


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


def evidence_operation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["evidence", "operands", "operation", "unit"],
        "properties": {
            "evidence": {"type": "array"},
            "operands": {"type": "array", "items": {"type": "number"}},
            "operation": {"type": "string", "enum": sorted(ALLOWED_OPERATIONS)},
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


def _select_evaluation_rows(
    dataset: Any, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[int] | None, int | None]:
    """Select a reproducible pilot subset without changing the source dataset."""

    source_size = len(dataset)
    requested_size = config.get("evaluation_size")
    if requested_size is None:
        return [dataset[index] for index in range(source_size)], None, None
    requested_size = int(requested_size)
    if requested_size <= 0 or requested_size > source_size:
        raise ValueError(
            f"evaluation_size must be between 1 and {source_size}, got {requested_size}"
        )
    seed = int(config.get("evaluation_seed", 0))
    indices = sorted(random.Random(seed).sample(range(source_size), requested_size))
    return [dataset[index] for index in indices], indices, seed


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
    pilot_variants = config.get("pilot_variants")
    if pilot_variants is not None:
        allowed_variants = {str(name) for name in pilot_variants}
        variants = [variant for variant in variants if variant.name in allowed_variants]
        if not variants:
            raise ValueError("pilot_variants did not match any configured variant")
    evaluation_rows, sample_indices, sample_seed = _select_evaluation_rows(
        dataset, config
    )
    calibration_rows = list(calibration or [])
    batch_size = int(config.get("batch_size", 4))
    if batch_size <= 0:
        raise ValueError("quality repair batch_size must be positive")
    results = []
    original_generation = service.config.get("generation")
    show_progress = bool(config.get("show_progress", True))
    total_variants = len(variants)
    total_rows = len(evaluation_rows)
    total_batches = math.ceil(total_rows / batch_size) if total_rows else 0
    configured_progress_interval = int(config.get("progress_every_batches", 0))
    progress_interval = (
        configured_progress_interval
        if configured_progress_interval > 0
        else max(1, math.ceil(total_batches / 10))
    )
    if isinstance(config.get("generation"), dict):
        service.config["generation"] = config["generation"]
    for variant_number, variant in enumerate(variants, start=1):
        started = time.perf_counter()
        if show_progress:
            print(
                f"[6.7.2] 方案 {variant_number}/{total_variants} "
                f"{variant.name} 开始，总进度 "
                f"{(variant_number - 1) / total_variants * 100:.1f}%",
                flush=True,
            )
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
            for batch_number, start in enumerate(
                range(0, len(evaluation_rows), batch_size), start=1
            ):
                rows = [
                    evaluation_rows[index]
                    for index in range(start, min(start + batch_size, len(evaluation_rows)))
                ]
                requests = [
                    InferenceRequest.from_finqa_row(
                        row, request_id=f"quality-repair-{variant.name}-{start + offset + 1:06d}"
                    )
                    for offset, row in enumerate(rows)
                ]
                generated = service.generate_batch(requests, prompt_suffix=prompt_suffix)
                for row, request, result in zip(rows, requests, generated):
                    raw_output = result["output_text"]
                    if variant.output_mode == "evidence_operation":
                        parsed = parse_evidence_operation_output(raw_output)
                    else:
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
                    elif (
                        parsed is None
                        and variant.format_retry
                        and variant.output_mode == "evidence_operation"
                    ):
                        retry = service.generate_one(
                            request,
                            prompt_suffix=(
                                f"{prompt_suffix}\nYour previous response did not match the required evidence-operation schema. "
                                "Return one valid JSON object only, with no markdown or explanation."
                            ),
                        )
                        raw_output = retry["output_text"]
                        parsed = parse_evidence_operation_output(raw_output)
                        format_retry_count += 1
                        retry_used = True
                    if parsed is None:
                        parsed = extract_final_numeric(raw_output)
                    if parsed.get("structured"):
                        structured_count += 1
                    if variant.output_mode == "evidence_operation":
                        repaired = repair_with_evidence_operation(parsed)
                        if repaired.get("calculator_used"):
                            calculator_count += 1
                        parsed = repaired
                    elif variant.calculator_enabled:
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
                    prediction_index = len(predictions)
                    record = {
                        "index": prediction_index,
                        "source_index": (
                            sample_indices[prediction_index]
                            if sample_indices is not None
                            else prediction_index
                        ),
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
                        "operands": parsed.get("operands"),
                        "operation": parsed.get("operation"),
                        "unit": parsed.get("unit"),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    predictions.append(extracted if extracted is not None else "")
                    references.append(reference)
                completed_rows = min(start + len(rows), total_rows)
                if show_progress and (
                    batch_number % progress_interval == 0
                    or completed_rows == total_rows
                ):
                    variant_percent = completed_rows / total_rows * 100 if total_rows else 100.0
                    overall_percent = (
                        (variant_number - 1 + completed_rows / total_rows)
                        / total_variants
                        * 100
                        if total_rows
                        else variant_number / total_variants * 100
                    )
                    print(
                        f"[6.7.2] {variant.name}: {completed_rows}/{total_rows} "
                        f"({variant_percent:.1f}%), 总进度 {overall_percent:.1f}%",
                        flush=True,
                    )
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
                "structured_output_rate": structured_count / len(evaluation_rows) if evaluation_rows else 0.0,
                "calculator_use_rate": calculator_count / len(evaluation_rows) if evaluation_rows else 0.0,
                "format_retry_count": format_retry_count,
                "duration_seconds": time.perf_counter() - started,
                "predictions_file": str(predictions_path.relative_to(output_path)),
                "predictions_sha256": _sha256(predictions_path),
            }
        )
        if show_progress:
            print(
                f"[6.7.2] 方案 {variant_number}/{total_variants} "
                f"{variant.name} 完成，总进度 "
                f"{variant_number / total_variants * 100:.1f}%",
                flush=True,
            )
    service.config["generation"] = original_generation
    report = {
        "schema_version": 1,
        "experiment_name": config.get("experiment_name"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "role": config.get("dataset_role"),
            "size": len(evaluation_rows),
            "source_size": len(dataset),
            "version": dataset_manifest.get("assets", {}).get(config.get("dataset_role"), {}).get("version"),
            "sha256": dataset_manifest.get("assets", {}).get(config.get("dataset_role"), {}).get("sha256"),
        },
        "variants": results,
        "schema": quality_repair_schema(),
        "evidence_operation_schema": evidence_operation_schema(),
        "generation": config.get("generation", original_generation),
        "note": "Few-shot examples must come from calibration/train, never evaluation dev/test.",
    }
    if sample_indices is not None:
        report["dataset"]["sample"] = {
            "seed": sample_seed,
            "indices": sample_indices,
            "indices_sha256": hashlib.sha256(
                json.dumps(sample_indices, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    (output_path / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
