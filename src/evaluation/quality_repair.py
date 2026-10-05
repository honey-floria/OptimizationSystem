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
    "times",
    "multiple",
    "mmboe",
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
OPERATION_OUTPUT_MODES = {
    "evidence_operation",
    "cell_ids_operation",
    "steps_operation",
    "candidate_cell_ids_operation",
}


@dataclass(frozen=True)
class PromptVariant:
    name: str
    prompt_suffix: str
    output_mode: str
    calculator_enabled: bool = False
    format_retry: bool = False
    stable_cell_ids: bool = False
    cell_ids_only: bool = False
    fallback_to_structured: bool = False
    fallback_prompt_suffix: str = ""
    question_routing: bool = False


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
            repaired_text = re.sub(
                r'("value"\s*:\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+))"(\s*[,}])',
                r"\1\2\3",
                text[index:],
            )
            if repaired_text == text[index:]:
                continue
            try:
                value, _ = decoder.raw_decode(repaired_text)
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
                "evidence": list(evidence),
                "operands": [str(value) for value in numeric_operands],
                "operation": operation,
                "unit": unit,
                "structured": True,
            }
    return None


def add_stable_table_ids(row: dict[str, Any]) -> dict[str, Any]:
    """Add deterministic row/column IDs to a copy of a FinQA row."""

    enriched = dict(row)
    table = row.get("table", [])
    column_labels = [
        str(value).replace("\n", " ").strip()
        for value in (table[0] if table else [])
    ]
    enriched["table"] = [
        [
            (
                f"[cell_id=r{row_index}c{column_index} "
                f"row={row_label} column={column_label}] {cell}"
            )
            for column_index, cell in enumerate(table_row)
            for row_label, column_label in [
                (
                    str(table_row[0]).replace("\n", " ").strip()
                    if table_row
                    else "",
                    column_labels[column_index]
                    if column_index < len(column_labels)
                    else f"column_{column_index}",
                )
            ]
        ]
        for row_index, table_row in enumerate(table)
    ]
    return enriched


def _table_cell_values(row: dict[str, Any]) -> dict[str, str]:
    return {
        f"r{row_index}c{column_index}": str(cell).strip()
        for row_index, table_row in enumerate(row.get("table", []))
        for column_index, cell in enumerate(table_row)
    }


def validate_evidence_operation(
    parsed: dict[str, Any], row: dict[str, Any]
) -> str | None:
    """Validate cell references, operand membership, operation arity, and units."""

    evidence = parsed.get("evidence")
    operation = parsed.get("operation")
    operands = parsed.get("operands")
    unit = str(parsed.get("unit", "")).lower()
    if not isinstance(evidence, list) or not evidence:
        return "evidence must be a non-empty list"
    if not isinstance(operation, str) or operation not in ALLOWED_OPERATIONS:
        return "operation is not allowed"
    if not isinstance(operands, list) or not operands:
        return "operands must be a non-empty list"
    if unit not in ALLOWED_UNITS:
        return "unit is not allowed"

    cell_values = _table_cell_values(row)
    evidence_numbers: list[Decimal] = []
    cell_id_pattern = re.compile(r"^r(\d+)c(\d+)$")
    for item in evidence:
        if not isinstance(item, dict):
            return "each evidence item must be an object with cell_id and value"
        cell_id = item.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id_pattern.fullmatch(cell_id):
            return "evidence cell_id must match rNcM"
        if cell_id not in cell_values:
            return f"evidence cell_id does not exist: {cell_id}"
        if "value" not in item:
            return f"evidence value is missing: {cell_id}"
        expected = parse_numeric_answer(cell_values[cell_id])
        actual = parse_numeric_answer(item["value"])
        if expected is not None or actual is not None:
            if expected is None or actual is None or expected != actual:
                return f"evidence value does not match table cell: {cell_id}"
            evidence_numbers.append(actual)
        elif str(item["value"]).strip() != cell_values[cell_id]:
            return f"evidence text does not match table cell: {cell_id}"

    parsed_operands: list[Decimal] = []
    for operand in operands:
        value = parse_numeric_answer(operand)
        if value is None:
            return "every operand must be numeric"
        parsed_operands.append(value)
    unmatched = list(evidence_numbers)
    for operand in parsed_operands:
        for index, evidence_number in enumerate(unmatched):
            if operand == evidence_number:
                unmatched.pop(index)
                break
        else:
            return "operands must be copied from numeric evidence cells"

    exact_two = {
        "subtract", "difference", "absolute_difference", "divide",
        "ratio", "percent", "percentage", "percent_change",
    }
    at_least_two = {"add", "sum", "multiply"}
    if operation in exact_two and len(parsed_operands) != 2:
        return f"{operation} requires exactly two operands"
    if operation in at_least_two and len(parsed_operands) < 2:
        return f"{operation} requires at least two operands"
    if operation in {"percent_change", "percent", "percentage"} and unit not in {"%", "percent"}:
        return f"{operation} requires percent unit"
    if operation == "ratio" and unit not in {"", "times", "multiple"}:
        return "ratio requires an empty, times, or multiple unit"
    return None


def parse_cell_ids_operation_output(text: str) -> dict[str, Any] | None:
    """Parse the compact contract that references cells without copying values."""

    for payload in _json_candidates(text):
        cell_ids = payload.get("cell_ids")
        operation = str(payload.get("operation", "")).strip().lower()
        unit = payload.get("unit", "")
        if (
            not isinstance(cell_ids, list)
            or not cell_ids
            or not all(isinstance(cell_id, str) for cell_id in cell_ids)
            or operation not in ALLOWED_OPERATIONS
            or not isinstance(unit, str)
            or unit.lower() not in ALLOWED_UNITS
        ):
            continue
        return {
            "cell_ids": cell_ids,
            "operation": operation,
            "unit": unit,
            "structured": True,
        }
    return None


def parse_steps_operation_output(text: str) -> dict[str, Any] | None:
    """Parse an ordered operation chain whose later steps reference prior results."""

    for payload in _json_candidates(text):
        steps = payload.get("steps")
        unit = payload.get("unit", "")
        if (
            not isinstance(steps, list)
            or not steps
            or not isinstance(unit, str)
            or unit.lower() not in ALLOWED_UNITS
            or not all(isinstance(step, dict) for step in steps)
        ):
            continue
        normalized_steps = []
        valid = True
        for step in steps:
            operation = str(step.get("operation", "")).strip().lower()
            operands = step.get("operands")
            if (
                operation not in ALLOWED_OPERATIONS
                or not isinstance(operands, list)
                or not operands
                or not all(isinstance(operand, str) for operand in operands)
            ):
                valid = False
                break
            normalized_steps.append(
                {"operation": operation, "operands": list(operands)}
            )
        if valid:
            return {
                "steps": normalized_steps,
                "operation": normalized_steps[-1]["operation"],
                "unit": unit,
                "structured": True,
            }
    return None


def _candidate_semantic_score(
    parsed: dict[str, Any], row: dict[str, Any], question: str
) -> tuple[int, int]:
    """Score candidate cells by matching question terms to row/column labels."""

    table = row.get("table", [])
    question_text = str(question).lower()
    question_tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{3,}", question_text)
        if token not in {"what", "which", "that", "from", "into", "were", "does"}
    }
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", question_text))
    score = 0
    for cell_id in parsed.get("cell_ids", []):
        match = re.fullmatch(r"r(\d+)c(\d+)", cell_id)
        if match is None:
            continue
        row_index, column_index = (int(value) for value in match.groups())
        row_label = str(table[row_index][0]).lower() if row_index < len(table) and table[row_index] else ""
        column_label = (
            str(table[0][column_index]).lower()
            if table and column_index < len(table[0])
            else ""
        )
        context = f"{row_label} {column_label}"
        score += sum(3 for year in years if year in context)
        score += sum(1 for token in question_tokens if token in context)
    return score, -int(parsed.get("candidate_index", 0))


def parse_candidate_cell_ids_operation_output(
    text: str, row: dict[str, Any], question: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and deterministically calculate the best model-generated candidate."""

    last_error: str | None = None
    for payload in _json_candidates(text):
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        valid: list[tuple[tuple[int, int], dict[str, Any]]] = []
        for candidate_index, candidate in enumerate(candidates[:3]):
            if not isinstance(candidate, dict):
                last_error = "each candidate must be an object"
                continue
            parsed = parse_cell_ids_operation_output(
                json.dumps(candidate, ensure_ascii=False)
            )
            if parsed is None:
                last_error = "candidate does not match cell-id operation schema"
                continue
            parsed["candidate_index"] = candidate_index
            error = validate_cell_ids_operation(parsed, row)
            if error is None:
                error = validate_question_operation(parsed, question)
            if error is not None:
                last_error = error
                continue
            materialized = materialize_cell_ids_operation(parsed, row)
            repaired = repair_with_evidence_operation(materialized)
            repaired["candidate_count"] = len(candidates)
            repaired["candidate_selected_index"] = candidate_index
            valid.append((_candidate_semantic_score(repaired, row, question), repaired))
        if valid:
            return max(valid, key=lambda item: item[0])[1], None
    return None, last_error or "unparseable candidate operation output"


def parse_validated_operation_output(
    text: str,
    row: dict[str, Any],
    output_mode: str,
    question: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Select the first operation candidate that is valid for this table.

    Small causal models often emit several JSON candidates while repairing their
    own answer. Parsing only the first candidate turns a later valid candidate
    into a false validation failure, so candidates are checked in output order.
    """

    if output_mode == "cell_ids_operation":
        parser = parse_cell_ids_operation_output
        validator = validate_cell_ids_operation
    elif output_mode == "steps_operation":
        parser = parse_steps_operation_output
        validator = validate_steps_operation
    elif output_mode == "candidate_cell_ids_operation":
        return parse_candidate_cell_ids_operation_output(text, row, question or "")
    elif output_mode == "evidence_operation":
        parser = parse_evidence_operation_output
        validator = validate_evidence_operation
    else:
        raise ValueError(f"Unsupported operation output mode: {output_mode}")

    last_error: str | None = None
    for payload in _json_candidates(text):
        candidate_text = json.dumps(payload, ensure_ascii=False)
        parsed = parser(candidate_text)
        if parsed is None:
            continue
        error = validator(parsed, row)
        if error is None and question is not None:
            error = validate_question_operation(parsed, question)
        if error is None:
            return parsed, None
        last_error = error
    if last_error is not None:
        return None, last_error
    return None, "unparseable operation output"


def validate_question_operation(parsed: dict[str, Any], question: str) -> str | None:
    """Reject operations that contradict an explicit question type."""

    family = _question_operation_family(question)
    operation = parsed.get("operation")
    operations = {operation}
    steps = parsed.get("steps")
    if isinstance(steps, list):
        operations.update(
            step.get("operation")
            for step in steps
            if isinstance(step, dict)
        )
    if family == "percent_change":
        if "percent_change" in operations or (
            operations.intersection({"subtract", "difference"})
            and "divide" in operations
        ):
            return None
        allowed = {"percent_change", "subtract + divide"}
    elif family == "divide":
        allowed = {"divide", "ratio"}
    elif family == "average":
        allowed = {"average"}
    elif family == "add":
        allowed = {"add", "sum"}
    elif family == "multiply":
        allowed = {"multiply"}
    elif family == "subtract":
        allowed = {"subtract", "difference", "absolute_difference"}
    else:
        return None
    if operations.intersection(allowed):
        return None
    if operation not in allowed:
        return f"operation {operation!r} conflicts with question type; expected one of {sorted(allowed)}"
    return None


def _question_operation_family(question: str) -> str | None:
    normalized = re.sub(r"\s+", " ", str(question).lower()).strip()
    has_percent = any(token in normalized for token in ("percent", "percentage", "percentual"))
    has_change = any(
        token in normalized
        for token in ("increase", "increased", "decrease", "decreased", "decline", "change", "difference", "variation", "growth")
    )
    if has_percent and has_change and not any(
        token in normalized for token in ("percent of", "percentage of", "percent to", "percentage to")
    ):
        return "percent_change"
    if any(token in normalized for token in ("what percentage", "what percent", "percent of", "percentage of", "percent to", "percentage to", "portion of", "proportion of")):
        return "divide"
    if any(token in normalized for token in ("ratio", "rate of return", "roi", "return on")):
        return "divide"
    if not has_change and any(token in normalized for token in ("average", "mean")):
        return "average"
    if any(
        token in normalized
        for token in (
            "interest expense",
            "interest cost",
            "interest payment",
            "annual interest",
            "yearly interest",
        )
    ):
        return "multiply"
    if any(token in normalized for token in ("growth rate", "growth in", "rate of growth")):
        return "percent_change"
    if has_change:
        return "subtract"
    if any(token in normalized for token in ("average", "mean")):
        return "average"
    if any(token in normalized for token in ("total", "combined", "sum of")):
        return "add"
    return None


def question_operation_hint(question: str) -> str:
    """Return concise, question-only routing guidance for operation selection."""

    normalized = re.sub(r"\s+", " ", str(question).lower()).strip()
    years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
    year_hint = ""
    if len(years) >= 2:
        year_hint = (
            f" The question names years {years[0]} and {years[1]}; when it asks "
            "for an increase, decrease, or change, use later year minus earlier year."
        )
    family = _question_operation_family(question)
    if family == "percent_change":
        rule = "Use percent_change with [new value, old value]; divide by the old value and keep the result as a fraction."
    elif family == "divide":
        rule = "Use divide with [part value, total value]; keep the result as a fraction even when unit is percent."
    elif family == "average":
        rule = "Use average over the requested numeric cells."
    elif family == "add":
        rule = "Use add over the requested numeric cells."
    elif family == "multiply":
        rule = (
            "Use multiply over the principal and rate cells for interest expense; "
            "the program converts percent-formatted rate cells to fractions."
        )
    elif family == "subtract":
        rule = "Use subtract in semantic order [later/new value, earlier/old value]."
    else:
        rule = "Choose the operation that directly matches the question wording."
    return f"\nQuestion routing hint: {rule}{year_hint}"


def numeric_cell_catalog(row: dict[str, Any]) -> str:
    """Build a compact allow-list of numeric cell IDs for constrained prompting."""

    entries = []
    for cell_id, value in _table_cell_values(row).items():
        if parse_numeric_answer(value) is not None:
            entries.append(f"{cell_id}={value}")
    return "\nValid numeric cell IDs (copy exactly): " + ", ".join(entries)


def validate_cell_ids_operation(
    parsed: dict[str, Any], row: dict[str, Any]
) -> str | None:
    """Validate compact cell references before resolving their numeric values."""

    cell_ids = parsed.get("cell_ids")
    operation = parsed.get("operation")
    unit = str(parsed.get("unit", "")).lower()
    if not isinstance(cell_ids, list) or not cell_ids:
        return "cell_ids must be a non-empty list"
    if not all(isinstance(cell_id, str) for cell_id in cell_ids):
        return "every cell_id must be a string"
    if len(set(cell_ids)) != len(cell_ids):
        return "cell_ids must not contain duplicates"
    if not isinstance(operation, str) or operation not in ALLOWED_OPERATIONS:
        return "operation is not allowed"
    if unit not in ALLOWED_UNITS:
        return "unit is not allowed"

    cell_values = _table_cell_values(row)
    cell_id_pattern = re.compile(r"^r(\d+)c(\d+)$")
    numeric_values: list[Decimal] = []
    for cell_id in cell_ids:
        if not cell_id_pattern.fullmatch(cell_id):
            return "cell_id must match rNcM"
        if cell_id not in cell_values:
            return f"cell_id does not exist: {cell_id}"
        value = parse_numeric_answer(cell_values[cell_id])
        if value is None:
            return f"cell is not numeric: {cell_id}"
        numeric_values.append(value)

    exact_two = {
        "subtract", "difference", "absolute_difference", "divide",
        "ratio", "percent", "percentage", "percent_change",
    }
    at_least_two = {"add", "sum", "multiply", "average"}
    if operation in exact_two and len(numeric_values) != 2:
        return f"{operation} requires exactly two cell_ids"
    if operation in at_least_two and len(numeric_values) < 2:
        return f"{operation} requires at least two cell_ids"
    if operation in {"percent_change", "percent", "percentage"} and unit not in {"", "%", "percent"}:
        return f"{operation} requires percent unit"
    if operation == "ratio" and unit not in {"", "times", "multiple"}:
        return "ratio requires an empty, times, or multiple unit"
    return None


def validate_steps_operation(
    parsed: dict[str, Any], row: dict[str, Any]
) -> str | None:
    """Validate cell references and arity for an ordered multi-step chain."""

    steps = parsed.get("steps")
    unit = str(parsed.get("unit", "")).lower()
    if not isinstance(steps, list) or not steps:
        return "steps must be a non-empty list"
    if unit not in ALLOWED_UNITS:
        return "unit is not allowed"
    cell_values = _table_cell_values(row)
    cell_id_pattern = re.compile(r"^r(\d+)c(\d+)$")
    step_pattern = re.compile(r"^step(\d+)$")
    for step_index, step in enumerate(steps):
        operation = step.get("operation")
        operands = step.get("operands")
        if operation not in ALLOWED_OPERATIONS:
            return "operation is not allowed"
        if not isinstance(operands, list) or not operands:
            return f"step {step_index} operands must be non-empty"
        exact_two = {
            "subtract", "difference", "absolute_difference", "divide",
            "ratio", "percent", "percentage", "percent_change",
        }
        at_least_two = {"add", "sum", "multiply", "average"}
        if operation in exact_two and len(operands) != 2:
            return f"{operation} requires exactly two operands"
        if operation in at_least_two and len(operands) < 2:
            return f"{operation} requires at least two operands"
        for operand in operands:
            if not isinstance(operand, str):
                return "step operands must be strings"
            if parse_numeric_answer(operand) is not None:
                continue
            cell_match = cell_id_pattern.fullmatch(operand)
            if cell_match:
                if operand not in cell_values:
                    return f"cell_id does not exist: {operand}"
                if parse_numeric_answer(cell_values[operand]) is None:
                    return f"cell is not numeric: {operand}"
                continue
            step_match = step_pattern.fullmatch(operand)
            if not step_match or int(step_match.group(1)) >= step_index:
                return f"step reference is invalid: {operand}"
    final_operation = steps[-1]["operation"]
    if final_operation in {"percent_change", "percent", "percentage"} and unit not in {"", "%", "percent"}:
        return f"{final_operation} requires percent unit"
    if final_operation == "ratio" and unit not in {"", "times", "multiple"}:
        return "ratio requires an empty, times, or multiple unit"
    return None


def materialize_cell_ids_operation(
    parsed: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Resolve validated cell IDs to evidence and operands for deterministic calculation."""

    cell_values = _table_cell_values(row)
    cell_ids = parsed["cell_ids"]
    unit = str(parsed.get("unit", "")).lower()
    operation = parsed["operation"]
    if not unit and operation in {"percent_change", "percent", "percentage"}:
        unit = "percent"
    evidence = [
        {"cell_id": cell_id, "value": cell_values[cell_id]}
        for cell_id in cell_ids
    ]
    operands = [str(parse_numeric_answer(cell_values[cell_id])) for cell_id in cell_ids]
    return {
        **parsed,
        "evidence": evidence,
        "operands": operands,
        "unit": unit,
    }


def materialize_steps_operation(
    parsed: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Resolve and execute an ordered operation chain deterministically."""

    cell_values = _table_cell_values(row)
    results: dict[str, Decimal] = {}
    formulas: list[str] = []
    final_operands: list[str] = []
    for step_index, step in enumerate(parsed["steps"]):
        operands: list[str] = []
        for reference in step["operands"]:
            if reference.startswith("step"):
                value = results.get(reference)
                if value is None:
                    return {**parsed, "normalized_value": None, "structured": False}
                operands.append(str(value))
            else:
                value = parse_numeric_answer(cell_values.get(reference, reference))
                if value is None:
                    return {**parsed, "normalized_value": None, "structured": False}
                operands.append(str(value))
        repaired = repair_with_evidence_operation(
            {
                "operands": operands,
                "operation": step["operation"],
                "unit": parsed.get("unit", ""),
            }
        )
        if not repaired.get("calculator_used"):
            return {**parsed, "normalized_value": None, "structured": False}
        result = Decimal(repaired["normalized_value"])
        results[f"step{step_index}"] = result
        formulas.append(repaired.get("formula", ""))
        final_operands = operands
    return {
        **parsed,
        "operands": final_operands,
        "formula": " -> ".join(formulas),
        "normalized_value": str(results[f"step{len(parsed['steps']) - 1}"]),
        "calculator_used": True,
    }


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
                stable_cell_ids=bool(item.get("stable_cell_ids", False)),
                cell_ids_only=bool(item.get("cell_ids_only", False)),
                fallback_to_structured=bool(item.get("fallback_to_structured", False)),
                fallback_prompt_suffix=str(item.get("fallback_prompt_suffix", "")),
                question_routing=bool(item.get("question_routing", False)),
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
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["cell_id", "value"],
                    "properties": {
                        "cell_id": {"type": "string", "pattern": "^r[0-9]+c[0-9]+$"},
                        "value": {"type": ["string", "number"]},
                    },
                    "additionalProperties": False,
                },
            },
            "operands": {"type": "array", "items": {"type": "number"}},
            "operation": {"type": "string", "enum": sorted(ALLOWED_OPERATIONS)},
            "unit": {"type": "string"},
        },
        "additionalProperties": False,
    }


def cell_ids_operation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["cell_ids", "operation", "unit"],
        "properties": {
            "cell_ids": {"type": "array", "items": {"type": "string"}},
            "operation": {"type": "string", "enum": sorted(ALLOWED_OPERATIONS)},
            "unit": {"type": "string"},
        },
        "additionalProperties": False,
    }


def steps_operation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["steps", "unit"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["operation", "operands"],
                    "properties": {
                        "operation": {"type": "string", "enum": sorted(ALLOWED_OPERATIONS)},
                        "operands": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "unit": {"type": "string"},
        },
        "additionalProperties": False,
    }


def candidate_cell_ids_operation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": cell_ids_operation_schema(),
            }
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
        validation_failure_count = 0
        fallback_count = 0
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
                request_rows = [
                    add_stable_table_ids(row) if variant.stable_cell_ids else row
                    for row in rows
                ]
                requests = [
                    InferenceRequest.from_finqa_row(
                        request_row,
                        request_id=f"quality-repair-{variant.name}-{start + offset + 1:06d}",
                    )
                    for offset, request_row in enumerate(request_rows)
                ]
                def suffix_for_row(row: dict[str, Any]) -> str:
                    if not variant.question_routing:
                        return prompt_suffix
                    question = row.get("question") or row.get("qa", {}).get("question", "")
                    return (
                        f"{prompt_suffix}{question_operation_hint(question)}"
                        f"{numeric_cell_catalog(row)}"
                    )

                if variant.question_routing:
                    generated = [
                        service.generate_one(request, prompt_suffix=suffix_for_row(row))
                        for request, row in zip(requests, rows)
                    ]
                else:
                    generated = service.generate_batch(requests, prompt_suffix=prompt_suffix)
                for row, request, result in zip(rows, requests, generated):
                    request_suffix = suffix_for_row(row)
                    raw_output = result["output_text"]
                    validation_error = None
                    validation_failure_reason = None
                    fallback_used = False
                    if variant.output_mode == "cell_ids_operation":
                        if variant.stable_cell_ids:
                            parsed, validation_error = parse_validated_operation_output(
                                raw_output,
                                row,
                                variant.output_mode,
                                request.question if variant.question_routing else None,
                            )
                        else:
                            parsed = parse_cell_ids_operation_output(raw_output)
                    elif variant.output_mode == "steps_operation":
                        parsed, validation_error = parse_validated_operation_output(
                            raw_output,
                            row,
                            variant.output_mode,
                            request.question if variant.question_routing else None,
                        )
                    elif variant.output_mode == "candidate_cell_ids_operation":
                        parsed, validation_error = parse_validated_operation_output(
                            raw_output,
                            row,
                            variant.output_mode,
                            request.question if variant.question_routing else None,
                        )
                    elif variant.output_mode == "evidence_operation":
                        if variant.stable_cell_ids:
                            parsed, validation_error = parse_validated_operation_output(
                                raw_output,
                                row,
                                variant.output_mode,
                                request.question if variant.question_routing else None,
                            )
                        else:
                            parsed = parse_evidence_operation_output(raw_output)
                    else:
                        parsed = parse_structured_output(raw_output)
                    retry_used = False
                    if parsed is None and variant.format_retry and variant.output_mode == "json":
                        retry = service.generate_one(
                            request,
                            prompt_suffix=(
                                f"{request_suffix}\nYour previous response did not match the required JSON schema. "
                                "Return one valid JSON object only, with no markdown or explanation."
                            ),
                        )
                        raw_output = retry["output_text"]
                        parsed = parse_structured_output(raw_output)
                        format_retry_count += 1
                        retry_used = True
                    elif (
                        variant.format_retry
                        and variant.output_mode in OPERATION_OUTPUT_MODES
                        and (parsed is None or validation_error is not None)
                    ):
                        validation_hint = (
                            f" Validation failure: {validation_error}."
                            if validation_error
                            else ""
                        )
                        contract_name = (
                            "cell-id operation schema"
                            if variant.output_mode == "cell_ids_operation"
                            else "steps operation schema"
                            if variant.output_mode == "steps_operation"
                            else "candidate cell-id operation schema"
                            if variant.output_mode == "candidate_cell_ids_operation"
                            else "evidence-operation schema"
                        )
                        retry = service.generate_one(
                            request,
                            prompt_suffix=(
                                f"{request_suffix}\nYour previous response did not match the required {contract_name}. "
                                f"Return one valid JSON object only, with no markdown or explanation.{validation_hint}"
                            ),
                        )
                        raw_output = retry["output_text"]
                        if variant.stable_cell_ids:
                            parsed, validation_error = parse_validated_operation_output(
                                raw_output,
                                row,
                                variant.output_mode,
                                request.question if variant.question_routing else None,
                            )
                        else:
                            parsed = (
                                parse_cell_ids_operation_output(raw_output)
                                if variant.output_mode == "cell_ids_operation"
                                else (
                                    parse_steps_operation_output(raw_output)
                                    if variant.output_mode == "steps_operation"
                                    else parse_evidence_operation_output(raw_output)
                                )
                            )
                            validation_error = None
                        format_retry_count += 1
                        retry_used = True
                    if (
                        variant.output_mode == "cell_ids_operation"
                        and variant.fallback_to_structured
                        and (parsed is None or validation_error is not None)
                    ):
                        validation_failure_reason = (
                            validation_error or "unparseable operation output"
                        )
                        fallback = service.generate_one(
                            request,
                            prompt_suffix=variant.fallback_prompt_suffix,
                        )
                        fallback_output = fallback["output_text"]
                        fallback_parsed = parse_structured_output(fallback_output)
                        fallback_count += 1
                        if fallback_parsed is not None:
                            raw_output = fallback_output
                            parsed = fallback_parsed
                            validation_error = None
                            fallback_used = True
                    if (
                        parsed is None
                        and variant.output_mode in OPERATION_OUTPUT_MODES
                        and variant.stable_cell_ids
                    ):
                        validation_error = validation_error or "unparseable operation output"
                        parsed = {
                            "normalized_value": None,
                            "structured": False,
                            "validation_error": validation_error,
                        }
                        validation_failure_reason = validation_failure_reason or validation_error
                    elif parsed is None:
                        parsed = extract_final_numeric(raw_output)
                    elif validation_error is not None:
                        parsed = {
                            **parsed,
                            "normalized_value": None,
                            "structured": False,
                            "validation_error": validation_error,
                        }
                        validation_failure_reason = validation_failure_reason or validation_error
                    if validation_failure_reason is not None:
                        validation_failure_count += 1
                    if parsed.get("structured"):
                        structured_count += 1
                    if (
                        variant.output_mode == "cell_ids_operation"
                        and validation_error is None
                        and not fallback_used
                    ):
                        parsed = materialize_cell_ids_operation(parsed, row)
                    elif (
                        variant.output_mode == "steps_operation"
                        and validation_error is None
                        and not fallback_used
                    ):
                        parsed = materialize_steps_operation(parsed, row)
                    if variant.output_mode in OPERATION_OUTPUT_MODES and not fallback_used:
                        repaired = parsed
                        if validation_error is None and variant.output_mode != "steps_operation":
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
                        "candidate_count": parsed.get("candidate_count"),
                        "candidate_selected_index": parsed.get("candidate_selected_index"),
                        "validation_error": validation_failure_reason or parsed.get("validation_error"),
                        "fallback_used": fallback_used,
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
                "stable_cell_ids": variant.stable_cell_ids,
                "cell_ids_only": variant.cell_ids_only,
                "fallback_to_structured": variant.fallback_to_structured,
                "question_routing": variant.question_routing,
                "prompt_suffix": prompt_suffix,
                "metrics": metrics,
                "structured_output_rate": structured_count / len(evaluation_rows) if evaluation_rows else 0.0,
                "calculator_use_rate": calculator_count / len(evaluation_rows) if evaluation_rows else 0.0,
                "format_retry_count": format_retry_count,
                "validation_failure_count": validation_failure_count,
                "fallback_count": fallback_count,
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
        "cell_ids_operation_schema": cell_ids_operation_schema(),
        "steps_operation_schema": steps_operation_schema(),
        "candidate_cell_ids_operation_schema": candidate_cell_ids_operation_schema(),
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
