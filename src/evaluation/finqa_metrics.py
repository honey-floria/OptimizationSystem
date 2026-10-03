"""Numeric answer metrics for FinQA-style evaluation and regression."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def parse_numeric_answer(value: Any) -> Decimal | None:
    text = str(value).strip()
    match = NUMBER_PATTERN.search(text)
    if not match:
        return None
    try:
        number = Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None
    if "%" in text[match.end() : match.end() + 2]:
        number /= Decimal("100")
    return number


def _is_correct(prediction: Any, reference: Any, tolerance: Decimal) -> bool:
    predicted_number = parse_numeric_answer(prediction)
    reference_number = parse_numeric_answer(reference)
    if predicted_number is None or reference_number is None:
        return False
    return abs(predicted_number - reference_number) <= tolerance


def evaluate_numeric_answers(
    predictions: Sequence[Any],
    references: Sequence[Any],
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")
    decimal_tolerance = Decimal(str(tolerance))
    parsed = [parse_numeric_answer(value) is not None for value in predictions]
    correct = [
        _is_correct(prediction, reference, decimal_tolerance)
        for prediction, reference in zip(predictions, references)
    ]
    total = len(references)
    return {
        "total": total,
        "parsed": sum(parsed),
        "correct": sum(correct),
        "parse_rate": sum(parsed) / total if total else 0.0,
        "numeric_accuracy": sum(correct) / total if total else 0.0,
    }


def classify_numeric_regression(
    baseline_prediction: Any,
    candidate_prediction: Any,
    reference: Any,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    decimal_tolerance = Decimal(str(tolerance))
    baseline_correct = _is_correct(
        baseline_prediction, reference, decimal_tolerance
    )
    candidate_number = parse_numeric_answer(candidate_prediction)
    candidate_correct = _is_correct(
        candidate_prediction, reference, decimal_tolerance
    )
    if candidate_number is None:
        error_type = "unparseable_numeric_output"
    elif not candidate_correct:
        error_type = "numeric_answer_mismatch"
    else:
        error_type = None
    return {
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "new_regression": baseline_correct and not candidate_correct,
        "critical_error_type": error_type,
    }
