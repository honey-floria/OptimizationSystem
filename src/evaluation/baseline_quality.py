"""Run and validate the FP16 FinQA business-quality baseline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.baseline.service import BaselineService, InferenceRequest
from src.evaluation.finqa_metrics import evaluate_numeric_answers
from src.input_validation.model_input import FAIL, PASS, CheckResult


@dataclass
class QualityValidationReport:
    report: str
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {file_path}: {exc}") from exc
    return file_path, payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer is None and isinstance(row.get("qa"), dict):
        answer = row["qa"].get("answer")
    if answer is None:
        raise ValueError("FinQA evaluation row has no reference answer.")
    return str(answer)


def run_baseline_quality(
    service: BaselineService,
    dataset: Any,
    dataset_manifest: dict[str, Any],
    quality_config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path / "baseline_predictions.jsonl"
    predictions_path.unlink(missing_ok=True)
    batch_size = int(quality_config["batch_size"])
    if batch_size <= 0:
        raise ValueError("Quality evaluation batch_size must be positive.")
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    predictions = []
    references = []
    prompt_suffix = str(quality_config.get("prompt_suffix", ""))
    original_generation = service.config.get("generation", {})
    if isinstance(quality_config.get("generation"), dict):
        service.config["generation"] = quality_config["generation"]
    effective_generation = service.config.get("generation", {})
    with predictions_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            rows = [dataset[index] for index in range(start, stop)]
            requests = [
                InferenceRequest.from_finqa_row(
                    row, request_id=f"quality-dev-{start + offset + 1:06d}"
                )
                for offset, row in enumerate(rows)
            ]
            results = service.generate_batch(requests, prompt_suffix=prompt_suffix)
            for index, (row, result) in enumerate(
                zip(rows, results), start=start
            ):
                reference = _reference_answer(row)
                prediction = result["output_text"]
                item_metrics = evaluate_numeric_answers(
                    [prediction],
                    [reference],
                    tolerance=float(quality_config.get("tolerance", 1e-4)),
                )
                record = {
                    "index": index,
                    "request_id": result["request_id"],
                    "question": requests[index - start].question,
                    "prediction": prediction,
                    "reference": reference,
                    "parsed": item_metrics["parsed"] == 1,
                    "correct": item_metrics["correct"] == 1,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                predictions.append(prediction)
                references.append(reference)
            print(f"Quality evaluation: {stop}/{len(dataset)}", flush=True)
    service.config["generation"] = original_generation
    metrics = evaluate_numeric_answers(
        predictions,
        references,
        tolerance=float(quality_config.get("tolerance", 1e-4)),
    )
    minimum_parse_rate = float(quality_config["minimum_parse_rate"])
    dataset_asset = dataset_manifest.get("assets", {}).get(
        quality_config["dataset_role"], {}
    )
    return {
        "schema_version": 1,
        "evaluation_name": quality_config.get("evaluation_name"),
        "scenario_id": quality_config.get("scenario_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "model_metadata": service.model_metadata,
        "environment": service.environment_metadata(),
        "generation": effective_generation,
        "prompt_suffix": prompt_suffix,
        "dataset": {
            "role": quality_config["dataset_role"],
            "version": dataset_asset.get("version"),
            "sha256": dataset_asset.get("sha256"),
            "expected_size": dataset_asset.get("size"),
            "evaluated_size": len(dataset),
        },
        "metrics": metrics,
        "quality_gate": {
            "minimum_parse_rate": minimum_parse_rate,
            "parse_rate_passed": metrics["parse_rate"] >= minimum_parse_rate,
            "numeric_accuracy_is_fp16_reference": True,
        },
        "predictions": {
            "file": predictions_path.name,
            "row_count": len(predictions),
            "sha256": _sha256(predictions_path),
        },
    }


def validate_quality_report(
    report_file: str | Path,
) -> QualityValidationReport:
    report_path, report = _load_json(report_file)
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", {})
    prediction_info = report.get("predictions", {})
    prediction_path = report_path.parent / str(prediction_info.get("file", ""))
    try:
        prediction_rows = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError):
        prediction_rows = []
    coverage_ready = (
        isinstance(dataset.get("expected_size"), int)
        and dataset.get("evaluated_size") == dataset.get("expected_size")
        and metrics.get("total") == dataset.get("expected_size")
    )
    metrics_ready = (
        isinstance(metrics.get("numeric_accuracy"), (int, float))
        and 0 <= metrics["numeric_accuracy"] <= 1
        and isinstance(metrics.get("parse_rate"), (int, float))
        and 0 <= metrics["parse_rate"] <= 1
    )
    gate_ready = report.get("quality_gate", {}).get("parse_rate_passed") is True
    traceability_ready = (
        all(
            report.get("model_metadata", {}).get(field)
            for field in ("model_id", "version", "commit")
        )
        and all(dataset.get(field) for field in ("role", "version", "sha256"))
        and report.get("environment", {}).get("dtype") in {"float16", "bfloat16"}
    )
    predictions_ready = (
        prediction_path.is_file()
        and len(prediction_rows) == prediction_info.get("row_count")
        == dataset.get("evaluated_size")
        and prediction_info.get("sha256") == _sha256(prediction_path)
    )
    checks = [
        CheckResult("full_development_set_evaluated", PASS if coverage_ready else FAIL, "Every versioned FinQA development sample was evaluated."),
        CheckResult("numeric_business_metrics_recorded", PASS if metrics_ready else FAIL, "Numeric accuracy and parse rate are recorded."),
        CheckResult("minimum_parse_rate_gate_passed", PASS if gate_ready else FAIL, "The FP16 baseline meets the configured parse-rate gate."),
        CheckResult("model_dataset_environment_traceable", PASS if traceability_ready else FAIL, "Model, dataset, and FP16 environment versions are traceable."),
        CheckResult("baseline_predictions_saved", PASS if predictions_ready else FAIL, "Per-sample predictions are saved and hash-verified."),
    ]
    return QualityValidationReport(str(report_path), checks)
