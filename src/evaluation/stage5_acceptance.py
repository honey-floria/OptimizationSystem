"""Validate the NVIDIA baseline-stage deliverables for Todo 5.5."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, PASS, CheckResult


@dataclass
class Stage5Report:
    scope: str
    checks: list[CheckResult]
    deliverables: dict[str, bool]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks) and all(
            self.deliverables.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
            "deliverables": self.deliverables,
        }


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid stage evidence {file_path}: {exc}") from exc
    return file_path, payload


def validate_stage5(
    model_report_file: str | Path,
    baseline_validation_file: str | Path,
    benchmark_validation_file: str | Path,
    benchmark_report_file: str | Path,
    quality_validation_file: str | Path,
    environment_validation_file: str | Path,
    dataset_validation_file: str | Path,
    business_validation_file: str | Path,
    deliverable_files: dict[str, str | Path],
) -> Stage5Report:
    _, model = _load(model_report_file)
    _, baseline = _load(baseline_validation_file)
    _, benchmark_validation = _load(benchmark_validation_file)
    _, benchmark = _load(benchmark_report_file)
    _, quality = _load(quality_validation_file)
    _, environment = _load(environment_validation_file)
    _, dataset = _load(dataset_validation_file)
    _, business = _load(business_validation_file)
    environment_status = {
        check.get("name"): check.get("status")
        for check in environment.get("checks", [])
    }
    stable_ready = model.get("complete") is True and baseline.get("complete") is True
    repeatable_ready = (
        benchmark_validation.get("complete") is True
        and benchmark.get("benchmark_config", {}).get("measured_runs", 0) >= 2
        and len(benchmark.get("cases", [])) >= 2
    )
    quality_ready = quality.get("complete") is True
    performance_ready = benchmark_validation.get("complete") is True
    required_environment_checks = {
        "nvidia_environment_file_created",
        "python_torch_transformers_versions_pinned",
        "inference_framework_version_pinned",
        "dependency_lock_saved",
        "hardware_and_driver_recorded",
    }
    versions_ready = (
        dataset.get("complete") is True
        and business.get("complete") is True
        and all(
            environment_status.get(name) == PASS
            for name in required_environment_checks
        )
        and not any(
            check.get("status") == FAIL
            for check in environment.get("checks", [])
        )
    )
    checks = [
        CheckResult("fp16_or_bf16_model_stable", PASS if stable_ready else FAIL, "Model loading and repeated baseline inference passed."),
        CheckResult("benchmark_repeatable", PASS if repeatable_ready else FAIL, "The versioned benchmark contains repeated measurements."),
        CheckResult("baseline_business_quality_confirmed", PASS if quality_ready else FAIL, "The complete FinQA development baseline passed quality validation."),
        CheckResult("baseline_performance_confirmed", PASS if performance_ready else FAIL, "The FP16 latency, throughput, and memory report passed validation."),
        CheckResult("nvidia_inputs_and_environment_versioned", PASS if versions_ready else FAIL, "NVIDIA-stage inputs and environment versions are registered."),
    ]
    deliverables = {
        name: Path(path).expanduser().is_file()
        for name, path in deliverable_files.items()
    }
    return Stage5Report("nvidia_t4_baseline", checks, deliverables)
