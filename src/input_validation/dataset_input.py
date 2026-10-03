"""Validate evaluation dataset assets declared by a dataset manifest."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.finqa_assets import dataset_hash, normalize_question
from src.input_validation.model_input import FAIL, PASS, CheckResult


ASSET_ROLES = (
    "calibration",
    "evaluation_dev",
    "evaluation_test",
    "high_risk_regression",
)


@dataclass
class DatasetValidationReport:
    manifest: str
    checks: list[CheckResult]
    actual_sizes: dict[str, int]
    actual_sha256: dict[str, str]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
            "actual_sizes": self.actual_sizes,
            "actual_sha256": self.actual_sha256,
        }


def _require_datasets():
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError("Install datasets to validate dataset assets.") from exc
    return load_from_disk


def _load_callable(value: Any) -> bool:
    if not isinstance(value, str) or ":" not in value:
        return False
    module_name, attribute_name = value.rsplit(":", 1)
    try:
        candidate = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError):
        return False
    return callable(candidate)


def _field_exists(row: dict[str, Any], dotted_path: str) -> bool:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return value is not None


def _question_key(row: dict[str, Any]) -> str:
    question = row.get("question")
    if question is None and isinstance(row.get("qa"), dict):
        question = row["qa"].get("question")
    if question is None:
        raise KeyError("Sample has no question or qa.question field.")
    return normalize_question(str(question))


def validate_dataset_input(manifest_file: str | Path) -> DatasetValidationReport:
    manifest_path = Path(manifest_file).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset manifest: {exc}") from exc
    assets = manifest.get("assets", {})
    manifest_ready = (
        manifest.get("schema_version") == 1
        and isinstance(assets, dict)
        and all(role in assets for role in ASSET_ROLES)
    )
    checks = [
        CheckResult(
            "dataset_manifest_received",
            PASS if manifest_ready else FAIL,
            "Dataset manifest schema and four asset roles are present."
            if manifest_ready
            else "Manifest schema or required asset roles are missing.",
        )
    ]

    load_from_disk = _require_datasets()
    loaded: dict[str, Any] = {}
    root = manifest_path.parent
    for role in ASSET_ROLES:
        entry = assets.get(role, {})
        path_value = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path_value, str) or not path_value:
            continue
        asset_path = Path(path_value)
        if not asset_path.is_absolute():
            asset_path = root / asset_path
        try:
            loaded[role] = load_from_disk(str(asset_path))
        except (FileNotFoundError, OSError, ValueError):
            continue

    def received(roles: tuple[str, ...]) -> bool:
        return all(
            role in loaded
            and isinstance(assets.get(role, {}).get("version"), str)
            and bool(assets[role]["version"])
            for role in roles
        )

    checks.extend(
        [
            CheckResult(
                "calibration_path_and_version_received",
                PASS if received(("calibration",)) else FAIL,
                "Calibration asset is loadable and versioned.",
            ),
            CheckResult(
                "evaluation_paths_and_versions_received",
                PASS if received(("evaluation_dev", "evaluation_test")) else FAIL,
                "Development and final evaluation assets are loadable and versioned.",
            ),
            CheckResult(
                "regression_path_and_version_received",
                PASS if received(("high_risk_regression",)) else FAIL,
                "High-risk regression asset is loadable and versioned.",
            ),
        ]
    )

    isolation_ready = False
    calibration_overlap_count = 0
    regression_overlap_count = 0
    if all(role in loaded for role in ASSET_ROLES):
        try:
            calibration_questions = {_question_key(row) for row in loaded["calibration"]}
            evaluation_questions = {
                _question_key(row)
                for role in ("evaluation_dev", "evaluation_test")
                for row in loaded[role]
            }
            regression_questions = {
                _question_key(row) for row in loaded["high_risk_regression"]
            }
            test_questions = {_question_key(row) for row in loaded["evaluation_test"]}
            calibration_overlap_count = len(
                calibration_questions & evaluation_questions
            )
            regression_overlap_count = len(regression_questions & test_questions)
            isolation_ready = (
                calibration_overlap_count == 0 and regression_overlap_count == 0
            )
        except KeyError:
            isolation_ready = False
    checks.append(
        CheckResult(
            "calibration_and_evaluation_isolated",
            PASS if isolation_ready else FAIL,
            "Calibration/evaluation overlap: "
            f"{calibration_overlap_count}; regression/final-test overlap: "
            f"{regression_overlap_count}.",
        )
    )

    schema_ready = bool(loaded)
    for role, dataset in loaded.items():
        required_fields = assets.get(role, {}).get("required_fields", [])
        schema_ready = schema_ready and bool(required_fields) and all(
            all(_field_exists(dict(row), field) for field in required_fields)
            for row in dataset
        )
    schema_ready = schema_ready and len(loaded) == len(ASSET_ROLES)
    checks.append(
        CheckResult(
            "sample_schema_and_fields_confirmed",
            PASS if schema_ready else FAIL,
            "All samples contain the declared required fields.",
        )
    )

    metric_interface = manifest.get("metric_interface", {})
    metric_ready = isinstance(metric_interface, dict) and _load_callable(
        metric_interface.get("callable")
    )
    checks.append(
        CheckResult(
            "business_metric_interface_confirmed",
            PASS if metric_ready else FAIL,
            "Business metric callable is importable.",
        )
    )

    regression_policy = manifest.get("regression_policy", {})
    policy_ready = (
        isinstance(regression_policy, dict)
        and _load_callable(regression_policy.get("callable"))
        and bool(regression_policy.get("critical_error_types"))
        and regression_policy.get("new_regression_definition")
        == "baseline_correct_and_candidate_incorrect"
    )
    checks.append(
        CheckResult(
            "high_risk_and_critical_error_policy_confirmed",
            PASS if policy_ready else FAIL,
            "High-risk regression set and critical error policy are executable.",
        )
    )

    actual_sizes = {role: len(dataset) for role, dataset in loaded.items()}
    actual_hashes = {
        role: dataset_hash(list(dataset)) for role, dataset in loaded.items()
    }
    integrity_ready = len(loaded) == len(ASSET_ROLES) and all(
        assets[role].get("size") == actual_sizes.get(role)
        and assets[role].get("sha256") == actual_hashes.get(role)
        for role in ASSET_ROLES
    )
    checks.append(
        CheckResult(
            "asset_versions_and_hashes_recorded",
            PASS if integrity_ready else FAIL,
            "All asset sizes and SHA-256 hashes match the manifest.",
        )
    )
    return DatasetValidationReport(
        manifest=str(manifest_path),
        checks=checks,
        actual_sizes=actual_sizes,
        actual_sha256=actual_hashes,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = validate_dataset_input(args.manifest)
    except (ValueError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
