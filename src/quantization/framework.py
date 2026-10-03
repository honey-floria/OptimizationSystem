"""Configuration, adapter and evidence contracts for quantization backends."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, NOT_RUN, PASS, CheckResult


SUPPORTED_METHODS = {"awq", "gptq", "smoothquant"}


@dataclass(frozen=True)
class QuantizationConfig:
    plan_id: str
    method: str
    bits: int
    weight_bits: int
    activation_bits: int
    group_size: int | None
    backend_package: str
    calibration_asset_role: str = "calibration"
    export_schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantizationFrameworkReport:
    evidence: str
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid quantization JSON {file_path}: {exc}") from exc
    return file_path, payload


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_quantization_config(
    plan: dict[str, Any],
    calibration_asset: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    method = str(plan.get("method", "")).lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported quantization method: {method}")
    required = ("plan_id", "bits", "weight_bits", "activation_bits")
    if any(plan.get(field) is None for field in required):
        raise ValueError("Quantization plan is missing required fields.")
    if method in {"awq", "gptq"} and plan.get("group_size") not in {
        32,
        64,
        128,
    }:
        raise ValueError("AWQ/GPTQ group_size must be 32, 64, or 128.")
    output_path = Path(output_dir).expanduser().resolve()
    return {
        "schema_version": 1,
        "plan": {
            **plan,
            "method": method,
            "output_dir": str(output_path),
        },
        "calibration": {
            "role": "calibration",
            "path": calibration_asset.get("path"),
            "version": calibration_asset.get("version"),
            "size": calibration_asset.get("size"),
            "sha256": calibration_asset.get("sha256"),
        },
        "export_contract": {
            "format": "safetensors_or_backend_native",
            "config_file": "quantization_config.json",
            "manifest_file": "quantized_model_manifest.json",
            "required_files": ["config.json", "quantization_config.json"],
        },
        "logging_contract": {
            "format": "jsonl",
            "file": "quantization_events.jsonl",
            "required_fields": [
                "event",
                "plan_id",
                "method",
                "timestamp",
                "status",
            ],
        },
    }


def collect_framework_evidence(
    config_file: str | Path,
    calibration_manifest_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_path, config = _load_json(config_file)
    calibration_path, calibration = _load_json(calibration_manifest_file)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    plans = config.get("plans", [])
    plan_configs = []
    for plan in plans:
        plan_configs.append(
            build_quantization_config(
                plan,
                calibration.get("assets", {}).get("calibration", {}),
                output_path / plan["plan_id"],
            )
        )
    package_versions = {
        package: _package_version(package)
        for package in (
            "autoawq",
            "auto-gptq",
            "smoothquant",
            "torch",
            "transformers",
        )
    }
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_file": str(config_path),
        "calibration_manifest": str(calibration_path),
        "calibration_asset": calibration.get("assets", {}).get("calibration"),
        "plans": plan_configs,
        "backend_packages": package_versions,
        "backend_available": {
            "awq": package_versions["autoawq"] is not None,
            "gptq": package_versions["auto-gptq"] is not None,
            "smoothquant": package_versions["smoothquant"] is not None,
        },
        "export_contract": {
            "manifest_file": "quantized_model_manifest.json",
            "config_file": "quantization_config.json",
            "event_log_file": "quantization_events.jsonl",
        },
        "status": "framework_ready_not_quantized",
    }


def validate_quantization_evidence(
    evidence_file: str | Path,
) -> QuantizationFrameworkReport:
    evidence_path, evidence = _load_json(evidence_file)
    plans = evidence.get("plans", [])
    methods = {plan.get("plan", {}).get("method") for plan in plans}
    config_ready = (
        len(plans) >= 3
        and {"awq", "gptq", "smoothquant"}.issubset(methods)
        and all(plan.get("calibration", {}).get("sha256") for plan in plans)
    )
    export_ready = all(
        plan.get("export_contract", {}).get("manifest_file")
        == "quantized_model_manifest.json"
        and plan.get("export_contract", {}).get("config_file")
        == "quantization_config.json"
        for plan in plans
    )
    logging_ready = all(
        plan.get("logging_contract", {}).get("format") == "jsonl"
        and len(plan.get("logging_contract", {}).get("required_fields", [])) >= 5
        for plan in plans
    )
    versions_ready = all(
        evidence.get("backend_packages", {}).get(name) is not None
        for name in ("torch", "transformers")
    )
    backend_status_ready = evidence.get("status") == "framework_ready_not_quantized"
    checks = [
        CheckResult("awq_gptq_smoothquant_configs", PASS if config_ready else FAIL, "AWQ, GPTQ, and SmoothQuant share one validated configuration contract."),
        CheckResult("unified_export_contract", PASS if export_ready else FAIL, "Quantized artifacts use one manifest and config naming contract."),
        CheckResult("unified_quantization_logging", PASS if logging_ready else FAIL, "All adapters declare the same JSONL event-log fields."),
        CheckResult("quantization_tool_versions_recorded", PASS if versions_ready else FAIL, "Core PyTorch and Transformers versions are recorded."),
        CheckResult("real_quantization_execution", NOT_RUN if backend_status_ready else FAIL, "Actual backend execution and exported weights are deferred to 6.2-6.4."),
    ]
    return QuantizationFrameworkReport(str(evidence_path), checks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--calibration-manifest", required=True)
    collect.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--evidence", required=True)
    validate.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "collect":
        payload = collect_framework_evidence(
            args.config, args.calibration_manifest, args.output
        )
        output = Path(args.output).expanduser() / "framework_evidence.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    report = validate_quantization_evidence(args.evidence)
    payload = report.to_dict()
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
