"""SmoothQuant INT8 and NVIDIA/Ascend FP8 capability evidence."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, NOT_RUN, PASS, CheckResult


@dataclass
class Int8Fp8ValidationReport:
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
        raise ValueError(f"Invalid INT8/FP8 JSON {file_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {file_path}")
    return file_path, payload


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_capabilities() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": False, "nvidia_fp8": False, "ascend_npu": False}
    float8_names = [
        name for name in ("float8_e4m3fn", "float8_e5m2")
        if hasattr(torch, name)
    ]
    cuda = bool(torch.cuda.is_available())
    try:
        import torch_npu  # type: ignore
        npu = bool(getattr(torch, "npu", None) and torch.npu.is_available())
    except ImportError:
        npu = False
    return {
        "torch": getattr(torch, "__version__", None),
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "nvidia_fp8": cuda and bool(float8_names),
        "fp8_dtypes": float8_names,
        "ascend_npu": npu,
        "ascend_fp8": npu and bool(float8_names),
    }


def collect_int8_fp8_evidence(
    config_file: str | Path,
    calibration_manifest_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_path, config = _load_json(config_file)
    calibration_path, calibration = _load_json(calibration_manifest_file)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    capabilities = _torch_capabilities()
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_file": str(config_path),
        "calibration_manifest": str(calibration_path),
        "calibration_asset": calibration.get("assets", {}).get("calibration"),
        "plans": [{
            "plan_id": config.get("plan_id"),
            "method": "smoothquant",
            "weight_bits": config.get("weight_bits"),
            "activation_bits": config.get("activation_bits"),
            "alpha": config.get("alpha"),
            "granularity": config.get("quantization_granularity"),
            "export_contract": {
                "config_file": "quantization_config.json",
                "manifest_file": "quantized_model_manifest.json",
                "event_log_file": "quantization_events.jsonl",
            },
        }],
        "backend_versions": {
            "smoothquant": _version("smoothquant"),
            "torch": capabilities["torch"],
            "transformers": _version("transformers"),
        },
        "hardware_capabilities": capabilities,
        "status": "framework_ready_not_quantized",
    }


def validate_int8_fp8_evidence(evidence_file: str | Path) -> Int8Fp8ValidationReport:
    evidence_path, evidence = _load_json(evidence_file)
    plan = (evidence.get("plans") or [{}])[0]
    calibration = evidence.get("calibration_asset", {})
    capabilities = evidence.get("hardware_capabilities", {})
    config_ready = (
        plan.get("method") == "smoothquant"
        and plan.get("weight_bits") == 8
        and plan.get("activation_bits") == 8
        and bool(calibration.get("sha256"))
    )
    export_ready = all(
        plan.get("export_contract", {}).get(name) == value
        for name, value in {
            "config_file": "quantization_config.json",
            "manifest_file": "quantized_model_manifest.json",
            "event_log_file": "quantization_events.jsonl",
        }.items()
    )
    checks = [
        CheckResult("smoothquant_int8_config", PASS if config_ready else FAIL, "SmoothQuant W8A8 configuration and calibration are recorded."),
        CheckResult("smoothquant_export_contract", PASS if export_ready else FAIL, "INT8 export and event-log names are fixed."),
        CheckResult("nvidia_fp8_capability", PASS if capabilities.get("nvidia_fp8") else NOT_RUN, "NVIDIA CUDA exposes a supported PyTorch FP8 dtype."),
        CheckResult("ascend_fp8_capability", PASS if capabilities.get("ascend_fp8") else NOT_RUN, "Ascend NPU and FP8 dtype support are detected."),
        CheckResult("smoothquant_real_execution", PASS if evidence.get("status") == "completed" else NOT_RUN, "SmoothQuant INT8 weights were exported."),
    ]
    return Int8Fp8ValidationReport(str(evidence_path), checks)
