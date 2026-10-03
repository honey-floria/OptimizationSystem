"""AWQ INT4 experiment runner and evidence validator."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.input_validation.model_input import FAIL, NOT_RUN, PASS, CheckResult
from src.baseline.service import BaselineService


REQUIRED_GROUP_SIZES = (32, 64, 128)


@dataclass(frozen=True)
class AWQPlan:
    plan_id: str
    method: str
    bits: int
    weight_bits: int
    activation_bits: int
    group_size: int
    quantization_granularity: str
    backend_package: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AWQValidationReport:
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
        raise ValueError(f"Invalid AWQ JSON {file_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"AWQ JSON must contain an object: {file_path}")
    return file_path, payload


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_awq_plans(config: dict[str, Any]) -> list[AWQPlan]:
    """Expand the AWQ experiment matrix into one plan per group size."""
    group_sizes = tuple(int(value) for value in config.get("group_sizes", []))
    if group_sizes != REQUIRED_GROUP_SIZES:
        raise ValueError(
            "AWQ experiment must test group sizes 32, 64, and 128."
        )
    if config.get("method") != "awq" or int(config.get("bits", 0)) != 4:
        raise ValueError("6.2 requires an AWQ INT4 configuration.")
    plans = []
    for group_size in group_sizes:
        plans.append(
            AWQPlan(
                plan_id=f"{config['plan_id']}-g{group_size}",
                method="awq",
                bits=4,
                weight_bits=int(config.get("weight_bits", 4)),
                activation_bits=int(config.get("activation_bits", 16)),
                group_size=group_size,
                quantization_granularity=str(
                    config.get("quantization_granularity", "groupwise")
                ),
                backend_package=str(config.get("backend_package", "autoawq")),
            )
        )
    return plans


def _append_event(log_file: Path, event: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _resolve_model_path(model_manifest_file: str | Path) -> Path:
    manifest_path, manifest = _load_json(model_manifest_file)
    value = manifest.get("model_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model manifest must define model_path.")
    model_path = Path(value).expanduser()
    if not model_path.is_absolute():
        model_path = manifest_path.parent / model_path
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"AWQ model directory does not exist: {model_path}")
    return model_path


def _supported_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return values
    return {name: value for name, value in values.items() if name in parameters}


def run_awq_plan(
    plan: AWQPlan,
    model_manifest_file: str | Path,
    calibration_prompts: Iterable[str],
    output_dir: str | Path,
    model_output_dir: str | Path | None = None,
    max_calibration_samples: int = 1024,
    max_calibration_seq_len: int = 512,
) -> dict[str, Any]:
    """Run one real AutoAWQ plan and write the common export contract."""
    output_path = Path(output_dir).expanduser().resolve()
    model_path_output = (
        Path(model_output_dir).expanduser().resolve()
        if model_output_dir is not None
        else output_path
    )
    output_path.mkdir(parents=True, exist_ok=True)
    model_path_output.mkdir(parents=True, exist_ok=True)
    event_log = output_path / "quantization_events.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    base_event = {
        "event": "quantization",
        "plan_id": plan.plan_id,
        "method": plan.method,
        "timestamp": started,
    }
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        _append_event(
            event_log,
            {**base_event, "status": "not_run", "error": repr(exc)},
        )
        return {
            "plan": plan.to_dict(),
            "status": "not_run",
            "detail": "autoawq is not installed; install it in Colab before running 6.2.",
            "output_dir": str(output_path),
        }

    model_path = _resolve_model_path(model_manifest_file)
    prompts = list(calibration_prompts)[:max_calibration_samples]
    if not prompts:
        raise ValueError("AWQ requires at least one calibration prompt.")
    _append_event(event_log, {**base_event, "status": "started"})
    started_clock = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoAWQForCausalLM.from_pretrained(
            str(model_path), safetensors=True, device_map="auto"
        )
        quant_config = {
            "zero_point": True,
            "q_group_size": plan.group_size,
            "w_bit": plan.weight_bits,
            "version": "GEMM",
        }
        quantize_kwargs = _supported_kwargs(
            model.quantize,
            {
                "calib_data": prompts,
                "max_calib_samples": len(prompts),
                "max_calib_seq_len": max_calibration_seq_len,
            },
        )
        model.quantize(tokenizer, quant_config=quant_config, **quantize_kwargs)
        model.save_quantized(str(model_path_output), safetensors=True)
        tokenizer.save_pretrained(str(model_path_output))
        quantization_config = {
            "schema_version": 1,
            "plan": plan.to_dict(),
            "quant_config": quant_config,
            "calibration": {
                "size": len(prompts),
                "max_seq_len": max_calibration_seq_len,
            },
        }
        (model_path_output / "quantization_config.json").write_text(
            json.dumps(quantization_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_path / "quantization_config.json").write_text(
            json.dumps(
                quantization_config,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "format": "autoawq",
            "plan": plan.to_dict(),
            "model_path": str(model_path_output),
            "source_model_path": str(model_path),
            "required_files": ["config.json", "quantization_config.json"],
            "analysis_artifacts": {
                "config": str(output_path / "quantization_config.json"),
                "manifest": str(output_path / "quantized_model_manifest.json"),
                "events": str(event_log),
            },
            "backend_version": _package_version("autoawq"),
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        (model_path_output / "quantized_model_manifest.json").write_text(
            manifest_text,
            encoding="utf-8",
        )
        (output_path / "quantized_model_manifest.json").write_text(
            manifest_text,
            encoding="utf-8",
        )
    except Exception as exc:
        _append_event(
            event_log,
            {
                **base_event,
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - started_clock,
            },
        )
        raise RuntimeError(f"AWQ plan {plan.plan_id} failed: {exc}") from exc
    _append_event(
        event_log,
        {
            **base_event,
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - started_clock,
        },
    )
    return {
        "plan": plan.to_dict(),
        "status": "pass",
        "output_dir": str(output_path),
        "model_output_dir": str(model_path_output),
        "backend_version": _package_version("autoawq"),
    }


def load_awq_service(
    baseline_config_file: str | Path,
    quantized_manifest_file: str | Path,
    source_model_manifest_file: str | Path,
    log_file: str | Path,
) -> BaselineService:
    """Load one exported AWQ model through the shared inference service."""
    _, baseline_config = _load_json(baseline_config_file)
    quantized_manifest_path, quantized_manifest = _load_json(
        quantized_manifest_file
    )
    _, source_manifest = _load_json(source_model_manifest_file)
    model_path_value = quantized_manifest.get("model_path")
    if not isinstance(model_path_value, str) or not model_path_value.strip():
        raise ValueError("Quantized manifest must define model_path.")
    model_path = Path(model_path_value).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"AWQ model directory does not exist: {model_path}")
    try:
        import torch
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install autoawq, torch, and transformers for AWQ evaluation.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    load_kwargs = _supported_kwargs(
        AutoAWQForCausalLM.from_quantized,
        {
            "safetensors": True,
            "device_map": "auto",
            "fuse_layers": False,
            "trust_remote_code": False,
        },
    )
    model = AutoAWQForCausalLM.from_quantized(str(model_path), **load_kwargs)
    model.eval()
    metadata = {
        "model_id": source_manifest.get("model_id"),
        "version": source_manifest.get("version"),
        "commit": source_manifest.get("commit"),
        "weight_files": quantized_manifest.get("required_files", []),
        "quantization": quantized_manifest.get("plan", {}),
        "quantized_manifest": str(quantized_manifest_path),
    }
    return BaselineService(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        config=baseline_config,
        model_metadata=metadata,
        log_file=log_file,
    )


def collect_awq_evidence(
    config_file: str | Path,
    calibration_manifest_file: str | Path,
    model_manifest_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Collect a reproducible matrix report without claiming execution."""
    config_path, config = _load_json(config_file)
    calibration_path, calibration = _load_json(calibration_manifest_file)
    model_path, model = _load_json(model_manifest_file)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    calibration_asset = calibration.get("assets", {}).get("calibration", {})
    plans = build_awq_plans(config)
    results = []
    for plan in plans:
        plan_dir = output_path / plan.plan_id
        manifest_file = plan_dir / "quantized_model_manifest.json"
        config_file_path = plan_dir / "quantization_config.json"
        event_file = plan_dir / "quantization_events.jsonl"
        artifact_status = "pass" if (
            manifest_file.is_file() and config_file_path.is_file() and event_file.is_file()
        ) else "not_run"
        results.append(
            {
                "plan": plan.to_dict(),
                "calibration": calibration_asset,
                "output_dir": str(plan_dir),
                "artifacts": {
                    "manifest": str(manifest_file),
                    "config": str(config_file_path),
                    "events": str(event_file),
                },
                "status": artifact_status,
            }
        )
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_file": str(config_path),
        "calibration_manifest": str(calibration_path),
        "model_manifest": str(model_path),
        "model_path": model.get("model_path"),
        "backend_package": "autoawq",
        "backend_version": _package_version("autoawq"),
        "plans": results,
        "status": "completed" if all(item["status"] == "pass" for item in results) else "not_run",
    }


def validate_awq_evidence(evidence_file: str | Path) -> AWQValidationReport:
    evidence_path, evidence = _load_json(evidence_file)

    def localize(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_file():
            return path
        parts = path.parts
        try:
            out_index = parts.index("out")
            if out_index + 1 < len(parts) and parts[out_index + 1] == "awq":
                relative = Path(*parts[out_index + 2 :])
                candidate = evidence_path.parent / relative
                if candidate.is_file():
                    return candidate
        except ValueError:
            pass
        return path

    plans = evidence.get("plans", [])
    groups = {item.get("plan", {}).get("group_size") for item in plans}
    matrix_ready = groups == set(REQUIRED_GROUP_SIZES)
    artifact_ready = all(
        item.get("status") == "pass"
        and all(
            localize(path).is_file()
            for path in item.get("artifacts", {}).values()
        )
        for item in plans
    )
    backend_ready = evidence.get("backend_version") is not None
    checks = [
        CheckResult(
            "awq_group_size_matrix",
            PASS if matrix_ready else FAIL,
            "AWQ plans cover group sizes 32, 64, and 128.",
        ),
        CheckResult(
            "awq_backend_recorded",
            PASS if backend_ready else NOT_RUN,
            "AutoAWQ version is recorded.",
        ),
        CheckResult(
            "awq_quantized_artifacts",
            PASS if artifact_ready else NOT_RUN,
            "Each plan has config, manifest, and JSONL event log.",
        ),
        CheckResult(
            "awq_real_execution",
            PASS if evidence.get("status") == "completed" else NOT_RUN,
            "All configured AWQ group sizes completed real export.",
        ),
    ]
    return AWQValidationReport(str(evidence_path), checks)
