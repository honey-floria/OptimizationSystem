"""GPTQ INT4 experiment runner and evidence validator."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.baseline.service import BaselineService
from src.input_validation.model_input import FAIL, NOT_RUN, PASS, CheckResult


REQUIRED_GROUP_SIZES = (32, 64, 128)


@dataclass(frozen=True)
class GPTQPlan:
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
class GPTQValidationReport:
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
        raise ValueError(f"Invalid GPTQ JSON {file_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"GPTQ JSON must contain an object: {file_path}")
    return file_path, payload


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_gptq_plans(config: dict[str, Any]) -> list[GPTQPlan]:
    group_sizes = tuple(int(value) for value in config.get("group_sizes", []))
    if group_sizes != REQUIRED_GROUP_SIZES:
        raise ValueError("GPTQ must use AWQ-comparable group sizes 32, 64, and 128.")
    if config.get("method") != "gptq" or int(config.get("bits", 0)) != 4:
        raise ValueError("6.3 requires a GPTQ INT4 configuration.")
    return [
        GPTQPlan(
            plan_id=f"{config['plan_id']}-g{group_size}",
            method="gptq",
            bits=4,
            weight_bits=int(config.get("weight_bits", 4)),
            activation_bits=int(config.get("activation_bits", 16)),
            group_size=group_size,
            quantization_granularity=str(
                config.get("quantization_granularity", "groupwise")
            ),
            backend_package=str(config.get("backend_package", "gptqmodel")),
        )
        for group_size in group_sizes
    ]


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
        raise FileNotFoundError(f"GPTQ model directory does not exist: {model_path}")
    return model_path


def run_gptq_plan(
    plan: GPTQPlan,
    model_manifest_file: str | Path,
    calibration_prompts: Iterable[str],
    output_dir: str | Path,
    model_output_dir: str | Path,
    max_calibration_samples: int = 1024,
    max_calibration_seq_len: int = 512,
) -> dict[str, Any]:
    """Quantize one real GPTQ matrix point and export its evidence."""
    output_path = Path(output_dir).expanduser().resolve()
    model_output_path = Path(model_output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_output_path.mkdir(parents=True, exist_ok=True)
    event_log = output_path / "quantization_events.jsonl"
    base_event = {
        "event": "quantization",
        "plan_id": plan.plan_id,
        "method": "gptq",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    except ImportError as exc:
        detail = traceback.format_exc()
        _append_event(event_log, {**base_event, "status": "not_run", "error": detail})
        return {"plan": plan.to_dict(), "status": "not_run", "detail": detail}
    if _package_version("gptqmodel") is None:
        detail = "gptqmodel is not installed."
        _append_event(event_log, {**base_event, "status": "not_run", "error": detail})
        return {"plan": plan.to_dict(), "status": "not_run", "detail": detail}
    model_path = _resolve_model_path(model_manifest_file)
    prompts = list(calibration_prompts)[:max_calibration_samples]
    if not prompts:
        raise ValueError("GPTQ requires at least one calibration prompt.")
    _append_event(event_log, {**base_event, "status": "started"})
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        quant_config = GPTQConfig(
            bits=plan.weight_bits,
            group_size=plan.group_size,
            dataset=prompts,
            tokenizer=tokenizer,
            model_seqlen=max_calibration_seq_len,
            desc_act=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
        )
        model.save_pretrained(str(model_output_path), safe_serialization=True)
        tokenizer.save_pretrained(str(model_output_path))
        config_payload = {
            "schema_version": 1,
            "plan": plan.to_dict(),
            "quant_config": {
                "bits": plan.weight_bits,
                "group_size": plan.group_size,
                "desc_act": False,
            },
            "calibration": {
                "size": len(prompts),
                "max_seq_len": max_calibration_seq_len,
            },
        }
        config_text = json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n"
        for directory in (output_path, model_output_path):
            (directory / "quantization_config.json").write_text(config_text, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "format": "gptqmodel_transformers",
            "plan": plan.to_dict(),
            "model_path": str(model_output_path),
            "source_model_path": str(model_path),
            "required_files": ["config.json", "quantization_config.json"],
            "backend_versions": {
                "gptqmodel": _package_version("gptqmodel"),
                "transformers": _package_version("transformers"),
            },
            "analysis_artifacts": {
                "config": str(output_path / "quantization_config.json"),
                "manifest": str(output_path / "quantized_model_manifest.json"),
                "events": str(event_log),
            },
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        for directory in (output_path, model_output_path):
            (directory / "quantized_model_manifest.json").write_text(manifest_text, encoding="utf-8")
    except Exception as exc:
        _append_event(event_log, {**base_event, "status": "failed", "error": repr(exc), "elapsed_seconds": time.perf_counter() - started})
        raise RuntimeError(f"GPTQ plan {plan.plan_id} failed: {exc}") from exc
    _append_event(event_log, {**base_event, "status": "completed", "elapsed_seconds": time.perf_counter() - started})
    return {
        "plan": plan.to_dict(),
        "status": "pass",
        "output_dir": str(output_path),
        "model_output_dir": str(model_output_path),
    }


def load_gptq_service(
    baseline_config_file: str | Path,
    quantized_manifest_file: str | Path,
    source_model_manifest_file: str | Path,
    log_file: str | Path,
) -> BaselineService:
    _, baseline_config = _load_json(baseline_config_file)
    manifest_path, manifest = _load_json(quantized_manifest_file)
    _, source_manifest = _load_json(source_model_manifest_file)
    model_path = Path(str(manifest.get("model_path", ""))).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"GPTQ model directory does not exist: {model_path}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install gptqmodel, torch, and transformers.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, device_map="auto"
    )
    model.eval()
    return BaselineService(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        config=baseline_config,
        model_metadata={
            "model_id": source_manifest.get("model_id"),
            "version": source_manifest.get("version"),
            "commit": source_manifest.get("commit"),
            "weight_files": manifest.get("required_files", []),
            "quantization": manifest.get("plan", {}),
            "quantized_manifest": str(manifest_path),
        },
        log_file=log_file,
    )


def collect_gptq_evidence(
    config_file: str | Path,
    calibration_manifest_file: str | Path,
    model_manifest_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_path, config = _load_json(config_file)
    calibration_path, calibration = _load_json(calibration_manifest_file)
    model_path, model = _load_json(model_manifest_file)
    output_path = Path(output_dir).expanduser().resolve()
    calibration_asset = calibration.get("assets", {}).get("calibration", {})
    plans = []
    for plan in build_gptq_plans(config):
        plan_dir = output_path / plan.plan_id
        artifacts = {
            "manifest": str(plan_dir / "quantized_model_manifest.json"),
            "config": str(plan_dir / "quantization_config.json"),
            "events": str(plan_dir / "quantization_events.jsonl"),
        }
        plans.append({
            "plan": plan.to_dict(),
            "calibration": calibration_asset,
            "output_dir": str(plan_dir),
            "artifacts": artifacts,
            "status": "pass" if all(Path(path).is_file() for path in artifacts.values()) else "not_run",
        })
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_file": str(config_path),
        "calibration_manifest": str(calibration_path),
        "model_manifest": str(model_path),
        "model_path": model.get("model_path"),
        "backend_versions": {
            "gptqmodel": _package_version("gptqmodel"),
            "transformers": _package_version("transformers"),
        },
        "plans": plans,
        "status": "completed" if all(item["status"] == "pass" for item in plans) else "not_run",
    }


def validate_gptq_evidence(evidence_file: str | Path) -> GPTQValidationReport:
    evidence_path, evidence = _load_json(evidence_file)

    def localize(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_file():
            return path
        parts = path.parts
        try:
            out_index = parts.index("out")
            if parts[out_index + 1] == "gptq":
                candidate = evidence_path.parent / Path(*parts[out_index + 2 :])
                if candidate.is_file():
                    return candidate
        except (ValueError, IndexError):
            pass
        return path

    plans = evidence.get("plans", [])
    groups = {item.get("plan", {}).get("group_size") for item in plans}
    artifacts_ready = all(
        item.get("status") == "pass"
        and all(localize(path).is_file() for path in item.get("artifacts", {}).values())
        for item in plans
    )
    checks = [
        CheckResult("gptq_awq_comparable_matrix", PASS if groups == set(REQUIRED_GROUP_SIZES) else FAIL, "GPTQ uses INT4 group sizes 32, 64, and 128."),
        CheckResult("gptq_backend_recorded", PASS if evidence.get("backend_versions", {}).get("gptqmodel") else NOT_RUN, "GPTQModel version is recorded."),
        CheckResult("gptq_quantized_artifacts", PASS if artifacts_ready else NOT_RUN, "Each GPTQ plan has config, manifest, and event log."),
        CheckResult("gptq_real_execution", PASS if evidence.get("status") == "completed" else NOT_RUN, "All GPTQ plans completed real export."),
    ]
    return GPTQValidationReport(str(evidence_path), checks)
