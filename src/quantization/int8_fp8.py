"""SmoothQuant INT8 execution, evaluation, and FP8 capability evidence."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.baseline.service import BaselineService
from src.evaluation.finqa_metrics import evaluate_numeric_answers
from src.input_validation.model_input import FAIL, NOT_RUN, PASS, CheckResult


@dataclass(frozen=True)
class SmoothQuantPlan:
    plan_id: str
    method: str
    weight_bits: int
    activation_bits: int
    alpha: float
    quantization_granularity: str
    backend_package: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _append_event(log_file: Path, event: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _resolve_model_path(model_manifest_file: str | Path) -> Path:
    manifest_path, manifest = _load_json(model_manifest_file)
    model_value = manifest.get("model_path")
    if not isinstance(model_value, str) or not model_value.strip():
        raise ValueError("Model manifest must define model_path.")
    model_path = Path(model_value).expanduser()
    if not model_path.is_absolute():
        model_path = manifest_path.parent / model_path
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"SmoothQuant model directory does not exist: {model_path}")
    return model_path


def build_smoothquant_plan(config: dict[str, Any]) -> SmoothQuantPlan:
    if config.get("method") != "smoothquant":
        raise ValueError("6.4 requires method=smoothquant.")
    if int(config.get("weight_bits", 0)) != 8 or int(config.get("activation_bits", 0)) != 8:
        raise ValueError("6.4 requires a W8A8 configuration.")
    alpha = float(config.get("alpha", -1))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be between 0 and 1.")
    return SmoothQuantPlan(
        plan_id=str(config["plan_id"]),
        method="smoothquant",
        weight_bits=8,
        activation_bits=8,
        alpha=alpha,
        quantization_granularity=str(config["quantization_granularity"]),
        backend_package=str(config.get("backend_package", "nvidia-modelopt")),
    )


def _probe_fp8_matmul(torch_module: Any, device: str) -> dict[str, Any]:
    dtype_names = [
        name for name in ("float8_e4m3fn", "float8_e5m2")
        if hasattr(torch_module, name)
    ]
    result = {"available": False, "dtypes": dtype_names, "detail": None}
    if not dtype_names:
        result["detail"] = "PyTorch exposes no FP8 dtype."
        return result
    try:
        dtype = getattr(torch_module, dtype_names[0])
        left = torch_module.randn((16, 16), device=device, dtype=torch_module.float16).to(dtype)
        right = torch_module.randn((16, 16), device=device, dtype=torch_module.float16).to(dtype)
        torch_module.mm(left, right)
        result["available"] = True
        result["detail"] = "A native FP8 matrix multiplication completed."
    except Exception as exc:
        result["detail"] = f"FP8 matrix multiplication failed: {type(exc).__name__}: {exc}"
    return result


def _torch_capabilities() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch": None,
            "cuda": False,
            "nvidia_fp8": False,
            "ascend_npu": False,
            "ascend_fp8": False,
        }
    cuda = bool(torch.cuda.is_available())
    cuda_probe = _probe_fp8_matmul(torch, "cuda:0") if cuda else {
        "available": False,
        "dtypes": [],
        "detail": "CUDA is unavailable.",
    }
    npu = False
    npu_name = None
    try:
        import torch_npu  # type: ignore  # noqa: F401

        npu = bool(getattr(torch, "npu", None) and torch.npu.is_available())
        if npu and hasattr(torch.npu, "get_device_name"):
            npu_name = torch.npu.get_device_name(0)
    except ImportError:
        pass
    npu_probe = _probe_fp8_matmul(torch, "npu:0") if npu else {
        "available": False,
        "dtypes": [],
        "detail": "Ascend NPU is unavailable.",
    }
    capability = list(torch.cuda.get_device_capability(0)) if cuda else None
    return {
        "torch": getattr(torch, "__version__", None),
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_compute_capability": capability,
        "nvidia_fp8": cuda_probe["available"],
        "nvidia_fp8_probe": cuda_probe,
        "ascend_npu": npu,
        "ascend_device": npu_name,
        "ascend_fp8": npu_probe["available"],
        "ascend_fp8_probe": npu_probe,
    }


def run_smoothquant_plan(
    plan: SmoothQuantPlan,
    model_manifest_file: str | Path,
    calibration_prompts: Iterable[str],
    output_dir: str | Path,
    model_output_dir: str | Path,
    max_calibration_samples: int = 1024,
    max_calibration_seq_len: int = 512,
    calibration_batch_size: int = 4,
) -> dict[str, Any]:
    """Run NVIDIA ModelOpt SmoothQuant and save a restorable W8A8 state."""
    output_path = Path(output_dir).expanduser().resolve()
    model_output_path = Path(model_output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_output_path.mkdir(parents=True, exist_ok=True)
    event_log = output_path / "quantization_events.jsonl"
    base_event = {
        "event": "quantization",
        "plan_id": plan.plan_id,
        "method": plan.method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        import modelopt.torch.opt as modelopt
        import modelopt.torch.quantization as mtq
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        detail = f"Install nvidia-modelopt[torch], torch, and transformers: {exc}"
        _append_event(event_log, {**base_event, "status": "not_run", "error": detail})
        return {"plan": plan.to_dict(), "status": "not_run", "detail": detail}
    if not torch.cuda.is_available():
        detail = "SmoothQuant export requires an NVIDIA CUDA device."
        _append_event(event_log, {**base_event, "status": "not_run", "error": detail})
        return {"plan": plan.to_dict(), "status": "not_run", "detail": detail}

    prompts = list(calibration_prompts)[:max_calibration_samples]
    if not prompts:
        raise ValueError("SmoothQuant requires at least one calibration prompt.")
    model_path = _resolve_model_path(model_manifest_file)
    _append_event(event_log, {**base_event, "status": "started"})
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        def forward_loop(active_model: Any) -> None:
            for start in range(0, len(prompts), calibration_batch_size):
                batch = prompts[start:start + calibration_batch_size]
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_calibration_seq_len,
                )
                encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
                with torch.no_grad():
                    active_model(**encoded)

        quant_config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
        quant_config["algorithm"] = {"method": "smoothquant", "alpha": plan.alpha}
        mtq.quantize(model, quant_config, forward_loop=forward_loop)
        state_file = model_output_path / "modelopt_state.pth"
        modelopt.save(model, state_file)
        tokenizer.save_pretrained(str(model_output_path))
        config_payload = {
            "schema_version": 1,
            "plan": plan.to_dict(),
            "calibration": {
                "size": len(prompts),
                "max_seq_len": max_calibration_seq_len,
                "batch_size": calibration_batch_size,
            },
            "backend_versions": {
                "nvidia-modelopt": _version("nvidia-modelopt"),
                "torch": getattr(torch, "__version__", None),
                "transformers": _version("transformers"),
            },
        }
        config_text = json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n"
        for directory in (output_path, model_output_path):
            (directory / "quantization_config.json").write_text(config_text, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "format": "nvidia-modelopt-smoothquant",
            "plan": plan.to_dict(),
            "model_path": str(model_output_path),
            "source_model_path": str(model_path),
            "modelopt_state": str(state_file),
            "required_files": ["modelopt_state.pth", "quantization_config.json"],
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
        _append_event(event_log, {
            **base_event,
            "status": "failed",
            "error": repr(exc),
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise RuntimeError(f"SmoothQuant plan {plan.plan_id} failed: {exc}") from exc
    _append_event(event_log, {
        **base_event,
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
    })
    return {
        "plan": plan.to_dict(),
        "status": "pass",
        "output_dir": str(output_path),
        "model_output_dir": str(model_output_path),
    }


def load_smoothquant_service(
    baseline_config_file: str | Path,
    quantized_manifest_file: str | Path,
    source_model_manifest_file: str | Path,
    log_file: str | Path,
) -> BaselineService:
    """Restore a ModelOpt SmoothQuant checkpoint through the shared service."""
    _, baseline_config = _load_json(baseline_config_file)
    manifest_path, manifest = _load_json(quantized_manifest_file)
    _, source_manifest = _load_json(source_model_manifest_file)
    model_path = _resolve_model_path(source_model_manifest_file)
    state_file = Path(str(manifest.get("modelopt_state", ""))).expanduser()
    if not state_file.is_file():
        candidate = manifest_path.parent / "modelopt_state.pth"
        if candidate.is_file():
            state_file = candidate
        else:
            raise FileNotFoundError(f"ModelOpt state does not exist: {state_file}")
    try:
        import torch
        import modelopt.torch.opt as modelopt
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install nvidia-modelopt[torch], torch, and transformers.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float16, device_map="auto"
    )
    model = modelopt.restore(model, state_file)
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
            "quantization": manifest.get("plan", {}),
            "quantized_manifest": str(manifest_path),
        },
        log_file=log_file,
    )


def collect_int8_fp8_evidence(
    config_file: str | Path,
    calibration_manifest_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_path, config = _load_json(config_file)
    calibration_path, calibration = _load_json(calibration_manifest_file)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    plan = build_smoothquant_plan(config)
    plan_dir = output_path / plan.plan_id
    artifacts = {
        "manifest": str(plan_dir / "quantized_model_manifest.json"),
        "config": str(plan_dir / "quantization_config.json"),
        "events": str(plan_dir / "quantization_events.jsonl"),
    }
    artifact_status = "pass" if all(Path(path).is_file() for path in artifacts.values()) else "not_run"
    capabilities = _torch_capabilities()
    return {
        "schema_version": 2,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_file": str(config_path),
        "calibration_manifest": str(calibration_path),
        "calibration_asset": calibration.get("assets", {}).get("calibration"),
        "plans": [{
            **plan.to_dict(),
            "output_dir": str(plan_dir),
            "artifacts": artifacts,
            "status": artifact_status,
            "export_contract": {
                "config_file": "quantization_config.json",
                "manifest_file": "quantized_model_manifest.json",
                "event_log_file": "quantization_events.jsonl",
            },
        }],
        "backend_versions": {
            "nvidia-modelopt": _version("nvidia-modelopt"),
            "torch": capabilities["torch"],
            "transformers": _version("transformers"),
        },
        "hardware_capabilities": capabilities,
        "status": "completed" if artifact_status == "pass" else "not_run",
    }


def _localize_artifact(evidence_path: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_file():
        return path
    parts = path.parts
    try:
        out_index = parts.index("out")
        if parts[out_index + 1] == "int8_fp8":
            candidate = evidence_path.parent / Path(*parts[out_index + 2:])
            if candidate.is_file():
                return candidate
    except (ValueError, IndexError):
        pass
    return path


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
    artifacts_ready = plan.get("status") == "pass" and all(
        _localize_artifact(evidence_path, path).is_file()
        for path in plan.get("artifacts", {}).values()
    )
    schema_version = int(evidence.get("schema_version", 0))
    nvidia_fp8_verified = (
        schema_version >= 2
        and capabilities.get("nvidia_fp8") is True
        and capabilities.get("nvidia_fp8_probe", {}).get("available") is True
    )
    ascend_fp8_verified = (
        schema_version >= 2
        and capabilities.get("ascend_fp8") is True
        and capabilities.get("ascend_fp8_probe", {}).get("available") is True
    )
    checks = [
        CheckResult("smoothquant_int8_config", PASS if config_ready else FAIL, "SmoothQuant W8A8 configuration and calibration are recorded."),
        CheckResult("smoothquant_export_contract", PASS if export_ready else FAIL, "INT8 export and event-log names are fixed."),
        CheckResult("smoothquant_backend_recorded", PASS if evidence.get("backend_versions", {}).get("nvidia-modelopt") else NOT_RUN, "NVIDIA ModelOpt version is recorded."),
        CheckResult("smoothquant_quantized_artifacts", PASS if artifacts_ready else NOT_RUN, "SmoothQuant config, manifest, and event log exist."),
        CheckResult("nvidia_fp8_capability", PASS if nvidia_fp8_verified else NOT_RUN, "A real FP8 matrix multiplication completed on NVIDIA CUDA."),
        CheckResult("ascend_fp8_capability", PASS if ascend_fp8_verified else NOT_RUN, "A real FP8 matrix multiplication completed on Ascend NPU."),
        CheckResult("smoothquant_real_execution", PASS if evidence.get("status") == "completed" and artifacts_ready else NOT_RUN, "SmoothQuant INT8 weights were exported."),
    ]
    return Int8Fp8ValidationReport(str(evidence_path), checks)


def evaluate_int8_outputs(
    predictions: list[str],
    references: list[str],
    long_context_results: list[dict[str, Any]],
    structured_results: list[dict[str, Any]],
    tolerance: float = 1e-4,
    baseline_numeric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute and compare the three quality views required by TODO 6.4."""
    numeric = evaluate_numeric_answers(predictions, references, tolerance=tolerance)
    long_context_passed = sum(bool(item.get("passed")) for item in long_context_results)
    structured_passed = sum(bool(item.get("passed")) for item in structured_results)
    comparison = None
    if baseline_numeric:
        comparison = {
            "baseline_numeric_accuracy": baseline_numeric.get("numeric_accuracy"),
            "int8_numeric_accuracy": numeric.get("numeric_accuracy"),
            "absolute_change": numeric.get("numeric_accuracy", 0.0) - baseline_numeric.get("numeric_accuracy", 0.0),
        }
    return {
        "schema_version": 2,
        "status": "completed",
        "numeric": numeric,
        "numeric_comparison": comparison,
        "long_context": {
            "total": len(long_context_results),
            "passed": long_context_passed,
            "pass_rate": long_context_passed / len(long_context_results) if long_context_results else 0.0,
            "cases": long_context_results,
        },
        "structured_output": {
            "total": len(structured_results),
            "passed": structured_passed,
            "pass_rate": structured_passed / len(structured_results) if structured_results else 0.0,
            "cases": structured_results,
        },
    }


def validate_int8_evaluation(evaluation_file: str | Path) -> Int8Fp8ValidationReport:
    evaluation_path, evaluation = _load_json(evaluation_file)
    numeric = evaluation.get("numeric", {})
    comparison = evaluation.get("numeric_comparison")
    long_context = evaluation.get("long_context", {})
    structured = evaluation.get("structured_output", {})
    long_cases = long_context.get("cases", [])
    structured_cases = structured.get("cases", [])
    checks = [
        CheckResult("int8_numeric_accuracy_compared", PASS if numeric.get("total", 0) > 0 and comparison else NOT_RUN, "INT8 numeric accuracy is compared with FP16."),
        CheckResult("int8_long_context_compared", PASS if long_cases and all("baseline_passed" in case for case in long_cases) else NOT_RUN, "INT8 long-context cases are compared with FP16."),
        CheckResult("int8_structured_output_compared", PASS if structured_cases and all("baseline_passed" in case for case in structured_cases) else NOT_RUN, "INT8 structured-output cases are compared with FP16."),
    ]
    return Int8Fp8ValidationReport(str(evaluation_path), checks)


def check_structured_json(text: str, required_fields: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"passed": False, "valid_json": False, "missing_fields": required_fields}
    if not isinstance(payload, dict):
        return {"passed": False, "valid_json": True, "missing_fields": required_fields}
    missing = [field for field in required_fields if field not in payload]
    return {"passed": not missing, "valid_json": True, "missing_fields": missing}
