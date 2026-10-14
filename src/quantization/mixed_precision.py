"""TODO 7.4 real mixed-precision candidate, export, evidence, and gating helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import time
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.baseline.service import BaselineService


ALLOWED_PRECISIONS = {"int4", "int8", "bf16"}
REQUIRED_SUITES = {"full_dev", "high_risk", "long_context", "benchmark"}


@dataclass(frozen=True)
class MixedPrecisionCandidate:
    plan_id: str
    description: str
    default_linear_precision: str
    overrides: dict[str, str]
    special_modules: dict[str, str]
    runnable: bool
    role: str
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_candidates(config: dict[str, Any]) -> list[MixedPrecisionCandidate]:
    rows = config.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("mixed_precision_search.candidates must be a non-empty list")
    candidates = []
    for row in rows:
        candidate = MixedPrecisionCandidate(
            plan_id=str(row["plan_id"]),
            description=str(row["description"]),
            default_linear_precision=str(row["default_linear_precision"]),
            overrides={str(k): str(v) for k, v in row.get("overrides", {}).items()},
            special_modules={str(k): str(v) for k, v in row.get("special_modules", {}).items()},
            runnable=bool(row.get("runnable", False)),
            role=str(row.get("role", "candidate")),
            unsupported_reason=row.get("unsupported_reason"),
        )
        precisions = {candidate.default_linear_precision, *candidate.overrides.values(), *candidate.special_modules.values()}
        if not precisions <= ALLOWED_PRECISIONS:
            raise ValueError(f"Unsupported precision in {candidate.plan_id}: {sorted(precisions - ALLOWED_PRECISIONS)}")
        if not candidate.runnable and not candidate.unsupported_reason:
            raise ValueError(f"Non-runnable candidate needs unsupported_reason: {candidate.plan_id}")
        candidates.append(candidate)
    ids = [row.plan_id for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Mixed-precision plan IDs must be unique")
    required_roles = {"deployment_baseline", "quality_reference", "primary", "high_precision_protection_reference", "protection_count_ablation"}
    roles = {row.role for row in candidates}
    if not required_roles <= roles:
        raise ValueError(f"Candidate matrix is missing roles: {sorted(required_roles - roles)}")
    return candidates


def candidate_by_id(config: dict[str, Any], plan_id: str) -> MixedPrecisionCandidate:
    matches = [row for row in build_candidates(config) if row.plan_id == plan_id]
    if not matches:
        raise ValueError(f"Unknown mixed-precision plan: {plan_id}")
    return matches[0]


def precision_for_module(candidate: MixedPrecisionCandidate, module_fqn: str) -> str:
    """Resolve longest-prefix override, so exact/layer-specific rules beat defaults."""
    matches = [
        (prefix, precision)
        for prefix, precision in candidate.overrides.items()
        if module_fqn == prefix or module_fqn.startswith(prefix + ".")
    ]
    if not matches:
        return candidate.default_linear_precision
    return max(matches, key=lambda item: len(item[0]))[1]


def build_module_assignment(model: Any, candidate: MixedPrecisionCandidate, torch_module: Any) -> dict[str, str]:
    assignment = {
        name: precision_for_module(candidate, name)
        for name, module in model.named_modules()
        if isinstance(module, torch_module.nn.Linear)
    }
    if not assignment:
        raise ValueError("No torch.nn.Linear modules found in the model")
    missing = [prefix for prefix in candidate.overrides if not any(name == prefix or name.startswith(prefix + ".") for name in assignment)]
    if missing:
        raise ValueError(f"Candidate overrides matched no Linear modules: {missing}")
    return assignment


def _observed_weight_precision(weight: Any, torch_module: Any) -> str:
    """Best-effort structural audit across TorchAO tensor layouts/versions."""
    objects = [weight]
    for attribute in ("tensor_impl", "layout_tensor", "int_data"):
        value = getattr(weight, attribute, None)
        if value is not None:
            objects.append(value)
    type_names = " ".join(type(value).__name__.lower() for value in objects)
    if "int4" in type_names or "uint4" in type_names:
        return "int4"
    if "int8" in type_names or "unpackedtoint8" in type_names:
        return "int8"
    for value in objects:
        quant_min = getattr(value, "quant_min", None)
        quant_max = getattr(value, "quant_max", None)
        if isinstance(quant_min, int) and isinstance(quant_max, int):
            if quant_max - quant_min <= 15:
                return "int4"
            if quant_max - quant_min <= 255:
                return "int8"
    if getattr(weight, "dtype", None) == torch_module.bfloat16:
        return "bf16"
    if getattr(weight, "dtype", None) == torch_module.float16:
        return "fp16"
    return "unknown"


def prepare_search(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / config["paths"]["output_root"]
    candidates = build_candidates(config)
    suites = set(config.get("evidence_suites", []))
    if suites != REQUIRED_SUITES:
        raise ValueError(f"evidence_suites must be exactly {sorted(REQUIRED_SUITES)}")
    manifest = {
        "schema_version": 1,
        "stage": "7.4",
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": config["backend"],
        "source_model_manifest": str((root / config["paths"]["source_model_manifest"]).resolve()),
        "module_policy": str((root / config["paths"]["module_policy"]).resolve()),
        "combination_ranking": str((root / config["paths"]["combination_ranking"]).resolve()),
        "candidates": [row.to_dict() for row in candidates],
        "evidence_suites": sorted(suites),
        "quality_gates": config["quality_gates"],
        "performance_gates": config["performance_gates"],
    }
    manifest["candidate_contract_sha256"] = hashlib.sha256(
        json.dumps(manifest["candidates"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(output / "experiment_manifest.json", manifest)
    for candidate in candidates:
        _write_json(output / "plans" / candidate.plan_id / "plan.json", candidate.to_dict())
        if not candidate.runnable:
            _write_json(output / "plans" / candidate.plan_id / "capability.json", {
                "status": "unsupported", "reason": candidate.unsupported_reason,
                "deployment_claims_allowed": False,
            })
    return manifest


def _resolve_source_model(root: Path, config: dict[str, Any], model_path: str | Path | None) -> Path:
    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve()
    else:
        manifest_path = (root / config["paths"]["source_model_manifest"]).resolve()
        manifest = _read_json(manifest_path)
        # The frozen 7.1 manifest stores the merged directory under
        # merged_model.path.  Also accept model_path for compatibility with
        # standalone model manifests produced by other stages.
        manifest_model_path = manifest.get("model_path")
        if manifest_model_path is None and isinstance(manifest.get("merged_model"), dict):
            manifest_model_path = manifest["merged_model"].get("path")
        if not isinstance(manifest_model_path, str) or not manifest_model_path.strip():
            raise ValueError(
                f"Source model manifest has neither model_path nor merged_model.path: {manifest_path}. "
                "Pass --model-path explicitly or set MODEL_PATH when using the shell scripts."
            )
        resolved = Path(manifest_model_path).expanduser()
        if not resolved.is_absolute():
            resolved = (manifest_path.parent / resolved).resolve()
        else:
            resolved = resolved.resolve()
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"Frozen merged model is unavailable: {resolved}")
    return resolved


def export_candidate(
    project_root: str | Path,
    config: dict[str, Any],
    plan_id: str,
    *,
    model_path: str | Path | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    candidate = candidate_by_id(config, plan_id)
    if not candidate.runnable:
        raise ValueError(f"Candidate is not deployable: {plan_id}: {candidate.unsupported_reason}")
    root = Path(project_root).resolve()
    source = _resolve_source_model(root, config, model_path)
    try:
        import torch
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Export requires torch, torchao, and transformers") from exc

    output = root / config["paths"]["output_root"] / "exports" / plan_id
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if config["backend"]["compute_dtype"] == "bfloat16" else torch.float16
    # Keep the source model on CPU. Loading a 7B BF16 model on a 16 GiB GPU and
    # then packing a large projection needs both the original weight and a
    # temporary padded tensor, which exceeds the card. TorchAO's `device=` path
    # streams modules through the accelerator and keeps only quantized outputs.
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    assignment = build_module_assignment(model, candidate, torch)
    int4 = Int4WeightOnlyConfig(
        group_size=int(config["backend"]["int4_group_size"]),
        int4_packing_format=str(config["backend"].get("int4_packing_format", "plain")),
    )
    int8 = Int8WeightOnlyConfig()
    named = dict(model.named_modules())
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    quantized_count = 0
    for name, precision in sorted(assignment.items()):
        if precision == "bf16":
            continue
        # Calling quantize_ on the individual Linear (instead of the full model)
        # makes `device=` move only this layer before packing. This is genuinely
        # streaming and avoids materializing the full BF16 model on a 16 GiB GPU.
        quantize_(
            named[name],
            int4 if precision == "int4" else int8,
            device=torch.device(device),
        )
        quantized_count += 1
        if str(device).startswith("cuda") and quantized_count % 16 == 0:
            torch.cuda.empty_cache()
    # FQN entries marked BF16 and non-Linear modules are intentionally skipped
    # by quantize_; move those remaining parameters after packed weights exist.
    model.to(device)

    audit = []
    failures = []
    named = dict(model.named_modules())
    for name, expected in sorted(assignment.items()):
        weight = named[name].weight
        weight_class = f"{type(weight).__module__}.{type(weight).__name__}"
        quantized = type(weight) is not torch.nn.Parameter and type(weight) is not torch.Tensor
        observed = _observed_weight_precision(weight, torch)
        if expected in {"int4", "int8"} and (not quantized or observed != expected):
            failures.append(name)
        if expected == "bf16" and observed != "bf16":
            failures.append(name)
        audit.append({"module": name, "expected_precision": expected, "observed_precision": observed, "weight_class": weight_class, "quantized_tensor_subclass": quantized})
    if failures:
        raise RuntimeError(f"TorchAO did not replace expected weights: {failures[:8]}")

    model.config.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    weights_path = output / "mixed_precision_state_dict.pt"
    temporary_weights_path = output / "mixed_precision_state_dict.pt.partial"
    # Save directly from the accelerator. Moving a tile-packed INT4 tensor back
    # to CPU is backend-dependent; torch.load(map_location="cpu") handles the
    # portable relocation when the artifact is loaded.
    # A previous interrupted torch.save may leave a corrupt destination. Write
    # atomically and remove only these known, plan-local incomplete artifacts.
    for incomplete in (temporary_weights_path, weights_path):
        if incomplete.exists():
            incomplete.unlink()
    try:
        torch.save(model.state_dict(), temporary_weights_path)
        temporary_weights_path.replace(weights_path)
    except (OSError, RuntimeError) as exc:
        if temporary_weights_path.exists():
            temporary_weights_path.unlink()
        free_gib = shutil.disk_usage(output).free / (1024 ** 3)
        raise RuntimeError(
            "Failed to save the mixed-precision state_dict. "
            f"Filesystem free space reported by df is {free_gib:.2f} GiB. "
            "Check both `df -h` and the account quota (`quota -s`). "
            "The artifact can be redirected to a large filesystem by setting "
            "MIXED_PRECISION_OUTPUT_ROOT=/path/to/scratch/mixed_precision_search."
        ) from exc
    _write_json(output / "module_assignment.json", {"plan_id": plan_id, "modules": audit})
    manifest = {
        "schema_version": 1,
        "format": "torchao-mixed-weight-only",
        "status": "completed",
        "plan": candidate.to_dict(),
        "model_path": str(output),
        "source_model_path": str(source),
        "state_dict": str(weights_path),
        "state_dict_bytes": weights_path.stat().st_size,
        "state_dict_sha256": _sha256(weights_path),
        "module_assignment": str(output / "module_assignment.json"),
        "backend_versions": {"torch": _package_version("torch"), "torchao": _package_version("torchao"), "transformers": _package_version("transformers")},
        "backend_artifact_valid": True,
        "deployment_claims_allowed": False,
    }
    _write_json(output / "quantized_model_manifest.json", manifest)
    return manifest


def load_mixed_precision_service(
    baseline_config_file: str | Path,
    manifest_file: str | Path,
    model_manifest_file: str | Path,
    log_file: str | Path,
) -> BaselineService:
    try:
        import torch
        import torchao  # noqa: F401 - register tensor subclasses before torch.load
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Loading mixed precision requires torch, torchao, and transformers") from exc
    baseline = _read_json(baseline_config_file)
    manifest = _read_json(manifest_file)
    model_metadata = _read_json(model_manifest_file)
    artifact = Path(manifest["model_path"]).resolve()
    state_path = Path(manifest["state_dict"]).resolve()
    if _sha256(state_path) != manifest["state_dict_sha256"]:
        raise ValueError("Mixed-precision state_dict SHA256 mismatch")
    dtype = torch.bfloat16 if baseline["dtype"] == "bfloat16" else torch.float16
    architecture = AutoConfig.from_pretrained(str(artifact), local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(architecture, torch_dtype=dtype)
    state = torch.load(
        state_path,
        map_location=baseline["device"],
        weights_only=False,
    )
    model.load_state_dict(state, assign=True)
    model.to(baseline["device"]).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(artifact), local_files_only=True)
    metadata = {
        "model_id": model_metadata.get("model_id", manifest["plan"]["plan_id"]),
        "version": manifest["state_dict_sha256"][:12],
        "commit": manifest["state_dict_sha256"],
        "weight_files": [str(state_path)],
        "quantization": manifest["format"],
    }
    return BaselineService(model, tokenizer, torch, baseline, metadata, log_file)


def record_evidence(project_root: str | Path, config: dict[str, Any], plan_id: str, suite: str, source: str | Path) -> dict[str, Any]:
    candidate_by_id(config, plan_id)
    if suite not in REQUIRED_SUITES:
        raise ValueError(f"Unknown evidence suite: {suite}")
    source_path = Path(source).expanduser().resolve()
    payload = _read_json(source_path)
    destination = Path(project_root).resolve() / config["paths"]["output_root"] / "evidence" / plan_id / f"{suite}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    receipt = {"plan_id": plan_id, "suite": suite, "source": str(source_path), "evidence": str(destination), "sha256": _sha256(destination), "top_level_keys": sorted(payload)}
    _write_json(destination.with_suffix(".receipt.json"), receipt)
    return receipt


def run_quality_evaluation(
    project_root: str | Path,
    config: dict[str, Any],
    plan_id: str,
    suite: str,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Evaluate one exported artifact on the frozen full/risk/long-context suite."""
    if suite not in {"full_dev", "high_risk", "long_context"}:
        raise ValueError(f"Not a quality suite: {suite}")
    from src.data.finqa_assets import load_assets
    from src.evaluation.quality_repair import run_quality_repair_experiment

    root = Path(project_root).resolve()
    export = root / config["paths"]["output_root"] / "exports" / plan_id
    quantized_manifest = export / "quantized_model_manifest.json"
    if not quantized_manifest.is_file():
        raise ValueError(f"Export is missing for {plan_id}: {quantized_manifest}")
    output = root / config["paths"]["output_root"] / "runs" / plan_id / suite
    output.mkdir(parents=True, exist_ok=True)
    baseline = _read_json(root / config["paths"]["baseline_config"])
    baseline["device"] = device
    baseline["dtype"] = config["backend"]["compute_dtype"]
    baseline_path = output / "baseline_config.effective.json"
    _write_json(baseline_path, baseline)
    quality = copy.deepcopy(_read_json(root / config["paths"]["quality_config"]))
    quality.pop("evaluation_size", None)
    quality.pop("evaluation_seed", None)
    quality["batch_size"] = 1
    quality["pilot_variants"] = ["structured_json"]
    quality["show_progress"] = True
    assets = load_assets(root / config["paths"]["dataset_dir"])
    if suite == "full_dev":
        dataset = assets.quality_dev
        dataset_role = "evaluation_dev"
    elif suite == "high_risk":
        dataset = assets.high_risk_regression
        dataset_role = "high_risk_regression"
    else:
        frozen = _read_json(root / config["paths"]["frozen_subsets"])["long_context"]
        dataset = []
        for entry in frozen:
            row = assets.quality_dev[int(entry["index"])]
            question = row.get("question") or row.get("qa", {}).get("question")
            if str(question) != str(entry["question"]):
                raise ValueError("Frozen long-context subset no longer matches evaluation_dev")
            dataset.append(row)
        dataset_role = "long_context"
    quality["dataset_role"] = dataset_role
    quality_path = output / "quality_config.effective.json"
    _write_json(quality_path, quality)
    started = time.perf_counter()
    service = load_mixed_precision_service(
        baseline_path,
        quantized_manifest,
        root / config["paths"]["source_model_manifest"],
        output / "quality_service.jsonl",
    )
    report = run_quality_repair_experiment(
        service, dataset, assets.manifest, quality, output, calibration=assets.calibration
    )
    variant = next(row for row in report["variants"] if row["variant"] == "structured_json")
    result = {
        "schema_version": 1,
        "stage": "7.4",
        "status": "completed",
        "plan_id": plan_id,
        "suite": suite,
        "metrics": variant["metrics"],
        "structured_output_rate": variant.get("structured_output_rate"),
        "format_retry_count": variant.get("format_retry_count"),
        "predictions_sha256": variant.get("predictions_sha256"),
        "duration_seconds": time.perf_counter() - started,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    record_evidence(root, config, plan_id, suite, result_path)
    return result


def run_performance_benchmark(
    project_root: str | Path,
    config: dict[str, Any],
    plan_id: str,
    prompt_file: str | Path,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    from src.benchmark.runner import run_benchmark, validate_benchmark_report

    root = Path(project_root).resolve()
    export = root / config["paths"]["output_root"] / "exports" / plan_id
    manifest = export / "quantized_model_manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Export is missing for {plan_id}: {manifest}")
    output = root / config["paths"]["output_root"] / "runs" / plan_id / "benchmark"
    output.mkdir(parents=True, exist_ok=True)
    baseline = _read_json(root / config["paths"]["baseline_config"])
    baseline["device"] = device
    baseline["dtype"] = config["backend"]["compute_dtype"]
    baseline_path = output / "baseline_config.effective.json"
    _write_json(baseline_path, baseline)
    service = load_mixed_precision_service(
        baseline_path,
        manifest,
        root / config["paths"]["source_model_manifest"],
        output / "benchmark_service.jsonl",
    )
    prompt = Path(prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    report = run_benchmark(service, prompt, _read_json(root / config["paths"]["benchmark_config"]))
    report_path = output / "benchmark.json"
    _write_json(report_path, report)
    validation = validate_benchmark_report(report_path).to_dict()
    _write_json(output / "validation.json", validation)
    if not validation["complete"]:
        raise ValueError(f"Benchmark validation failed for {plan_id}")
    record_evidence(root, config, plan_id, "benchmark", report_path)
    return report


def pending_evidence(project_root: str | Path, config: dict[str, Any]) -> dict[str, list[str]]:
    output = Path(project_root).resolve() / config["paths"]["output_root"] / "evidence"
    candidates = build_candidates(config)
    candidate_by_plan = {candidate.plan_id: candidate for candidate in candidates}
    active_ids = config.get("evaluation_plan_ids")
    if active_ids is None:
        active = [candidate for candidate in candidates if candidate.runnable]
    else:
        unknown = sorted(set(active_ids) - set(candidate_by_plan))
        if unknown:
            raise ValueError(f"Unknown evaluation_plan_ids: {unknown}")
        active = [candidate_by_plan[plan_id] for plan_id in active_ids]
        non_runnable = [candidate.plan_id for candidate in active if not candidate.runnable]
        if non_runnable:
            raise ValueError(f"Non-runnable evaluation plans: {non_runnable}")
    return {
        candidate.plan_id: sorted(
            suite for suite in REQUIRED_SUITES
            if not (output / candidate.plan_id / f"{suite}.json").is_file()
            or not (output / candidate.plan_id / f"{suite}.receipt.json").is_file()
        )
        for candidate in active
    }


def _load_frozen_evidence(evidence_root: Path, plan_id: str, suite: str) -> dict[str, Any]:
    evidence = evidence_root / plan_id / f"{suite}.json"
    receipt = _read_json(evidence.with_suffix(".receipt.json"))
    if receipt.get("plan_id") != plan_id or receipt.get("suite") != suite:
        raise ValueError(f"Evidence receipt identity mismatch: {plan_id}/{suite}")
    if _sha256(evidence) != receipt.get("sha256"):
        raise ValueError(f"Evidence changed after registration: {plan_id}/{suite}")
    return _read_json(evidence)


def _quality_summary(payload: dict[str, Any]) -> dict[str, float | int]:
    if "metrics" in payload:
        metrics = payload["metrics"]
        transitions = payload.get("transitions_vs_all_int4", {})
    else:
        variants = payload.get("variants", [])
        variant = next((row for row in variants if row.get("variant") == "structured_json"), variants[0] if variants else {})
        metrics = variant.get("metrics", {})
        transitions = payload.get("transitions_vs_all_int4", variant.get("transitions_vs_all_int4", {}))
    return {
        "parse_rate": float(metrics.get("parse_rate", 0)),
        "correct": int(metrics.get("correct", 0)),
        "total": int(metrics.get("total", 0)),
        "net_correct": int(transitions.get("net_correct_gain", transitions.get("net_correct_change", 0))),
    }


def _benchmark_summary(payload: dict[str, Any]) -> dict[str, float]:
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("Benchmark evidence has no cases")
    return {
        "peak_memory_mib": max(float(row["memory"]["peak_allocated_mib"]) for row in cases),
        "p95_latency_ms": max(float(row["latency_ms"]["end_to_end"]["p95"]) for row in cases),
        "throughput": sum(float(row["throughput"]["mean_output_tokens_per_second"]) for row in cases) / len(cases),
    }


def finalize_search(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / config["paths"]["output_root"]
    pending = pending_evidence(root, config)
    incomplete = {key: value for key, value in pending.items() if value}
    if incomplete:
        raise ValueError(f"Mixed-precision evidence is incomplete: {incomplete}")
    prepared = _read_json(output / "experiment_manifest.json")
    current_contract = hashlib.sha256(
        json.dumps([row.to_dict() for row in build_candidates(config)], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if prepared.get("candidate_contract_sha256") != current_contract:
        raise ValueError("Candidate matrix changed after prepare; prepare a new 7.4 contract")
    evidence_root = output / "evidence"
    active_ids = set(pending)
    baseline_id = config["performance_gates"]["baseline_plan_id"]
    if baseline_id not in active_ids:
        raise ValueError(f"Performance baseline is not active: {baseline_id}")
    baseline_benchmark = _benchmark_summary(_load_frozen_evidence(evidence_root, baseline_id, "benchmark"))
    baseline_quality = {
        suite: _quality_summary(_load_frozen_evidence(evidence_root, baseline_id, suite))
        for suite in ("full_dev", "high_risk", "long_context")
    }
    rows = []
    for candidate in build_candidates(config):
        if not candidate.runnable:
            rows.append({"plan_id": candidate.plan_id, "status": "unsupported", "reason": candidate.unsupported_reason, "selected": False})
            continue
        if candidate.plan_id not in active_ids:
            rows.append({"plan_id": candidate.plan_id, "status": "not_evaluated", "reason": "outside evaluation_plan_ids", "selected": False})
            continue
        quality = {suite: _quality_summary(_load_frozen_evidence(evidence_root, candidate.plan_id, suite)) for suite in ("full_dev", "high_risk", "long_context")}
        for suite, summary in quality.items():
            summary["net_correct"] = int(summary["correct"]) - int(baseline_quality[suite]["correct"])
        benchmark = _benchmark_summary(_load_frozen_evidence(evidence_root, candidate.plan_id, "benchmark"))
        perf = {
            "peak_memory_ratio": benchmark["peak_memory_mib"] / baseline_benchmark["peak_memory_mib"],
            "p95_latency_ratio": benchmark["p95_latency_ms"] / baseline_benchmark["p95_latency_ms"],
            "throughput_ratio": benchmark["throughput"] / baseline_benchmark["throughput"],
        }
        qg, pg = config["quality_gates"], config["performance_gates"]
        quality_pass = quality["full_dev"]["parse_rate"] >= qg["min_parse_rate"] and quality["full_dev"]["net_correct"] >= qg["min_full_dev_net_correct"] and quality["high_risk"]["net_correct"] >= qg["min_high_risk_net_correct"] and quality["long_context"]["net_correct"] >= qg["min_long_context_net_correct"]
        performance_pass = perf["peak_memory_ratio"] <= pg["max_peak_memory_ratio"] and perf["p95_latency_ratio"] <= pg["max_p95_latency_ratio"] and perf["throughput_ratio"] >= pg["min_throughput_ratio"]
        rows.append({"plan_id": candidate.plan_id, "status": "completed", "role": candidate.role, "quality": quality, "benchmark": benchmark, "performance_vs_all_int4": perf, "quality_pass": quality_pass, "performance_pass": performance_pass, "selected": quality_pass and performance_pass})
    eligible = [row for row in rows if row.get("selected")]
    eligible.sort(key=lambda row: (-row["quality"]["full_dev"]["net_correct"], row["benchmark"]["peak_memory_mib"], row["benchmark"]["p95_latency_ms"]))
    for rank, row in enumerate(eligible, start=1):
        row["eligible_rank"] = rank
    result = {"schema_version": 1, "stage": "7.4", "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "candidates": rows, "eligible_ranking": eligible, "recommendation": eligible[0]["plan_id"] if eligible else None, "deployment_claims_allowed": bool(eligible)}
    _write_json(output / "ranking.json", result)
    report_lines = [
        "# TODO 7.4 mixed-precision search",
        "",
        f"Recommendation: `{result['recommendation']}`" if result["recommendation"] else "Recommendation: none (no candidate passed both gates)",
        "",
        "| Plan | Full-dev net | High-risk net | Long-context net | Peak memory ratio | P95 latency ratio | Throughput ratio | Selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["status"] != "completed":
            report_lines.append(f"| `{row['plan_id']}` | — | — | — | — | — | — | no (`{row['status']}`) |")
            continue
        report_lines.append(
            f"| `{row['plan_id']}` | {row['quality']['full_dev']['net_correct']} | "
            f"{row['quality']['high_risk']['net_correct']} | {row['quality']['long_context']['net_correct']} | "
            f"{row['performance_vs_all_int4']['peak_memory_ratio']:.3f} | "
            f"{row['performance_vs_all_int4']['p95_latency_ratio']:.3f} | "
            f"{row['performance_vs_all_int4']['throughput_ratio']:.3f} | "
            f"{'yes' if row['selected'] else 'no'} |"
        )
    _write_text(output / "report.md", "\n".join(report_lines))
    _write_json(output / "validation.json", {
        "complete": True,
        "candidate_contract_sha256": current_contract,
        "evidence_complete": True,
        "eligible_candidate_count": len(eligible),
        "deployment_claims_allowed": bool(eligible),
    })
    return result
