"""Build and run the TODO 7.1.2 layer-sensitivity experiment matrix."""

from __future__ import annotations

import copy
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from src.quantization.layer_sensitivity_setup import (
    canonical_json_sha256,
    discover_transformer_blocks,
    load_json,
    resolve_project_path,
    row_fingerprint,
    sha256_file,
    write_json,
)


REFERENCE_KINDS = {"ref_fp", "ref_nf4"}
RUNNABLE_KINDS = {"all_int4", "all_int8", "layer_int8"}
ERROR_METRICS_SCHEMA_VERSION = 2
ERROR_METRICS_CHUNK_SIZE = 1_048_576
ACTIVATION_METRICS_SCHEMA_VERSION = 1


def build_plan_matrix(block_count: int) -> list[dict[str, Any]]:
    if block_count <= 0:
        raise ValueError("block_count must be positive.")
    plans: list[dict[str, Any]] = [
        {
            "plan_id": "ref_fp",
            "kind": "ref_fp",
            "runnable": False,
            "description": "Merged FP16 mathematical reference.",
        },
        {
            "plan_id": "ref_nf4",
            "kind": "ref_nf4",
            "runnable": False,
            "description": "Accepted QLoRA/NF4 business reference.",
        },
        {
            "plan_id": "all_int4",
            "kind": "all_int4",
            "runnable": True,
            "default_precision": "int4",
            "overrides": {},
        },
        {
            "plan_id": "all_int8",
            "kind": "all_int8",
            "runnable": True,
            "default_precision": "int8",
            "overrides": {},
        },
    ]
    for layer_index in range(block_count):
        plans.append(
            {
                "plan_id": f"layer_{layer_index:02d}_int8",
                "kind": "layer_int8",
                "runnable": True,
                "layer_index": layer_index,
                "default_precision": "int4",
                "overrides": {str(layer_index): "int8"},
            }
        )
    validate_plan_matrix(plans, block_count)
    return plans


def validate_plan_matrix(plans: list[dict[str, Any]], block_count: int) -> None:
    ids = [str(plan.get("plan_id")) for plan in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("Layer-sensitivity plan IDs must be unique.")
    kinds = [plan.get("kind") for plan in plans]
    for required in ("ref_fp", "ref_nf4", "all_int4", "all_int8"):
        if kinds.count(required) != 1:
            raise ValueError(f"Matrix must contain exactly one {required} plan.")
    layer_plans = [plan for plan in plans if plan.get("kind") == "layer_int8"]
    indices = sorted(plan.get("layer_index") for plan in layer_plans)
    if indices != list(range(block_count)):
        raise ValueError("Layer INT8 plans must cover every transformer block exactly once.")
    for plan in layer_plans:
        index = int(plan["layer_index"])
        if plan.get("default_precision") != "int4":
            raise ValueError("Layer plans must default all blocks to INT4.")
        if plan.get("overrides") != {str(index): "int8"}:
            raise ValueError("Layer plans may override exactly one block to INT8.")


def precision_for_layer(plan: dict[str, Any], layer_index: int) -> str | None:
    if plan.get("kind") in REFERENCE_KINDS:
        return None
    return plan.get("overrides", {}).get(
        str(layer_index), plan.get("default_precision")
    )


def plan_precision_vector(
    plan: dict[str, Any], block_count: int
) -> list[str | None]:
    return [precision_for_layer(plan, index) for index in range(block_count)]


def validate_plan_isolation(plans: list[dict[str, Any]], block_count: int) -> None:
    all_int4 = next(plan for plan in plans if plan["kind"] == "all_int4")
    baseline = plan_precision_vector(all_int4, block_count)
    for plan in plans:
        if plan.get("kind") != "layer_int8":
            continue
        vector = plan_precision_vector(plan, block_count)
        changed = [index for index, pair in enumerate(zip(baseline, vector)) if pair[0] != pair[1]]
        if changed != [plan["layer_index"]] or vector[changed[0]] != "int8":
            raise ValueError(f"Plan {plan['plan_id']} changes more than its target layer.")


def aggregate_error_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = sum(int(row["numel"]) for row in rows)
    if total <= 0:
        raise ValueError("Quantization error aggregation requires positive numel.")
    weighted = {}
    for field in ("mse", "nmse", "cosine_similarity", "max_abs_error", "saturation_rate"):
        if field == "max_abs_error":
            weighted[field] = max(float(row[field]) for row in rows)
        else:
            weighted[field] = sum(
                float(row[field]) * int(row["numel"]) for row in rows
            ) / total
    return {"module_count": len(rows), "numel": total, **weighted}


def _stable_error_statistics(
    source: Any,
    measured: Any,
    *,
    torch_module: Any,
    chunk_size: int = ERROR_METRICS_CHUNK_SIZE,
) -> dict[str, float]:
    """Accumulate large-vector error statistics in bounded FP64 chunks."""

    if source.shape != measured.shape:
        raise ValueError("Error statistics require tensors with identical shapes.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    source_flat = source.reshape(-1)
    measured_flat = measured.reshape(-1)
    dot = 0.0
    source_square_sum = 0.0
    measured_square_sum = 0.0
    error_square_sum = 0.0
    max_abs_error = 0.0
    for start in range(0, source_flat.numel(), chunk_size):
        stop = min(start + chunk_size, source_flat.numel())
        left = source_flat[start:stop].to(dtype=torch_module.float64)
        right = measured_flat[start:stop].to(dtype=torch_module.float64)
        difference = right - left
        dot += float(torch_module.sum(left * right).item())
        source_square_sum += float(torch_module.sum(left * left).item())
        measured_square_sum += float(torch_module.sum(right * right).item())
        error_square_sum += float(torch_module.sum(difference * difference).item())
        if difference.numel():
            max_abs_error = max(
                max_abs_error, float(difference.abs().max().item())
            )
    numel = source_flat.numel()
    if numel <= 0:
        raise ValueError("Error statistics require at least one value.")
    denominator = math.sqrt(source_square_sum * measured_square_sum)
    if denominator > 0:
        raw_cosine = dot / denominator
        cosine = min(1.0, max(-1.0, raw_cosine))
    else:
        cosine = 1.0 if source_square_sum == measured_square_sum else 0.0
    source_power = source_square_sum / numel
    mse = error_square_sum / numel
    return {
        "source_power": source_power,
        "mse": mse,
        "cosine_similarity": cosine,
        "max_abs_error": max_abs_error,
    }


def select_evenly_spaced_positions(token_count: int, limit: int) -> list[int]:
    """Select deterministic token positions, always including both endpoints."""

    if token_count <= 0 or limit <= 0:
        raise ValueError("token_count and limit must be positive.")
    if token_count <= limit:
        return list(range(token_count))
    return [index * (token_count - 1) // (limit - 1) for index in range(limit)]


def _empty_error_sums() -> dict[str, float | int]:
    return {
        "numel": 0,
        "dot": 0.0,
        "source_square_sum": 0.0,
        "measured_square_sum": 0.0,
        "error_square_sum": 0.0,
        "max_abs_error": 0.0,
    }


def _accumulate_error_sums(
    accumulator: dict[str, float | int],
    source: Any,
    measured: Any,
    *,
    torch_module: Any,
    chunk_size: int = ERROR_METRICS_CHUNK_SIZE,
) -> None:
    if source.shape != measured.shape:
        raise ValueError("Activation tensors have different shapes.")
    left_flat = source.reshape(-1)
    right_flat = measured.reshape(-1)
    for start in range(0, left_flat.numel(), chunk_size):
        stop = min(start + chunk_size, left_flat.numel())
        left = left_flat[start:stop].to(dtype=torch_module.float64)
        right = right_flat[start:stop].to(dtype=torch_module.float64)
        difference = right - left
        accumulator["dot"] += float(torch_module.sum(left * right).item())
        accumulator["source_square_sum"] += float(torch_module.sum(left * left).item())
        accumulator["measured_square_sum"] += float(torch_module.sum(right * right).item())
        accumulator["error_square_sum"] += float(
            torch_module.sum(difference * difference).item()
        )
        if difference.numel():
            accumulator["max_abs_error"] = max(
                float(accumulator["max_abs_error"]),
                float(difference.abs().max().item()),
            )
    accumulator["numel"] += int(left_flat.numel())


def _metrics_from_error_sums(sums: dict[str, float | int]) -> dict[str, Any]:
    numel = int(sums["numel"])
    if numel <= 0:
        raise ValueError("Activation error statistics require at least one value.")
    source_square_sum = float(sums["source_square_sum"])
    measured_square_sum = float(sums["measured_square_sum"])
    denominator = math.sqrt(source_square_sum * measured_square_sum)
    if denominator > 0:
        cosine = min(1.0, max(-1.0, float(sums["dot"]) / denominator))
    else:
        cosine = 1.0 if source_square_sum == measured_square_sum else 0.0
    mse = float(sums["error_square_sum"]) / numel
    source_power = source_square_sum / numel
    return {
        "numel": numel,
        "mse": mse,
        "nmse": mse / source_power if source_power > 0 else 0.0,
        "cosine_similarity": cosine,
        "max_abs_error": float(sums["max_abs_error"]),
    }


def _fake_quantize_weight(
    weight: Any, *, bits: int, group_size: int | None, torch_module: Any
) -> tuple[Any, dict[str, float]]:
    if bits not in {4, 8}:
        raise ValueError("Fake quantization supports only 4-bit and 8-bit weights.")
    original_dtype = weight.dtype
    source = weight.detach().to(dtype=torch_module.float32, device="cpu")
    if source.ndim != 2:
        raise ValueError("Only 2-D Linear weights are supported.")
    rows, columns = source.shape
    effective_group = columns if group_size is None else int(group_size)
    if effective_group <= 0:
        raise ValueError("group_size must be positive or null.")
    padded_columns = math.ceil(columns / effective_group) * effective_group
    if padded_columns != columns:
        padded = torch_module.zeros((rows, padded_columns), dtype=source.dtype)
        padded[:, :columns] = source
    else:
        padded = source
    grouped = padded.reshape(rows, -1, effective_group)
    qmax = (1 << (bits - 1)) - 1
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch_module.where(absmax > 0, absmax / qmax, torch_module.ones_like(absmax))
    quantized = torch_module.round(grouped / scale).clamp(-qmax, qmax)
    restored = (quantized * scale).reshape(rows, padded_columns)[:, :columns]
    stored = restored.to(dtype=original_dtype)
    measured = stored.to(dtype=torch_module.float32)
    statistics = _stable_error_statistics(
        source,
        measured,
        torch_module=torch_module,
    )
    quantized_unpadded = quantized.reshape(rows, padded_columns)[:, :columns]
    saturation = float(
        (quantized_unpadded.abs() == qmax).to(torch_module.float32).mean().item()
    )
    metrics = {
        "mse": statistics["mse"],
        "nmse": (
            statistics["mse"] / statistics["source_power"]
            if statistics["source_power"] > 0
            else 0.0
        ),
        "cosine_similarity": statistics["cosine_similarity"],
        "max_abs_error": statistics["max_abs_error"],
        "saturation_rate": saturation,
    }
    return stored, metrics


def validate_quantization_error_report(report: dict[str, Any]) -> None:
    if report.get("error_metrics_schema_version") != ERROR_METRICS_SCHEMA_VERSION:
        raise ValueError("Quantization error report uses an obsolete metrics schema.")
    records = [
        *report.get("modules", []),
        *report.get("layers", []),
        report.get("aggregate", {}),
    ]
    if not report.get("modules") or not report.get("layers"):
        raise ValueError("Quantization error report is missing module or layer records.")
    for record in records:
        for field in ("mse", "nmse", "cosine_similarity", "max_abs_error", "saturation_rate"):
            value = float(record[field])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite quantization metric: {field}")
        cosine = float(record["cosine_similarity"])
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("Quantization cosine similarity must be in [-1, 1].")
        if float(record["mse"]) < 0 or float(record["nmse"]) < 0:
            raise ValueError("Quantization MSE and NMSE must be non-negative.")
        saturation = float(record["saturation_rate"])
        if not 0.0 <= saturation <= 1.0:
            raise ValueError("Quantization saturation rate must be in [0, 1].")


def quantize_model_for_plan(
    model: Any,
    plan: dict[str, Any],
    precision: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    block_path, blocks = discover_transformer_blocks(model)
    module_records = []
    layer_records = []
    for layer_index, block in enumerate(blocks):
        precision_name = precision_for_layer(plan, layer_index)
        if precision_name is None:
            continue
        contract = precision[precision_name]
        layer_modules = []
        for module_name, module in block.named_modules():
            if not isinstance(module, torch_module.nn.Linear):
                continue
            restored, metrics = _fake_quantize_weight(
                module.weight.data,
                bits=int(contract["weight_bits"]),
                group_size=contract.get("group_size"),
                torch_module=torch_module,
            )
            module.weight.data.copy_(restored)
            record = {
                "layer_index": layer_index,
                "module": module_name,
                "precision": precision_name,
                "bits": int(contract["weight_bits"]),
                "group_size": contract.get("group_size"),
                "numel": module.weight.numel(),
                **metrics,
            }
            layer_modules.append(record)
            module_records.append(record)
        if not layer_modules:
            raise ValueError(f"Transformer block {layer_index} has no Linear modules.")
        layer_records.append(
            {
                "layer_index": layer_index,
                "precision": precision_name,
                **aggregate_error_metrics(layer_modules),
            }
        )
    report = {
        "error_metrics_schema_version": ERROR_METRICS_SCHEMA_VERSION,
        "metric_accumulation": {
            "strategy": "chunked_sum",
            "accumulator_dtype": "float64",
            "input_dtype": "float32",
            "chunk_size": ERROR_METRICS_CHUNK_SIZE,
            "cosine_clamped_to_valid_range": True,
        },
        "block_path": block_path,
        "block_count": len(blocks),
        "plan_id": plan["plan_id"],
        "backend": "symmetric_fake_quant",
        "deployment_performance_claims_allowed": False,
        "layers": layer_records,
        "modules": module_records,
        "aggregate": aggregate_error_metrics(module_records),
    }
    validate_quantization_error_report(report)
    return report


def apply_plan_with_callback(
    blocks: list[Any], plan: dict[str, Any], callback: Callable[[int, Any, str], None]
) -> None:
    for layer_index, block in enumerate(blocks):
        precision = precision_for_layer(plan, layer_index)
        if precision is not None:
            callback(layer_index, block, precision)


def _structured_variant(report: dict[str, Any]) -> dict[str, Any]:
    variants = [
        item for item in report.get("variants", [])
        if item.get("variant") == "structured_json"
    ]
    if len(variants) != 1:
        raise ValueError("Expected exactly one structured_json result.")
    return variants[0]


def reference_record(
    plan: dict[str, Any], report_path: Path, expected_sample_hash: str
) -> dict[str, Any]:
    report = load_json(report_path)
    sample = report.get("dataset", {}).get("sample", {})
    if sample.get("indices_sha256") != expected_sample_hash:
        raise ValueError(f"Reference {plan['plan_id']} uses the wrong pilot sample.")
    model = report.get("model", {})
    if plan["kind"] == "ref_fp":
        if (
            model.get("quantization") != "none"
            or model.get("adapter_path") is not None
            or bool(model.get("load_in_4bit"))
        ):
            raise ValueError("REF-FP must be the standalone merged non-quantized model.")
    elif plan["kind"] == "ref_nf4":
        if not model.get("load_in_4bit") or not model.get("adapter_path"):
            raise ValueError("REF-NF4 must use the accepted BitsAndBytes NF4 path.")
    variant = _structured_variant(report)
    return {
        "plan": plan,
        "status": "reference_registered",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "predictions_sha256": variant.get("predictions_sha256"),
        "metrics": variant["metrics"],
        "structured_output_rate": variant.get("structured_output_rate"),
        "normalized_runtime_quantization": (
            "bitsandbytes-nf4" if plan["kind"] == "ref_nf4" else "none"
        ),
    }


def prepare_matrix(
    *, project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = config["paths"]
    expected = config["expected_identity"]
    merged_manifest_path = resolve_project_path(root, paths["merged_manifest"])
    input_freeze_path = resolve_project_path(root, paths["input_freeze"])
    merged = load_json(merged_manifest_path)
    frozen = load_json(input_freeze_path)
    if merged.get("status") != "merged" or frozen.get("status") != "inputs_frozen":
        raise ValueError("TODO 7.1.2 requires completed merge and input freeze artifacts.")
    if merged.get("merged_model", {}).get("tree_sha256") != expected["merged_tree_sha256"]:
        raise ValueError("Merged model tree SHA256 does not match the accepted 7.1.1 artifact.")
    if merged.get("adapter", {}).get("tree_sha256") != expected["adapter_tree_sha256"]:
        raise ValueError("Adapter tree SHA256 does not match the accepted 7.1.1 artifact.")
    blocks = merged.get("transformer_blocks", {})
    if blocks != {"path": expected["block_path"], "count": expected["block_count"]}:
        raise ValueError("Transformer block discovery differs from the 7.1.1 result.")
    if frozen.get("frozen_adapter_record", {}).get("adapter_weights_sha256") != expected["adapter_weights_sha256"]:
        raise ValueError("Frozen adapter weight identity differs from the 7.1.1 contract.")
    plans = build_plan_matrix(int(blocks["count"]))
    validate_plan_isolation(plans, int(blocks["count"]))
    output_root = resolve_project_path(root, paths["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    references = {}
    reference_paths = {
        "ref_fp": resolve_project_path(root, paths["ref_fp_report"]),
        "ref_nf4": resolve_project_path(root, paths["ref_nf4_report"]),
    }
    for plan in plans:
        plan_dir = output_root / plan["plan_id"]
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_payload = {
            "schema_version": 1,
            "stage": "7.1.2",
            "plan": plan,
            "precision_vector": plan_precision_vector(plan, int(blocks["count"])),
            "precision_contract": config["precision"],
            "backend": config["backend"],
        }
        plan_payload["plan_sha256"] = canonical_json_sha256(plan_payload)
        write_json(plan_dir / "plan.json", plan_payload)
        if plan["kind"] in REFERENCE_KINDS:
            registered = reference_record(
                plan,
                reference_paths[plan["plan_id"]],
                expected["pilot_indices_sha256"],
            )
            write_json(plan_dir / "reference.json", registered)
            references[plan["plan_id"]] = registered
    experiment = {
        "schema_version": 1,
        "stage": "7.1.2",
        "status": "matrix_prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": config["backend"],
        "merged_manifest": {
            "path": str(merged_manifest_path),
            "sha256": sha256_file(merged_manifest_path),
            "tree_sha256": expected["merged_tree_sha256"],
        },
        "input_freeze": {
            "path": str(input_freeze_path),
            "sha256": sha256_file(input_freeze_path),
        },
        "block_path": blocks["path"],
        "block_count": blocks["count"],
        "plan_count": len(plans),
        "runnable_plan_count": sum(bool(plan["runnable"]) for plan in plans),
        "plans": plans,
        "references": references,
    }
    experiment["matrix_sha256"] = canonical_json_sha256(experiment["plans"])
    write_json(output_root / "experiment_manifest.json", experiment)
    return experiment


def load_prepared_plan(
    project_root: str | Path, config: dict[str, Any], plan_id: str
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    root = Path(project_root).resolve()
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    experiment = load_json(output_root / "experiment_manifest.json")
    if experiment.get("status") != "matrix_prepared":
        raise ValueError("Experiment matrix is not prepared.")
    matches = [plan for plan in experiment["plans"] if plan["plan_id"] == plan_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown plan_id: {plan_id}")
    plan = matches[0]
    if not plan.get("runnable"):
        raise ValueError(f"Reference plan {plan_id} is not runnable.")
    plan_dir = output_root / plan_id
    plan_payload = load_json(plan_dir / "plan.json")
    expected_hash = plan_payload.pop("plan_sha256")
    if canonical_json_sha256(plan_payload) != expected_hash:
        raise ValueError(f"Plan contract changed after preparation: {plan_id}")
    plan_payload["plan_sha256"] = expected_hash
    return plan, plan_dir, experiment


def pending_plan_ids(project_root: str | Path, config: dict[str, Any]) -> list[str]:
    root = Path(project_root).resolve()
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    experiment = load_json(output_root / "experiment_manifest.json")
    pending = []
    for plan in experiment["plans"]:
        if not plan.get("runnable"):
            continue
        result_path = output_root / plan["plan_id"] / "result.json"
        if not result_path.is_file():
            pending.append(plan["plan_id"])
            continue
        result = load_json(result_path)
        if result.get("status") != "completed":
            pending.append(plan["plan_id"])
    return pending


def _evaluation_dtype(torch_module: Any, name: str) -> Any:
    dtype = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }.get(name)
    if dtype is None:
        raise ValueError("Evaluation dtype must be float16 or bfloat16.")
    return dtype


def _validate_runtime_blocks(
    error_report: dict[str, Any], experiment: dict[str, Any]
) -> None:
    if (
        error_report["block_path"] != experiment["block_path"]
        or error_report["block_count"] != experiment["block_count"]
    ):
        raise ValueError("Runtime block discovery differs from the prepared matrix.")


def recompute_plan_error(
    *, project_root: str | Path, config: dict[str, Any], plan_id: str
) -> dict[str, Any]:
    """Repair one error report without repeating the GPU quality evaluation."""

    root = Path(project_root).resolve()
    plan, plan_dir, experiment = load_prepared_plan(root, config, plan_id)
    result_path = plan_dir / "result.json"
    if not result_path.is_file():
        raise ValueError(f"Plan {plan_id} has no completed result to repair.")
    result = load_json(result_path)
    if result.get("status") != "completed" or result.get("plan") != plan:
        raise ValueError(f"Plan {plan_id} result is not a matching completed run.")
    report = load_json(plan_dir / "comparison_report.json")
    variant = _structured_variant(report)
    predictions_path = plan_dir / variant["predictions_file"]
    predictions_sha256 = sha256_file(predictions_path)
    if (
        predictions_sha256 != result.get("predictions_sha256")
        or predictions_sha256 != variant.get("predictions_sha256")
    ):
        raise ValueError(f"Plan {plan_id} prediction evidence changed before repair.")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Error recomputation requires torch and transformers.") from exc

    evaluation = config["evaluation"]
    dtype = _evaluation_dtype(torch, evaluation["dtype"])
    merged_manifest = load_json(
        resolve_project_path(root, config["paths"]["merged_manifest"])
    )
    model_path = Path(merged_manifest["merged_model"]["path"]).resolve()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    error_report = quantize_model_for_plan(
        model, plan, config["precision"], torch
    )
    _validate_runtime_blocks(error_report, experiment)
    error_path = plan_dir / "quantization_error.json"
    previous_error_sha256 = sha256_file(error_path) if error_path.is_file() else None
    backup_path = plan_dir / "quantization_error.schema_v1.json"
    previous_error = load_json(error_path) if error_path.is_file() else {}
    if (
        error_path.is_file()
        and previous_error.get("error_metrics_schema_version")
        != ERROR_METRICS_SCHEMA_VERSION
        and not backup_path.exists()
    ):
        shutil.copy2(error_path, backup_path)
    write_json(error_path, error_report)
    repaired_at = datetime.now(timezone.utc).isoformat()
    if "original_quantization_error_sha256" not in result:
        result["original_quantization_error_sha256"] = previous_error_sha256
    result.update(
        {
            "quantization_error_sha256": sha256_file(error_path),
            "quantization_error_metrics_schema_version": ERROR_METRICS_SCHEMA_VERSION,
            "quantization_error_recomputed_at": repaired_at,
            "previous_quantization_error_sha256": previous_error_sha256,
            "weighted_nmse": error_report["aggregate"]["nmse"],
        }
    )
    write_json(result_path, result)
    return {
        "schema_version": 1,
        "stage": "7.1.2",
        "status": "quantization_error_recomputed",
        "plan_id": plan_id,
        "predictions_preserved": True,
        "predictions_sha256": predictions_sha256,
        "previous_quantization_error_sha256": previous_error_sha256,
        "quantization_error_sha256": result["quantization_error_sha256"],
        "error_metrics_schema_version": ERROR_METRICS_SCHEMA_VERSION,
        "weighted_nmse": result["weighted_nmse"],
        "duration_seconds": time.perf_counter() - started,
    }


def audit_error_reports(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    experiment = load_json(output_root / "experiment_manifest.json")
    invalid = []
    checked = 0
    cosines = []
    for plan in experiment["plans"]:
        if not plan.get("runnable"):
            continue
        plan_id = plan["plan_id"]
        plan_dir = output_root / plan_id
        error_path = plan_dir / "quantization_error.json"
        result_path = plan_dir / "result.json"
        try:
            if not error_path.is_file() or not result_path.is_file():
                raise ValueError("missing quantization_error.json or result.json")
            error_report = load_json(error_path)
            result = load_json(result_path)
            validate_quantization_error_report(error_report)
            if (
                result.get("quantization_error_metrics_schema_version")
                != ERROR_METRICS_SCHEMA_VERSION
            ):
                raise ValueError("result does not declare the current metrics schema")
            if result.get("quantization_error_sha256") != sha256_file(error_path):
                raise ValueError("result quantization-error SHA256 does not match")
            cosines.extend(
                float(record["cosine_similarity"])
                for record in error_report["modules"]
            )
            checked += 1
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            invalid.append({"plan_id": plan_id, "reason": str(exc)})
    audit = {
        "schema_version": 1,
        "stage": "7.1.2",
        "status": "pass" if not invalid else "fail",
        "error_metrics_schema_version": ERROR_METRICS_SCHEMA_VERSION,
        "expected_runnable_plans": experiment["runnable_plan_count"],
        "valid_plans": checked,
        "invalid_plans": invalid,
        "module_cosine_min": min(cosines) if cosines else None,
        "module_cosine_max": max(cosines) if cosines else None,
    }
    write_json(output_root / "error_audit.json", audit)
    return audit


def run_plan(
    *, project_root: str | Path, config: dict[str, Any], plan_id: str
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan, plan_dir, experiment = load_prepared_plan(root, config, plan_id)
    result_path = plan_dir / "result.json"
    if result_path.is_file() and load_json(result_path).get("status") == "completed":
        return {**load_json(result_path), "resume_action": "skipped_completed"}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Plan execution requires torch and transformers.") from exc
    from src.baseline.service import BaselineService
    from src.data.finqa_assets import load_assets
    from src.evaluation.quality_repair import run_quality_repair_experiment

    paths = config["paths"]
    evaluation = config["evaluation"]
    merged_manifest = load_json(resolve_project_path(root, paths["merged_manifest"]))
    model_path = Path(merged_manifest["merged_model"]["path"]).resolve()
    baseline = load_json(resolve_project_path(root, paths["baseline_config"]))
    baseline["device"] = evaluation["device"]
    baseline["dtype"] = evaluation["dtype"]
    quality = load_json(resolve_project_path(root, paths["quality_config"]))
    quality["evaluation_size"] = int(evaluation["pilot_size"])
    quality["evaluation_seed"] = int(evaluation["pilot_seed"])
    quality["batch_size"] = int(evaluation["batch_size"])
    quality["generation"]["max_new_tokens"] = int(evaluation["max_new_tokens"])
    quality["pilot_variants"] = [evaluation["variant"]]
    quality["dataset_role"] = evaluation["dataset_role"]
    write_json(plan_dir / "baseline_config.effective.json", baseline)
    write_json(plan_dir / "quality_config.effective.json", quality)
    model_manifest = {
        "schema_version": 1,
        "model_id": "Qwen2.5-7B-Instruct-QLoRA-step724-merged-fp16",
        "model_path": str(model_path),
        "version": experiment["merged_manifest"]["tree_sha256"],
        "runtime": {
            "framework": "transformers",
            "model_class": "causal_lm",
            "dtype": evaluation["dtype"],
            "device": evaluation["device"],
            "quantization": "symmetric_fake_quant",
            "trust_remote_code": False,
        },
    }
    write_json(plan_dir / "model_manifest.json", model_manifest)
    dtype = _evaluation_dtype(torch, evaluation["dtype"])
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    error_report = quantize_model_for_plan(
        model, plan, config["precision"], torch
    )
    _validate_runtime_blocks(error_report, experiment)
    write_json(plan_dir / "quantization_error.json", error_report)
    model.to(evaluation["device"])
    model.eval()
    service = BaselineService(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        config=baseline,
        model_metadata={
            "model_id": model_manifest["model_id"],
            "version": model_manifest["version"],
            "quantization": {
                "backend": "symmetric_fake_quant",
                "plan_id": plan_id,
                "deployment_performance_claims_allowed": False,
            },
        },
        log_file=plan_dir / "quality_service.jsonl",
    )
    assets = load_assets(resolve_project_path(root, paths["dataset_dir"]))
    report = run_quality_repair_experiment(
        service,
        assets.quality_dev,
        assets.manifest,
        quality,
        plan_dir,
        calibration=assets.calibration,
    )
    sample_hash = report.get("dataset", {}).get("sample", {}).get("indices_sha256")
    if sample_hash != config["expected_identity"]["pilot_indices_sha256"]:
        raise ValueError("Plan evaluation used the wrong pilot sample.")
    variant = _structured_variant(report)
    result = {
        "schema_version": 1,
        "stage": "7.1.2",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "backend": config["backend"],
        "sample_indices_sha256": sample_hash,
        "metrics": variant["metrics"],
        "structured_output_rate": variant.get("structured_output_rate"),
        "predictions_sha256": variant.get("predictions_sha256"),
        "quantization_error_sha256": sha256_file(plan_dir / "quantization_error.json"),
        "quantization_error_metrics_schema_version": ERROR_METRICS_SCHEMA_VERSION,
        "weighted_nmse": error_report["aggregate"]["nmse"],
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    return result


def _activation_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("activation_analysis")
    if not isinstance(settings, dict):
        raise ValueError("activation_analysis is missing from the sensitivity config.")
    indices = settings.get("sample_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(not isinstance(index, int) or index < 0 for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise ValueError("activation sample_indices must be unique non-negative integers.")
    if int(settings.get("max_input_tokens", 0)) <= 0:
        raise ValueError("activation max_input_tokens must be positive.")
    if int(settings.get("token_positions_per_sample", 0)) < 2:
        raise ValueError("activation token_positions_per_sample must be at least 2.")
    if settings.get("truncation_side") not in {"left", "right"}:
        raise ValueError("activation truncation_side must be left or right.")
    return settings


def _activation_reference_dir(root: Path, config: dict[str, Any]) -> Path:
    return resolve_project_path(root, _activation_settings(config)["reference_dir"])


def _block_hidden_state(output: Any) -> Any:
    if hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0]
    raise ValueError("Transformer block hook did not receive a hidden-state tensor.")


def _capture_block_outputs(
    model: Any,
    encoded: dict[str, Any],
    positions: list[int],
    *,
    torch_module: Any,
    cache_dtype: Any,
) -> tuple[str, list[Any]]:
    block_path, blocks = discover_transformer_blocks(model)
    captured: list[Any | None] = [None] * len(blocks)
    device = encoded["input_ids"].device
    position_tensor = torch_module.tensor(positions, dtype=torch_module.long, device=device)
    handles = []
    for layer_index, block in enumerate(blocks):
        def capture(_module: Any, _inputs: Any, output: Any, index: int = layer_index) -> None:
            hidden = _block_hidden_state(output)
            captured[index] = (
                hidden.detach()
                .index_select(1, position_tensor)
                .contiguous()
            )

        handles.append(block.register_forward_hook(capture))
    try:
        backbone = getattr(model, "model", model)
        with torch_module.inference_mode():
            backbone(**encoded, use_cache=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise ValueError("One or more transformer block hooks did not run.")
    captured_values = [value for value in captured if value is not None]
    return block_path, [
        value.to(device="cpu", dtype=cache_dtype).contiguous()
        for value in captured_values
    ]


def _tokenize_activation_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_input_tokens: int,
    device: str,
) -> tuple[dict[str, Any], list[int]]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"][0].tolist()
    if not input_ids:
        raise ValueError("Tokenizer produced an empty activation input.")
    return {name: tensor.to(device) for name, tensor in encoded.items()}, input_ids


def _validate_activation_reference(
    root: Path, config: dict[str, Any], manifest: dict[str, Any]
) -> None:
    settings = _activation_settings(config)
    expected = config["expected_identity"]
    if manifest.get("status") != "completed":
        raise ValueError("Activation reference is not complete.")
    if manifest.get("merged_tree_sha256") != expected["merged_tree_sha256"]:
        raise ValueError("Activation reference uses the wrong merged model.")
    if manifest.get("calibration_subset_sha256") != expected["calibration_subset_sha256"]:
        raise ValueError("Activation reference uses the wrong calibration set.")
    if manifest.get("sample_indices") != settings["sample_indices"]:
        raise ValueError("Activation reference uses different calibration indices.")
    if manifest.get("sample_indices_sha256") != canonical_json_sha256(
        settings["sample_indices"]
    ):
        raise ValueError("Activation reference sample-index SHA256 differs.")
    contract = manifest.get("probe_contract", {})
    for field in (
        "max_input_tokens",
        "truncation_side",
        "token_positions_per_sample",
        "cache_dtype",
    ):
        if contract.get(field) != settings[field]:
            raise ValueError(f"Activation reference contract changed: {field}.")
    if contract.get("model_dtype") != settings["dtype"]:
        raise ValueError("Activation reference contract changed: model_dtype.")
    if (
        contract.get("scope") != "sampled_block_output_token_states"
        or contract.get("position_selection") != "evenly_spaced_include_endpoints"
        or contract.get("use_cache") is not False
    ):
        raise ValueError("Activation reference probe semantics changed.")
    if manifest.get("block_count") != expected["block_count"]:
        raise ValueError("Activation reference block count changed.")
    if manifest.get("block_path") != expected["block_path"]:
        raise ValueError("Activation reference block path changed.")
    samples = manifest.get("samples", [])
    if [sample.get("source_index") for sample in samples] != settings["sample_indices"]:
        raise ValueError("Activation reference sample records are incomplete or reordered.")
    reference_dir = _activation_reference_dir(root, config)
    for sample in samples:
        shard = reference_dir / sample["cache_file"]
        if not shard.is_file() or sha256_file(shard) != sample["cache_sha256"]:
            raise ValueError(f"Activation reference shard is missing or changed: {shard.name}")


def prepare_activation_reference(
    *, project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Cache sampled FP16 block outputs for fixed calibration prompts."""

    root = Path(project_root).resolve()
    settings = _activation_settings(config)
    reference_dir = _activation_reference_dir(root, config)
    manifest_path = reference_dir / "manifest.json"
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        try:
            _validate_activation_reference(root, config, existing)
            return {**existing, "resume_action": "skipped_completed"}
        except (KeyError, TypeError, ValueError):
            pass
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Activation collection requires torch and transformers.") from exc
    from src.data.finqa_assets import load_assets

    paths = config["paths"]
    expected = config["expected_identity"]
    frozen_subsets = load_json(
        resolve_project_path(root, paths["setup_dir"]) / "frozen_subsets.json"
    )
    calibration_entries = frozen_subsets["calibration"]
    if canonical_json_sha256(calibration_entries) != expected["calibration_subset_sha256"]:
        raise ValueError("Frozen calibration entries do not match the accepted SHA256.")
    assets = load_assets(resolve_project_path(root, paths["dataset_dir"]))
    indices = settings["sample_indices"]
    if max(indices) >= len(assets.calibration):
        raise ValueError("Activation sample index exceeds the calibration dataset.")
    for index in indices:
        if row_fingerprint(dict(assets.calibration[index])) != calibration_entries[index]["row_sha256"]:
            raise ValueError(f"Calibration row identity changed at index {index}.")

    merged_manifest = load_json(resolve_project_path(root, paths["merged_manifest"]))
    model_path = Path(merged_manifest["merged_model"]["path"]).resolve()
    dtype = _evaluation_dtype(torch, settings["dtype"])
    cache_dtype = _evaluation_dtype(torch, settings["cache_dtype"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    tokenizer.truncation_side = settings["truncation_side"]
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        dtype=dtype, low_cpu_mem_usage=True,
    )
    model.to(settings["device"])
    model.eval()
    reference_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    observed_block_path = None
    started = time.perf_counter()
    for ordinal, source_index in enumerate(indices):
        encoded, input_ids = _tokenize_activation_prompt(
            tokenizer,
            assets.calibration_prompts[source_index],
            max_input_tokens=int(settings["max_input_tokens"]),
            device=settings["device"],
        )
        positions = select_evenly_spaced_positions(
            len(input_ids), int(settings["token_positions_per_sample"])
        )
        block_path, layers = _capture_block_outputs(
            model, encoded, positions, torch_module=torch, cache_dtype=cache_dtype
        )
        observed_block_path = observed_block_path or block_path
        if block_path != observed_block_path or len(layers) != expected["block_count"]:
            raise ValueError("Runtime transformer block discovery changed.")
        shard_path = reference_dir / f"sample_{ordinal:03d}.pt"
        temporary_path = shard_path.with_suffix(".pt.tmp")
        torch.save({"layers": layers}, temporary_path)
        temporary_path.replace(shard_path)
        samples.append(
            {
                "ordinal": ordinal,
                "source_index": source_index,
                "row_sha256": calibration_entries[source_index]["row_sha256"],
                "input_token_count": len(input_ids),
                "input_ids_sha256": canonical_json_sha256(input_ids),
                "selected_positions": positions,
                "selected_positions_sha256": canonical_json_sha256(positions),
                "cache_file": shard_path.name,
                "cache_sha256": sha256_file(shard_path),
                "cache_bytes": shard_path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "stage": "7.1.3",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference": "merged_fp16",
        "merged_tree_sha256": expected["merged_tree_sha256"],
        "calibration_subset_sha256": expected["calibration_subset_sha256"],
        "sample_indices": indices,
        "sample_indices_sha256": canonical_json_sha256(indices),
        "selected_rows_sha256": canonical_json_sha256(
            [calibration_entries[index] for index in indices]
        ),
        "probe_contract": {
            "scope": "sampled_block_output_token_states",
            "max_input_tokens": settings["max_input_tokens"],
            "truncation_side": settings["truncation_side"],
            "token_positions_per_sample": settings["token_positions_per_sample"],
            "position_selection": "evenly_spaced_include_endpoints",
            "model_dtype": settings["dtype"],
            "cache_dtype": settings["cache_dtype"],
            "use_cache": False,
        },
        "block_path": observed_block_path,
        "block_count": expected["block_count"],
        "samples": samples,
        "total_cache_bytes": sum(sample["cache_bytes"] for sample in samples),
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    return manifest


def _merge_error_sums(
    destination: dict[str, float | int], source: dict[str, float | int]
) -> None:
    for field in ("numel", "dot", "source_square_sum", "measured_square_sum", "error_square_sum"):
        destination[field] += source[field]
    destination["max_abs_error"] = max(
        float(destination["max_abs_error"]), float(source["max_abs_error"])
    )


def validate_activation_error_report(report: dict[str, Any]) -> None:
    if report.get("activation_metrics_schema_version") != ACTIVATION_METRICS_SCHEMA_VERSION:
        raise ValueError("Activation report uses an obsolete metrics schema.")
    if report.get("status") != "completed" or not report.get("layers"):
        raise ValueError("Activation report is incomplete.")
    for record in [*report["layers"], report.get("aggregate", {})]:
        for field in ("mse", "nmse", "cosine_similarity", "max_abs_error"):
            if not math.isfinite(float(record[field])):
                raise ValueError(f"Non-finite activation metric: {field}")
        if float(record["mse"]) < 0 or float(record["nmse"]) < 0:
            raise ValueError("Activation MSE and NMSE must be non-negative.")
        if not -1.0 <= float(record["cosine_similarity"]) <= 1.0:
            raise ValueError("Activation cosine similarity must be in [-1, 1].")


def run_activation_plan(
    *, project_root: str | Path, config: dict[str, Any], plan_id: str
) -> dict[str, Any]:
    """Compare one fake-quant plan with the cached FP16 block outputs."""

    root = Path(project_root).resolve()
    settings = _activation_settings(config)
    plan, plan_dir, experiment = load_prepared_plan(root, config, plan_id)
    result_path = plan_dir / "activation_result.json"
    report_path = plan_dir / settings["report_filename"]
    reference_dir = _activation_reference_dir(root, config)
    reference_manifest_path = reference_dir / "manifest.json"
    reference = load_json(reference_manifest_path)
    _validate_activation_reference(root, config, reference)
    if result_path.is_file() and report_path.is_file():
        existing = load_json(result_path)
        existing_report = load_json(report_path)
        if (
            existing.get("status") == "completed"
            and existing.get("activation_error_sha256") == sha256_file(report_path)
            and existing_report.get("plan") == plan
            and existing_report.get("reference_manifest_sha256")
            == sha256_file(reference_manifest_path)
        ):
            validate_activation_error_report(existing_report)
            return {**existing, "resume_action": "skipped_completed"}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Activation collection requires torch and transformers.") from exc
    from src.data.finqa_assets import load_assets

    merged_manifest = load_json(
        resolve_project_path(root, config["paths"]["merged_manifest"])
    )
    model_path = Path(merged_manifest["merged_model"]["path"]).resolve()
    dtype = _evaluation_dtype(torch, settings["dtype"])
    cache_dtype = _evaluation_dtype(torch, settings["cache_dtype"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    tokenizer.truncation_side = settings["truncation_side"]
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        dtype=dtype, low_cpu_mem_usage=True,
    )
    quantization_error = quantize_model_for_plan(model, plan, config["precision"], torch)
    _validate_runtime_blocks(quantization_error, experiment)
    model.to(settings["device"])
    model.eval()
    assets = load_assets(resolve_project_path(root, config["paths"]["dataset_dir"]))
    layer_sums = [_empty_error_sums() for _ in range(experiment["block_count"])]
    layer_token_states = [0] * experiment["block_count"]
    started = time.perf_counter()
    for sample in reference["samples"]:
        source_index = int(sample["source_index"])
        encoded, input_ids = _tokenize_activation_prompt(
            tokenizer,
            assets.calibration_prompts[source_index],
            max_input_tokens=int(settings["max_input_tokens"]),
            device=settings["device"],
        )
        if canonical_json_sha256(input_ids) != sample["input_ids_sha256"]:
            raise ValueError(f"Tokenized activation input changed at index {source_index}.")
        positions = select_evenly_spaced_positions(
            len(input_ids), int(settings["token_positions_per_sample"])
        )
        if positions != sample["selected_positions"]:
            raise ValueError(f"Activation token positions changed at index {source_index}.")
        block_path, measured_layers = _capture_block_outputs(
            model, encoded, positions, torch_module=torch, cache_dtype=cache_dtype
        )
        if block_path != reference["block_path"]:
            raise ValueError("Candidate block path differs from the FP16 reference.")
        cached = torch.load(
            reference_dir / sample["cache_file"], map_location="cpu", weights_only=True
        )
        source_layers = cached["layers"]
        if len(source_layers) != len(measured_layers):
            raise ValueError("Candidate and reference block counts differ.")
        for layer_index, (source, measured) in enumerate(zip(source_layers, measured_layers)):
            _accumulate_error_sums(
                layer_sums[layer_index], source, measured,
                torch_module=torch,
            )
            layer_token_states[layer_index] += len(positions)
    aggregate_sums = _empty_error_sums()
    layers = []
    for layer_index, sums in enumerate(layer_sums):
        metrics = _metrics_from_error_sums(sums)
        layers.append(
            {
                "layer_index": layer_index,
                "precision": precision_for_layer(plan, layer_index),
                "sample_count": len(reference["samples"]),
                "token_state_count": layer_token_states[layer_index],
                **metrics,
            }
        )
        _merge_error_sums(aggregate_sums, sums)
    report = {
        "schema_version": 1,
        "activation_metrics_schema_version": ACTIVATION_METRICS_SCHEMA_VERSION,
        "stage": "7.1.3",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "probe_contract": reference["probe_contract"],
        "sample_indices_sha256": reference["sample_indices_sha256"],
        "block_path": reference["block_path"],
        "block_count": reference["block_count"],
        "metric_accumulation": {
            "strategy": "paired_chunked_sum",
            "accumulator_dtype": "float64",
            "cached_activation_dtype": settings["cache_dtype"],
            "chunk_size": ERROR_METRICS_CHUNK_SIZE,
            "cosine_clamped_to_valid_range": True,
        },
        "layers": layers,
        "aggregate": {
            "layer_count": len(layers),
            "sample_count": len(reference["samples"]),
            "token_state_count": sum(layer_token_states),
            **_metrics_from_error_sums(aggregate_sums),
        },
        "duration_seconds": time.perf_counter() - started,
    }
    validate_activation_error_report(report)
    write_json(report_path, report)
    result = {
        "schema_version": 1,
        "stage": "7.1.3",
        "status": "completed",
        "plan_id": plan_id,
        "activation_error_sha256": sha256_file(report_path),
        "reference_manifest_sha256": report["reference_manifest_sha256"],
        "sample_indices_sha256": report["sample_indices_sha256"],
        "aggregate": report["aggregate"],
        "duration_seconds": report["duration_seconds"],
    }
    write_json(result_path, result)
    return result


def pending_activation_plan_ids(
    project_root: str | Path, config: dict[str, Any]
) -> list[str]:
    root = Path(project_root).resolve()
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    experiment = load_json(output_root / "experiment_manifest.json")
    pending = []
    for plan in experiment["plans"]:
        if not plan.get("runnable"):
            continue
        result_path = output_root / plan["plan_id"] / "activation_result.json"
        if not result_path.is_file() or load_json(result_path).get("status") != "completed":
            pending.append(plan["plan_id"])
    return pending


def audit_activation_reports(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = _activation_settings(config)
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    experiment = load_json(output_root / "experiment_manifest.json")
    reference_path = _activation_reference_dir(root, config) / "manifest.json"
    reference = load_json(reference_path)
    _validate_activation_reference(root, config, reference)
    invalid = []
    cosines = []
    for plan in experiment["plans"]:
        if not plan.get("runnable"):
            continue
        plan_dir = output_root / plan["plan_id"]
        report_path = plan_dir / settings["report_filename"]
        result_path = plan_dir / "activation_result.json"
        try:
            report = load_json(report_path)
            result = load_json(result_path)
            validate_activation_error_report(report)
            if report.get("plan") != plan:
                raise ValueError("activation report plan contract differs")
            if report.get("reference_manifest_sha256") != sha256_file(reference_path):
                raise ValueError("activation reference manifest SHA256 differs")
            if result.get("activation_error_sha256") != sha256_file(report_path):
                raise ValueError("activation result SHA256 differs")
            cosines.extend(float(layer["cosine_similarity"]) for layer in report["layers"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            invalid.append({"plan_id": plan["plan_id"], "reason": str(exc)})
    audit = {
        "schema_version": 1,
        "stage": "7.1.3",
        "status": "pass" if not invalid else "fail",
        "expected_runnable_plans": experiment["runnable_plan_count"],
        "valid_plans": experiment["runnable_plan_count"] - len(invalid),
        "invalid_plans": invalid,
        "layer_cosine_min": min(cosines) if cosines else None,
        "layer_cosine_max": max(cosines) if cosines else None,
    }
    write_json(output_root / "activation_audit.json", audit)
    return audit


def summarize_activation_reports(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = _activation_settings(config)
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    experiment = load_json(output_root / "experiment_manifest.json")
    reports = {}
    for plan in experiment["plans"]:
        if not plan.get("runnable"):
            continue
        path = output_root / plan["plan_id"] / settings["report_filename"]
        if path.is_file():
            report = load_json(path)
            validate_activation_error_report(report)
            reports[plan["plan_id"]] = report
    baseline = reports.get("all_int4")
    if baseline is None:
        raise ValueError("ALL-I4 activation report is required for summarization.")
    candidates = []
    for plan in experiment["plans"]:
        if plan.get("kind") != "layer_int8" or plan["plan_id"] not in reports:
            continue
        report = reports[plan["plan_id"]]
        layer_index = int(plan["layer_index"])
        candidates.append(
            {
                "layer_index": layer_index,
                "plan_id": plan["plan_id"],
                "aggregate_nmse": report["aggregate"]["nmse"],
                "aggregate_cosine_similarity": report["aggregate"]["cosine_similarity"],
                "aggregate_nmse_recovery_vs_all_int4": (
                    baseline["aggregate"]["nmse"] - report["aggregate"]["nmse"]
                ),
                "target_layer_nmse": report["layers"][layer_index]["nmse"],
                "target_layer_cosine_similarity": report["layers"][layer_index]["cosine_similarity"],
                "target_layer_nmse_recovery_vs_all_int4": (
                    baseline["layers"][layer_index]["nmse"]
                    - report["layers"][layer_index]["nmse"]
                ),
            }
        )
    candidates.sort(
        key=lambda row: (-row["aggregate_nmse_recovery_vs_all_int4"], row["layer_index"])
    )
    for rank, row in enumerate(candidates, start=1):
        row["activation_error_rank"] = rank
    summary = {
        "schema_version": 1,
        "stage": "7.1.3",
        "status": "complete" if len(reports) == experiment["runnable_plan_count"] else "partial",
        "completed_plan_count": len(reports),
        "expected_plan_count": experiment["runnable_plan_count"],
        "baseline": {
            "plan_id": "all_int4",
            "aggregate": baseline["aggregate"],
        },
        "all_int8": reports.get("all_int8", {}).get("aggregate"),
        "candidates": candidates,
        "interpretation": "Activation error is diagnostic evidence, not a replacement for paired business quality.",
    }
    write_json(output_root / "activation_summary.json", summary)
    return summary


def paired_prediction_transitions(
    baseline_rows: Iterable[dict[str, Any]], candidate_rows: Iterable[dict[str, Any]]
) -> dict[str, int]:
    baseline = list(baseline_rows)
    candidate = list(candidate_rows)
    if len(baseline) != len(candidate):
        raise ValueError("Paired prediction files have different row counts.")
    transitions = {
        "recovered": 0,
        "regressed": 0,
        "stable_correct": 0,
        "stable_wrong": 0,
        "changed_raw_predictions": 0,
    }
    for before, after in zip(baseline, candidate):
        if before.get("source_index") != after.get("source_index"):
            raise ValueError("Paired prediction files are not source-index aligned.")
        before_correct = bool(before.get("correct"))
        after_correct = bool(after.get("correct"))
        if before.get("raw_prediction") != after.get("raw_prediction"):
            transitions["changed_raw_predictions"] += 1
        if not before_correct and after_correct:
            transitions["recovered"] += 1
        elif before_correct and not after_correct:
            transitions["regressed"] += 1
        elif before_correct:
            transitions["stable_correct"] += 1
        else:
            transitions["stable_wrong"] += 1
    return transitions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _plan_predictions(plan_dir: Path) -> list[dict[str, Any]]:
    report = load_json(plan_dir / "comparison_report.json")
    variant = _structured_variant(report)
    return _read_jsonl(plan_dir / variant["predictions_file"])


def summarize_matrix(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    experiment = load_json(output_root / "experiment_manifest.json")
    pending = pending_plan_ids(root, config)
    results: dict[str, dict[str, Any]] = {}
    for plan in experiment["plans"]:
        if not plan.get("runnable"):
            continue
        result_path = output_root / plan["plan_id"] / "result.json"
        if result_path.is_file():
            results[plan["plan_id"]] = load_json(result_path)
    baseline = results.get("all_int4")
    if baseline is None or baseline.get("status") != "completed":
        raise ValueError("ALL-I4 must complete before matrix summarization.")
    baseline_rows = _plan_predictions(output_root / "all_int4")
    enriched = []
    for plan_id, result in results.items():
        if result.get("status") != "completed":
            continue
        item = copy.deepcopy(result)
        item["transitions"] = paired_prediction_transitions(
            baseline_rows, _plan_predictions(output_root / plan_id)
        )
        enriched.append(item)
        write_json(
            output_root / plan_id / "comparison_to_all_int4.json",
            {
                "schema_version": 1,
                "baseline": "all_int4",
                "candidate": plan_id,
                "transitions": item["transitions"],
                "baseline_metrics": baseline["metrics"],
                "candidate_metrics": item["metrics"],
            },
        )
    ranking = rank_completed_results(enriched, baseline)
    summary = {
        "schema_version": 1,
        "stage": "7.1.2",
        "status": "complete" if not pending else "partial",
        "backend": config["backend"],
        "deployment_performance_claims_allowed": False,
        "completed_runnable_plans": len(results),
        "expected_runnable_plans": experiment["runnable_plan_count"],
        "pending": pending,
        "all_int4": baseline,
        "all_int8": results.get("all_int8"),
        "ranking_preview": ranking,
        "note": (
            "This is a pilot sensitivity ranking from dequantized fake weights, "
            "not a deployment memory, latency, or throughput result."
        ),
    }
    write_json(output_root / "matrix_summary.json", summary)
    write_json(output_root / "ranking_preview.json", {"ranking": ranking})
    return summary


def rank_completed_results(
    results: Iterable[dict[str, Any]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    baseline_correct = int(baseline["metrics"]["correct"])
    baseline_nmse = float(baseline.get("weighted_nmse", 0.0))
    ranked = []
    for result in results:
        if result.get("status") != "completed":
            continue
        plan = result.get("plan", {})
        if plan.get("kind") != "layer_int8":
            continue
        transitions = result.get("transitions", {})
        recovered = int(transitions.get("recovered", 0))
        regressed = int(transitions.get("regressed", 0))
        item = copy.deepcopy(result)
        item["recovered_count"] = recovered
        item["regressed_count"] = regressed
        item["net_correct_gain"] = recovered - regressed
        item["correct_gain"] = int(result["metrics"]["correct"]) - baseline_correct
        item["quantization_error_recovery"] = baseline_nmse - float(
            result.get("weighted_nmse", baseline_nmse)
        )
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -item["net_correct_gain"],
            -item["recovered_count"],
            -float(item["metrics"]["numeric_accuracy"]),
            -float(item["quantization_error_recovery"]),
            int(item["plan"]["layer_index"]),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked
