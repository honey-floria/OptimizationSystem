"""Build and run the TODO 7.1.2 layer-sensitivity experiment matrix."""

from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from src.quantization.layer_sensitivity_setup import (
    canonical_json_sha256,
    discover_transformer_blocks,
    load_json,
    resolve_project_path,
    sha256_file,
    write_json,
)


REFERENCE_KINDS = {"ref_fp", "ref_nf4"}
RUNNABLE_KINDS = {"all_int4", "all_int8", "layer_int8"}


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
    difference = measured - source
    source_power = torch_module.mean(source * source).item()
    mse = torch_module.mean(difference * difference).item()
    source_flat = source.reshape(-1)
    restored_flat = measured.reshape(-1)
    denominator = torch_module.linalg.vector_norm(source_flat) * torch_module.linalg.vector_norm(restored_flat)
    cosine = (
        float(torch_module.dot(source_flat, restored_flat).item() / denominator.item())
        if denominator.item() > 0
        else 1.0
    )
    quantized_unpadded = quantized.reshape(rows, padded_columns)[:, :columns]
    saturation = float(
        (quantized_unpadded.abs() == qmax).to(torch_module.float32).mean().item()
    )
    metrics = {
        "mse": float(mse),
        "nmse": float(mse / source_power) if source_power > 0 else 0.0,
        "cosine_similarity": cosine,
        "max_abs_error": float(difference.abs().max().item()),
        "saturation_rate": saturation,
    }
    return stored, metrics


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
    return {
        "block_path": block_path,
        "block_count": len(blocks),
        "plan_id": plan["plan_id"],
        "backend": "symmetric_fake_quant",
        "deployment_performance_claims_allowed": False,
        "layers": layer_records,
        "modules": module_records,
        "aggregate": aggregate_error_metrics(module_records),
    }


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
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(evaluation["dtype"])
    if dtype is None:
        raise ValueError("Evaluation dtype must be float16 or bfloat16.")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    error_report = quantize_model_for_plan(
        model, plan, config["precision"], torch
    )
    if error_report["block_path"] != experiment["block_path"] or error_report["block_count"] != experiment["block_count"]:
        raise ValueError("Runtime block discovery differs from the prepared matrix.")
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
        "weighted_nmse": error_report["aggregate"]["nmse"],
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    return result


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
