"""TODO 7.2.2/7.2.3 controlled module and special-component experiments."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.quantization.layer_sensitivity import (
    ERROR_METRICS_SCHEMA_VERSION,
    _fake_quantize_weight,
    _stable_error_statistics,
    aggregate_error_metrics,
    paired_prediction_transitions,
    validate_quantization_error_report,
)
from src.quantization.layer_sensitivity_setup import (
    canonical_json_sha256,
    load_json,
    resolve_project_path,
    sha256_file,
    write_json,
)
from src.quantization.module_sensitivity import (
    ATTENTION_MODULES,
    BLOCK_NORMS,
    MLP_MODULES,
    validate_upstream_contract,
)


TERMINAL_RESULT_STATUSES = {"completed", "unsupported"}
RECOVERY_KINDS = {
    "attention_all_int8",
    "mlp_all_int8",
    "combined_all_int8_audit",
    "layer_attention_int8",
    "layer_mlp_int8",
    "candidate_mixed_int8",
}
SPECIAL_WEIGHT_KINDS = {
    "embedding_quantized",
    "lm_head_quantized",
    "shared_embedding_lm_head_quantized",
    "block_norm_quantized",
    "final_norm_quantized",
}


def build_experiment_plans(
    block_count: int,
    embedding_lm_head_tied: bool,
    candidate_recovery_plans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if block_count <= 0:
        raise ValueError("block_count must be positive.")
    plans: list[dict[str, Any]] = [
        {
            "plan_id": "all_int4_ref",
            "kind": "all_int4_reference",
            "runnable": False,
            "comparison": "reference",
        },
        {
            "plan_id": "attn_all_int8",
            "kind": "attention_all_int8",
            "runnable": True,
            "comparison": "recovery",
        },
        {
            "plan_id": "mlp_all_int8",
            "kind": "mlp_all_int8",
            "runnable": True,
            "comparison": "recovery",
        },
        {
            "plan_id": "attn_mlp_all_int8_audit",
            "kind": "combined_all_int8_audit",
            "runnable": True,
            "comparison": "implementation_audit",
        },
    ]
    for index in range(block_count):
        plans.extend(
            [
                {
                    "plan_id": f"layer_{index:02d}_attn_int8",
                    "kind": "layer_attention_int8",
                    "runnable": True,
                    "comparison": "recovery",
                    "layer_index": index,
                },
                {
                    "plan_id": f"layer_{index:02d}_mlp_int8",
                    "kind": "layer_mlp_int8",
                    "runnable": True,
                    "comparison": "recovery",
                    "layer_index": index,
                },
            ]
        )
    for candidate in candidate_recovery_plans or []:
        plan_id = candidate.get("plan_id")
        attention_layers = candidate.get("int8_attention_layers")
        mlp_layers = candidate.get("int8_mlp_layers")
        if not isinstance(plan_id, str) or not plan_id.startswith("candidate_"):
            raise ValueError("Candidate recovery plan IDs must start with candidate_.")
        if not isinstance(attention_layers, list) or not isinstance(mlp_layers, list):
            raise ValueError("Candidate recovery plans must list INT8 Attention and MLP layers.")
        if not attention_layers and not mlp_layers:
            raise ValueError("Candidate recovery plans must change at least one module family.")
        for field, indices in (
            ("int8_attention_layers", attention_layers),
            ("int8_mlp_layers", mlp_layers),
        ):
            if (
                any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
                or len(indices) != len(set(indices))
                or any(index < 0 or index >= block_count for index in indices)
            ):
                raise ValueError(f"Candidate recovery plan has invalid {field}.")
        plans.append(
            {
                "plan_id": plan_id,
                "kind": "candidate_mixed_int8",
                "runnable": True,
                "comparison": "recovery",
                "int8_attention_layers": sorted(attention_layers),
                "int8_mlp_layers": sorted(mlp_layers),
                "selection_basis": candidate.get("selection_basis"),
            }
        )
    if embedding_lm_head_tied:
        for precision in ("int8", "int4"):
            plans.append(
                {
                    "plan_id": f"shared_embedding_lm_head_{precision}",
                    "kind": "shared_embedding_lm_head_quantized",
                    "runnable": True,
                    "comparison": "degradation",
                    "target_precision": precision,
                }
            )
    else:
        for target in ("embedding", "lm_head"):
            for precision in ("int8", "int4"):
                plans.append(
                    {
                        "plan_id": f"{target}_{precision}",
                        "kind": f"{target}_quantized",
                        "runnable": True,
                        "comparison": "degradation",
                        "target_precision": precision,
                    }
                )
    for target in ("block_norm", "final_norm"):
        for precision in ("int8", "int4"):
            plans.append(
                {
                    "plan_id": f"{target}_{precision}",
                    "kind": f"{target}_quantized",
                    "runnable": True,
                    "comparison": "degradation",
                    "target_precision": precision,
                }
            )
    plans.extend(
        [
            {
                "plan_id": "kv_cache_fp16_ref",
                "kind": "kv_cache_reference",
                "runnable": True,
                "comparison": "reference",
                "target_precision": "fp16",
            },
            {
                "plan_id": "kv_cache_int8",
                "kind": "kv_cache_quantized",
                "runnable": True,
                "comparison": "degradation",
                "target_precision": "int8",
            },
            {
                "plan_id": "kv_cache_int4",
                "kind": "kv_cache_quantized",
                "runnable": True,
                "comparison": "degradation",
                "target_precision": "int4",
            },
        ]
    )
    validate_experiment_plans(plans, block_count, embedding_lm_head_tied)
    return plans


def validate_experiment_plans(
    plans: list[dict[str, Any]], block_count: int, embedding_lm_head_tied: bool
) -> None:
    ids = [plan.get("plan_id") for plan in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("Module-sensitivity plan IDs must be unique.")
    for kind in ("layer_attention_int8", "layer_mlp_int8"):
        indices = sorted(
            int(plan["layer_index"]) for plan in plans if plan.get("kind") == kind
        )
        if indices != list(range(block_count)):
            raise ValueError(f"{kind} plans must cover every block exactly once.")
    independent = {
        plan["kind"] for plan in plans
        if plan.get("kind") in {"embedding_quantized", "lm_head_quantized"}
    }
    joint = [plan for plan in plans if plan.get("kind") == "shared_embedding_lm_head_quantized"]
    if embedding_lm_head_tied:
        if independent or len(joint) != 2:
            raise ValueError("Tied embedding/lm_head weights require exactly two joint plans.")
    elif joint or independent != {"embedding_quantized", "lm_head_quantized"}:
        raise ValueError("Untied embedding/lm_head weights require independent plans.")


def linear_precision_for(plan: dict[str, Any], layer_index: int, family: str) -> str:
    kind = plan["kind"]
    if kind == "attention_all_int8" and family == "attention":
        return "int8"
    if kind == "mlp_all_int8" and family == "mlp":
        return "int8"
    if kind == "combined_all_int8_audit":
        return "int8"
    if kind == "layer_attention_int8" and family == "attention" and plan["layer_index"] == layer_index:
        return "int8"
    if kind == "layer_mlp_int8" and family == "mlp" and plan["layer_index"] == layer_index:
        return "int8"
    if kind == "candidate_mixed_int8":
        target_layers = (
            plan["int8_attention_layers"]
            if family == "attention"
            else plan["int8_mlp_layers"]
        )
        if layer_index in target_layers:
            return "int8"
    return "int4"


def effective_weight_assignment(
    plan: dict[str, Any], block_count: int, embedding_lm_head_tied: bool
) -> dict[str, str]:
    assignment = {}
    for index in range(block_count):
        for relative in ATTENTION_MODULES:
            assignment[f"model.layers.{index}.{relative}"] = linear_precision_for(
                plan, index, "attention"
            )
        for relative in MLP_MODULES:
            assignment[f"model.layers.{index}.{relative}"] = linear_precision_for(
                plan, index, "mlp"
            )
    assignment.update(
        {
            "embedding": "fp16",
            "lm_head": "fp16",
            "block_norm": "fp16",
            "final_norm": "fp16",
            "kv_cache": "fp16",
        }
    )
    kind = plan["kind"]
    target = plan.get("target_precision")
    if kind == "embedding_quantized":
        assignment["embedding"] = target
    elif kind == "lm_head_quantized":
        assignment["lm_head"] = target
    elif kind == "shared_embedding_lm_head_quantized":
        if not embedding_lm_head_tied:
            raise ValueError("Joint embedding/lm_head plan requires tied weights.")
        assignment["embedding"] = target
        assignment["lm_head"] = target
    elif kind == "block_norm_quantized":
        assignment["block_norm"] = target
    elif kind == "final_norm_quantized":
        assignment["final_norm"] = target
    elif kind == "kv_cache_quantized":
        assignment["kv_cache"] = target
    return assignment


def validate_plan_isolation(
    plans: list[dict[str, Any]], block_count: int, embedding_lm_head_tied: bool
) -> None:
    baseline_plan = next(plan for plan in plans if plan["kind"] == "all_int4_reference")
    baseline = effective_weight_assignment(baseline_plan, block_count, embedding_lm_head_tied)
    for plan in plans:
        if plan["kind"] == "all_int4_reference":
            continue
        assignment = effective_weight_assignment(plan, block_count, embedding_lm_head_tied)
        changed = {key for key in baseline if baseline[key] != assignment[key]}
        if plan["kind"] == "layer_attention_int8" and len(changed) != len(ATTENTION_MODULES):
            raise ValueError(f"{plan['plan_id']} does not isolate one Attention group.")
        if plan["kind"] == "layer_mlp_int8" and len(changed) != len(MLP_MODULES):
            raise ValueError(f"{plan['plan_id']} does not isolate one MLP group.")
        if plan["kind"] == "candidate_mixed_int8":
            expected_count = (
                len(plan["int8_attention_layers"]) * len(ATTENTION_MODULES)
                + len(plan["int8_mlp_layers"]) * len(MLP_MODULES)
            )
            if len(changed) != expected_count:
                raise ValueError(f"{plan['plan_id']} does not match its candidate module set.")
        if plan["kind"] in SPECIAL_WEIGHT_KINDS | {"kv_cache_quantized"}:
            expected_count = 2 if plan["kind"] == "shared_embedding_lm_head_quantized" else 1
            if len(changed) != expected_count:
                raise ValueError(f"{plan['plan_id']} changes an unexpected component set.")


def _read_model_path(root: Any, path: str) -> Any:
    current = root
    for name in path.split("."):
        if name.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(name)]
        elif hasattr(current, name):
            current = getattr(current, name)
        else:
            raise ValueError(f"Runtime module path is missing: {path}")
    return current


def _fake_quantize_vector(
    weight: Any, *, bits: int, torch_module: Any
) -> tuple[Any, dict[str, float]]:
    if bits not in {4, 8}:
        raise ValueError("Norm fake quantization supports only 4 or 8 bits.")
    source = weight.detach().to(dtype=torch_module.float32, device="cpu")
    if source.ndim != 1:
        raise ValueError("Norm quantization requires a 1-D weight.")
    qmax = (1 << (bits - 1)) - 1
    absmax = source.abs().max()
    scale = absmax / qmax if float(absmax.item()) > 0 else torch_module.ones_like(absmax)
    quantized = torch_module.round(source / scale).clamp(-qmax, qmax)
    stored = (quantized * scale).to(dtype=weight.dtype)
    measured = stored.to(dtype=torch_module.float32)
    statistics = _stable_error_statistics(source, measured, torch_module=torch_module)
    saturation = float((quantized.abs() == qmax).to(torch_module.float32).mean().item())
    return stored, {
        "mse": statistics["mse"],
        "nmse": statistics["mse"] / statistics["source_power"] if statistics["source_power"] > 0 else 0.0,
        "cosine_similarity": statistics["cosine_similarity"],
        "max_abs_error": statistics["max_abs_error"],
        "saturation_rate": saturation,
    }


def _quantize_module(
    module: Any,
    *,
    path: str,
    family: str,
    precision_name: str,
    contract: dict[str, Any],
    torch_module: Any,
    layer_index: int | None = None,
) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(f"Quantized module has no weight: {path}")
    bits = 4 if precision_name == "int4" else 8
    rule = contract[precision_name]
    restored, metrics = _fake_quantize_weight(
        weight.data,
        bits=bits,
        group_size=rule.get("group_size"),
        torch_module=torch_module,
    )
    weight.data.copy_(restored)
    record = {
        "module": path,
        "family": family,
        "precision": precision_name,
        "bits": bits,
        "group_size": rule.get("group_size"),
        "numel": int(weight.numel()),
        **metrics,
    }
    if layer_index is not None:
        record["layer_index"] = layer_index
    return record


def _quantize_special_module(
    module: Any,
    *,
    path: str,
    family: str,
    precision_name: str,
    rules: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(f"Special module has no weight: {path}")
    bits = 4 if precision_name == "int4" else 8
    rule = rules[precision_name]
    source = weight.detach().to(dtype=torch_module.float32, device="cpu")
    finite = torch_module.isfinite(source)
    source_min = float(source[finite].min().item()) if bool(finite.any()) else None
    source_max = float(source[finite].max().item()) if bool(finite.any()) else None
    if weight.ndim == 1:
        restored, metrics = _fake_quantize_vector(weight.data, bits=bits, torch_module=torch_module)
    elif weight.ndim == 2:
        restored, metrics = _fake_quantize_weight(
            weight.data,
            bits=bits,
            group_size=rule.get("group_size"),
            torch_module=torch_module,
        )
    else:
        raise ValueError(f"Unsupported special-module weight rank at {path}: {weight.ndim}")
    weight.data.copy_(restored)
    if weight.ndim == 1:
        scale_count = 1
    elif rule.get("group_size"):
        scale_count = int(weight.shape[0]) * math.ceil(
            int(weight.shape[1]) / int(rule["group_size"])
        )
    else:
        scale_count = int(weight.shape[0])
    packed_weight_bytes = math.ceil(int(weight.numel()) * bits / 8)
    scale_bytes = scale_count * 2
    return {
        "module": path,
        "family": family,
        "precision": precision_name,
        "bits": bits,
        "group_size": rule.get("group_size"),
        "axis": rule.get("axis"),
        "scheme": rule["scheme"],
        "zero_point": rule["zero_point"],
        "numel": int(weight.numel()),
        "source_min": source_min,
        "source_max": source_max,
        "source_nan_count": int(torch_module.isnan(source).sum().item()),
        "source_inf_count": int(torch_module.isinf(source).sum().item()),
        "quantized_nan_count": int(torch_module.isnan(restored).sum().item()),
        "quantized_inf_count": int(torch_module.isinf(restored).sum().item()),
        "theoretical_storage": {
            "packed_weight_bytes": packed_weight_bytes,
            "scale_count": scale_count,
            "scale_bytes_fp16": scale_bytes,
            "total_bytes": packed_weight_bytes + scale_bytes,
            "fp16_reference_bytes": int(weight.numel()) * 2,
            "savings_bytes": int(weight.numel()) * 2 - packed_weight_bytes - scale_bytes,
        },
        **metrics,
    }


def _tensor_storage_profile(value: Any, torch_module: Any) -> dict[str, Any]:
    seen_tensors: set[int] = set()
    seen_objects: set[int] = set()
    dtype_bytes: dict[str, int] = {}

    def visit(item: Any) -> None:
        if torch_module.is_tensor(item):
            identity = int(item.data_ptr()) if item.numel() else id(item)
            if identity in seen_tensors:
                return
            seen_tensors.add(identity)
            dtype = str(item.dtype).replace("torch.", "")
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + int(item.numel()) * int(item.element_size())
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        attributes = getattr(item, "__dict__", None)
        if isinstance(attributes, dict):
            identity = id(item)
            if identity in seen_objects:
                return
            seen_objects.add(identity)
            for child in attributes.values():
                visit(child)

    visit(value)
    return {
        "total_bytes": sum(dtype_bytes.values()),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "tensor_count": len(seen_tensors),
    }


def benchmark_kv_cache_runtime(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    generation_arguments: dict[str, Any] | None,
    config: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    settings = config["kv_cache"].get("benchmark", {})
    device = config["pilot_evaluation"]["device"]
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(settings.get("max_input_tokens", 1024)),
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    base_arguments = {
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "use_cache": True,
    }
    base_arguments.update(generation_arguments or {})

    def timed_generate(tokens: int) -> tuple[Any, float, int | None]:
        arguments = {**base_arguments, "max_new_tokens": tokens}
        if device.startswith("cuda"):
            torch_module.cuda.empty_cache()
            torch_module.cuda.reset_peak_memory_stats()
            torch_module.cuda.synchronize()
        started = time.perf_counter()
        with torch_module.no_grad():
            output = model.generate(**encoded, **arguments)
        if device.startswith("cuda"):
            torch_module.cuda.synchronize()
            peak = int(torch_module.cuda.max_memory_allocated())
        else:
            peak = None
        return output, (time.perf_counter() - started) * 1000.0, peak

    warmup_tokens = int(settings.get("warmup_new_tokens", 2))
    if warmup_tokens:
        timed_generate(warmup_tokens)
    _, ttft_ms, first_peak = timed_generate(1)
    decode_tokens = int(settings.get("decode_new_tokens", 32))
    full, total_ms, full_peak = timed_generate(decode_tokens)
    cache = getattr(full, "past_key_values", None)
    cache_class = type(cache).__name__ if cache is not None else None
    storage = (
        _tensor_storage_profile(cache, torch_module)
        if cache is not None
        else {"total_bytes": 0, "dtype_bytes": {}, "tensor_count": 0}
    )
    cache_bytes = int(storage["total_bytes"])
    quantized = bool(generation_arguments)
    requested_bits = (
        (generation_arguments or {}).get("cache_config", {}).get("nbits")
        if quantized else 16
    )
    integer_cache_bytes = sum(
        byte_count
        for dtype, byte_count in storage["dtype_bytes"].items()
        if dtype.startswith(("int", "uint"))
    )
    low_bit_residency = bool(
        quantized
        and cache_bytes > 0
        and cache_class
        and ("quant" in cache_class.lower() or "hqq" in cache_class.lower())
        and requested_bits in {4, 8}
        and integer_cache_bytes > 0
    )
    total_cache_tokens = int(encoded["attention_mask"].sum().item()) + decode_tokens
    return {
        "status": "completed",
        "method": "generate_one_token_plus_fixed_decode",
        "input_tokens": int(encoded["attention_mask"].sum().item()),
        "generated_tokens": decode_tokens,
        "ttft_ms": ttft_ms,
        "end_to_end_ms": total_ms,
        "tpot_ms": (total_ms - ttft_ms) / max(decode_tokens - 1, 1),
        "throughput_tokens_per_second": decode_tokens / (total_ms / 1000.0),
        "peak_memory_bytes": full_peak,
        "first_token_peak_memory_bytes": first_peak,
        "cache_class": cache_class,
        "cache_tensor_bytes": cache_bytes,
        "cache_bytes_per_token": cache_bytes / total_cache_tokens if total_cache_tokens else None,
        "cache_tensor_count": storage["tensor_count"],
        "cache_dtype_bytes": storage["dtype_bytes"],
        "integer_cache_bytes": integer_cache_bytes,
        "requested_bits": requested_bits,
        "low_bit_residency_verified": low_bit_residency,
        "generation_arguments": generation_arguments,
        "performance_claims_allowed": (not quantized) or low_bit_residency,
    }


def _numeric_token_ids(tokenizer: Any) -> list[int]:
    numeric = re.compile(r"^[\s▁Ġ]*[-+]?[$€£]?[0-9][0-9,._%]*$")
    return sorted(
        int(index)
        for token, index in tokenizer.get_vocab().items()
        if numeric.match(str(token))
    )


def _chunked_logits(hidden: Any, weight: Any, torch_module: Any, chunk_size: int) -> Any:
    pieces = []
    vector = hidden.detach().to(device="cpu", dtype=torch_module.float32)
    for start in range(0, int(weight.shape[0]), chunk_size):
        chunk = weight[start : start + chunk_size].to(dtype=torch_module.float32)
        pieces.append(torch_module.mv(chunk, vector))
    return torch_module.cat(pieces)


def probe_lm_head_logits(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    original_weight: Any,
    config: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    from src.baseline.service import InferenceRequest

    settings = config.get("special_probes", {}).get("lm_head", {})
    sample_count = min(int(settings.get("sample_count", 4)), len(rows))
    suffix = str(settings.get("prompt_suffix", ""))
    numeric_ids = _numeric_token_ids(tokenizer)
    records = []
    target = getattr(model, "model", None)
    if target is None:
        raise ValueError("lm_head probe requires the decoder body at model.model.")
    candidate_weight = model.lm_head.weight.detach().to(device="cpu")
    for index, row in enumerate(rows[:sample_count]):
        prompt = InferenceRequest.from_finqa_row(row).render_prompt() + suffix
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(settings.get("max_input_tokens", 1024)),
        )
        encoded = {name: tensor.to(config["pilot_evaluation"]["device"]) for name, tensor in encoded.items()}
        with torch_module.no_grad():
            output = target(**encoded, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[0, -1].to(device="cpu")
        chunk_size = int(settings.get("vocab_chunk_size", 4096))
        reference_logits = _chunked_logits(hidden, original_weight, torch_module, chunk_size)
        candidate_logits = _chunked_logits(hidden, candidate_weight, torch_module, chunk_size)
        delta = candidate_logits - reference_logits
        reference_top = int(reference_logits.argmax().item())
        top_k = int(settings.get("top_k", 5))
        reference_topk = set(reference_logits.topk(top_k).indices.tolist())
        candidate_topk = set(candidate_logits.topk(top_k).indices.tolist())
        numeric_rank_change = None
        if numeric_ids:
            numeric_tensor = torch_module.tensor(numeric_ids, dtype=torch_module.long)
            best_numeric = int(numeric_tensor[reference_logits[numeric_tensor].argmax()].item())
            reference_rank = int((reference_logits > reference_logits[best_numeric]).sum().item()) + 1
            candidate_rank = int((candidate_logits > candidate_logits[best_numeric]).sum().item()) + 1
            numeric_rank_change = candidate_rank - reference_rank
        records.append(
            {
                "sample_index": index,
                "logits_mse": float((delta * delta).mean().item()),
                "top1_consistent": reference_top == int(candidate_logits.argmax().item()),
                "topk_overlap": len(reference_topk & candidate_topk) / top_k,
                "reference_top_numeric_rank_change": numeric_rank_change,
            }
        )
    return {
        "status": "completed",
        "sample_count": len(records),
        "numeric_token_count": len(numeric_ids),
        "mean_logits_mse": sum(row["logits_mse"] for row in records) / len(records),
        "top1_consistency": sum(row["top1_consistent"] for row in records) / len(records),
        "mean_topk_overlap": sum(row["topk_overlap"] for row in records) / len(records),
        "mean_abs_numeric_rank_change": (
            sum(abs(row["reference_top_numeric_rank_change"]) for row in records if row["reference_top_numeric_rank_change"] is not None)
            / max(sum(row["reference_top_numeric_rank_change"] is not None for row in records), 1)
        ),
        "records": records,
    }


def quantize_model_for_module_plan(
    model: Any,
    plan: dict[str, Any],
    config: dict[str, Any],
    inventory: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    module_records = []
    layer_records = []
    block_path = inventory["block_path"]
    for index in range(inventory["block_count"]):
        layer_modules = []
        for family, relatives in (("attention", ATTENTION_MODULES), ("mlp", MLP_MODULES)):
            precision_name = linear_precision_for(plan, index, family)
            contract = config["precision"][f"linear_{precision_name}"]
            for relative in relatives:
                path = f"{block_path}.{index}.{relative}"
                record = _quantize_module(
                    _read_model_path(model, path),
                    path=path,
                    family=family,
                    precision_name=precision_name,
                    contract={precision_name: contract},
                    torch_module=torch_module,
                    layer_index=index,
                )
                layer_modules.append(record)
                module_records.append(record)
        layer_records.append({"layer_index": index, **aggregate_error_metrics(layer_modules)})

    special_records = []
    kind = plan["kind"]
    precision_name = plan.get("target_precision")
    inventory_groups = inventory["module_groups"]
    if kind in {"embedding_quantized", "shared_embedding_lm_head_quantized"}:
        record = inventory_groups["embedding"][0]
        special_records.append(
            _quantize_special_module(
                _read_model_path(model, record["path"]),
                path=record["path"], family="embedding", precision_name=precision_name,
                rules=config["precision"]["embedding"], torch_module=torch_module,
            )
        )
    if kind == "lm_head_quantized":
        record = inventory_groups["lm_head"][0]
        special_records.append(
            _quantize_special_module(
                _read_model_path(model, record["path"]),
                path=record["path"], family="lm_head", precision_name=precision_name,
                rules=config["precision"]["lm_head"], torch_module=torch_module,
            )
        )
    if kind == "shared_embedding_lm_head_quantized":
        special_records[0]["family"] = "shared_embedding_lm_head"
        special_records[0]["aliases"] = [
            inventory_groups["embedding"][0]["path"],
            inventory_groups["lm_head"][0]["path"],
        ]
    if kind in {"block_norm_quantized", "final_norm_quantized"}:
        role = "block_norm" if kind == "block_norm_quantized" else "final_norm"
        for record in inventory_groups["norm"]:
            if record["role"] != role:
                continue
            special_records.append(
                _quantize_special_module(
                    _read_model_path(model, record["path"]),
                    path=record["path"], family=role, precision_name=precision_name,
                    rules=config["precision"]["norm"], torch_module=torch_module,
                )
            )
        if not special_records:
            raise ValueError(f"No modules matched special plan role: {role}")
    module_records.extend(special_records)
    report = {
        "error_metrics_schema_version": ERROR_METRICS_SCHEMA_VERSION,
        "stage": "7.2.2-7.2.3",
        "plan_id": plan["plan_id"],
        "backend": "symmetric_fake_quant",
        "deployment_performance_claims_allowed": False,
        "block_path": block_path,
        "block_count": inventory["block_count"],
        "layers": layer_records,
        "modules": module_records,
        "special_modules": special_records,
        "aggregate": aggregate_error_metrics(module_records),
    }
    validate_quantization_error_report(report)
    return report


def audit_combined_all_int8_equivalence(
    module_report: dict[str, Any], layer_report: dict[str, Any]
) -> dict[str, Any]:
    left = {
        (int(row["layer_index"]), str(row["module"]).split(f"model.layers.{row['layer_index']}.", 1)[-1]): row
        for row in module_report["modules"]
        if row.get("family") in {"attention", "mlp"}
    }
    right = {
        (int(row["layer_index"]), str(row["module"])): row
        for row in layer_report["modules"]
    }
    if set(left) != set(right):
        raise ValueError("Combined Attention+MLP INT8 modules differ from 7.1 ALL-I8.")
    fields = ("bits", "group_size", "numel", "mse", "nmse", "cosine_similarity", "max_abs_error", "saturation_rate")
    mismatches = []
    for key in sorted(left):
        for field in fields:
            if left[key].get(field) != right[key].get(field):
                mismatches.append({"layer_index": key[0], "module": key[1], "field": field})
    if mismatches:
        raise ValueError(f"Combined Attention+MLP INT8 does not reproduce ALL-I8: {mismatches[:3]}")
    return {
        "status": "pass",
        "matched_module_count": len(left),
        "reference_plan_id": layer_report.get("plan_id"),
        "comparison_fields": list(fields),
    }


def _setup_paths(root: Path, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    setup_root = resolve_project_path(root, config["paths"]["output_root"])
    return (
        setup_root / "contract.json",
        setup_root / "module_inventory.json",
        setup_root / "validation.json",
    )


def prepare_experiment_matrix(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    validate_upstream_contract(root, config)
    contract_path, inventory_path, validation_path = _setup_paths(root, config)
    for path in (contract_path, inventory_path, validation_path):
        if not path.is_file():
            raise ValueError(f"Run 7.2.1 inspect-model before preparing experiments: {path}")
    contract = load_json(contract_path)
    inventory = load_json(inventory_path)
    validation = load_json(validation_path)
    if validation.get("status") != "pass":
        raise ValueError("7.2.1 model inspection has not passed.")
    if validation.get("contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("7.2.1 validation references a different contract.")
    if validation.get("module_inventory_sha256") != sha256_file(inventory_path):
        raise ValueError("7.2.1 module inventory changed after validation.")
    all_int8_error_path = resolve_project_path(root, config["paths"]["all_int8_error"])
    if not all_int8_error_path.is_file():
        raise ValueError("The 7.1 ALL-I8 quantization-error reference is missing.")
    all_int8_error = load_json(all_int8_error_path)
    if (
        all_int8_error.get("plan_id") != "all_int8"
        or all_int8_error.get("block_count") != inventory.get("block_count")
    ):
        raise ValueError("The 7.1 ALL-I8 quantization-error reference is incompatible.")
    tied = bool(inventory["embedding_lm_head_tied"])
    block_count = int(inventory["block_count"])
    plans = build_experiment_plans(
        block_count,
        tied,
        config.get("candidate_recovery_plans", []),
    )
    validate_plan_isolation(plans, block_count, tied)
    output_root = resolve_project_path(root, config["paths"]["experiment_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        plan_dir = output_root / plan["plan_id"]
        plan_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "stage": "7.2.2-7.2.3",
            "plan": plan,
            "effective_weight_assignment": effective_weight_assignment(plan, block_count, tied),
            "precision": config["precision"],
            "parent_all_int4": contract["upstream"]["all_int4_reference"],
        }
        payload["plan_sha256"] = canonical_json_sha256(payload)
        write_json(plan_dir / "plan.json", payload)
    manifest = {
        "schema_version": 1,
        "stage": "7.2.2-7.2.3",
        "status": "matrix_prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract["contract_sha256"],
        "module_inventory_sha256": sha256_file(inventory_path),
        "embedding_lm_head_tied": tied,
        "block_count": block_count,
        "plan_count": len(plans),
        "runnable_plan_count": sum(bool(plan["runnable"]) for plan in plans),
        "plans": plans,
        "all_int4_reference": contract["upstream"]["all_int4_reference"],
        "all_int4_report": config["paths"]["all_int4_report"],
        "all_int8_error": {
            "path": str(all_int8_error_path),
            "sha256": sha256_file(all_int8_error_path),
        },
        "fake_quant_performance_claims_allowed": False,
    }
    manifest["matrix_sha256"] = canonical_json_sha256(plans)
    write_json(output_root / "experiment_manifest.json", manifest)
    return manifest


def _load_prepared_plan(
    root: Path, config: dict[str, Any], plan_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    output_root = resolve_project_path(root, config["paths"]["experiment_output_root"])
    manifest = load_json(output_root / "experiment_manifest.json")
    if manifest.get("status") != "matrix_prepared":
        raise ValueError("Module-sensitivity experiment matrix is not prepared.")
    matches = [plan for plan in manifest["plans"] if plan["plan_id"] == plan_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown module-sensitivity plan: {plan_id}")
    plan = matches[0]
    if not plan.get("runnable"):
        raise ValueError(f"Reference plan is not runnable: {plan_id}")
    plan_dir = output_root / plan_id
    payload = load_json(plan_dir / "plan.json")
    stored_hash = payload.get("plan_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if canonical_json_sha256(unhashed) != stored_hash or payload.get("plan") != plan:
        raise ValueError(f"Plan contract changed after preparation: {plan_id}")
    setup_root = resolve_project_path(root, config["paths"]["output_root"])
    inventory = load_json(setup_root / "module_inventory.json")
    if sha256_file(setup_root / "module_inventory.json") != manifest["module_inventory_sha256"]:
        raise ValueError("Module inventory changed after matrix preparation.")
    return plan, manifest, inventory, plan_dir


def pending_plan_ids(project_root: str | Path, config: dict[str, Any]) -> list[str]:
    root = Path(project_root).resolve()
    output_root = resolve_project_path(root, config["paths"]["experiment_output_root"])
    manifest = load_json(output_root / "experiment_manifest.json")
    pending = []
    for plan in manifest["plans"]:
        if not plan.get("runnable"):
            continue
        result_path = output_root / plan["plan_id"] / "result.json"
        if not result_path.is_file() or load_json(result_path).get("status") not in TERMINAL_RESULT_STATUSES:
            pending.append(plan["plan_id"])
    return pending


def _structured_variant(report: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in report.get("variants", []) if row.get("variant") == "structured_json"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one structured_json result.")
    return matches[0]


def _prediction_rows(report_path: Path) -> list[dict[str, Any]]:
    report = load_json(report_path)
    variant = _structured_variant(report)
    path = report_path.parent / variant["predictions_file"]
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if sha256_file(path) != variant.get("predictions_sha256"):
        raise ValueError(f"Prediction file hash mismatch: {path}")
    return rows


def _kv_cache_runtime_arguments(
    plan: dict[str, Any], config: dict[str, Any], capability: dict[str, Any]
) -> dict[str, Any] | None:
    if plan["kind"] != "kv_cache_quantized":
        return None
    precision = plan["target_precision"]
    backend = config["kv_cache"]["real_backend"]
    if precision not in capability.get("runtime_verified_precisions", []):
        return None
    return backend.get("runtime_arguments", {}).get(precision)


def run_experiment_plan(
    project_root: str | Path,
    config: dict[str, Any],
    plan_id: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan, manifest, inventory, plan_dir = _load_prepared_plan(root, config, plan_id)
    result_path = plan_dir / "result.json"
    if result_path.is_file() and load_json(result_path).get("status") in TERMINAL_RESULT_STATUSES:
        return {**load_json(result_path), "resume_action": "skipped_terminal"}
    setup_root = resolve_project_path(root, config["paths"]["output_root"])
    capability = load_json(setup_root / "kv_cache_capability.json")
    kv_arguments = _kv_cache_runtime_arguments(plan, config, capability)
    if plan["kind"] == "kv_cache_quantized" and kv_arguments is None:
        result = {
            "schema_version": 1,
            "stage": "7.2.3",
            "status": "unsupported",
            "plan": plan,
            "reason": "Real KV Cache backend and runtime probe are not verified for this precision.",
            "capability_sha256": sha256_file(setup_root / "kv_cache_capability.json"),
            "deployment_performance_claims_allowed": False,
        }
        write_json(result_path, result)
        return result
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Module-sensitivity execution requires torch and transformers.") from exc
    from src.baseline.service import BaselineService
    from src.data.finqa_assets import load_assets
    from src.evaluation.quality_repair import run_quality_repair_experiment

    merged = load_json(resolve_project_path(root, config["paths"]["merged_manifest"]))
    resolved_model_path = Path(model_path or merged["merged_model"]["path"]).expanduser().resolve()
    if not resolved_model_path.is_dir():
        raise ValueError(f"Merged model directory is unavailable: {resolved_model_path}")
    evaluation = config["pilot_evaluation"]
    baseline = load_json(resolve_project_path(root, config["paths"]["baseline_config"]))
    baseline["device"] = evaluation["device"]
    baseline["dtype"] = evaluation["dtype"]
    baseline["generation"]["max_new_tokens"] = evaluation["max_new_tokens"]
    if kv_arguments:
        baseline["generation"].update(kv_arguments)
    quality = load_json(resolve_project_path(root, config["paths"]["quality_config"]))
    quality["evaluation_size"] = evaluation["pilot_size"]
    quality["evaluation_seed"] = evaluation["pilot_seed"]
    quality["batch_size"] = evaluation["batch_size"]
    quality["generation"]["max_new_tokens"] = evaluation["max_new_tokens"]
    quality["pilot_variants"] = [evaluation["variant"]]
    quality["dataset_role"] = evaluation["dataset_role"]
    write_json(plan_dir / "baseline_config.effective.json", baseline)
    write_json(plan_dir / "quality_config.effective.json", quality)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(evaluation["dtype"])
    if dtype is None:
        raise ValueError("pilot_evaluation.dtype must be float16 or bfloat16.")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model_path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_model_path), local_files_only=True, trust_remote_code=False,
        dtype=dtype, low_cpu_mem_usage=True,
    )
    original_lm_head_weight = None
    if plan["kind"] == "lm_head_quantized":
        original_lm_head_weight = model.lm_head.weight.detach().to(device="cpu").clone()
    error_report = quantize_model_for_module_plan(model, plan, config, inventory, torch)
    all_int8_equivalence = None
    if plan["kind"] == "combined_all_int8_audit":
        all_int8_equivalence = audit_combined_all_int8_equivalence(
            error_report,
            load_json(resolve_project_path(root, config["paths"]["all_int8_error"])),
        )
    write_json(plan_dir / "quantization_error.json", error_report)
    model.to(evaluation["device"])
    model.eval()
    assets = load_assets(resolve_project_path(root, config["paths"]["dataset_dir"]))
    lm_head_probe = None
    if original_lm_head_weight is not None:
        lm_head_probe = probe_lm_head_logits(
            model,
            tokenizer,
            list(assets.calibration),
            original_weight=original_lm_head_weight,
            config=config,
            torch_module=torch,
        )
        write_json(plan_dir / "lm_head_probe.json", lm_head_probe)
        del original_lm_head_weight
    kv_cache_benchmark = None
    if plan["kind"] in {"kv_cache_reference", "kv_cache_quantized"}:
        from src.baseline.service import InferenceRequest

        benchmark_row = max(
            assets.calibration,
            key=lambda row: len(InferenceRequest.from_finqa_row(row).render_prompt()),
        )
        kv_cache_benchmark = benchmark_kv_cache_runtime(
            model,
            tokenizer,
            InferenceRequest.from_finqa_row(benchmark_row).render_prompt(),
            generation_arguments=kv_arguments,
            config=config,
            torch_module=torch,
        )
        write_json(plan_dir / "kv_cache_benchmark.json", kv_cache_benchmark)
    service = BaselineService(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        config=baseline,
        model_metadata={
            "model_id": "Qwen2.5-7B-Instruct-QLoRA-step724-module-sensitivity",
            "version": config["expected_identity"]["merged_tree_sha256"],
            "quantization": {"plan_id": plan_id, "deployment_performance_claims_allowed": False},
        },
        log_file=plan_dir / "quality_service.jsonl",
    )
    report = run_quality_repair_experiment(
        service, assets.quality_dev, assets.manifest, quality, plan_dir,
        calibration=assets.calibration,
    )
    if report.get("dataset", {}).get("sample", {}).get("indices_sha256") != config["expected_identity"]["pilot_indices_sha256"]:
        raise ValueError("Module plan evaluation used the wrong pilot sample.")
    variant = _structured_variant(report)
    baseline_report_path = resolve_project_path(root, config["paths"]["all_int4_report"])
    transitions = paired_prediction_transitions(
        _prediction_rows(baseline_report_path),
        _prediction_rows(plan_dir / "comparison_report.json"),
    )
    baseline_metrics = manifest["all_int4_reference"]["metrics"]
    correct_delta = int(variant["metrics"]["correct"]) - int(baseline_metrics["correct"])
    if plan["comparison"] == "degradation":
        transition_summary = {
            "recovered_from_fp16_component_baseline": transitions["recovered"],
            "degraded_from_fp16_component_baseline": transitions["regressed"],
            "net_correct_change": correct_delta,
            "stable_correct": transitions["stable_correct"],
            "stable_wrong": transitions["stable_wrong"],
            "changed_raw_predictions": transitions["changed_raw_predictions"],
        }
    elif plan["comparison"] == "reference":
        transition_summary = {
            "recovered": transitions["recovered"],
            "regressed": transitions["regressed"],
            "net_correct_change": correct_delta,
            "stable_correct": transitions["stable_correct"],
            "stable_wrong": transitions["stable_wrong"],
            "changed_raw_predictions": transitions["changed_raw_predictions"],
        }
    else:
        transition_summary = {
            "recovered": transitions["recovered"],
            "regressed": transitions["regressed"],
            "net_correct_gain": correct_delta,
            "stable_correct": transitions["stable_correct"],
            "stable_wrong": transitions["stable_wrong"],
            "changed_raw_predictions": transitions["changed_raw_predictions"],
        }
    result = {
        "schema_version": 1,
        "stage": "7.2.2" if plan["kind"] in RECOVERY_KINDS else "7.2.3",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "metrics": variant["metrics"],
        "structured_output_rate": variant.get("structured_output_rate"),
        "format_retry_count": variant.get("format_retry_count"),
        "predictions_sha256": variant.get("predictions_sha256"),
        "quantization_error_sha256": sha256_file(plan_dir / "quantization_error.json"),
        "transition_semantics": plan["comparison"],
        "transitions_vs_all_int4": transition_summary,
        "duration_seconds": time.perf_counter() - started,
        "kv_cache_runtime_arguments": kv_arguments,
        "kv_cache_benchmark": kv_cache_benchmark,
        "lm_head_probe": lm_head_probe,
        "special_storage": {
            "fp16_reference_bytes": sum(
                row.get("theoretical_storage", {}).get("fp16_reference_bytes", 0)
                for row in error_report["special_modules"]
            ),
            "quantized_total_bytes": sum(
                row.get("theoretical_storage", {}).get("total_bytes", 0)
                for row in error_report["special_modules"]
            ),
            "savings_bytes": sum(
                row.get("theoretical_storage", {}).get("savings_bytes", 0)
                for row in error_report["special_modules"]
            ),
        } if error_report["special_modules"] else None,
        "all_int8_equivalence": all_int8_equivalence,
        "deployment_performance_claims_allowed": bool(
            kv_cache_benchmark
            and kv_cache_benchmark.get("performance_claims_allowed")
        ),
    }
    write_json(result_path, result)
    return result

