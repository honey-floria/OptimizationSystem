"""Prepare and validate the TODO 7.2.1 module-sensitivity contract."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.quantization.layer_sensitivity_setup import (
    canonical_json_sha256,
    discover_transformer_blocks,
    load_json,
    resolve_project_path,
    sha256_file,
    write_json,
)


CONTRACT_SCHEMA_VERSION = 1
ATTENTION_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
MLP_MODULES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
BLOCK_NORMS = ("input_layernorm", "post_attention_layernorm")


def _read_attr_path(root: Any, path: str) -> Any:
    current = root
    for name in path.split("."):
        if name.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(name)]
            continue
        if not hasattr(current, name):
            raise ValueError(f"Required module path is missing: {path}")
        current = getattr(current, name)
    return current


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("A discovered quantizable module has no weight shape.")
    return [int(item) for item in shape]


def _numel(value: Any) -> int:
    method = getattr(value, "numel", None)
    if callable(method):
        return int(method())
    total = 1
    for dimension in _shape_list(value):
        total *= dimension
    return total


def _module_record(path: str, module: Any, role: str, layer_index: int | None = None) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(f"Quantizable module has no weight: {path}")
    shape = _shape_list(weight)
    record: dict[str, Any] = {
        "path": path,
        "role": role,
        "class_name": type(module).__name__,
        "weight_shape": shape,
        "weight_ndim": len(shape),
        "weight_numel": _numel(weight),
        "weight_dtype": str(getattr(weight, "dtype", "unknown")),
    }
    if layer_index is not None:
        record["layer_index"] = layer_index
    return record


def _weight_storage_identity(weight: Any) -> tuple[str, int] | tuple[str, object]:
    data_ptr = getattr(weight, "data_ptr", None)
    if callable(data_ptr):
        try:
            pointer = int(data_ptr())
            if pointer > 0:
                return ("data_ptr", pointer)
        except (RuntimeError, TypeError, ValueError):
            pass
    return ("object", id(weight))


def weights_are_tied(left: Any, right: Any) -> bool:
    left_weight = getattr(left, "weight", None)
    right_weight = getattr(right, "weight", None)
    if left_weight is None or right_weight is None:
        raise ValueError("Tied-weight inspection requires two weighted modules.")
    return left_weight is right_weight or _weight_storage_identity(left_weight) == _weight_storage_identity(right_weight)


def _module_names(model: Any) -> dict[int, list[str]]:
    names: dict[int, list[str]] = {}
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return names
    try:
        rows: Iterable[tuple[str, Any]] = named_modules(remove_duplicate=False)
    except TypeError:
        rows = named_modules()
    for name, module in rows:
        names.setdefault(id(module), []).append(str(name))
    return names


def _preferred_module_path(module: Any, names: dict[int, list[str]], fallback: str) -> str:
    candidates = [name for name in names.get(id(module), []) if name]
    if not candidates:
        return fallback
    return sorted(candidates, key=lambda value: (len(value.split(".")), len(value), value))[0]


def validate_module_group_contract(groups: dict[str, Any]) -> None:
    expected = {
        "attention": list(ATTENTION_MODULES),
        "mlp": list(MLP_MODULES),
        "block_norms": list(BLOCK_NORMS),
    }
    if groups != expected:
        raise ValueError("Module groups must match the frozen Qwen block contract exactly.")


def validate_precision_contract(precision: dict[str, Any], backend: dict[str, Any], kv_cache: dict[str, Any]) -> None:
    int4 = precision.get("linear_int4", {})
    int8 = precision.get("linear_int8", {})
    if (
        int4.get("weight_bits"),
        int4.get("activation_bits"),
        int4.get("group_size"),
        int4.get("axis"),
        int4.get("scheme"),
    ) != (4, 16, 32, "input_dimension", "symmetric_groupwise"):
        raise ValueError("Module sensitivity requires Linear INT4=W4A16 g32 along the input dimension.")
    if (
        int8.get("weight_bits"),
        int8.get("activation_bits"),
        int8.get("group_size"),
        int8.get("axis"),
        int8.get("scheme"),
    ) != (8, 16, None, "output_channel", "symmetric_per_output_channel"):
        raise ValueError("Module sensitivity requires Linear INT8=W8A16 per output channel.")
    for name in ("embedding", "lm_head", "norm"):
        module_precision = precision.get(name, {})
        allowed = module_precision.get("allowed")
        if allowed != ["fp16", "int8", "int4"]:
            raise ValueError(f"{name} must explicitly compare fp16, int8, and int4.")
        for bit_name in ("int4", "int8"):
            rule = module_precision.get(bit_name, {})
            if rule.get("zero_point") != 0 or not rule.get("scheme") or "axis" not in rule:
                raise ValueError(f"{name}.{bit_name} must freeze scheme, axis, and zero_point=0.")
        if name in {"embedding", "lm_head"} and module_precision["int4"].get("group_size") != 32:
            raise ValueError(f"{name} INT4 must use group_size=32.")
    if precision.get("shared_weight_policy") != "joint_plan_only":
        raise ValueError("Tied embedding/lm_head weights must use a joint-only plan policy.")
    if backend.get("deployment_performance_claims_allowed") is not False:
        raise ValueError("Fake-quant weight results must prohibit deployment performance claims.")
    if kv_cache.get("weight_fake_quant_allowed") is not False:
        raise ValueError("KV Cache must not be represented by weight fake-quant.")
    required_kv = {
        "cache_tensors": ["key", "value"],
        "phases": ["prefill", "decode"],
        "quantize_at": "cache_write",
        "dequantize_at": "attention_read",
    }
    for field, value in required_kv.items():
        if kv_cache.get(field) != value:
            raise ValueError(f"KV Cache contract must freeze {field}={value}.")
    for field in ("scale_mode", "granularity", "cache_layout"):
        if not kv_cache.get(field):
            raise ValueError(f"KV Cache contract must record {field}.")
    real_backend = kv_cache.get("real_backend", {})
    if real_backend.get("performance_claims_allowed_before_runtime_probe") is not False:
        raise ValueError("KV Cache performance claims require a successful runtime probe.")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CONTRACT_SCHEMA_VERSION or config.get("stage") != "7.2.1":
        raise ValueError("Expected a schema v1 TODO 7.2.1 configuration.")
    required_paths = {
        "merged_manifest",
        "input_freeze",
        "layer_matrix_manifest",
        "all_int4_plan",
        "all_int4_result",
        "all_int4_report",
        "output_root",
    }
    missing = sorted(required_paths - set(config.get("paths", {})))
    if missing:
        raise ValueError(f"Module-sensitivity config is missing paths: {missing}")
    validate_module_group_contract(config.get("module_groups", {}))
    validate_precision_contract(
        config.get("precision", {}),
        config.get("weight_backend", {}),
        config.get("kv_cache", {}),
    )


def _structured_variant(report: dict[str, Any]) -> dict[str, Any]:
    matches = [item for item in report.get("variants", []) if item.get("variant") == "structured_json"]
    if len(matches) != 1:
        raise ValueError("ALL-I4 report must contain exactly one structured_json variant.")
    return matches[0]


def _verified_file(root: Path, paths: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_project_path(root, paths[key])
    if not path.is_file():
        raise ValueError(f"Required upstream artifact is missing: {path}")
    return path, load_json(path)


def validate_upstream_contract(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    root = Path(project_root).resolve()
    paths = config["paths"]
    expected = config["expected_identity"]
    merged_path, merged = _verified_file(root, paths, "merged_manifest")
    freeze_path, frozen = _verified_file(root, paths, "input_freeze")
    matrix_path, matrix = _verified_file(root, paths, "layer_matrix_manifest")
    plan_path, plan_payload = _verified_file(root, paths, "all_int4_plan")
    result_path, result = _verified_file(root, paths, "all_int4_result")
    report_path, report = _verified_file(root, paths, "all_int4_report")

    if merged.get("status") != "merged" or frozen.get("status") != "inputs_frozen":
        raise ValueError("7.2.1 requires completed 7.1.1 merge and input-freeze artifacts.")
    if merged.get("merged_model", {}).get("tree_sha256") != expected["merged_tree_sha256"]:
        raise ValueError("Merged model identity differs from the frozen contract.")
    if merged.get("adapter", {}).get("tree_sha256") != expected["adapter_tree_sha256"]:
        raise ValueError("Adapter tree identity differs from the frozen contract.")
    if frozen.get("frozen_adapter_record", {}).get("adapter_weights_sha256") != expected["adapter_weights_sha256"]:
        raise ValueError("Adapter weight identity differs from the frozen contract.")
    if frozen.get("identities", {}).get("tokenizer", {}).get("tree_sha256") != expected["tokenizer_tree_sha256"]:
        raise ValueError("Tokenizer identity differs from the frozen contract.")
    blocks = merged.get("transformer_blocks", {})
    if blocks != {"path": expected["block_path"], "count": expected["block_count"]}:
        raise ValueError("Transformer block identity differs from 7.1.1.")
    expected_subsets = {
        "calibration": expected["calibration_subset_sha256"],
        "evaluation_dev": expected["evaluation_dev_subset_sha256"],
        "contextual_high_risk": expected["high_risk_subset_sha256"],
        "long_context": expected["long_context_subset_sha256"],
    }
    subset_hashes = frozen.get("subset_hashes", {})
    for name, digest in expected_subsets.items():
        if subset_hashes.get(name) != digest:
            raise ValueError(f"Frozen {name} subset identity differs from 7.1.1.")
    if frozen.get("evaluation_contract") != config.get("evaluation_contract"):
        raise ValueError("Prompt, generation, sampling, or scorer contract differs from 7.1.1.")
    if matrix.get("status") != "matrix_prepared" or (
        matrix.get("block_path"), matrix.get("block_count")
    ) != (expected["block_path"], expected["block_count"]):
        raise ValueError("7.1 layer matrix is missing or incompatible.")

    stored_plan_hash = plan_payload.get("plan_sha256")
    unhashed_plan = {key: value for key, value in plan_payload.items() if key != "plan_sha256"}
    if not stored_plan_hash or canonical_json_sha256(unhashed_plan) != stored_plan_hash:
        raise ValueError("ALL-I4 plan contract hash is invalid.")
    plan = plan_payload.get("plan", {})
    if plan.get("plan_id") != "all_int4" or plan.get("kind") != "all_int4" or plan.get("default_precision") != "int4":
        raise ValueError("The reused baseline is not the frozen ALL-I4 plan.")
    if result.get("status") != "completed" or result.get("plan") != plan:
        raise ValueError("ALL-I4 does not have a matching completed result.")
    if result.get("sample_indices_sha256") != expected["pilot_indices_sha256"]:
        raise ValueError("ALL-I4 used the wrong pilot sample.")
    variant = _structured_variant(report)
    if report.get("dataset", {}).get("sample", {}).get("indices_sha256") != expected["pilot_indices_sha256"]:
        raise ValueError("ALL-I4 report used the wrong pilot sample.")
    if variant.get("predictions_sha256") != result.get("predictions_sha256"):
        raise ValueError("ALL-I4 report and result reference different predictions.")
    if variant.get("metrics", {}).get("total") != config["evaluation_contract"]["pilot_size"]:
        raise ValueError("ALL-I4 report has the wrong pilot size.")

    artifacts = {}
    for key, path in (
        ("merged_manifest", merged_path),
        ("input_freeze", freeze_path),
        ("layer_matrix_manifest", matrix_path),
        ("all_int4_plan", plan_path),
        ("all_int4_result", result_path),
        ("all_int4_report", report_path),
    ):
        artifacts[key] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "status": "pass",
        "model_identity": {
            "merged_tree_sha256": expected["merged_tree_sha256"],
            "adapter_tree_sha256": expected["adapter_tree_sha256"],
            "adapter_weights_sha256": expected["adapter_weights_sha256"],
            "tokenizer_tree_sha256": expected["tokenizer_tree_sha256"],
        },
        "transformer_blocks": blocks,
        "subset_hashes": {name: subset_hashes[name] for name in expected_subsets},
        "evaluation_contract": frozen["evaluation_contract"],
        "all_int4_reference": {
            "plan_sha256": stored_plan_hash,
            "sample_indices_sha256": result["sample_indices_sha256"],
            "predictions_sha256": result["predictions_sha256"],
            "metrics": result["metrics"],
        },
        "artifacts": artifacts,
        "merged_model_path": merged["merged_model"]["path"],
    }


def discover_module_inventory(model: Any, groups: dict[str, Any]) -> dict[str, Any]:
    validate_module_group_contract(groups)
    block_path, blocks = discover_transformer_blocks(model)
    names = _module_names(model)
    attention: list[dict[str, Any]] = []
    mlp: list[dict[str, Any]] = []
    norms: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        for relative in groups["attention"]:
            module = _read_attr_path(block, relative)
            record = _module_record(f"{block_path}.{index}.{relative}", module, "attention", index)
            if record["weight_ndim"] != 2:
                raise ValueError(f"Attention projection must have a 2D weight: {record['path']}")
            attention.append(record)
        for relative in groups["mlp"]:
            module = _read_attr_path(block, relative)
            record = _module_record(f"{block_path}.{index}.{relative}", module, "mlp", index)
            if record["weight_ndim"] != 2:
                raise ValueError(f"MLP projection must have a 2D weight: {record['path']}")
            mlp.append(record)
        for relative in groups["block_norms"]:
            module = _read_attr_path(block, relative)
            norms.append(_module_record(f"{block_path}.{index}.{relative}", module, "block_norm", index))

    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise ValueError("Model must expose input and output embedding accessors.")
    embedding = get_input()
    lm_head = get_output()
    if embedding is None or lm_head is None:
        raise ValueError("Embedding and lm_head must both be discoverable.")
    embedding_path = _preferred_module_path(embedding, names, "<input_embeddings>")
    lm_head_path = _preferred_module_path(lm_head, names, "<output_embeddings>")
    embedding_record = _module_record(embedding_path, embedding, "embedding")
    lm_head_record = _module_record(lm_head_path, lm_head, "lm_head")
    if embedding_record["weight_ndim"] != 2 or lm_head_record["weight_ndim"] != 2:
        raise ValueError("Embedding and lm_head must have 2D weights.")

    known_norm_ids = set()
    for record in norms:
        try:
            known_norm_ids.add(id(_read_attr_path(model, record["path"])))
        except ValueError:
            pass
    additional_norms = []
    final_norms = []
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for name, module in named_modules():
            lowered = f"{name}.{type(module).__name__}".lower()
            if id(module) in known_norm_ids or "norm" not in lowered or not hasattr(module, "weight"):
                continue
            role = "additional_block_norm" if str(name).startswith(f"{block_path}.") else "final_norm"
            record = _module_record(str(name), module, role)
            if role == "final_norm":
                final_norms.append(record)
            else:
                additional_norms.append(record)
    if not final_norms:
        raise ValueError("No final normalization module was discovered outside the blocks.")
    norms.extend(additional_norms)
    norms.extend(final_norms)

    tied = weights_are_tied(embedding, lm_head)
    inventory = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "stage": "7.2.1",
        "block_path": block_path,
        "block_count": len(blocks),
        "module_groups": {
            "attention": attention,
            "mlp": mlp,
            "norm": norms,
            "embedding": [embedding_record],
            "lm_head": [lm_head_record],
        },
        "counts": {
            "attention": len(attention),
            "mlp": len(mlp),
            "norm": len(norms),
            "embedding": 1,
            "lm_head": 1,
        },
        "embedding_lm_head_tied": tied,
        "shared_weight_policy": "joint_plan_only" if tied else "independent_plans_allowed",
    }
    inventory["inventory_sha256"] = canonical_json_sha256(inventory)
    return inventory


def inspect_kv_cache_capability(model: Any, contract: dict[str, Any], transformers_module: Any | None = None) -> dict[str, Any]:
    forward = getattr(model, "forward", None)
    parameters = set()
    if callable(forward):
        try:
            parameters = set(inspect.signature(forward).parameters)
        except (TypeError, ValueError):
            parameters = set()
    model_support = {
        "forward_accepts_use_cache": "use_cache" in parameters,
        "forward_accepts_past_key_values": "past_key_values" in parameters,
        "config_use_cache": bool(getattr(getattr(model, "config", None), "use_cache", False)),
    }
    cache_api = {"QuantizedCache": False, "QuantizedCacheConfig": False}
    transformers_version = None
    if transformers_module is not None:
        transformers_version = getattr(transformers_module, "__version__", None)
        cache_utils = getattr(transformers_module, "cache_utils", None)
        for name in cache_api:
            cache_api[name] = bool(cache_utils is not None and hasattr(cache_utils, name))
    supports_model_cache = model_support["forward_accepts_use_cache"] and model_support["forward_accepts_past_key_values"]
    has_quantized_api = all(cache_api.values())
    real_backend = contract["real_backend"]
    declared_low_bit = bool(real_backend.get("supports_real_int8") or real_backend.get("supports_real_int4"))
    if not supports_model_cache:
        status = "unsupported_model_cache_interface"
    elif declared_low_bit and has_quantized_api:
        status = "runtime_probe_required"
    elif contract.get("sensitivity_qdq_hook_allowed"):
        status = "sensitivity_only"
    else:
        status = "unsupported"
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "stage": "7.2.1",
        "status": status,
        "requested_precisions": contract["requested_precisions"],
        "model_cache_interface": model_support,
        "transformers_version": transformers_version,
        "transformers_quantized_cache_api": cache_api,
        "real_backend_contract": real_backend,
        "weight_fake_quant_allowed": False,
        "deployment_performance_claims_allowed": False,
        "runtime_probe_completed": False,
    }


def prepare_contract(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    upstream = validate_upstream_contract(root, config)
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "stage": "7.2.1",
        "status": "awaiting_model_inspection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "upstream": upstream,
        "module_groups": config["module_groups"],
        "precision": config["precision"],
        "weight_backend": config["weight_backend"],
        "kv_cache": config["kv_cache"],
        "runtime_inspection": config["runtime_inspection"],
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    write_json(output_root / "contract.json", contract)
    return contract


def inspect_model_contract(
    project_root: str | Path,
    config: dict[str, Any],
    *,
    model: Any | None = None,
    model_path: str | Path | None = None,
    transformers_module: Any | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract = prepare_contract(root, config)
    output_root = resolve_project_path(root, config["paths"]["output_root"])
    loaded_here = model is None
    if loaded_here:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Model inspection requires torch and transformers.") from exc
        transformers_module = transformers
        configured_path = model_path or contract["upstream"]["merged_model_path"]
        resolved_model_path = Path(configured_path).expanduser().resolve()
        if not resolved_model_path.is_dir():
            raise ValueError(
                f"Merged model directory is unavailable on this host: {resolved_model_path}. "
                "Pass --model-path on the server that stores the frozen model."
            )
        dtype_name = config["runtime_inspection"]["dtype"]
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_name)
        if dtype is None:
            raise ValueError("runtime_inspection.dtype must be float16 or bfloat16.")
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_model_path),
            local_files_only=bool(config["runtime_inspection"]["local_files_only"]),
            trust_remote_code=bool(config["runtime_inspection"]["trust_remote_code"]),
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=config["runtime_inspection"]["device_map"],
        )
    inventory = discover_module_inventory(model, config["module_groups"])
    expected = config["expected_identity"]
    if (inventory["block_path"], inventory["block_count"]) != (
        expected["block_path"],
        expected["block_count"],
    ):
        raise ValueError("Runtime transformer block discovery differs from the frozen identity.")
    if inventory["counts"]["attention"] != expected["block_count"] * len(ATTENTION_MODULES):
        raise ValueError("Runtime Attention module coverage is incomplete.")
    if inventory["counts"]["mlp"] != expected["block_count"] * len(MLP_MODULES):
        raise ValueError("Runtime MLP module coverage is incomplete.")
    kv_capability = inspect_kv_cache_capability(model, config["kv_cache"], transformers_module)
    write_json(output_root / "module_inventory.json", inventory)
    write_json(output_root / "kv_cache_capability.json", kv_capability)
    validation = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "stage": "7.2.1",
        "status": "pass",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract["contract_sha256"],
        "module_inventory_sha256": sha256_file(output_root / "module_inventory.json"),
        "kv_cache_capability_sha256": sha256_file(output_root / "kv_cache_capability.json"),
        "block_count": inventory["block_count"],
        "attention_module_count": inventory["counts"]["attention"],
        "mlp_module_count": inventory["counts"]["mlp"],
        "norm_module_count": inventory["counts"]["norm"],
        "embedding_lm_head_tied": inventory["embedding_lm_head_tied"],
        "kv_cache_status": kv_capability["status"],
        "deployment_performance_claims_allowed": False,
    }
    write_json(output_root / "validation.json", validation)
    if loaded_here:
        del model
    return validation

