"""Prepare and validate the frozen inputs required by TODO 7.1.1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


BLOCK_PATH_CANDIDATES = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "decoder.layers",
)
TOKENIZER_FILE_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "generation_config.json",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_directory(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    files = []
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "tree_sha256": canonical_json_sha256(files),
        "files": files,
    }


def tokenizer_identity(model_identity: dict[str, Any]) -> dict[str, Any]:
    files = [
        item
        for item in model_identity.get("files", [])
        if Path(item["path"]).name in TOKENIZER_FILE_NAMES
    ]
    if not files:
        raise ValueError("Base model contains no recognized tokenizer files.")
    return {
        "path": model_identity["path"],
        "file_count": len(files),
        "tree_sha256": canonical_json_sha256(files),
        "files": files,
    }


def resolve_project_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root).expanduser().resolve() / path
    return path.resolve()


def validate_precision_contract(contract: dict[str, Any]) -> None:
    int4 = contract.get("int4", {})
    int8 = contract.get("int8", {})
    if (int4.get("weight_bits"), int4.get("activation_bits"), int4.get("group_size")) != (
        4,
        16,
        32,
    ):
        raise ValueError("TODO 7.1.1 requires INT4=W4A16 group_size=32.")
    if (int8.get("weight_bits"), int8.get("activation_bits")) != (8, 16):
        raise ValueError("TODO 7.1.1 requires INT8=W8A16.")
    if contract.get("forbidden_direct_comparison") != "awq_w4a16_vs_smoothquant_w8a8":
        raise ValueError("The AWQ W4A16 vs SmoothQuant W8A8 confound must be explicit.")


def _read_attr_path(root: Any, path: str) -> Any:
    current = root
    for name in path.split("."):
        if not hasattr(current, name):
            raise AttributeError(path)
        current = getattr(current, name)
    return current


def discover_transformer_blocks(model: Any) -> tuple[str, Sequence[Any]]:
    """Find the repeated transformer blocks without hard-coding a layer count."""

    matches: list[tuple[str, Sequence[Any]]] = []
    for path in BLOCK_PATH_CANDIDATES:
        try:
            value = _read_attr_path(model, path)
        except AttributeError:
            continue
        if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
            continue
        if not hasattr(value, "__getitem__") or len(value) == 0:
            continue
        matches.append((path, value))
    if not matches:
        raise ValueError(
            "Unable to discover transformer blocks; checked: "
            + ", ".join(BLOCK_PATH_CANDIDATES)
        )
    matches.sort(key=lambda item: len(item[1]), reverse=True)
    return matches[0]


def question_from_row(row: dict[str, Any]) -> str:
    question = row.get("question")
    if question is None and isinstance(row.get("qa"), dict):
        question = row["qa"].get("question")
    if question is None:
        raise ValueError("Dataset row has no question or qa.question.")
    return str(question)


def row_fingerprint(row: dict[str, Any]) -> str:
    return canonical_json_sha256(row)


def build_sample_entries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "question": question_from_row(dict(row)),
            "row_sha256": row_fingerprint(dict(row)),
        }
        for index, row in enumerate(rows)
    ]


def classify_business_question(
    question: str, taxonomy: dict[str, Any]
) -> list[str]:
    normalized = re.sub(r"\s+", " ", question.lower())
    labels = []
    for category, keywords in taxonomy.get("categories", {}).items():
        if any(re.search(rf"\b{re.escape(str(keyword).lower())}\b", normalized) for keyword in keywords):
            labels.append(category)
    if re.search(r"\b(?:19|20)\d{2}\b", normalized) and "date_or_term" not in labels:
        labels.append("date_or_term")
    return labels


def build_business_labels(
    rows: Iterable[dict[str, Any]], taxonomy: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        question = question_from_row(row)
        records.append(
            {
                "source_index": index,
                "question": question,
                "labels": classify_business_question(question, taxonomy),
                "row_sha256": row_fingerprint(row),
            }
        )
    return records


def long_context_indices_from_log(
    log_file: str | Path, total: int, fraction: float
) -> list[int]:
    first_request: dict[str, tuple[int, int]] = {}
    with Path(log_file).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = str(record.get("request_id", ""))
            match = re.search(r"(\d+)$", request_id)
            input_tokens = record.get("input_tokens")
            if match is None or not isinstance(input_tokens, int):
                continue
            source_index = int(match.group(1)) - 1
            if 0 <= source_index < total and request_id not in first_request:
                first_request[request_id] = (source_index, input_tokens)
    by_source: dict[int, int] = {}
    for source_index, input_tokens in first_request.values():
        by_source.setdefault(source_index, input_tokens)
    if len(by_source) != total:
        raise ValueError(
            f"Long-context log covers {len(by_source)}/{total} primary samples."
        )
    count = math.ceil(total * fraction)
    ranked = sorted(by_source.items(), key=lambda item: (-item[1], item[0]))
    return sorted(index for index, _ in ranked[:count])


def validate_pilot_indices(indices: list[int], size: int, expected_hash: str) -> None:
    if len(indices) != size or len(set(indices)) != size:
        raise ValueError(f"Pilot indices must contain {size} unique values.")
    actual_hash = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Pilot index hash mismatch: expected {expected_hash}, got {actual_hash}."
        )


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_evaluation_contract(
    quality_config: dict[str, Any],
    baseline_config: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    variants = [
        item
        for item in quality_config.get("variants", [])
        if item.get("name") == expected.get("variant")
    ]
    if len(variants) != 1:
        raise ValueError("Quality config must contain exactly one frozen variant.")
    generation = quality_config.get("generation")
    if generation != expected.get("generation"):
        raise ValueError("Quality generation settings do not match the 7.1.1 contract.")
    if float(quality_config.get("tolerance")) != float(expected.get("tolerance")):
        raise ValueError("Quality tolerance does not match the 7.1.1 contract.")
    if not bool(baseline_config.get("deterministic")):
        raise ValueError("Layer-sensitivity evaluation requires deterministic inference.")
    return {
        "variant": variants[0],
        "generation": generation,
        "tolerance": quality_config["tolerance"],
        "baseline_seed": baseline_config.get("seed"),
        "deterministic": baseline_config.get("deterministic"),
        "max_input_tokens": baseline_config.get("max_input_tokens"),
        "scorer": expected.get("scorer"),
    }


def freeze_inputs(
    *,
    project_root: str | Path,
    config: dict[str, Any],
    assets: Any,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze identities, evaluation contracts, sample lists, and business labels."""

    root = Path(project_root).resolve()
    paths = config["paths"]
    expected = config["expected_identity"]
    evaluation = config["evaluation_contract"]
    validate_precision_contract(config["precision_contract"])
    validate_pilot_indices(
        list(config["pilot_indices"]),
        int(evaluation["pilot_size"]),
        expected["pilot_indices_sha256"],
    )
    base_identity = hash_directory(resolve_project_path(root, paths["base_model"]))
    adapter_identity = hash_directory(resolve_project_path(root, paths["adapter"]))
    adapter_weight = next(
        (item for item in adapter_identity["files"] if item["path"] == "adapter_model.safetensors"),
        None,
    )
    if adapter_weight is None:
        raise ValueError("Adapter directory has no adapter_model.safetensors.")
    if adapter_weight["sha256"] != expected["adapter_weights_sha256"]:
        raise ValueError("Frozen adapter SHA256 does not match the 6.7.5 acceptance record.")
    manifest = assets.manifest
    dev_asset = manifest.get("assets", {}).get("evaluation_dev", {})
    high_risk_asset = manifest.get("assets", {}).get("high_risk_regression", {})
    if dev_asset.get("sha256") != expected["evaluation_dev_sha256"]:
        raise ValueError("FinQA dev SHA256 does not match the frozen contract.")
    if high_risk_asset.get("sha256") != expected["contextual_high_risk_sha256"]:
        raise ValueError("Contextual high-risk SHA256 does not match the frozen contract.")
    calibration_entries = build_sample_entries(assets.calibration)
    dev_entries = build_sample_entries(assets.quality_dev)
    high_risk_entries = build_sample_entries(assets.high_risk_regression)
    if len(calibration_entries) != int(evaluation["calibration_size"]):
        raise ValueError("Calibration size does not match the 7.1.1 contract.")
    if len(dev_entries) != int(evaluation["dev_size"]):
        raise ValueError("FinQA dev size does not match the 7.1.1 contract.")
    if len(high_risk_entries) != int(evaluation["high_risk_size"]):
        raise ValueError("High-risk size does not match the 7.1.1 contract.")
    long_indices = long_context_indices_from_log(
        resolve_project_path(root, paths["long_context_log"]),
        len(dev_entries),
        float(evaluation["long_context_fraction"]),
    )
    subsets = {
        "schema_version": 1,
        "calibration": calibration_entries,
        "pilot": [dev_entries[index] for index in config["pilot_indices"]],
        "evaluation_dev": dev_entries,
        "contextual_high_risk": high_risk_entries,
        "long_context": [dev_entries[index] for index in long_indices],
    }
    subset_hashes = {
        name: canonical_json_sha256(entries) for name, entries in subsets.items() if name != "schema_version"
    }
    labels = build_business_labels(assets.quality_dev, config["business_taxonomy"])
    label_counts: dict[str, int] = {
        name: 0 for name in config["business_taxonomy"]["categories"]
    }
    for record in labels:
        for label in record["labels"]:
            label_counts[label] += 1
    quality_config_path = resolve_project_path(root, paths["quality_config"])
    baseline_config_path = resolve_project_path(root, paths["baseline_config"])
    quality_config = load_json(quality_config_path)
    baseline_config = load_json(baseline_config_path)
    frozen_evaluation = freeze_evaluation_contract(
        quality_config, baseline_config, evaluation
    )
    frozen_adapter_path = resolve_project_path(root, paths["frozen_adapter"])
    frozen_adapter = load_json(frozen_adapter_path)
    recorded_adapter_sha = frozen_adapter.get("adapter_weights_sha256")
    if recorded_adapter_sha != expected["adapter_weights_sha256"]:
        raise ValueError("6.7.5 frozen-adapter record does not match the 7.1.1 contract.")
    scorer_path = root / "src" / "evaluation" / "finqa_metrics.py"
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_subsets.json", subsets)
    write_jsonl(output / "business_labels.jsonl", labels)
    freeze_manifest = {
        "schema_version": 1,
        "stage": "7.1.1",
        "status": "inputs_frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "identities": {
            "base_model": base_identity,
            "adapter": adapter_identity,
            "tokenizer": tokenizer_identity(base_identity),
            "dataset_dir": str(resolve_project_path(root, paths["dataset_dir"])),
            "dataset_manifest_sha256": canonical_json_sha256(manifest),
            "dataset_assets": manifest.get("assets", {}),
        },
        "precision_contract": config["precision_contract"],
        "evaluation_contract": config["evaluation_contract"],
        "frozen_evaluation": frozen_evaluation,
        "frozen_adapter_record": {
            "path": str(frozen_adapter_path),
            "sha256": sha256_file(frozen_adapter_path),
            "status": frozen_adapter.get("status"),
            "global_step": frozen_adapter.get("selection", {}).get("global_step"),
            "adapter_weights_sha256": recorded_adapter_sha,
        },
        "quality_config": {
            "path": str(quality_config_path),
            "sha256": sha256_file(quality_config_path),
        },
        "baseline_config": {
            "path": str(baseline_config_path),
            "sha256": sha256_file(baseline_config_path),
        },
        "scorer": {
            "callable": evaluation["scorer"],
            "path": str(scorer_path),
            "sha256": sha256_file(scorer_path),
        },
        "subsets_file": str(output / "frozen_subsets.json"),
        "subset_hashes": subset_hashes,
        "business_labels_file": str(output / "business_labels.jsonl"),
        "business_labels_sha256": sha256_file(output / "business_labels.jsonl"),
        "business_label_counts": label_counts,
    }
    write_json(output / "input_freeze.json", freeze_manifest)
    return freeze_manifest


def compare_pilot_reports(
    reference_report: dict[str, Any],
    candidate_report: dict[str, Any],
    reference_predictions: Iterable[dict[str, Any]],
    candidate_predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    reference_sample = reference_report.get("dataset", {}).get("sample", {})
    candidate_sample = candidate_report.get("dataset", {}).get("sample", {})
    if reference_sample.get("indices") != candidate_sample.get("indices"):
        raise ValueError("Pilot reports do not use identical source indices.")
    reference_rows = list(reference_predictions)
    candidate_rows = list(candidate_predictions)
    if len(reference_rows) != len(candidate_rows):
        raise ValueError("Pilot prediction counts differ.")
    transitions = {"recovered": 0, "regressed": 0, "stable_correct": 0, "stable_wrong": 0}
    changed_predictions = 0
    for reference, candidate in zip(reference_rows, candidate_rows):
        if reference.get("source_index") != candidate.get("source_index"):
            raise ValueError("Pilot prediction source indices are not aligned.")
        if reference.get("raw_prediction") != candidate.get("raw_prediction"):
            changed_predictions += 1
        before = bool(reference.get("correct"))
        after = bool(candidate.get("correct"))
        key = (
            "recovered" if not before and after else
            "regressed" if before and not after else
            "stable_correct" if before else
            "stable_wrong"
        )
        transitions[key] += 1
    reference_variants = [
        item for item in reference_report.get("variants", [])
        if item.get("variant") == "structured_json"
    ]
    candidate_variants = [
        item for item in candidate_report.get("variants", [])
        if item.get("variant") == "structured_json"
    ]
    if len(reference_variants) != 1 or len(candidate_variants) != 1:
        raise ValueError("Both pilot reports must contain one structured_json result.")
    return {
        "schema_version": 1,
        "stage": "7.1.1",
        "status": "pilot_compared",
        "sample_indices_sha256": reference_sample.get("indices_sha256"),
        "total": len(reference_rows),
        "changed_raw_predictions": changed_predictions,
        "transitions": transitions,
        "reference_metrics": reference_variants[0]["metrics"],
        "candidate_metrics": candidate_variants[0]["metrics"],
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def predictions_path(report_path: str | Path, report: dict[str, Any]) -> Path:
    variants = report.get("variants", [])
    matches = [item for item in variants if item.get("variant") == "structured_json"]
    if len(matches) != 1:
        raise ValueError("Report must contain exactly one structured_json result.")
    return Path(report_path).resolve().parent / matches[0]["predictions_file"]


def merge_frozen_adapter(
    *,
    project_root: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge the accepted LoRA adapter into an FP16/BF16 base model."""

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Merge requires torch, transformers, peft, and safetensors.") from exc
    root = Path(project_root).resolve()
    paths = config["paths"]
    merge = config["merge"]
    base_path = resolve_project_path(root, paths["base_model"])
    adapter_path = resolve_project_path(root, paths["adapter"])
    merged_path = resolve_project_path(root, paths["merged_model"])
    if merged_path.exists() and any(merged_path.iterdir()):
        raise FileExistsError(
            f"Merged model directory is not empty; refusing to overwrite: {merged_path}"
        )
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(merge["dtype"])
    if dtype is None:
        raise ValueError("Merge dtype must be float16 or bfloat16.")
    base_identity = hash_directory(base_path)
    adapter_identity = hash_directory(adapter_path)
    expected_adapter = config["expected_identity"]["adapter_weights_sha256"]
    weights = next(
        (item for item in adapter_identity["files"] if item["path"] == "adapter_model.safetensors"),
        None,
    )
    if weights is None or weights["sha256"] != expected_adapter:
        raise ValueError("Adapter identity does not match the frozen step 724 artifact.")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map=merge.get("device_map", "cpu"),
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    block_path, blocks = discover_transformer_blocks(merged_model)
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        str(merged_path),
        safe_serialization=bool(merge.get("safe_serialization", True)),
        max_shard_size=str(merge.get("max_shard_size", "4GB")),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    tokenizer.save_pretrained(str(merged_path))
    merged_identity = hash_directory(merged_path)
    manifest = {
        "schema_version": 1,
        "stage": "7.1.1",
        "status": "merged",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_identity,
        "adapter": adapter_identity,
        "merged_model": merged_identity,
        "merge": merge,
        "transformer_blocks": {"path": block_path, "count": len(blocks)},
        "precision_contract": config["precision_contract"],
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "merged_model_manifest.json", manifest)
    return manifest
