"""TODO 8.1 Qwen2-style structured pruning and evidence helpers.

The implementation physically removes projection rows/columns and decoder
blocks.  It does not mask weights, so parameter counts and serialized model
size reflect the pruned structure.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PruningCandidate:
    plan_id: str
    target_rate: float
    attention: bool
    mlp: bool
    layer_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["layer_indices"] = list(self.layer_indices)
        return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_key(value: str) -> tuple[int, ...]:
    numbers = tuple(int(part) for part in re.findall(r"\d+", value))
    return numbers or (0,)


def build_candidates(config: dict[str, Any]) -> list[PruningCandidate]:
    rows = config.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("structured_pruning.candidates must be a non-empty list")
    candidates: list[PruningCandidate] = []
    for row in rows:
        components = set(row.get("components", []))
        if not components or not components <= {"attention", "mlp", "layers"}:
            raise ValueError(f"Invalid pruning components for {row.get('plan_id')}: {sorted(components)}")
        rate = float(row["target_rate"])
        if not 0 < rate < 1:
            raise ValueError(f"target_rate must be in (0, 1): {rate}")
        layers = tuple(sorted({int(index) for index in row.get("layer_indices", [])}))
        if "layers" in components and not layers:
            raise ValueError("Layer pruning requires explicit layer_indices; it is never inferred from rate")
        candidates.append(PruningCandidate(
            plan_id=str(row["plan_id"]), target_rate=rate,
            attention="attention" in components, mlp="mlp" in components,
            layer_indices=layers,
        ))
    ids = [row.plan_id for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Pruning plan IDs must be unique")
    required = {0.05, 0.10, 0.15, 0.20}
    present = {round(row.target_rate, 2) for row in candidates if row.attention and row.mlp}
    if not required <= present:
        raise ValueError(f"Combined candidate matrix is missing rates: {sorted(required - present)}")
    return candidates


def candidate_by_id(config: dict[str, Any], plan_id: str) -> PruningCandidate:
    matches = [row for row in build_candidates(config) if row.plan_id == plan_id]
    if not matches:
        raise ValueError(f"Unknown pruning plan: {plan_id}")
    return matches[0]


def feasible_gqa_shape(num_heads: int, num_kv_heads: int, target_rate: float) -> tuple[int, int, float]:
    """Return a legal, rate-close, balanced GQA shape and its width reduction.

    A legal shape must retain at least one Q head per KV head and satisfy
    ``Q % KV == 0``.  The rate weights Q and KV widths equally, matching the
    q/o and k/v projection parameter contributions for a fixed head dimension.
    The objective also penalizes a large difference between the Q- and KV-head
    removal fractions.  Without that guard, a nominal 10% request on Qwen2
    28Q/4KV would choose 28Q/1KV and destroy 75% of the KV heads merely because
    its aggregate width happens to be close to 10%.
    """
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError("Source GQA shape requires positive heads and num_heads % num_kv_heads == 0")
    if not 0 <= target_rate < 1:
        raise ValueError("target_rate must be in [0, 1)")
    options: list[tuple[float, float, bool, int, int, float]] = []
    total = num_heads + num_kv_heads
    for keep_kv in range(1, num_kv_heads + 1):
        for keep_q in range(keep_kv, num_heads + 1):
            if keep_q % keep_kv:
                continue
            realized = 1.0 - (keep_q + keep_kv) / total
            if target_rate > 0 and realized == 0:
                continue
            q_rate = 1.0 - keep_q / num_heads
            kv_rate = 1.0 - keep_kv / num_kv_heads
            imbalance = abs(q_rate - kv_rate)
            score = abs(realized - target_rate) + 0.25 * imbalance
            options.append((score, abs(realized - target_rate), realized > target_rate,
                            -keep_q, -keep_kv, realized))
    _, _, _, neg_q, neg_kv, realized = min(options)
    return -neg_q, -neg_kv, realized


def aligned_keep_size(size: int, target_rate: float, alignment: int) -> int:
    if size <= 0 or alignment <= 0 or not 0 <= target_rate < 1:
        raise ValueError("size/alignment must be positive and target_rate must be in [0, 1)")
    choices = list(range(alignment, size + 1, alignment))
    if size not in choices:
        choices.append(size)
    target = size * (1.0 - target_rate)
    return min(choices, key=lambda value: (abs(value - target), -value))


def prepare_experiment(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / config["paths"]["output_root"]
    candidates = build_candidates(config)
    manifest = {
        "schema_version": 1,
        "stage": "8.1",
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_manifest": str((root / config["paths"]["source_model_manifest"]).resolve()),
        "architecture": config["architecture"],
        "constraints": config["constraints"],
        "quality_gate": config["quality_gate"],
        "candidates": [row.to_dict() for row in candidates],
    }
    manifest["contract_sha256"] = hashlib.sha256(
        json.dumps(manifest["candidates"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(output / "experiment_manifest.json", manifest)
    for candidate in candidates:
        _write_json(output / "plans" / candidate.plan_id / "plan.json", candidate.to_dict())
    return manifest


def _resolve_source_model(root: Path, config: dict[str, Any], model_path: str | Path | None) -> Path:
    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve()
    else:
        manifest_path = (root / config["paths"]["source_model_manifest"]).resolve()
        manifest = _read_json(manifest_path)
        value = manifest.get("model_path")
        if value is None and isinstance(manifest.get("merged_model"), dict):
            value = manifest["merged_model"].get("path")
        if not isinstance(value, str) or not value:
            raise ValueError("Source manifest has neither model_path nor merged_model.path")
        resolved = Path(value).expanduser()
        resolved = resolved.resolve() if resolved.is_absolute() else (manifest_path.parent / resolved).resolve()
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"Model directory is unavailable: {resolved}")
    return resolved


def _module_by_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _new_linear(linear: Any, *, out_indices: Iterable[int] | None = None, in_indices: Iterable[int] | None = None) -> Any:
    import torch

    out_index = torch.tensor(list(out_indices), device=linear.weight.device) if out_indices is not None else None
    in_index = torch.tensor(list(in_indices), device=linear.weight.device) if in_indices is not None else None
    weight = linear.weight.detach()
    if out_index is not None:
        weight = weight.index_select(0, out_index)
    if in_index is not None:
        weight = weight.index_select(1, in_index)
    replacement = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=linear.bias is not None,
                                  device=weight.device, dtype=weight.dtype)
    replacement.weight.requires_grad_(linear.weight.requires_grad)
    with torch.no_grad():
        replacement.weight.copy_(weight)
        if linear.bias is not None:
            bias = linear.bias.detach()
            if out_index is not None:
                bias = bias.index_select(0, out_index)
            replacement.bias.copy_(bias)
            replacement.bias.requires_grad_(linear.bias.requires_grad)
    return replacement


def _highest_indices(scores: Any, keep: int) -> list[int]:
    # Keep the highest-magnitude structures; stable argsort makes ties auditable.
    order = scores.detach().float().cpu().argsort(descending=True, stable=True)
    return sorted(int(value) for value in order[:keep].tolist())


def prune_attention(attention: Any, target_rate: float) -> dict[str, Any]:
    """Physically prune Q/KV heads from one Qwen2-compatible attention module."""
    head_dim = int(attention.head_dim)
    q_heads = int(getattr(attention, "num_heads", attention.q_proj.out_features // head_dim))
    kv_heads = int(getattr(attention, "num_key_value_heads", attention.k_proj.out_features // head_dim))
    keep_q, keep_kv, realized = feasible_gqa_shape(q_heads, kv_heads, target_rate)
    q_score = attention.q_proj.weight.detach().reshape(q_heads, head_dim, -1).float().pow(2).sum((1, 2))
    q_score += attention.o_proj.weight.detach().reshape(-1, q_heads, head_dim).float().pow(2).sum((0, 2))
    k_score = attention.k_proj.weight.detach().reshape(kv_heads, head_dim, -1).float().pow(2).sum((1, 2))
    v_score = attention.v_proj.weight.detach().reshape(kv_heads, head_dim, -1).float().pow(2).sum((1, 2))
    q_keep = _highest_indices(q_score, keep_q)
    kv_keep = _highest_indices(k_score + v_score, keep_kv)
    q_dims = [head * head_dim + offset for head in q_keep for offset in range(head_dim)]
    kv_dims = [head * head_dim + offset for head in kv_keep for offset in range(head_dim)]
    attention.q_proj = _new_linear(attention.q_proj, out_indices=q_dims)
    attention.k_proj = _new_linear(attention.k_proj, out_indices=kv_dims)
    attention.v_proj = _new_linear(attention.v_proj, out_indices=kv_dims)
    attention.o_proj = _new_linear(attention.o_proj, in_indices=q_dims)
    attention.num_heads = keep_q
    attention.num_key_value_heads = keep_kv
    attention.num_key_value_groups = keep_q // keep_kv
    return {"before_q_heads": q_heads, "after_q_heads": keep_q,
            "before_kv_heads": kv_heads, "after_kv_heads": keep_kv,
            "realized_projection_rate": realized, "kept_q_heads": q_keep, "kept_kv_heads": kv_keep}


def prune_mlp(mlp: Any, target_rate: float, alignment: int) -> dict[str, Any]:
    """Prune the same SwiGLU channels from gate/up rows and down columns."""
    import torch

    before = int(mlp.gate_proj.out_features)
    keep = aligned_keep_size(before, target_rate, alignment)
    score = mlp.gate_proj.weight.detach().float().pow(2).sum(1)
    score += mlp.up_proj.weight.detach().float().pow(2).sum(1)
    score += mlp.down_proj.weight.detach().float().pow(2).sum(0)
    indices = _highest_indices(score, keep)
    mlp.gate_proj = _new_linear(mlp.gate_proj, out_indices=indices)
    mlp.up_proj = _new_linear(mlp.up_proj, out_indices=indices)
    mlp.down_proj = _new_linear(mlp.down_proj, in_indices=indices)
    return {"before_intermediate_size": before, "after_intermediate_size": keep,
            "realized_rate": 1.0 - keep / before, "kept_channels_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()).hexdigest()}


def prune_layers(model: Any, block_path: str, indices: Iterable[int]) -> dict[str, Any]:
    import torch

    blocks = _module_by_path(model, block_path)
    removed = sorted(set(int(index) for index in indices))
    if any(index < 0 or index >= len(blocks) for index in removed):
        raise ValueError(f"Layer indices outside [0, {len(blocks)}): {removed}")
    if len(removed) >= len(blocks):
        raise ValueError("Layer pruning must retain at least one decoder block")
    kept = [block for index, block in enumerate(blocks) if index not in removed]
    replacement = torch.nn.ModuleList(kept)
    parent_path, attribute = block_path.rsplit(".", 1)
    setattr(_module_by_path(model, parent_path), attribute, replacement)
    for index, block in enumerate(replacement):
        if hasattr(block, "self_attn") and hasattr(block.self_attn, "layer_idx"):
            block.self_attn.layer_idx = index
    return {"before_layers": len(blocks), "after_layers": len(replacement), "removed_layers": removed}


def validate_structure(model: Any, config: dict[str, Any], candidate: PruningCandidate) -> dict[str, Any]:
    blocks = _module_by_path(model, config["architecture"]["block_path"])
    attention_path = config["architecture"]["attention_module"]
    mlp_path = config["architecture"]["mlp_module"]
    errors: list[str] = []
    signatures = []
    for index, block in enumerate(blocks):
        attention = _module_by_path(block, attention_path)
        mlp = _module_by_path(block, mlp_path)
        head_dim = int(attention.head_dim)
        q_heads = int(getattr(attention, "num_heads", attention.q_proj.out_features // head_dim))
        kv_heads = int(getattr(attention, "num_key_value_heads", attention.k_proj.out_features // head_dim))
        q_width = q_heads * head_dim
        kv_width = kv_heads * head_dim
        checks = {
            "q_out": attention.q_proj.out_features == q_width,
            "k_out": attention.k_proj.out_features == kv_width,
            "v_out": attention.v_proj.out_features == kv_width,
            "o_in": attention.o_proj.in_features == q_width,
            "gqa_divisible": q_heads % kv_heads == 0,
            "mlp_equal": mlp.gate_proj.out_features == mlp.up_proj.out_features == mlp.down_proj.in_features,
        }
        errors.extend(f"layer {index}: {name}" for name, passed in checks.items() if not passed)
        signatures.append({"layer": index, "q_heads": q_heads, "kv_heads": kv_heads,
                           "head_dim": head_dim,
                           "intermediate_size": mlp.gate_proj.out_features})
    if signatures:
        first = signatures[0]
        expected_config = {
            "num_hidden_layers": len(blocks),
            "num_attention_heads": first["q_heads"],
            "num_key_value_heads": first["kv_heads"],
            "head_dim": first["head_dim"],
            "intermediate_size": first["intermediate_size"],
        }
        for name, expected in expected_config.items():
            if getattr(model.config, name, None) != expected:
                errors.append(f"config {name}: expected {expected}, got {getattr(model.config, name, None)}")
    return {"status": "pass" if not errors else "fail", "errors": errors,
            "block_count": len(blocks), "candidate": candidate.plan_id, "layer_signatures": signatures}


def export_candidate(project_root: str | Path, config: dict[str, Any], plan_id: str, *,
                     model_path: str | Path | None = None, device: str = "cpu") -> dict[str, Any]:
    """Load, prune, smoke-test, and save a standalone Transformers model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Pruning export requires torch and transformers") from exc
    transformers_version = importlib.metadata.version("transformers")
    minimum_version = str(config["constraints"]["minimum_transformers_version"])
    if _version_key(transformers_version) < _version_key(minimum_version):
        raise RuntimeError(
            f"Structured Qwen2 head pruning requires transformers>={minimum_version}; "
            f"found {transformers_version}."
        )
    root = Path(project_root).resolve()
    candidate = candidate_by_id(config, plan_id)
    source = _resolve_source_model(root, config, model_path)
    output = root / config["paths"]["output_root"] / "exports" / plan_id
    dtype_name = config.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True, torch_dtype=dtype,
                                                 low_cpu_mem_usage=True).eval()
    before_parameters = sum(value.numel() for value in model.parameters())
    blocks = _module_by_path(model, config["architecture"]["block_path"])
    attention_path = config["architecture"]["attention_module"]
    mlp_path = config["architecture"]["mlp_module"]
    attention_audit, mlp_audit = [], []
    for index, block in enumerate(blocks):
        attention = _module_by_path(block, attention_path)
        mlp = _module_by_path(block, mlp_path)
        if candidate.attention:
            attention_audit.append({"layer": index, **prune_attention(attention, candidate.target_rate)})
        if candidate.mlp:
            mlp_audit.append({"layer": index, **prune_mlp(
                mlp, candidate.target_rate, int(config["constraints"]["mlp_alignment"]))})
    layer_audit = None
    if candidate.layer_indices:
        layer_audit = prune_layers(model, config["architecture"]["block_path"], candidate.layer_indices)
    remaining_blocks = _module_by_path(model, config["architecture"]["block_path"])
    first_attention = _module_by_path(remaining_blocks[0], attention_path)
    first_mlp = _module_by_path(remaining_blocks[0], mlp_path)
    model.config.num_hidden_layers = len(remaining_blocks)
    model.config.num_attention_heads = first_attention.num_heads
    model.config.num_key_value_heads = first_attention.num_key_value_heads
    # Qwen2 releases predating explicit head_dim support always re-derived it
    # as hidden_size // num_heads and therefore cannot reload reduced Q width.
    # Persisting it is mandatory and the standalone reload below is the final
    # compatibility gate for the installed Transformers version.
    model.config.head_dim = first_attention.head_dim
    model.config.intermediate_size = first_mlp.gate_proj.out_features
    audit = validate_structure(model, config, candidate)
    if audit["status"] != "pass":
        raise ValueError(f"Pruned structure is invalid: {audit['errors']}")
    model.to(device)
    vocab_size = int(model.config.vocab_size)
    smoke_tokens = torch.tensor([[min(int(getattr(model.config, "bos_token_id", 1) or 1), vocab_size - 1)]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=smoke_tokens, use_cache=False).logits
    if tuple(logits.shape[:2]) != (1, 1) or logits.shape[-1] != vocab_size:
        raise ValueError(f"Unexpected smoke-forward logits shape: {tuple(logits.shape)}")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(source), local_files_only=True).save_pretrained(output)
    after_parameters = sum(value.numel() for value in model.parameters())
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(output), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).eval().to(device)
    with torch.inference_mode():
        reload_logits = reloaded(input_ids=smoke_tokens, use_cache=False).logits
    if tuple(reload_logits.shape) != tuple(logits.shape):
        raise ValueError(f"Reloaded logits shape changed: {tuple(logits.shape)} -> {tuple(reload_logits.shape)}")
    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1, "stage": "8.1", "status": "completed", "plan_id": plan_id,
        "source_model": str(source), "device": device, "dtype": dtype_name,
        "transformers_version": transformers_version,
        "before_parameters": before_parameters, "after_parameters": after_parameters,
        "realized_parameter_rate": 1.0 - after_parameters / before_parameters,
        "attention_audit": attention_audit, "mlp_audit": mlp_audit, "layer_audit": layer_audit,
        "structure_validation": audit,
        "smoke_forward": {"status": "pass", "logits_shape": list(logits.shape)},
        "standalone_reload": {"status": "pass", "logits_shape": list(reload_logits.shape)},
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
    }
    _write_json(output / "pruning_manifest.json", manifest)
    return manifest


def record_quality_evidence(project_root: str | Path, config: dict[str, Any], plan_id: str,
                            source: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate_by_id(config, plan_id)
    source_path = Path(source).expanduser().resolve()
    payload = _read_json(source_path)
    metrics = _quality_metrics(payload, str(config["quality_gate"].get("variant", "structured_json")))
    if "baseline_accuracy" not in metrics:
        baseline_path = (root / config["paths"]["baseline_quality_report"]).resolve()
        baseline = _quality_metrics(
            _read_json(baseline_path), str(config["quality_gate"].get("variant", "structured_json")),
        )
        metrics["baseline_accuracy"] = baseline["accuracy"]
        metrics["baseline_total"] = baseline.get("total")
        metrics["baseline_source"] = str(baseline_path)
        metrics["baseline_sha256"] = _sha256(baseline_path)
    output = root / config["paths"]["output_root"] / "evidence" / plan_id / "quality.json"
    normalized = {"schema_version": 1, "stage": "8.1", "status": "completed", "plan_id": plan_id,
                  "source": str(source_path), "source_sha256": _sha256(source_path), "metrics": metrics}
    _write_json(output, normalized)
    receipt = {"plan_id": plan_id, "source": str(source_path), "destination": str(output),
               "sha256": _sha256(output), "metrics": metrics}
    _write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def run_quality_evaluation(project_root: str | Path, config: dict[str, Any], plan_id: str, *,
                           device: str = "cuda:0") -> dict[str, Any]:
    """Run the frozen full FinQA dev evaluation for one pruned artifact."""
    root = Path(project_root).resolve()
    candidate_by_id(config, plan_id)
    export = root / config["paths"]["output_root"] / "exports" / plan_id
    artifact = _read_json(export / "pruning_manifest.json")
    if (artifact.get("status") != "completed"
            or artifact.get("standalone_reload", {}).get("status") != "pass"):
        raise ValueError(f"A validated pruning export is required before evaluation: {plan_id}")
    output = root / config["paths"]["output_root"] / "runs" / plan_id / "full_dev"
    report_path, report = _run_quality_for_model(root, config, plan_id, export, output, device)
    receipt = record_quality_evidence(root, config, plan_id, report_path)
    variant = str(config["quality_gate"].get("variant", "structured_json"))
    return {"plan_id": plan_id, "report": str(report_path), "evidence": receipt,
            "metrics": _quality_metrics(report, variant)}


def run_baseline_quality_evaluation(project_root: str | Path, config: dict[str, Any], *,
                                    model_path: str | Path | None = None,
                                    device: str = "cuda:0") -> dict[str, Any]:
    """Generate the same-source BF16 full-dev reference required by 8.1."""
    root = Path(project_root).resolve()
    source = _resolve_source_model(root, config, model_path)
    output = root / config["paths"]["output_root"] / "baseline"
    report_path, report = _run_quality_for_model(root, config, "structured_pruning_bf16", source, output, device)
    configured = (root / config["paths"]["baseline_quality_report"]).resolve()
    if report_path.resolve() != configured:
        raise ValueError(f"baseline_quality_report must point to {report_path}, got {configured}")
    variant = str(config["quality_gate"].get("variant", "structured_json"))
    return {"model_path": str(source), "report": str(report_path),
            "report_sha256": _sha256(report_path), "metrics": _quality_metrics(report, variant)}


def _run_quality_for_model(root: Path, config: dict[str, Any], model_id: str, model_path: Path,
                           output: Path, device: str) -> tuple[Path, dict[str, Any]]:
    from src.baseline.service import BaselineService
    from src.data.finqa_assets import load_assets
    from src.evaluation.quality_repair import run_quality_repair_experiment

    output.mkdir(parents=True, exist_ok=True)
    baseline = _read_json((root / config["paths"]["baseline_config"]).resolve())
    baseline["device"] = device
    baseline["dtype"] = config.get("dtype", "bfloat16")
    baseline_path = output / "baseline_config.effective.json"
    _write_json(baseline_path, baseline)
    quality = copy.deepcopy(_read_json((root / config["paths"]["quality_config"]).resolve()))
    quality.pop("evaluation_size", None)
    quality.pop("evaluation_seed", None)
    quality["batch_size"] = 1
    quality["pilot_variants"] = [str(config["quality_gate"].get("variant", "structured_json"))]
    quality["show_progress"] = True
    quality["dataset_role"] = "evaluation_dev"
    _write_json(output / "quality_config.effective.json", quality)
    model_manifest = {
        "model_path": str(model_path), "model_id": model_id, "version": "structured-pruning-8.1",
        "commit": _sha256(model_path / "config.json"), "runtime": {"trust_remote_code": False},
    }
    model_manifest_path = output / "model_manifest.json"
    _write_json(model_manifest_path, model_manifest)
    assets = load_assets(root / config["paths"]["dataset_dir"])
    service = BaselineService.from_local_model(
        baseline_path, model_manifest_path, output / "quality_service.jsonl",
    )
    report = run_quality_repair_experiment(
        service, assets.quality_dev, assets.manifest, quality, output, calibration=assets.calibration,
    )
    report_path = output / "comparison_report.json"
    if not report_path.is_file():
        _write_json(report_path, report)
    return report_path, report


def _quality_metrics(payload: dict[str, Any], variant_name: str) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        variants = payload.get("variants", [])
        match = next((row for row in variants if row.get("variant") == variant_name), None)
        if not isinstance(match, dict) or not isinstance(match.get("metrics"), dict):
            raise ValueError(f"Quality report has no metrics for variant {variant_name}")
        metrics = match["metrics"]
    result = dict(metrics)
    accuracy = result.get("accuracy", result.get("numeric_accuracy"))
    if accuracy is None and isinstance(result.get("correct"), int) and int(result.get("total", 0)) > 0:
        accuracy = result["correct"] / result["total"]
    if accuracy is None:
        raise ValueError("Quality metrics require accuracy, numeric_accuracy, or correct/total")
    result["accuracy"] = float(accuracy)
    return result


def pending(project_root: str | Path, config: dict[str, Any]) -> dict[str, list[str]]:
    root = Path(project_root).resolve()
    output = root / config["paths"]["output_root"]
    result: dict[str, list[str]] = {}
    for candidate in build_candidates(config):
        missing = []
        if not (output / "exports" / candidate.plan_id / "pruning_manifest.json").is_file():
            missing.append("export")
        quality = output / "evidence" / candidate.plan_id / "quality.json"
        if not quality.is_file() or not quality.with_suffix(".receipt.json").is_file():
            missing.append("quality")
        result[candidate.plan_id] = missing
    return result


def finalize(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / config["paths"]["output_root"]
    missing = pending(root, config)
    incomplete = {key: value for key, value in missing.items() if value}
    if incomplete:
        raise ValueError(f"Structured pruning evidence is incomplete: {incomplete}")
    rows = []
    max_drop = float(config["quality_gate"]["max_absolute_accuracy_drop"])
    for candidate in build_candidates(config):
        artifact = _read_json(output / "exports" / candidate.plan_id / "pruning_manifest.json")
        structure_pass = (
            artifact.get("status") == "completed"
            and artifact.get("structure_validation", {}).get("status") == "pass"
            and artifact.get("standalone_reload", {}).get("status") == "pass"
        )
        if not structure_pass:
            raise ValueError(f"Invalid or non-reloadable pruning artifact: {candidate.plan_id}")
        quality_path = output / "evidence" / candidate.plan_id / "quality.json"
        quality = _read_json(quality_path)
        receipt = _read_json(quality_path.with_suffix(".receipt.json"))
        if receipt.get("plan_id") != candidate.plan_id or receipt.get("sha256") != _sha256(quality_path):
            raise ValueError(f"Quality evidence receipt mismatch: {candidate.plan_id}")
        metrics = quality.get("metrics", {})
        accuracy = float(metrics["accuracy"])
        baseline_accuracy = float(metrics["baseline_accuracy"])
        drop = baseline_accuracy - accuracy
        parse_rate = float(metrics.get("parse_rate", 1.0))
        parse_pass = parse_rate >= float(config["quality_gate"].get("min_parse_rate", 0.0))
        sample_count_match = (
            metrics.get("baseline_total") is None
            or metrics.get("total") == metrics.get("baseline_total")
        )
        accuracy_pass = drop <= max_drop
        rows.append({"plan_id": candidate.plan_id, "target_rate": candidate.target_rate,
                     "realized_parameter_rate": artifact["realized_parameter_rate"],
                     "accuracy": accuracy, "baseline_accuracy": baseline_accuracy,
                     "absolute_accuracy_drop": drop, "parse_rate": parse_rate,
                     "accuracy_pass": accuracy_pass, "parse_pass": parse_pass,
                     "sample_count_match": sample_count_match,
                     "quality_pass": accuracy_pass and parse_pass and sample_count_match,
                     "structure_pass": structure_pass})
    rows.sort(key=lambda row: row["target_rate"])
    turning = next((row for row in rows if not row["accuracy_pass"]), None)
    report = {"schema_version": 1, "stage": "8.1", "status": "completed",
              "quality_turning_point": turning["plan_id"] if turning else None,
              "quality_gate": config["quality_gate"], "candidates": rows}
    _write_json(output / "final_report.json", report)
    return report

