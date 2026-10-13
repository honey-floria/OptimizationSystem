"""Finalize TODO 7.2 module sensitivity with frozen promotion and paired evaluation."""

from __future__ import annotations

import copy
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.quantization.layer_sensitivity_final import (
    _group_metrics,
    _transition_detail,
    paired_bootstrap_interval,
)
from src.quantization.layer_sensitivity_setup import (
    build_sample_entries,
    canonical_json_sha256,
    load_json,
    resolve_project_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from src.quantization.module_sensitivity_experiments import (
    TERMINAL_RESULT_STATUSES,
    _kv_cache_runtime_arguments,
    _load_prepared_plan,
    _structured_variant,
    pending_plan_ids,
    quantize_model_for_module_plan,
)


FINAL_SCHEMA_VERSION = 1
REQUIRED_ARTIFACTS = (
    "experiment_manifest.json",
    "module_metrics.jsonl",
    "sample_transitions.jsonl",
    "ranking.json",
    "ranking.csv",
    "protection_policy.json",
    "report.md",
    "validation.json",
)


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("final_evaluation")
    if not isinstance(settings, dict):
        raise ValueError("final_evaluation is missing from module sensitivity config.")
    if set(settings.get("suites", {})) != {"full_dev", "high_risk", "determinism"}:
        raise ValueError("Module final evaluation must define all three suites.")
    for key in (
        "output_root", "advancement_output", "reference_baselines",
        "required_special_pilot_plans", "recovery_candidate_plans",
        "determinism_sentinel_plans", "advancement_rules", "bootstrap", "quality_gates",
    ):
        if key not in settings:
            raise ValueError(f"final_evaluation.{key} is required.")
    if set(settings["reference_baselines"]) != {"full_dev", "high_risk"}:
        raise ValueError("Final evaluation requires frozen full-dev and high-risk references.")
    if len(settings["required_special_pilot_plans"]) != len(
        set(settings["required_special_pilot_plans"])
    ):
        raise ValueError("Special pilot plan IDs must be unique.")
    if len(settings["recovery_candidate_plans"]) != len(
        set(settings["recovery_candidate_plans"])
    ):
        raise ValueError("Recovery candidate plan IDs must be unique.")
    bootstrap = settings["bootstrap"]
    if int(bootstrap.get("iterations", 0)) <= 0 or not 0 < float(
        bootstrap.get("confidence", 0)
    ) < 1:
        raise ValueError("Final bootstrap requires positive iterations and 0 < confidence < 1.")
    return settings


def _output_root(root: Path, config: dict[str, Any]) -> Path:
    return resolve_project_path(root, _settings(config)["output_root"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _predictions(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = load_json(directory / "comparison_report.json")
    variant = _structured_variant(report)
    path = directory / variant["predictions_file"]
    if sha256_file(path) != variant["predictions_sha256"]:
        raise ValueError(f"Prediction SHA256 mismatch: {path}")
    return _read_jsonl(path), report


def _pilot_result(root: Path, config: dict[str, Any], plan_id: str) -> dict[str, Any]:
    path = resolve_project_path(root, config["paths"]["experiment_output_root"]) / plan_id / "result.json"
    if not path.is_file():
        raise ValueError(f"Pilot result is missing: {plan_id}")
    return load_json(path)


def _pilot_result_path(root: Path, config: dict[str, Any], plan_id: str) -> Path:
    return (
        resolve_project_path(root, config["paths"]["experiment_output_root"])
        / plan_id
        / "result.json"
    )


def _storage_benefit(result: dict[str, Any], kv_reference: dict[str, Any] | None) -> int:
    storage = result.get("special_storage") or {}
    if int(storage.get("savings_bytes", 0)) > 0:
        return int(storage["savings_bytes"])
    benchmark = result.get("kv_cache_benchmark") or {}
    reference = (kv_reference or {}).get("kv_cache_benchmark") or {}
    candidate_bytes = benchmark.get("cache_tensor_bytes")
    reference_bytes = reference.get("cache_tensor_bytes")
    if isinstance(candidate_bytes, int) and isinstance(reference_bytes, int):
        return reference_bytes - candidate_bytes
    return 0


def select_advancement(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Freeze pilot-based promotion before any TODO 7.2 full-dev runs."""

    root = Path(project_root).resolve()
    settings = _settings(config)
    rules = settings["advancement_rules"]
    required = list(settings["required_special_pilot_plans"])
    if settings.get("require_complete_pilot_matrix"):
        incomplete = pending_plan_ids(root, config)
        if incomplete:
            raise ValueError(
                f"Pilot matrix must be complete before advancement; pending: {incomplete}"
            )
    if settings.get("recovery_candidate_scope") == "all_isolated_attention_mlp":
        matrix_path = (
            resolve_project_path(root, config["paths"]["experiment_output_root"])
            / "experiment_manifest.json"
        )
        matrix = load_json(matrix_path)
        allowed_kinds = {
            "attention_all_int8", "mlp_all_int8",
            "layer_attention_int8", "layer_mlp_int8",
        }
        recovery_ids = [
            plan["plan_id"] for plan in matrix["plans"]
            if plan.get("kind") in allowed_kinds
        ]
    else:
        recovery_ids = list(settings["recovery_candidate_plans"])
    all_required = recovery_ids + required
    results: dict[str, dict[str, Any]] = {}
    pending = []
    for plan_id in all_required:
        try:
            result = _pilot_result(root, config, plan_id)
        except ValueError:
            pending.append(plan_id)
            continue
        if result.get("status") not in TERMINAL_RESULT_STATUSES:
            pending.append(plan_id)
        results[plan_id] = result
    if pending:
        raise ValueError(f"Required pilot plans are still pending: {pending}")

    kv_reference = results.get("kv_cache_fp16_ref")
    selected_recovery = []
    selected_special = []
    decisions = []
    for plan_id in recovery_ids:
        result = results[plan_id]
        gain = int((result.get("transitions_vs_all_int4") or {}).get("net_correct_gain", -10**9))
        eligible = result.get("status") == "completed" and gain >= int(
            rules["recovery_min_net_correct_gain"]
        )
        decisions.append(
            {
                "plan_id": plan_id,
                "category": "recovery",
                "eligible": eligible,
                "status": result.get("status"),
                "pilot_net_correct_change": gain,
                "pilot_result_sha256": sha256_file(
                    _pilot_result_path(root, config, plan_id)
                ),
            }
        )
        if eligible:
            selected_recovery.append(plan_id)

    special_ids = [plan_id for plan_id in required if plan_id != "kv_cache_fp16_ref"]
    for plan_id in special_ids:
        result = results[plan_id]
        transitions = result.get("transitions_vs_all_int4") or {}
        change = int(transitions.get("net_correct_change", -10**9))
        benefit = _storage_benefit(result, kv_reference)
        is_kv = result.get("plan", {}).get("kind") == "kv_cache_quantized"
        residency = bool((result.get("kv_cache_benchmark") or {}).get("low_bit_residency_verified"))
        eligible = (
            result.get("status") == "completed"
            and change >= int(rules["special_min_net_correct_change"])
            and (not rules.get("require_positive_storage_savings") or benefit > 0)
            and (not is_kv or not rules.get("kv_cache_require_low_bit_residency") or residency)
        )
        decisions.append(
            {
                "plan_id": plan_id,
                "category": "special",
                "eligible": eligible,
                "status": result.get("status"),
                "pilot_net_correct_change": change if result.get("status") == "completed" else None,
                "benefit_bytes": benefit,
                "low_bit_residency_verified": residency if is_kv else None,
                "reason": result.get("reason") if result.get("status") == "unsupported" else None,
                "pilot_result_sha256": sha256_file(
                    _pilot_result_path(root, config, plan_id)
                ),
            }
        )
        if eligible:
            selected_special.append(plan_id)

    payload = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.2.4",
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rules": rules,
        "selected_recovery_plans": selected_recovery,
        "selected_special_plans": selected_special,
        "excluded_plans": [row["plan_id"] for row in decisions if not row["eligible"]],
        "decisions": decisions,
    }
    payload["selection_sha256"] = canonical_json_sha256(payload)
    write_json(resolve_project_path(root, settings["advancement_output"]), payload)
    return payload


def prepare_final_evaluation(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = _settings(config)
    advancement_path = resolve_project_path(root, settings["advancement_output"])
    advancement = load_json(advancement_path)
    stored_hash = advancement.get("selection_sha256")
    unhashed = {key: value for key, value in advancement.items() if key != "selection_sha256"}
    if advancement.get("status") != "frozen" or canonical_json_sha256(unhashed) != stored_hash:
        raise ValueError("Advancement selection is not a valid frozen contract.")
    selected = advancement["selected_recovery_plans"] + advancement["selected_special_plans"]
    if not selected:
        raise ValueError("No module-sensitivity plan passed pilot advancement.")
    for plan_id in selected:
        _load_prepared_plan(root, config, plan_id)

    determinism = list(settings["determinism_sentinel_plans"])
    if len(determinism) != len(set(determinism)):
        raise ValueError("Determinism sentinel plan IDs must be unique.")
    for plan_id in determinism:
        _load_prepared_plan(root, config, plan_id)
    suites = {
        "full_dev": {**settings["suites"]["full_dev"], "plan_ids": selected},
        "high_risk": {**settings["suites"]["high_risk"], "plan_ids": selected},
        "determinism": {**settings["suites"]["determinism"], "plan_ids": determinism},
    }
    freeze_path = resolve_project_path(root, config["paths"]["input_freeze"])
    freeze = load_json(freeze_path)
    setup_dir = resolve_project_path(root, config["paths"]["setup_dir"])
    subsets_path = setup_dir / "frozen_subsets.json"
    labels_path = setup_dir / "business_labels.jsonl"
    subsets = load_json(subsets_path)
    if sha256_file(labels_path) != freeze["business_labels_sha256"]:
        raise ValueError("Business-label evidence changed after input freeze.")
    for name, expected_hash in freeze["subset_hashes"].items():
        if canonical_json_sha256(subsets[name]) != expected_hash:
            raise ValueError(f"Frozen subset changed: {name}")

    references = {}
    for name, value in settings["reference_baselines"].items():
        directory = resolve_project_path(root, value)
        references[name] = {
            "directory": str(directory),
            "result_sha256": sha256_file(directory / "result.json"),
            "report_sha256": sha256_file(directory / "comparison_report.json"),
        }
    matrix_path = resolve_project_path(root, config["paths"]["experiment_output_root"]) / "experiment_manifest.json"
    manifest = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.2.4-7.2.5",
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "advancement": str(advancement_path),
        "advancement_sha256": sha256_file(advancement_path),
        "matrix_manifest": str(matrix_path),
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "input_freeze": str(freeze_path),
        "input_freeze_sha256": sha256_file(freeze_path),
        "frozen_subsets": str(subsets_path),
        "frozen_subsets_sha256": sha256_file(subsets_path),
        "business_labels": str(labels_path),
        "business_labels_sha256": sha256_file(labels_path),
        "subset_hashes": freeze["subset_hashes"],
        "selected_recovery_plans": advancement["selected_recovery_plans"],
        "selected_special_plans": advancement["selected_special_plans"],
        "suites": suites,
        "references": references,
        "bootstrap": settings["bootstrap"],
        "quality_gates": settings["quality_gates"],
    }
    manifest["contract_sha256"] = canonical_json_sha256(
        {key: manifest[key] for key in (
            "advancement_sha256", "matrix_manifest_sha256", "input_freeze_sha256",
            "frozen_subsets_sha256", "business_labels_sha256", "subset_hashes",
            "selected_recovery_plans",
            "selected_special_plans", "suites", "references", "bootstrap", "quality_gates"
        )}
    )
    write_json(_output_root(root, config) / "experiment_manifest.json", manifest)
    return manifest


def _load_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = load_json(_output_root(root, config) / "experiment_manifest.json")
    if manifest.get("status") not in {"prepared", "completed"}:
        raise ValueError("Module final evaluation is not prepared.")
    if sha256_file(Path(manifest["advancement"])) != manifest["advancement_sha256"]:
        raise ValueError("Frozen advancement evidence changed.")
    advancement = load_json(Path(manifest["advancement"]))
    for decision in advancement["decisions"]:
        if sha256_file(
            _pilot_result_path(root, config, decision["plan_id"])
        ) != decision["pilot_result_sha256"]:
            raise ValueError(
                f"Pilot result changed after advancement: {decision['plan_id']}"
            )
    if sha256_file(Path(manifest["matrix_manifest"])) != manifest["matrix_manifest_sha256"]:
        raise ValueError("Module matrix changed after final preparation.")
    for field in ("input_freeze", "frozen_subsets", "business_labels"):
        if sha256_file(Path(manifest[field])) != manifest[f"{field}_sha256"]:
            raise ValueError(f"Frozen final-evaluation input changed: {field}")
    for reference in manifest["references"].values():
        directory = Path(reference["directory"])
        if sha256_file(directory / "result.json") != reference["result_sha256"]:
            raise ValueError("Frozen reference result changed after final preparation.")
        if sha256_file(directory / "comparison_report.json") != reference["report_sha256"]:
            raise ValueError("Frozen reference report changed after final preparation.")
    contract_fields = (
        "advancement_sha256", "matrix_manifest_sha256", "input_freeze_sha256",
        "frozen_subsets_sha256", "business_labels_sha256", "subset_hashes",
        "selected_recovery_plans", "selected_special_plans", "suites", "references",
        "bootstrap", "quality_gates",
    )
    if canonical_json_sha256({key: manifest[key] for key in contract_fields}) != manifest["contract_sha256"]:
        raise ValueError("Module final-evaluation contract changed after preparation.")
    return manifest


def _validate_dataset_against_freeze(
    dataset: Any,
    *,
    frozen_entries: list[dict[str, Any]],
    expected_hash: str,
    suite: str,
) -> None:
    entries = build_sample_entries(dataset)
    if entries != frozen_entries or canonical_json_sha256(entries) != expected_hash:
        raise ValueError(f"{suite} dataset differs from the frozen sample list.")


def pending_final_plan_ids(
    project_root: str | Path, config: dict[str, Any], suite: str | None = None
) -> dict[str, list[str]]:
    root = Path(project_root).resolve()
    manifest = _load_manifest(root, config)
    names = [suite] if suite else list(manifest["suites"])
    pending = {}
    for name in names:
        if name not in manifest["suites"]:
            raise ValueError(f"Unknown final suite: {name}")
        pending[name] = []
        for plan_id in manifest["suites"][name]["plan_ids"]:
            path = _output_root(root, config) / name / plan_id / "result.json"
            if not path.is_file() or load_json(path).get("status") != "completed":
                pending[name].append(plan_id)
    return pending


def run_final_plan(
    project_root: str | Path,
    config: dict[str, Any],
    suite: str,
    plan_id: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = _load_manifest(root, config)
    if suite not in manifest["suites"] or plan_id not in manifest["suites"][suite]["plan_ids"]:
        raise ValueError(f"Plan {plan_id} is not registered for suite {suite}.")
    plan, _, inventory, _ = _load_prepared_plan(root, config, plan_id)
    output = _output_root(root, config) / suite / plan_id
    result_path = output / "result.json"
    if result_path.is_file() and load_json(result_path).get("status") == "completed":
        return {**load_json(result_path), "resume_action": "skipped_completed"}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Module final evaluation requires torch and transformers.") from exc
    from src.baseline.service import BaselineService
    from src.data.finqa_assets import load_assets
    from src.evaluation.quality_repair import run_quality_repair_experiment

    suite_contract = manifest["suites"][suite]
    evaluation = config["pilot_evaluation"]
    assets = load_assets(resolve_project_path(root, config["paths"]["dataset_dir"]))
    role = suite_contract["dataset_role"]
    if role == "evaluation_dev":
        dataset = assets.quality_dev
    elif role == "high_risk_regression":
        dataset = assets.high_risk_regression
    else:
        raise ValueError(f"Unsupported final evaluation dataset role: {role}")
    expected_size = int(suite_contract.get("pilot_size", suite_contract.get("expected_size")))
    frozen_name = suite_contract.get("frozen_subset")
    if frozen_name:
        frozen_subsets = load_json(Path(manifest["frozen_subsets"]))
        _validate_dataset_against_freeze(
            dataset,
            frozen_entries=frozen_subsets[frozen_name],
            expected_hash=manifest["subset_hashes"][frozen_name],
            suite=suite,
        )
        if len(dataset) != expected_size:
            raise ValueError(f"{suite} dataset size differs from its frozen contract.")

    baseline = load_json(resolve_project_path(root, config["paths"]["baseline_config"]))
    baseline["device"] = evaluation["device"]
    baseline["dtype"] = evaluation["dtype"]
    baseline["generation"]["max_new_tokens"] = evaluation["max_new_tokens"]
    capability = load_json(resolve_project_path(root, config["paths"]["output_root"]) / "kv_cache_capability.json")
    kv_arguments = _kv_cache_runtime_arguments(plan, config, capability)
    if plan["kind"] == "kv_cache_quantized" and kv_arguments is None:
        raise ValueError(f"KV Cache runtime support disappeared after pilot: {plan_id}")
    if kv_arguments:
        baseline["generation"].update(kv_arguments)
    quality = load_json(resolve_project_path(root, config["paths"]["quality_config"]))
    quality["batch_size"] = evaluation["batch_size"]
    quality["generation"]["max_new_tokens"] = evaluation["max_new_tokens"]
    quality["pilot_variants"] = [evaluation["variant"]]
    quality["dataset_role"] = role
    quality["show_progress"] = True
    if "pilot_size" in suite_contract:
        quality["evaluation_size"] = expected_size
        quality["evaluation_seed"] = int(suite_contract["pilot_seed"])
    else:
        quality.pop("evaluation_size", None)
        quality.pop("evaluation_seed", None)
    write_json(output / "baseline_config.effective.json", baseline)
    write_json(output / "quality_config.effective.json", quality)

    merged = load_json(resolve_project_path(root, config["paths"]["merged_manifest"]))
    resolved_model = Path(model_path or merged["merged_model"]["path"]).expanduser().resolve()
    if not resolved_model.is_dir():
        raise ValueError(f"Merged model directory is unavailable: {resolved_model}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[evaluation["dtype"]]
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_model), local_files_only=True, trust_remote_code=False,
        dtype=dtype, low_cpu_mem_usage=True,
    )
    error_report = quantize_model_for_module_plan(model, plan, config, inventory, torch)
    write_json(output / "quantization_error.json", error_report)
    model.to(evaluation["device"])
    model.eval()
    service = BaselineService(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        config=baseline,
        model_metadata={
            "model_id": "Qwen2.5-7B-Instruct-QLoRA-step724-module-final",
            "version": config["expected_identity"]["merged_tree_sha256"],
            "quantization": {"plan_id": plan_id, "kv_cache": kv_arguments},
        },
        log_file=output / "quality_service.jsonl",
    )
    report = run_quality_repair_experiment(
        service, dataset, assets.manifest, quality, output, calibration=assets.calibration
    )
    if report["dataset"]["size"] != expected_size:
        raise ValueError(f"{suite} produced the wrong sample count.")
    if "expected_indices_sha256" in suite_contract:
        observed = report.get("dataset", {}).get("sample", {}).get("indices_sha256")
        if observed != suite_contract["expected_indices_sha256"]:
            raise ValueError("Determinism suite used the wrong pilot sample.")
    variant = _structured_variant(report)
    result = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.2.4",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "plan": plan,
        "metrics": variant["metrics"],
        "structured_output_rate": variant.get("structured_output_rate"),
        "format_retry_count": variant.get("format_retry_count"),
        "predictions_sha256": variant["predictions_sha256"],
        "comparison_report_sha256": sha256_file(output / "comparison_report.json"),
        "quantization_error_sha256": sha256_file(output / "quantization_error.json"),
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    return result


def _audit_runs(output: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    invalid = []
    for suite, contract in manifest["suites"].items():
        for plan_id in contract["plan_ids"]:
            directory = output / suite / plan_id
            try:
                result = load_json(directory / "result.json")
                rows, report = _predictions(directory)
                variant = _structured_variant(report)
                expected = int(contract.get("pilot_size", contract.get("expected_size")))
                if (
                    result.get("status") != "completed"
                    or result.get("suite") != suite
                    or result.get("plan", {}).get("plan_id") != plan_id
                ):
                    raise ValueError("result status, suite, or plan differs")
                if len(rows) != expected or report["dataset"]["size"] != expected:
                    raise ValueError("prediction count differs")
                if report.get("dataset", {}).get("role") != contract["dataset_role"]:
                    raise ValueError("dataset role differs")
                if result.get("comparison_report_sha256") != sha256_file(
                    directory / "comparison_report.json"
                ):
                    raise ValueError("comparison report hash differs")
                prediction_path = directory / variant["predictions_file"]
                prediction_hash = sha256_file(prediction_path)
                if (
                    result["predictions_sha256"] != prediction_hash
                    or variant["predictions_sha256"] != prediction_hash
                ):
                    raise ValueError("prediction hash differs")
                if "expected_indices_sha256" in contract and (
                    report.get("dataset", {}).get("sample", {}).get("indices_sha256")
                    != contract["expected_indices_sha256"]
                ):
                    raise ValueError("determinism sample hash differs")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid.append({"suite": suite, "plan_id": plan_id, "reason": str(exc)})
    return invalid


def _write_ranking_csv(path: Path, recovery: list[dict[str, Any]], special: list[dict[str, Any]]) -> None:
    fields = [
        "category", "rank", "plan_id", "kind", "target_precision", "full_dev_correct",
        "full_dev_total", "net_correct_change", "recovered", "regressed",
        "high_risk_net", "long_context_net", "bootstrap_lower", "bootstrap_upper",
        "benefit_bytes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for category, rows in (("recovery", recovery), ("special", special)):
            for row in rows:
                writer.writerow(
                    {
                        "category": category,
                        "rank": row["rank"],
                        "plan_id": row["plan_id"],
                        "kind": row["kind"],
                        "target_precision": row.get("target_precision"),
                        "full_dev_correct": row["full_dev"]["candidate_correct"],
                        "full_dev_total": row["full_dev"]["total"],
                        "net_correct_change": row["full_dev"]["net_correct_gain"],
                        "recovered": row["full_dev"]["recovered"],
                        "regressed": row["full_dev"]["regressed"],
                        "high_risk_net": row["high_risk"]["net_correct_gain"],
                        "long_context_net": row["long_context"]["net_correct_gain"],
                        "bootstrap_lower": row["bootstrap"]["lower"],
                        "bootstrap_upper": row["bootstrap"]["upper"],
                        "benefit_bytes": row["benefit_bytes"],
                    }
                )


def finalize_module_sensitivity(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = _settings(config)
    output = _output_root(root, config)
    manifest = _load_manifest(root, config)
    pending = pending_final_plan_ids(root, config)
    if any(pending.values()):
        raise ValueError(f"Module final evaluation still has pending plans: {pending}")
    invalid = _audit_runs(output, manifest)
    if invalid:
        raise ValueError(f"Module final evaluation audit failed: {invalid}")

    full_base, _ = _predictions(Path(manifest["references"]["full_dev"]["directory"]))
    risk_base, _ = _predictions(Path(manifest["references"]["high_risk"]["directory"]))
    labels = _read_jsonl(Path(manifest["business_labels"]))
    frozen_subsets = load_json(Path(manifest["frozen_subsets"]))
    label_indices = {
        label: {int(row["source_index"]) for row in labels if label in row["labels"]}
        for label in ("amount", "rate_or_ratio", "date_or_term")
    }
    long_indices = {int(row["index"]) for row in frozen_subsets["long_context"]}
    bootstrap = settings["bootstrap"]
    selected_recovery = set(manifest["selected_recovery_plans"])
    rows = []
    transitions = []
    selected = manifest["selected_recovery_plans"] + manifest["selected_special_plans"]
    for ordinal, plan_id in enumerate(selected):
        full_candidate, full_report = _predictions(output / "full_dev" / plan_id)
        risk_candidate, risk_report = _predictions(output / "high_risk" / plan_id)
        pilot = _pilot_result(root, config, plan_id)
        full_metrics = _group_metrics(full_base, full_candidate, {int(row["source_index"]) for row in full_base})
        risk_metrics = _group_metrics(risk_base, risk_candidate, {int(row["source_index"]) for row in risk_base})
        row = {
            "plan_id": plan_id,
            "category": "recovery" if plan_id in selected_recovery else "special",
            "kind": pilot["plan"]["kind"],
            "target_precision": pilot["plan"].get("target_precision"),
            "full_dev": full_metrics,
            "high_risk": risk_metrics,
            "long_context": _group_metrics(full_base, full_candidate, long_indices),
            "business_subsets": {
                label: _group_metrics(full_base, full_candidate, indices)
                for label, indices in label_indices.items()
            },
            "bootstrap": paired_bootstrap_interval(
                full_base,
                full_candidate,
                iterations=int(bootstrap["iterations"]),
                confidence=float(bootstrap["confidence"]),
                seed=int(bootstrap["seed"]) + ordinal,
            ),
            "full_dev_format_retry_count": _structured_variant(full_report).get("format_retry_count"),
            "high_risk_format_retry_count": _structured_variant(risk_report).get("format_retry_count"),
            "benefit_bytes": _storage_benefit(
                pilot,
                _pilot_result(root, config, "kv_cache_fp16_ref")
                if pilot["plan"]["kind"] == "kv_cache_quantized" else None,
            ),
            "pilot_evidence": {
                "metrics": pilot.get("metrics"),
                "transitions": pilot.get("transitions_vs_all_int4"),
                "special_storage": pilot.get("special_storage"),
                "kv_cache_benchmark": pilot.get("kv_cache_benchmark"),
                "lm_head_probe": pilot.get("lm_head_probe"),
            },
        }
        rows.append(row)
        for suite_name, before_rows, after_rows in (
            ("full_dev", full_base, full_candidate),
            ("high_risk", risk_base, risk_candidate),
        ):
            for before, after in zip(before_rows, after_rows):
                transitions.append(
                    {"suite": suite_name, "plan_id": plan_id, **_transition_detail(before, after)}
                )

    recovery = sorted(
        (copy.deepcopy(row) for row in rows if row["category"] == "recovery"),
        key=lambda row: (
            -row["full_dev"]["net_correct_gain"], -row["full_dev"]["recovered"],
            -row["high_risk"]["net_correct_gain"], -row["long_context"]["net_correct_gain"],
            row["plan_id"],
        ),
    )
    special = sorted(
        (copy.deepcopy(row) for row in rows if row["category"] == "special"),
        key=lambda row: (
            -row["full_dev"]["net_correct_gain"], -row["high_risk"]["net_correct_gain"],
            -row["benefit_bytes"], row["plan_id"],
        ),
    )
    for ranked in (recovery, special):
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank

    gates = settings["quality_gates"]
    protected = [
        row["plan_id"] for row in recovery
        if (not gates["recovery_require_positive_full_dev_gain"] or row["full_dev"]["net_correct_gain"] > 0)
        and (not gates["require_nonnegative_high_risk_gain"] or row["high_risk"]["net_correct_gain"] >= 0)
        and (not gates["require_nonnegative_long_context_gain"] or row["long_context"]["net_correct_gain"] >= 0)
    ]
    accepted_special = [
        row for row in special
        if row["full_dev"]["net_correct_gain"] >= -int(gates["special_max_full_dev_loss"])
        and row["high_risk"]["net_correct_gain"] >= -int(gates["special_max_high_risk_loss"])
        and row["benefit_bytes"] > 0
    ]
    special_module_precisions: dict[str, str] = {}
    for row in accepted_special:
        if row["kind"] != "kv_cache_quantized":
            special_module_precisions.setdefault(row["kind"], row["target_precision"])
    policy = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.2.5",
        "status": "candidate_policy",
        "protected_recovery_plans": protected,
        "special_module_precisions": special_module_precisions,
        "kv_cache_precision": next(
            (row["target_precision"] for row in accepted_special if row["kind"] == "kv_cache_quantized"),
            "fp16",
        ),
        "quality_gates": gates,
        "backend_constraints": {
            "weight_results_are_fake_quant": True,
            "weight_deployment_benchmark_required": True,
            "kv_cache_requires_verified_low_bit_residency": True,
        },
        "prohibited_combinations": [
            {
                "plans": ["candidate_l23_all_l10_mlp_int8"],
                "reason": "Pilot added storage without incremental correct answers over layer 23 alone.",
            }
        ],
    }
    ranking_payload = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.2.5",
        "status": "candidate_subset_final",
        "claim_boundary": "Only pre-registered promoted plans were evaluated; this is not a full 28-layer module ranking.",
        "recovery_ranking": recovery,
        "special_loss_benefit_ranking": special,
    }
    write_jsonl(output / "module_metrics.jsonl", rows)
    write_jsonl(output / "sample_transitions.jsonl", sorted(transitions, key=lambda row: (row["suite"], row["plan_id"], row["source_index"])))
    write_json(output / "ranking.json", ranking_payload)
    _write_ranking_csv(output / "ranking.csv", recovery, special)
    write_json(output / "protection_policy.json", policy)

    determinism = {}
    pilot_root = resolve_project_path(root, config["paths"]["experiment_output_root"])
    for plan_id in manifest["suites"]["determinism"]["plan_ids"]:
        original, original_report = _predictions(pilot_root / plan_id)
        repeated, repeated_report = _predictions(output / "determinism" / plan_id)
        determinism[plan_id] = {
            "status": "pass" if original == repeated else "fail",
            "original_sha256": _structured_variant(original_report)["predictions_sha256"],
            "repeat_sha256": _structured_variant(repeated_report)["predictions_sha256"],
            "exact_row_match": original == repeated,
        }
    determinism_pass = all(row["status"] == "pass" for row in determinism.values())
    validation = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.2.5",
        "status": "pass" if determinism_pass else "fail",
        "pending": pending,
        "invalid_runs": invalid,
        "determinism": determinism,
        "selected_recovery_count": len(recovery),
        "selected_special_count": len(special),
        "full_dev_size": len(full_base),
        "high_risk_size": len(risk_base),
        "long_context_size": len(long_indices),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
    }
    write_json(output / "validation.json", validation)
    report_lines = [
        "# TODO 7.2 Module Sensitivity Candidate Report", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Claim boundary", "",
        "Only the frozen promoted subset is ranked. Weight experiments use fake-quant and do not prove deployment latency or memory gains.", "",
        "## Attention/MLP recovery ranking", "",
        "| Rank | Plan | Full dev | Net | High-risk net | Long-context net |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    report_lines.extend(
        f"| {row['rank']} | `{row['plan_id']}` | {row['full_dev']['candidate_correct']}/{row['full_dev']['total']} | {row['full_dev']['net_correct_gain']} | {row['high_risk']['net_correct_gain']} | {row['long_context']['net_correct_gain']} |"
        for row in recovery
    )
    report_lines.extend([
        "", "## Special module / KV Cache loss-benefit ranking", "",
        "| Rank | Plan | Full dev net | High-risk net | Benefit bytes |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    report_lines.extend(
        f"| {row['rank']} | `{row['plan_id']}` | {row['full_dev']['net_correct_gain']} | {row['high_risk']['net_correct_gain']} | {row['benefit_bytes']} |"
        for row in special
    )
    report_lines.extend([
        "", "## Candidate protection policy", "",
        f"Protected recovery plans: `{protected}`", "",
        f"Special module precisions: `{policy['special_module_precisions']}`", "",
        f"KV Cache precision: `{policy['kv_cache_precision']}`", "",
    ])
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    if not determinism_pass:
        raise ValueError("Module sensitivity determinism audit failed.")
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["artifact_sha256"] = {
        name: sha256_file(output / name)
        for name in REQUIRED_ARTIFACTS if name != "experiment_manifest.json"
    }
    write_json(output / "experiment_manifest.json", manifest)
    return {"validation": validation, "ranking": ranking_payload, "protection_policy": policy}
