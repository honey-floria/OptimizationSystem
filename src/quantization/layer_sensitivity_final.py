"""Run and finalize the remaining TODO 7.1 business evaluations."""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.quantization.layer_sensitivity import (
    _evaluation_dtype,
    _structured_variant,
    _validate_runtime_blocks,
    load_prepared_plan,
    paired_prediction_transitions,
    quantize_model_for_plan,
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


FINAL_SCHEMA_VERSION = 1
REQUIRED_FINAL_ARTIFACTS = (
    "experiment_manifest.json",
    "layer_metrics.jsonl",
    "sample_transitions.jsonl",
    "ranking.json",
    "ranking.csv",
    "report.md",
    "validation.json",
)


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("final_evaluation")
    if not isinstance(settings, dict):
        raise ValueError("final_evaluation is missing from the sensitivity config.")
    required_suites = {"full_dev", "high_risk", "determinism"}
    if set(settings.get("suites", {})) != required_suites:
        raise ValueError("final_evaluation must define full_dev, high_risk, and determinism.")
    return settings


def _output_root(root: Path, config: dict[str, Any]) -> Path:
    return resolve_project_path(root, _settings(config)["output_root"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _predictions_from_dir(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = load_json(directory / "comparison_report.json")
    variant = _structured_variant(report)
    path = directory / variant["predictions_file"]
    if sha256_file(path) != variant["predictions_sha256"]:
        raise ValueError(f"Prediction SHA256 changed: {path}")
    return _read_jsonl(path), report


def _suite_plan_ids(
    suite: str, experiment: dict[str, Any], settings: dict[str, Any]
) -> list[str]:
    contract = settings["suites"][suite]
    runnable = [
        plan["plan_id"] for plan in experiment["plans"] if plan.get("runnable")
    ]
    requested = contract["plans"]
    plan_ids = runnable if requested == "all_runnable" else list(requested)
    unknown = sorted(set(plan_ids) - set(runnable))
    if unknown:
        raise ValueError(f"Suite {suite} contains unknown plans: {unknown}")
    return plan_ids


def _analysis_scope(
    experiment: dict[str, Any], settings: dict[str, Any], suites: dict[str, Any]
) -> dict[str, Any]:
    layer_plans = {
        int(plan["layer_index"]): plan["plan_id"]
        for plan in experiment["plans"]
        if plan.get("runnable") and plan.get("kind") == "layer_int8"
    }
    configured = settings.get("analysis_scope", {"mode": "full_layer_ranking"})
    mode = configured.get("mode")
    if mode == "full_layer_ranking":
        indices = sorted(layer_plans)
        claim_allowed = True
    elif mode == "candidate_subset":
        raw_indices = configured.get("candidate_layer_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError(
                "candidate_subset requires non-empty candidate_layer_indices."
            )
        indices = [int(index) for index in raw_indices]
        if len(indices) != len(set(indices)):
            raise ValueError("candidate_layer_indices contains duplicates.")
        unknown = sorted(set(indices) - set(layer_plans))
        if unknown:
            raise ValueError(f"Unknown candidate layer indices: {unknown}")
        claim_allowed = False
        if configured.get("full_layer_ranking_claim_allowed") is True:
            raise ValueError(
                "A candidate subset cannot allow a full-layer ranking claim."
            )
    else:
        raise ValueError(f"Unsupported final evaluation analysis scope: {mode}")

    candidate_plan_ids = [layer_plans[index] for index in indices]
    for suite_name in ("full_dev", "high_risk"):
        missing = sorted(set(candidate_plan_ids) - set(suites[suite_name]["plan_ids"]))
        if missing:
            raise ValueError(
                f"Suite {suite_name} is missing scoped candidate plans: {missing}"
            )
    for suite_name in ("full_dev", "high_risk"):
        if "all_int4" not in suites[suite_name]["plan_ids"]:
            raise ValueError(f"Suite {suite_name} must include the all_int4 baseline.")

    return {
        "mode": mode,
        "candidate_layer_indices": indices,
        "candidate_plan_ids": candidate_plan_ids,
        "candidate_layer_count": len(indices),
        "total_model_layers": len(layer_plans),
        "complete_layer_coverage": len(indices) == len(layer_plans),
        "full_layer_ranking_claim_allowed": claim_allowed,
        "selection_basis": configured.get(
            "selection_basis",
            "all runnable layer plans" if mode == "full_layer_ranking" else None,
        ),
    }


def prepare_final_evaluation(
    *, project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = _settings(config)
    matrix_root = resolve_project_path(root, config["paths"]["output_root"])
    matrix = load_json(matrix_root / "experiment_manifest.json")
    if matrix.get("status") != "matrix_prepared":
        raise ValueError("The layer-sensitivity matrix is not prepared.")
    freeze_path = resolve_project_path(root, config["paths"]["input_freeze"])
    freeze = load_json(freeze_path)
    subsets_path = resolve_project_path(root, config["paths"]["setup_dir"]) / "frozen_subsets.json"
    subsets = load_json(subsets_path)
    labels_path = resolve_project_path(root, config["paths"]["setup_dir"]) / "business_labels.jsonl"
    if sha256_file(labels_path) != freeze["business_labels_sha256"]:
        raise ValueError("Business-label evidence changed after input freeze.")
    for name, expected_hash in freeze["subset_hashes"].items():
        if canonical_json_sha256(subsets[name]) != expected_hash:
            raise ValueError(f"Frozen subset changed: {name}")
    suites = {}
    for suite, contract in settings["suites"].items():
        suites[suite] = {
            **contract,
            "plan_ids": _suite_plan_ids(suite, matrix, settings),
        }
    analysis_scope = _analysis_scope(matrix, settings, suites)
    runtime_contract = {
        "evaluation": config["evaluation"],
        "precision": config["precision"],
        "backend": config["backend"],
    }
    manifest = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.1.4-7.1.5",
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix_manifest": str(matrix_root / "experiment_manifest.json"),
        "matrix_manifest_sha256": sha256_file(matrix_root / "experiment_manifest.json"),
        "input_freeze": str(freeze_path),
        "input_freeze_sha256": sha256_file(freeze_path),
        "subset_hashes": freeze["subset_hashes"],
        "business_labels_sha256": freeze["business_labels_sha256"],
        "business_label_counts": freeze["business_label_counts"],
        "analysis_scope": analysis_scope,
        "suites": suites,
        "runtime_contract": runtime_contract,
        "bootstrap": settings["bootstrap"],
        "ranking_tie_break": settings["ranking_tie_break"],
        "protection_policy": settings["protection_policy"],
    }
    manifest["contract_sha256"] = canonical_json_sha256(
        {
            "analysis_scope": analysis_scope,
            "suites": suites,
            "runtime_contract": runtime_contract,
            "bootstrap": settings["bootstrap"],
            "ranking_tie_break": settings["ranking_tie_break"],
            "protection_policy": settings["protection_policy"],
        }
    )
    output = _output_root(root, config)
    write_json(output / "experiment_manifest.json", manifest)
    return manifest


def _load_final_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = _output_root(root, config) / "experiment_manifest.json"
    manifest = load_json(path)
    if manifest.get("status") not in {"prepared", "completed"}:
        raise ValueError("Final evaluation manifest is not prepared.")
    settings = _settings(config)
    matrix = load_json(
        resolve_project_path(root, config["paths"]["output_root"])
        / "experiment_manifest.json"
    )
    current_suites = {
        name: {
            **contract,
            "plan_ids": _suite_plan_ids(name, matrix, settings),
        }
        for name, contract in settings["suites"].items()
    }
    current_scope = _analysis_scope(matrix, settings, current_suites)
    expected_contract = canonical_json_sha256(
        {
            "analysis_scope": current_scope,
            "suites": current_suites,
            "runtime_contract": {
                "evaluation": config["evaluation"],
                "precision": config["precision"],
                "backend": config["backend"],
            },
            "bootstrap": settings["bootstrap"],
            "ranking_tie_break": settings["ranking_tie_break"],
            "protection_policy": settings["protection_policy"],
        }
    )
    if manifest.get("contract_sha256") != expected_contract:
        raise ValueError("Final evaluation contract changed after preparation.")
    if manifest.get("analysis_scope") != current_scope:
        raise ValueError("Final evaluation analysis scope changed after preparation.")
    return manifest


def _audit_completed_final_runs(
    output: Path, manifest: dict[str, Any]
) -> list[dict[str, str]]:
    invalid = []
    for suite, contract in manifest["suites"].items():
        for plan_id in contract["plan_ids"]:
            directory = output / suite / plan_id
            try:
                result = load_json(directory / "result.json")
                report = load_json(directory / "comparison_report.json")
                variant = _structured_variant(report)
                predictions_path = directory / variant["predictions_file"]
                if result.get("status") != "completed":
                    raise ValueError("result status is not completed")
                if result.get("suite") != suite or result.get("plan", {}).get("plan_id") != plan_id:
                    raise ValueError("result suite or plan differs")
                if result.get("comparison_report_sha256") != sha256_file(
                    directory / "comparison_report.json"
                ):
                    raise ValueError("comparison report SHA256 differs")
                prediction_sha = sha256_file(predictions_path)
                if (
                    variant.get("predictions_sha256") != prediction_sha
                    or result.get("predictions_sha256") != prediction_sha
                ):
                    raise ValueError("prediction SHA256 differs")
                if report.get("dataset", {}).get("role") != contract["dataset_role"]:
                    raise ValueError("dataset role differs")
                expected_size = int(
                    contract.get("pilot_size", contract.get("expected_size"))
                )
                if report.get("dataset", {}).get("size") != expected_size:
                    raise ValueError("dataset size differs")
                if "expected_indices_sha256" in contract and (
                    report.get("dataset", {}).get("sample", {}).get("indices_sha256")
                    != contract["expected_indices_sha256"]
                ):
                    raise ValueError("pilot sample SHA256 differs")
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                invalid.append({"suite": suite, "plan_id": plan_id, "reason": str(exc)})
    return invalid


def pending_final_plan_ids(
    project_root: str | Path, config: dict[str, Any], suite: str | None = None
) -> dict[str, list[str]]:
    root = Path(project_root).resolve()
    manifest = _load_final_manifest(root, config)
    output = _output_root(root, config)
    names = [suite] if suite else list(manifest["suites"])
    pending: dict[str, list[str]] = {}
    for name in names:
        if name not in manifest["suites"]:
            raise ValueError(f"Unknown final evaluation suite: {name}")
        pending[name] = []
        for plan_id in manifest["suites"][name]["plan_ids"]:
            result_path = output / name / plan_id / "result.json"
            if not result_path.is_file() or load_json(result_path).get("status") != "completed":
                pending[name].append(plan_id)
    return pending


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


def run_final_plan(
    *,
    project_root: str | Path,
    config: dict[str, Any],
    suite: str,
    plan_id: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = _load_final_manifest(root, config)
    if suite not in manifest["suites"]:
        raise ValueError(f"Unknown final evaluation suite: {suite}")
    suite_contract = manifest["suites"][suite]
    if plan_id not in suite_contract["plan_ids"]:
        raise ValueError(f"Plan {plan_id} is not part of suite {suite}.")
    plan, _, matrix = load_prepared_plan(root, config, plan_id)
    output = _output_root(root, config) / suite / plan_id
    result_path = output / "result.json"
    if result_path.is_file() and load_json(result_path).get("status") == "completed":
        result = load_json(result_path)
        predictions, report = _predictions_from_dir(output)
        variant = _structured_variant(report)
        if result.get("predictions_sha256") == variant["predictions_sha256"]:
            return {**result, "resume_action": "skipped_completed"}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Final sensitivity evaluation requires torch and transformers.") from exc
    from src.baseline.service import BaselineService
    from src.data.finqa_assets import load_assets
    from src.evaluation.quality_repair import run_quality_repair_experiment

    settings = _settings(config)
    paths = config["paths"]
    evaluation = config["evaluation"]
    frozen = load_json(resolve_project_path(root, paths["input_freeze"]))
    frozen_subsets = load_json(
        resolve_project_path(root, paths["setup_dir"]) / "frozen_subsets.json"
    )
    assets = load_assets(resolve_project_path(root, paths["dataset_dir"]))
    role = suite_contract["dataset_role"]
    if role == "evaluation_dev":
        dataset = assets.quality_dev
    elif role == "high_risk_regression":
        dataset = assets.high_risk_regression
    else:
        raise ValueError(f"Unsupported final evaluation dataset role: {role}")
    frozen_name = suite_contract.get("frozen_subset")
    if frozen_name:
        _validate_dataset_against_freeze(
            dataset,
            frozen_entries=frozen_subsets[frozen_name],
            expected_hash=frozen["subset_hashes"][frozen_name],
            suite=suite,
        )
        if len(dataset) != int(suite_contract["expected_size"]):
            raise ValueError(f"{suite} dataset size differs from the contract.")

    baseline = load_json(resolve_project_path(root, paths["baseline_config"]))
    baseline["device"] = evaluation["device"]
    baseline["dtype"] = evaluation["dtype"]
    quality = load_json(resolve_project_path(root, paths["quality_config"]))
    quality["batch_size"] = int(evaluation["batch_size"])
    quality["generation"]["max_new_tokens"] = int(evaluation["max_new_tokens"])
    quality["pilot_variants"] = [evaluation["variant"]]
    quality["dataset_role"] = role
    quality["show_progress"] = True
    if "pilot_size" in suite_contract:
        quality["evaluation_size"] = int(suite_contract["pilot_size"])
        quality["evaluation_seed"] = int(suite_contract["pilot_seed"])
    else:
        quality.pop("evaluation_size", None)
        quality.pop("evaluation_seed", None)
    write_json(output / "baseline_config.effective.json", baseline)
    write_json(output / "quality_config.effective.json", quality)

    merged_manifest = load_json(resolve_project_path(root, paths["merged_manifest"]))
    model_path = Path(merged_manifest["merged_model"]["path"]).resolve()
    dtype = _evaluation_dtype(torch, evaluation["dtype"])
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        dtype=dtype, low_cpu_mem_usage=True,
    )
    error_report = quantize_model_for_plan(model, plan, config["precision"], torch)
    _validate_runtime_blocks(error_report, matrix)
    model.to(evaluation["device"])
    model.eval()
    service = BaselineService(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        config=baseline,
        model_metadata={
            "model_id": "Qwen2.5-7B-Instruct-QLoRA-step724-merged-fp16",
            "version": matrix["merged_manifest"]["tree_sha256"],
            "quantization": {
                "backend": "symmetric_fake_quant",
                "plan_id": plan_id,
                "deployment_performance_claims_allowed": False,
            },
        },
        log_file=output / "quality_service.jsonl",
    )
    report = run_quality_repair_experiment(
        service, dataset, assets.manifest, quality, output,
        calibration=assets.calibration,
    )
    if report["dataset"]["size"] != (
        int(suite_contract.get("pilot_size", suite_contract.get("expected_size")))
    ):
        raise ValueError(f"{suite} report size differs from the contract.")
    if "expected_indices_sha256" in suite_contract:
        observed = report.get("dataset", {}).get("sample", {}).get("indices_sha256")
        if observed != suite_contract["expected_indices_sha256"]:
            raise ValueError("Determinism rerun used different pilot indices.")
    variant = _structured_variant(report)
    result = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.1.4",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "plan": plan,
        "dataset": report["dataset"],
        "metrics": variant["metrics"],
        "structured_output_rate": variant["structured_output_rate"],
        "format_retry_count": variant["format_retry_count"],
        "predictions_sha256": variant["predictions_sha256"],
        "comparison_report_sha256": sha256_file(output / "comparison_report.json"),
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    return result


def _transition_detail(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if before["source_index"] != after["source_index"]:
        raise ValueError("Prediction rows are not source-index aligned.")
    before_correct = bool(before.get("correct"))
    after_correct = bool(after.get("correct"))
    if not before_correct and after_correct:
        transition = "recovered"
    elif before_correct and not after_correct:
        transition = "regressed"
    elif before_correct:
        transition = "stable_correct"
    else:
        transition = "stable_wrong"
    return {
        "source_index": before["source_index"],
        "question": after.get("question"),
        "transition": transition,
        "baseline_correct": before_correct,
        "candidate_correct": after_correct,
        "baseline_parsed": bool(before.get("parsed")),
        "candidate_parsed": bool(after.get("parsed")),
        "baseline_structured": bool(before.get("structured_output")),
        "candidate_structured": bool(after.get("structured_output")),
        "raw_prediction_changed": before.get("raw_prediction") != after.get("raw_prediction"),
    }


def _group_metrics(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    indices: set[int],
) -> dict[str, Any]:
    paired = [
        (before, after)
        for before, after in zip(baseline, candidate)
        if int(before["source_index"]) in indices
    ]
    total = len(paired)
    baseline_correct = sum(bool(before.get("correct")) for before, _ in paired)
    candidate_correct = sum(bool(after.get("correct")) for _, after in paired)
    recovered = sum(
        not bool(before.get("correct")) and bool(after.get("correct"))
        for before, after in paired
    )
    regressed = sum(
        bool(before.get("correct")) and not bool(after.get("correct"))
        for before, after in paired
    )
    baseline_parsed = sum(bool(before.get("parsed")) for before, _ in paired)
    candidate_parsed = sum(bool(after.get("parsed")) for _, after in paired)
    baseline_structured = sum(
        bool(before.get("structured_output")) for before, _ in paired
    )
    candidate_structured = sum(
        bool(after.get("structured_output")) for _, after in paired
    )
    baseline_retries = sum(
        bool(before.get("format_retry_used")) for before, _ in paired
    )
    candidate_retries = sum(
        bool(after.get("format_retry_used")) for _, after in paired
    )
    return {
        "total": total,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / total if total else 0.0,
        "candidate_correct": candidate_correct,
        "candidate_accuracy": candidate_correct / total if total else 0.0,
        "recovered": recovered,
        "regressed": regressed,
        "net_correct_gain": recovered - regressed,
        "baseline_parsed": baseline_parsed,
        "baseline_parse_rate": baseline_parsed / total if total else 0.0,
        "candidate_parsed": candidate_parsed,
        "candidate_parse_rate": candidate_parsed / total if total else 0.0,
        "parse_rate_delta": (
            (candidate_parsed - baseline_parsed) / total if total else 0.0
        ),
        "baseline_structured": baseline_structured,
        "baseline_structured_rate": baseline_structured / total if total else 0.0,
        "candidate_structured": candidate_structured,
        "candidate_structured_rate": candidate_structured / total if total else 0.0,
        "structured_rate_delta": (
            (candidate_structured - baseline_structured) / total if total else 0.0
        ),
        "baseline_format_retry_count": baseline_retries,
        "candidate_format_retry_count": candidate_retries,
        "format_retry_delta": candidate_retries - baseline_retries,
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Percentile requires values.")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_interval(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired bootstrap requires aligned non-empty predictions.")
    deltas = []
    for before, after in zip(baseline, candidate):
        if before["source_index"] != after["source_index"]:
            raise ValueError("Paired bootstrap predictions are not aligned.")
        deltas.append(int(bool(after.get("correct"))) - int(bool(before.get("correct"))))
    rng = random.Random(seed)
    count = len(deltas)
    estimates = sorted(
        sum(rng.choices(deltas, k=count)) / count for _ in range(iterations)
    )
    alpha = (1.0 - confidence) / 2.0
    observed = sum(deltas) / count
    return {
        "method": "paired_nonparametric_percentile",
        "iterations": iterations,
        "confidence": confidence,
        "seed": seed,
        "point_estimate": observed,
        "lower": _percentile(estimates, alpha),
        "upper": _percentile(estimates, 1.0 - alpha),
    }


def _structured_transitions(
    baseline: Iterable[dict[str, Any]], candidate: Iterable[dict[str, Any]]
) -> dict[str, int]:
    restored = 0
    lost = 0
    for before, after in zip(baseline, candidate):
        before_value = bool(before.get("structured_output"))
        after_value = bool(after.get("structured_output"))
        restored += int(not before_value and after_value)
        lost += int(before_value and not after_value)
    return {"restored": restored, "lost": lost, "net": restored - lost}


def _ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["full_dev"]["net_correct_gain"]),
        -int(row["full_dev"]["recovered"]),
        -int(row["high_risk"]["recovered"]),
        -int(row["long_context"]["recovered"]),
        -int(row["structured_output_transitions"]["net"]),
        -float(row["quantization_error_recovery"]),
        int(row["layer_index"]),
    )


def _write_ranking_csv(path: Path, ranking: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "layer_index", "full_dev_correct", "full_dev_total",
        "net_correct_gain", "recovered", "regressed", "high_risk_recovered",
        "high_risk_regressed", "long_context_recovered", "long_context_regressed",
        "structured_output_net", "bootstrap_lower", "bootstrap_upper",
        "quantization_error_recovery",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ranking:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "layer_index": row["layer_index"],
                    "full_dev_correct": row["full_dev"]["candidate_correct"],
                    "full_dev_total": row["full_dev"]["total"],
                    "net_correct_gain": row["full_dev"]["net_correct_gain"],
                    "recovered": row["full_dev"]["recovered"],
                    "regressed": row["full_dev"]["regressed"],
                    "high_risk_recovered": row["high_risk"]["recovered"],
                    "high_risk_regressed": row["high_risk"]["regressed"],
                    "long_context_recovered": row["long_context"]["recovered"],
                    "long_context_regressed": row["long_context"]["regressed"],
                    "structured_output_net": row["structured_output_transitions"]["net"],
                    "bootstrap_lower": row["bootstrap"]["lower"],
                    "bootstrap_upper": row["bootstrap"]["upper"],
                    "quantization_error_recovery": row["quantization_error_recovery"],
                }
            )


def _protection_recommendation(
    ranking: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    eligible = []
    for row in ranking:
        if policy.get("require_positive_full_dev_gain") and row["full_dev"]["net_correct_gain"] <= 0:
            continue
        if policy.get("require_nonnegative_high_risk_gain") and row["high_risk"]["net_correct_gain"] < 0:
            continue
        if policy.get("require_nonnegative_long_context_gain") and row["long_context"]["net_correct_gain"] < 0:
            continue
        eligible.append(row["layer_index"])
    selected = eligible[: int(policy["max_layers"])]
    return {
        "policy": policy,
        "eligible_layers_in_rank_order": eligible,
        "recommended_protected_layers": selected,
        "status": "selected" if selected else "no_eligible_layer",
    }


def audit_determinism(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = _output_root(root, config)
    manifest = _load_final_manifest(root, config)
    pending = pending_final_plan_ids(root, config, "determinism")["determinism"]
    if pending:
        raise ValueError(f"Determinism suite still has pending plans: {pending}")
    matrix_root = resolve_project_path(root, config["paths"]["output_root"])
    plans = {}
    for plan_id in manifest["suites"]["determinism"]["plan_ids"]:
        original, original_report = _predictions_from_dir(matrix_root / plan_id)
        repeated, repeat_report = _predictions_from_dir(output / "determinism" / plan_id)
        exact = original == repeated
        original_sha = _structured_variant(original_report)["predictions_sha256"]
        repeat_sha = _structured_variant(repeat_report)["predictions_sha256"]
        plans[plan_id] = {
            "status": "pass" if exact and original_sha == repeat_sha else "fail",
            "row_count": len(original),
            "exact_row_match": exact,
            "original_sha256": original_sha,
            "repeat_sha256": repeat_sha,
        }
    audit = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.1.4",
        "status": "pass" if all(row["status"] == "pass" for row in plans.values()) else "fail",
        "plans": plans,
    }
    write_json(output / "determinism_audit.json", audit)
    if audit["status"] != "pass":
        raise ValueError(
            "Determinism audit failed; inspect determinism_audit.json before continuing."
        )
    return audit


def finalize_layer_sensitivity(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = _settings(config)
    output = _output_root(root, config)
    manifest = _load_final_manifest(root, config)
    pending = pending_final_plan_ids(root, config)
    if any(pending.values()):
        raise ValueError(f"Final evaluation still has pending plans: {pending}")
    invalid_runs = _audit_completed_final_runs(output, manifest)
    if invalid_runs:
        raise ValueError(f"Final evaluation audit failed: {invalid_runs}")

    matrix_root = resolve_project_path(root, config["paths"]["output_root"])
    matrix_summary = load_json(matrix_root / "matrix_summary.json")
    all_int4_weight_nmse = float(matrix_summary["all_int4"]["weighted_nmse"])
    frozen_subsets = load_json(
        resolve_project_path(root, config["paths"]["setup_dir"]) / "frozen_subsets.json"
    )
    labels = _read_jsonl(
        resolve_project_path(root, config["paths"]["setup_dir"]) / "business_labels.jsonl"
    )
    label_indices: dict[str, set[int]] = {
        label: {int(row["source_index"]) for row in labels if label in row["labels"]}
        for label in ("amount", "rate_or_ratio", "date_or_term")
    }
    long_indices = {int(row["index"]) for row in frozen_subsets["long_context"]}

    full_baseline, full_baseline_report = _predictions_from_dir(output / "full_dev" / "all_int4")
    risk_baseline, _ = _predictions_from_dir(output / "high_risk" / "all_int4")
    if len(full_baseline) != 883 or len(risk_baseline) != 91:
        raise ValueError("Final baseline prediction sizes are invalid.")

    determinism_audit = audit_determinism(root, config)
    determinism = determinism_audit["plans"]
    if determinism_audit["status"] != "pass":
        raise ValueError("Determinism rerun does not exactly reproduce pilot predictions.")

    layer_metrics = []
    transition_rows = []
    bootstrap_settings = settings["bootstrap"]
    scoped_plan_ids = set(manifest["analysis_scope"]["candidate_plan_ids"])
    layer_plans = [
        plan for plan in load_json(matrix_root / "experiment_manifest.json")["plans"]
        if plan.get("kind") == "layer_int8" and plan["plan_id"] in scoped_plan_ids
    ]
    if len(layer_plans) != manifest["analysis_scope"]["candidate_layer_count"]:
        raise ValueError("Scoped candidate layer plans are incomplete in the matrix.")
    for plan in layer_plans:
        plan_id = plan["plan_id"]
        layer_index = int(plan["layer_index"])
        full_candidate, full_report = _predictions_from_dir(output / "full_dev" / plan_id)
        risk_candidate, risk_report = _predictions_from_dir(output / "high_risk" / plan_id)
        full_metrics = _group_metrics(
            full_baseline, full_candidate,
            {int(row["source_index"]) for row in full_baseline},
        )
        risk_metrics = _group_metrics(
            risk_baseline, risk_candidate,
            {int(row["source_index"]) for row in risk_baseline},
        )
        business = {
            label: _group_metrics(full_baseline, full_candidate, indices)
            for label, indices in label_indices.items()
        }
        long_metrics = _group_metrics(full_baseline, full_candidate, long_indices)
        bootstrap = paired_bootstrap_interval(
            full_baseline,
            full_candidate,
            iterations=int(bootstrap_settings["iterations"]),
            confidence=float(bootstrap_settings["confidence"]),
            seed=int(bootstrap_settings["seed"]) + layer_index,
        )
        structured = _structured_transitions(full_baseline, full_candidate)
        weight_result = load_json(matrix_root / plan_id / "result.json")
        row = {
            "layer_index": layer_index,
            "plan_id": plan_id,
            "full_dev": full_metrics,
            "business_subsets": business,
            "long_context": long_metrics,
            "high_risk": risk_metrics,
            "structured_output_transitions": structured,
            "full_dev_format_retry_count": _structured_variant(full_report)["format_retry_count"],
            "high_risk_format_retry_count": _structured_variant(risk_report)["format_retry_count"],
            "bootstrap": bootstrap,
            "weighted_nmse": float(weight_result["weighted_nmse"]),
            "quantization_error_recovery": (
                all_int4_weight_nmse - float(weight_result["weighted_nmse"])
            ),
        }
        layer_metrics.append(row)
        for suite_name, before_rows, after_rows in (
            ("full_dev", full_baseline, full_candidate),
            ("high_risk", risk_baseline, risk_candidate),
        ):
            for before, after in zip(before_rows, after_rows):
                transition_rows.append(
                    {
                        "suite": suite_name,
                        "layer_index": layer_index,
                        "plan_id": plan_id,
                        **_transition_detail(before, after),
                    }
                )

    ranking = sorted(copy.deepcopy(layer_metrics), key=_ranking_key)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    intervals_overlap = False
    if len(ranking) >= 2:
        first = ranking[0]["bootstrap"]
        second = ranking[1]["bootstrap"]
        intervals_overlap = not (
            first["upper"] < second["lower"] or second["upper"] < first["lower"]
        )
    tied_primary = sum(
        row["full_dev"]["net_correct_gain"]
        == ranking[0]["full_dev"]["net_correct_gain"]
        for row in ranking
    )
    conclusion_strength = (
        "low" if intervals_overlap or tied_primary > 1 else "moderate"
    )
    recommendation = _protection_recommendation(
        ranking, settings["protection_policy"]
    )

    write_jsonl(output / "layer_metrics.jsonl", sorted(layer_metrics, key=lambda row: row["layer_index"]))
    write_jsonl(
        output / "sample_transitions.jsonl",
        sorted(transition_rows, key=lambda row: (row["suite"], row["layer_index"], row["source_index"])),
    )
    ranking_payload = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.1.5",
        "status": (
            "final"
            if manifest["analysis_scope"]["complete_layer_coverage"]
            else "candidate_subset_final"
        ),
        "analysis_scope": manifest["analysis_scope"],
        "ranking_tie_break": settings["ranking_tie_break"],
        "ranking_conclusion_strength": conclusion_strength,
        "top_two_bootstrap_intervals_overlap": intervals_overlap,
        "top_primary_tie_count": tied_primary,
        "recommendation": recommendation,
        "ranking": ranking,
    }
    write_json(output / "ranking.json", ranking_payload)
    _write_ranking_csv(output / "ranking.csv", ranking)

    validation = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "stage": "7.1.5",
        "status": "pass",
        "pending": pending,
        "invalid_runs": invalid_runs,
        "determinism": determinism,
        "analysis_scope": manifest["analysis_scope"],
        "expected_candidate_layers": manifest["analysis_scope"][
            "candidate_layer_count"
        ],
        "total_model_layers": manifest["analysis_scope"]["total_model_layers"],
        "ranked_layers": len(ranking),
        "full_dev_size": len(full_baseline),
        "high_risk_size": len(risk_baseline),
        "long_context_size": len(long_indices),
        "business_label_counts": {key: len(value) for key, value in label_indices.items()},
        "required_artifacts": list(REQUIRED_FINAL_ARTIFACTS),
        "ranking_conclusion_strength": conclusion_strength,
    }
    write_json(output / "validation.json", validation)

    top_lines = [
        f"| {row['rank']} | {row['layer_index']} | "
        f"{row['full_dev']['candidate_correct']}/{row['full_dev']['total']} | "
        f"{row['full_dev']['recovered']} | {row['full_dev']['regressed']} | "
        f"{row['high_risk']['recovered']} | {row['long_context']['recovered']} |"
        for row in ranking[:10]
    ]
    scope = manifest["analysis_scope"]
    scope_lines = [
        f"- Analysis scope: `{scope['mode']}`",
        f"- Evaluated candidate layers: `{scope['candidate_layer_indices']}`",
        f"- Candidate coverage: {scope['candidate_layer_count']}/{scope['total_model_layers']} layers",
    ]
    if not scope["full_layer_ranking_claim_allowed"]:
        scope_lines.append(
            "- Claim boundary: candidate validation only; this is not a complete "
            f"{scope['total_model_layers']}-layer ranking"
        )
    report_title = (
        "# TODO 7.1 Final Layer Sensitivity Report"
        if scope["complete_layer_coverage"]
        else "# TODO 7.1 Layer Sensitivity Candidate Validation Report"
    )
    ranking_heading = (
        "## Layer ranking"
        if scope["complete_layer_coverage"]
        else "## Candidate ranking"
    )
    report = "\n".join(
        [
            report_title,
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Analysis scope",
            "",
            *scope_lines,
            "",
            "## Validation",
            "",
            "- Full dev: 883 rows",
            "- Contextual high risk: 91 rows",
            "- Long context: 89 rows",
            "- Determinism reruns: exact match",
            f"- Ranking conclusion strength: {conclusion_strength}",
            "",
            ranking_heading,
            "",
            "| Rank | Layer | Full dev | Recovered | Regressed | High-risk recovered | Long-context recovered |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *top_lines,
            "",
            "## Recommended protection set",
            "",
            f"`{recommendation['recommended_protected_layers']}`",
            "",
            "The ranking is quality-first. Weight and activation errors remain diagnostic tie-break evidence.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["final_artifact_sha256"] = {
        name: sha256_file(output / name)
        for name in REQUIRED_FINAL_ARTIFACTS
        if name != "experiment_manifest.json"
    }
    write_json(output / "experiment_manifest.json", manifest)
    return {"validation": validation, "ranking": ranking_payload}
