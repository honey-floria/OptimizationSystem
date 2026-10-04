"""Build the TODO 6.5 cross-method quantization comparison report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.input_validation.model_input import NOT_RUN, PASS, CheckResult


@dataclass
class QuantizationComparisonValidation:
    report: str
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid comparison input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _benchmark_summary(
    report: dict[str, Any] | None,
    hourly_costs: dict[str, float | None],
) -> dict[str, Any]:
    if not report:
        return {
            "status": "not_run",
            "gpu": None,
            "case_count": 0,
            "matrix_signature": [],
            "mean_output_tokens_per_second": None,
            "mean_end_to_end_p50_ms": None,
            "max_model_allocated_mib": None,
            "max_peak_allocated_mib": None,
            "gpu_hours_per_million_output_tokens": None,
            "usd_per_million_output_tokens": None,
        }
    cases = report.get("cases", [])
    gpu = report.get("environment", {}).get("gpu")
    throughput = _mean([
        float(case["throughput"]["mean_output_tokens_per_second"])
        for case in cases
    ])
    gpu_hours = 1_000_000 / throughput / 3600 if throughput else None
    hourly_cost = hourly_costs.get(str(gpu))
    return {
        "status": "pass" if cases else "not_run",
        "gpu": gpu,
        "case_count": len(cases),
        "matrix_signature": sorted([
            [case.get("batch_size"), case.get("input_tokens"), case.get("output_tokens")]
            for case in cases
        ]),
        "mean_output_tokens_per_second": throughput,
        "mean_end_to_end_p50_ms": _mean([
            float(case["latency_ms"]["end_to_end"]["p50"])
            for case in cases
        ]),
        "max_model_allocated_mib": max(
            (float(case["memory"]["model_allocated_mib"]) for case in cases),
            default=None,
        ),
        "max_peak_allocated_mib": max(
            (float(case["memory"]["peak_allocated_mib"]) for case in cases),
            default=None,
        ),
        "gpu_hours_per_million_output_tokens": gpu_hours,
        "usd_per_million_output_tokens": (
            gpu_hours * float(hourly_cost)
            if gpu_hours is not None and hourly_cost is not None
            else None
        ),
    }


def _quality_summary(
    metrics: dict[str, Any] | None,
    baseline_accuracy: float,
) -> dict[str, Any]:
    if not metrics:
        return {
            "status": "not_run",
            "total": None,
            "parse_rate": None,
            "numeric_accuracy": None,
            "absolute_change_from_fp16": None,
            "quality_drop_percentage_points": None,
        }
    accuracy = float(metrics.get("numeric_accuracy", 0.0))
    delta = accuracy - baseline_accuracy
    return {
        "status": "pass",
        "total": metrics.get("total"),
        "parse_rate": metrics.get("parse_rate"),
        "numeric_accuracy": accuracy,
        "absolute_change_from_fp16": delta,
        "quality_drop_percentage_points": max(0.0, -delta * 100),
    }


def _risk_assessment(
    method: str,
    quality: dict[str, Any],
    regression: dict[str, Any] | None,
    max_drop_percentage_points: float,
    minimum_parse_rate: float,
    structured_regression: bool = False,
) -> dict[str, Any]:
    reasons = []
    if quality.get("status") != "pass":
        reasons.append("quality_not_run")
    elif float(quality.get("parse_rate") or 0.0) < minimum_parse_rate:
        reasons.append("parse_rate_below_gate")
    if float(quality.get("quality_drop_percentage_points") or 0.0) > max_drop_percentage_points:
        reasons.append("quality_drop_above_gate")
    if regression and int(regression.get("new_regressions") or 0) > 0:
        reasons.append("new_critical_regressions")
    if structured_regression:
        reasons.append("structured_output_regression")
    l2_candidate = not reasons and quality.get("status") == "pass"
    l3_candidate = l2_candidate and method in {"fp16", "int8"}
    return {
        "l2_candidate": l2_candidate,
        "l3_candidate": l3_candidate,
        "reasons": reasons,
        "deployment_ready": False,
        "deployment_note": (
            "The FP16 reference accuracy is too low to claim production readiness."
        ),
    }


def _candidate(
    plan_id: str,
    method: str,
    precision: str,
    quality_metrics: dict[str, Any] | None,
    benchmark_report: dict[str, Any] | None,
    baseline_accuracy: float,
    baseline_performance: dict[str, Any],
    hourly_costs: dict[str, float | None],
    thresholds: dict[str, float],
    regression: dict[str, Any] | None = None,
    structured_regression: bool = False,
    applicability: str = "applicable",
) -> dict[str, Any]:
    quality = _quality_summary(quality_metrics, baseline_accuracy)
    performance = _benchmark_summary(benchmark_report, hourly_costs)
    comparable = (
        performance["status"] == "pass"
        and performance["gpu"] == baseline_performance["gpu"]
        and performance["matrix_signature"] == baseline_performance["matrix_signature"]
    )
    risk_assessment = _risk_assessment(
        method,
        quality,
        regression,
        thresholds["max_quality_drop_percentage_points"],
        thresholds["minimum_parse_rate"],
        structured_regression,
    )
    if applicability != "applicable":
        risk_assessment = {
            "l2_candidate": False,
            "l3_candidate": False,
            "reasons": [applicability],
            "deployment_ready": False,
            "deployment_note": "The method is not supported on the target hardware.",
        }
    return {
        "plan_id": plan_id,
        "method": method,
        "precision": precision,
        "applicability": applicability,
        "quality": quality,
        "regression": regression,
        "performance": performance,
        "performance_comparable_to_fp16": comparable,
        "performance_delta_from_fp16": {
            "memory_change_mib": (
                performance["max_model_allocated_mib"]
                - baseline_performance["max_model_allocated_mib"]
                if comparable else None
            ),
            "latency_change_ms": (
                performance["mean_end_to_end_p50_ms"]
                - baseline_performance["mean_end_to_end_p50_ms"]
                if comparable else None
            ),
            "throughput_change_tokens_per_second": (
                performance["mean_output_tokens_per_second"]
                - baseline_performance["mean_output_tokens_per_second"]
                if comparable else None
            ),
            "memory_change_percent": (
                (performance["max_model_allocated_mib"] / baseline_performance["max_model_allocated_mib"] - 1) * 100
                if comparable else None
            ),
            "latency_change_percent": (
                (performance["mean_end_to_end_p50_ms"] / baseline_performance["mean_end_to_end_p50_ms"] - 1) * 100
                if comparable else None
            ),
            "throughput_change_percent": (
                (performance["mean_output_tokens_per_second"] / baseline_performance["mean_output_tokens_per_second"] - 1) * 100
                if comparable else None
            ),
        },
        "risk_assessment": risk_assessment,
    }


def build_quantization_comparison(
    project_root: str | Path,
    config_file: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = _load_json(Path(config_file).expanduser().resolve())
    hourly_costs = config.get("hourly_gpu_cost_usd", {})
    method_disposition = config.get("method_disposition", {})
    thresholds = config["quality_gates"]
    baseline_quality_report = _load_json(
        root / "out/baseline_quality_experiment/baseline_quality_report.json"
    )
    baseline_metrics = baseline_quality_report["metrics"]
    baseline_accuracy = float(baseline_metrics["numeric_accuracy"])
    comparable_baseline = _optional_json(
        root / "out/quantization_comparison/fp16_a100_benchmark_report.json"
    )
    baseline_performance = _benchmark_summary(
        comparable_baseline
        or _load_json(root / "out/benchmark/baseline_fp16_report.json"),
        hourly_costs,
    )
    candidates = []

    awq_summary = _load_json(root / "out/awq/experiment_summary.json")
    for item in awq_summary.get("candidates", []):
        plan_id = item["plan_id"]
        candidates.append(_candidate(
            plan_id,
            "awq",
            "INT4 W4A16",
            item.get("quality"),
            _optional_json(root / f"out/awq/{plan_id}/comparison_benchmark_report.json"),
            baseline_accuracy,
            baseline_performance,
            hourly_costs,
            thresholds,
            item.get("regression"),
        ))

    for group_size in (32, 64, 128):
        plan_id = f"gptq-int4-w4a16-g{group_size}"
        quality_report = _optional_json(
            root / f"out/gptq/{plan_id}/quality/baseline_quality_report.json"
        )
        candidates.append(_candidate(
            plan_id,
            "gptq",
            "INT4 W4A16",
            quality_report.get("metrics") if quality_report else None,
            _optional_json(root / f"out/gptq/{plan_id}/comparison_benchmark_report.json"),
            baseline_accuracy,
            baseline_performance,
            hourly_costs,
            thresholds,
            _optional_json(root / f"out/gptq/{plan_id}/regression.json"),
        ))

    int8_evaluation = _optional_json(root / "out/int8_fp8/evaluation.json")
    structured = (int8_evaluation or {}).get("structured_output", {})
    baseline_structured_passed = sum(
        bool(case.get("baseline_passed")) for case in structured.get("cases", [])
    )
    int8_structured_passed = int(structured.get("passed") or 0)
    candidates.append(_candidate(
        "smoothquant-int8-w8a8",
        "int8",
        "INT8 W8A8",
        (int8_evaluation or {}).get("numeric"),
        _optional_json(root / "out/int8_fp8/benchmark_report.json"),
        baseline_accuracy,
        baseline_performance,
        hourly_costs,
        thresholds,
        structured_regression=int8_structured_passed < baseline_structured_passed,
    ))
    candidates.append(_candidate(
        "fp8-not-run",
        "fp8",
        "FP8",
        None,
        None,
        baseline_accuracy,
        baseline_performance,
        hourly_costs,
        thresholds,
        applicability=str(method_disposition.get("fp8", "applicable")),
    ))

    methods = {candidate["method"] for candidate in candidates}
    applicable_candidates = [
        candidate for candidate in candidates
        if candidate.get("applicability") == "applicable"
    ]
    performance_missing = [
        candidate["plan_id"] for candidate in applicable_candidates
        if candidate["performance"]["status"] != "pass"
    ]
    performance_incomparable = [
        candidate["plan_id"] for candidate in applicable_candidates
        if candidate["performance"]["status"] == "pass"
        and not candidate["performance_comparable_to_fp16"]
    ]
    monetary_cost_missing = (
        ["fp16-baseline"]
        if baseline_performance["usd_per_million_output_tokens"] is None
        else []
    ) + [
        candidate["plan_id"] for candidate in applicable_candidates
        if candidate["performance"]["status"] == "pass"
        and candidate["performance"]["usd_per_million_output_tokens"] is None
    ]
    comparable_candidates = [
        candidate["plan_id"] for candidate in candidates
        if candidate["performance_comparable_to_fp16"]
    ]
    l2_candidates = [
        candidate["plan_id"] for candidate in candidates
        if candidate["risk_assessment"]["l2_candidate"]
    ]
    validated_findings = [
        {
            "plan_id": candidate["plan_id"],
            **candidate["performance_delta_from_fp16"],
        }
        for candidate in candidates
        if candidate["performance_comparable_to_fp16"]
    ]
    limitations = [
        "The FP16 reference accuracy is not production-quality."
    ]
    next_evidence = []
    if performance_missing:
        limitations.append(
            "Some applicable candidates have no performance benchmark."
        )
        next_evidence.append("Run the missing candidate benchmarks.")
    if performance_incomparable:
        limitations.append(
            "Some INT4 benchmarks use a different A100 memory variant from FP16."
        )
        next_evidence.append(
            "Rerun AWQ/GPTQ on A100 80GB, or rerun FP16/INT8 on A100 40GB."
        )
    if monetary_cost_missing:
        limitations.append("Approved hourly GPU prices are missing.")
        next_evidence.append("Record approved hourly GPU prices for monetary cost.")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cost_basis": {
            "resource_metric": "gpu_hours_per_million_output_tokens",
            "formula": "1000000 / mean_output_tokens_per_second / 3600",
            "hourly_gpu_cost_usd": hourly_costs,
            "monetary_cost_status": (
                "complete" if not monetary_cost_missing else "not_run"
            ),
        },
        "baseline": {
            "plan_id": "fp16-baseline",
            "method": "fp16",
            "precision": "FP16",
            "quality": _quality_summary(baseline_metrics, baseline_accuracy),
            "performance": baseline_performance,
        },
        "candidates": candidates,
        "coverage": {
            "methods": sorted(methods | {"fp16"}),
            "comparison_table_complete": methods == {"awq", "gptq", "int8", "fp8"},
            "quality_missing": [
                candidate["plan_id"] for candidate in applicable_candidates
                if candidate["quality"]["status"] != "pass"
            ],
            "performance_missing": performance_missing,
            "performance_incomparable": performance_incomparable,
            "monetary_cost_missing": monetary_cost_missing,
            "performance_comparable_to_fp16": comparable_candidates,
        },
        "recommendation": {
            "l2_candidates_by_current_gates": l2_candidates,
            "deployment_recommendation": None,
            "reason": " ".join(limitations) + " No deployment winner can be selected.",
            "validated_performance_findings": validated_findings,
            "next_evidence": next_evidence,
        },
    }


def validate_quantization_comparison(
    report_file: str | Path,
) -> QuantizationComparisonValidation:
    report_path = Path(report_file).expanduser().resolve()
    report = _load_json(report_path)
    candidates = report.get("candidates", [])
    coverage = report.get("coverage", {})
    quality_complete = not coverage.get("quality_missing")
    performance_complete = (
        not coverage.get("performance_missing")
        and not coverage.get("performance_incomparable")
    )
    monetary_cost_complete = report.get("cost_basis", {}).get("monetary_cost_status") == "complete"
    risks_recorded = bool(candidates) and all(
        isinstance(candidate.get("risk_assessment", {}).get("reasons"), list)
        for candidate in candidates
    )
    checks = [
        CheckResult("fp16_int8_int4_fp8_comparison_table", PASS if coverage.get("comparison_table_complete") else NOT_RUN, "FP16, INT8, INT4, and FP8 rows are present."),
        CheckResult("quality_changes_recorded", PASS if quality_complete else NOT_RUN, "Every candidate has measured quality."),
        CheckResult("memory_changes_recorded", PASS if performance_complete else NOT_RUN, "Every candidate has measured memory."),
        CheckResult("latency_changes_recorded", PASS if performance_complete else NOT_RUN, "Every candidate has measured latency."),
        CheckResult("throughput_changes_recorded", PASS if performance_complete else NOT_RUN, "Every candidate has measured throughput."),
        CheckResult("unit_token_cost_recorded", PASS if monetary_cost_complete else NOT_RUN, "Every measured candidate has an approved monetary cost."),
        CheckResult("high_risk_unsuitable_plans_marked", PASS if risks_recorded else NOT_RUN, "Every candidate has an L2/L3 risk assessment."),
    ]
    return QuantizationComparisonValidation(str(report_path), checks)


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    baseline = report["baseline"]
    all_rows = [baseline] + report["candidates"]
    for item in all_rows:
        quality = item["quality"]
        performance = item["performance"]
        risk = item.get("risk_assessment", {})
        rows.append(
            "| {plan} | {precision} | {accuracy} | {parse} | {gpu} | {memory} | "
            "{latency} | {throughput} | {gpu_hours} | {l2} |".format(
                plan=item["plan_id"],
                precision=item["precision"],
                accuracy=_format_number(quality.get("numeric_accuracy"), 6),
                parse=_format_number(quality.get("parse_rate"), 4),
                gpu=performance.get("gpu") or "N/A",
                memory=_format_number(performance.get("max_model_allocated_mib"), 1),
                latency=_format_number(performance.get("mean_end_to_end_p50_ms"), 1),
                throughput=_format_number(performance.get("mean_output_tokens_per_second"), 1),
                gpu_hours=_format_number(performance.get("gpu_hours_per_million_output_tokens"), 3),
                l2=str(risk.get("l2_candidate", False)) if risk else "baseline",
            )
        )
    return "\n".join([
        "# Quantization Comparison (TODO 6.5)",
        "",
        "| Plan | Precision | Accuracy | Parse rate | GPU | Model MiB | P50 ms | tok/s | GPU h / 1M tok | L2 candidate |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## Conclusion",
        "",
        report["recommendation"]["reason"],
        "",
        "Performance deltas against FP16 are intentionally blank unless hardware and benchmark matrices match.",
        "Monetary cost remains `not_run` until approved hourly GPU prices are provided.",
        "",
    ])


def _format_number(value: Any, digits: int) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/quantization_comparison.json")
    parser.add_argument("--output-dir", default="out/quantization_comparison")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_quantization_comparison(root, root / args.config)
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "comparison_report.md").write_text(render_markdown(report), encoding="utf-8")
    validation = validate_quantization_comparison(report_path)
    (output_dir / "validation.json").write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
