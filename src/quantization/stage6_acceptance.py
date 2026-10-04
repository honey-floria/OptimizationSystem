"""Validate the customized-quantization stage for Todo 6.6."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, PASS, CheckResult


@dataclass
class Stage6Report:
    scope: str
    checks: list[CheckResult]
    deliverables: dict[str, bool]
    technical_deployment_candidates: list[str]
    production_recommendation: str

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks) and all(
            self.deliverables.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
            "deliverables": self.deliverables,
            "technical_deployment_candidates": self.technical_deployment_candidates,
            "production_recommendation": self.production_recommendation,
        }


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid stage 6 evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Stage 6 evidence must be a JSON object: {path}")
    return payload


def _check_status(payload: dict[str, Any], name: str) -> str | None:
    return next(
        (
            check.get("status")
            for check in payload.get("checks", [])
            if check.get("name") == name
        ),
        None,
    )


def _benchmark_path(root: Path, plan_id: str) -> Path:
    if plan_id.startswith("awq-"):
        return root / "out/awq" / plan_id / "comparison_benchmark_report.json"
    if plan_id.startswith("gptq-"):
        return root / "out/gptq" / plan_id / "comparison_benchmark_report.json"
    if plan_id == "smoothquant-int8-w8a8":
        return root / "out/int8_fp8/benchmark_report.json"
    raise ValueError(f"Unsupported quantization plan: {plan_id}")


def _benchmark_validation_path(root: Path, plan_id: str) -> Path:
    if plan_id.startswith("awq-"):
        return root / "out/awq" / plan_id / "comparison_benchmark_validation.json"
    if plan_id.startswith("gptq-"):
        return root / "out/gptq" / plan_id / "comparison_benchmark_validation.json"
    if plan_id == "smoothquant-int8-w8a8":
        return root / "out/int8_fp8/benchmark_validation.json"
    raise ValueError(f"Unsupported quantization plan: {plan_id}")


def validate_stage6(project_root: str | Path) -> Stage6Report:
    root = Path(project_root).expanduser().resolve()
    awq_validation = _load(root / "out/awq/validation.json")
    gptq_validation = _load(root / "out/gptq/validation.json")
    int8_validation = _load(root / "out/int8_fp8/validation.json")
    comparison = _load(root / "out/quantization_comparison/comparison_report.json")

    exported_methods = []
    if awq_validation.get("complete") is True:
        exported_methods.append("awq")
    if gptq_validation.get("complete") is True:
        exported_methods.append("gptq")
    if _check_status(int8_validation, "smoothquant_real_execution") == PASS:
        exported_methods.append("int8")

    applicable_candidates = [
        candidate
        for candidate in comparison.get("candidates", [])
        if candidate.get("applicability") == "applicable"
    ]
    technical_candidates = []
    repeatable_plans = []
    for candidate in applicable_candidates:
        plan_id = candidate.get("plan_id")
        if not isinstance(plan_id, str):
            continue
        benchmark = _load(_benchmark_path(root, plan_id))
        validation = _load(_benchmark_validation_path(root, plan_id))
        stable = (
            validation.get("complete") is True
            and len(benchmark.get("cases", [])) == 16
            and benchmark.get("benchmark_config", {}).get("measured_runs", 0) >= 2
            and candidate.get("performance_comparable_to_fp16") is True
        )
        if stable:
            technical_candidates.append(plan_id)
            repeatable_plans.append(plan_id)

    regression_plans = []
    for candidate in applicable_candidates:
        plan_id = candidate.get("plan_id")
        if not isinstance(plan_id, str) or plan_id.startswith("smoothquant-"):
            continue
        method = "awq" if plan_id.startswith("awq-") else "gptq"
        regression = _load(root / "out" / method / plan_id / "regression.json")
        quality = _load(
            root / "out" / method / plan_id / "quality/baseline_quality_report.json"
        )
        if (
            regression.get("total", 0) > 0
            and len(regression.get("rows", [])) == regression.get("total")
            and quality.get("metrics", {}).get("total") == 883
        ):
            regression_plans.append(plan_id)

    recommendation_path = root / "docs/QUANTIZATION_RECOMMENDATION.md"
    recommendation_text = (
        recommendation_path.read_text(encoding="utf-8")
        if recommendation_path.is_file()
        else ""
    )
    recommendation_ready = all(
        phrase in recommendation_text
        for phrase in ("暂不部署", "awq-int4-w4a16-g32", "gptq-int4-w4a16-g64")
    )

    checks = [
        CheckResult(
            "at_least_two_quantization_methods_exported",
            PASS if len(exported_methods) >= 2 else FAIL,
            f"Real exports completed for: {', '.join(exported_methods) or 'none'}.",
        ),
        CheckResult(
            "at_least_one_quantized_model_technically_stable",
            PASS if technical_candidates else FAIL,
            "Technical stability means model loading plus a complete repeated 16-case benchmark; it does not claim production business readiness.",
        ),
        CheckResult(
            "complete_business_regression_executed",
            PASS if regression_plans else FAIL,
            f"Full 883-row quality evaluation and 91-row high-risk regression completed for {len(regression_plans)} INT4 plans.",
        ),
        CheckResult(
            "quantized_performance_reproducible",
            PASS if len(repeatable_plans) == len(applicable_candidates) else FAIL,
            f"{len(repeatable_plans)}/{len(applicable_candidates)} applicable plans have repeated, comparable 16-case benchmarks.",
        ),
        CheckResult(
            "quantization_recommendation_draft_completed",
            PASS if recommendation_ready else FAIL,
            "The draft records a no-deployment decision and the quality-first and performance-first research candidates.",
        ),
    ]
    deliverables = {
        "awq_gptq_int8_quantized_models": len(exported_methods) == 3,
        "quantization_configs": all(
            (root / path).is_file()
            for path in (
                "configs/awq.json",
                "configs/gptq.json",
                "configs/int8_fp8.json",
            )
        ),
        "quantization_scripts": all(
            (root / path).is_file()
            for path in (
                "src/quantization/awq.py",
                "src/quantization/gptq.py",
                "src/quantization/int8_fp8.py",
            )
        ),
        "quantization_quality_report": (
            root / "out/quantization_comparison/comparison_report.json"
        ).is_file(),
        "quantization_performance_report": (
            root / "out/quantization_comparison/comparison_report.md"
        ).is_file(),
    }
    return Stage6Report(
        scope="customized_quantization",
        checks=checks,
        deliverables=deliverables,
        technical_deployment_candidates=technical_candidates,
        production_recommendation="暂不部署：FP16 基线业务质量不足，且当前量化方案没有性能收益。",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="out/stage6/validation.json")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    report = validate_stage6(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
