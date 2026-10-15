"""Validate TODO 7.5 degradation-localization and mixed-precision deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, PASS, CheckResult


@dataclass
class Stage7Report:
    scope: str
    checks: list[CheckResult]
    deliverables: dict[str, bool]
    selected_deployment_plan: str | None
    claim_boundaries: list[str]
    evidence_sha256: dict[str, str]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks) and all(self.deliverables.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": "7.5",
            "scope": self.scope,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
            "deliverables": self.deliverables,
            "selected_deployment_plan": self.selected_deployment_plan,
            "claim_boundaries": self.claim_boundaries,
            "evidence_sha256": self.evidence_sha256,
        }


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid stage 7 evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Stage 7 evidence must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_candidate(ranking: dict[str, Any]) -> dict[str, Any] | None:
    selected = [row for row in ranking.get("candidates", []) if row.get("selected") is True]
    return selected[0] if len(selected) == 1 else None


def _plan_is_mixed_precision(plan: dict[str, Any]) -> bool:
    precisions = {str(plan.get("default_linear_precision", ""))}
    precisions.update(str(value) for value in plan.get("overrides", {}).values())
    precisions.update(str(value) for value in plan.get("special_modules", {}).values())
    precisions.discard("")
    return len(precisions) >= 2


def _quality_suite_complete(payload: dict[str, Any], expected_total: int) -> bool:
    metrics = payload.get("metrics", {})
    return (
        payload.get("status") == "completed"
        and metrics.get("total") == expected_total
        and isinstance(metrics.get("correct"), int)
        and isinstance(metrics.get("parse_rate"), (int, float))
        and isinstance(payload.get("predictions_sha256"), str)
        and len(payload["predictions_sha256"]) == 64
    )


def validate_stage7(project_root: str | Path) -> Stage7Report:
    root = Path(project_root).expanduser().resolve()
    paths = {
        "layer_matrix": root / "out/layer_sensitivity/matrix/matrix_summary.json",
        "layer_final_validation": root / "out/layer_sensitivity/final_evaluation/validation.json",
        "layer_ranking": root / "out/layer_sensitivity/final_evaluation/ranking.json",
        "module_validation": root / "out/module_sensitivity/final_evaluation/validation.json",
        "module_ranking": root / "out/module_sensitivity/final_evaluation/ranking.json",
        "financial_validation": root / "out/financial_error_localization/validation.json",
        "financial_statistics": root / "out/financial_error_localization/error_statistics.json",
        "mixed_validation": root / "out/mixed_precision_search/validation.json",
        "mixed_ranking": root / "out/mixed_precision_search/ranking.json",
    }
    evidence = {name: _load(path) for name, path in paths.items()}

    layer_preview = evidence["layer_matrix"].get("ranking_preview", [])
    final_layers = evidence["layer_ranking"].get("ranking", [])
    layer_ready = (
        evidence["layer_matrix"].get("status") == "complete"
        and evidence["layer_matrix"].get("pending") == []
        and len(layer_preview) == 28
        and {row.get("plan", {}).get("layer_index") for row in layer_preview} == set(range(28))
        and evidence["layer_final_validation"].get("status") == "pass"
        and evidence["layer_ranking"].get("status") == "candidate_subset_final"
        and len(final_layers) >= 1
    )

    module_recovery = evidence["module_ranking"].get("recovery_ranking", [])
    module_special = evidence["module_ranking"].get("special_loss_benefit_ranking", [])
    module_ready = (
        evidence["module_validation"].get("status") == "pass"
        and evidence["module_validation"].get("pending") == {
            "full_dev": [], "high_risk": [], "determinism": [],
        }
        and evidence["module_ranking"].get("status") == "candidate_subset_final"
        and len(module_recovery) >= 1
        and len(module_special) >= 1
    )

    degradation_rows = evidence["financial_statistics"].get("degradation", [])
    localized = [
        row for row in degradation_rows
        if row.get("net_correct_gain", 0) < 0
        and any(value < 0 for value in row.get("business_net_change", {}).values())
    ]
    localization_ready = (
        evidence["financial_validation"].get("status") == "pass"
        and evidence["financial_statistics"].get("status") == "completed_with_declared_coverage_gaps"
        and bool(localized)
        and evidence["financial_statistics"].get("coverage", {}).get(
            "zero_errors_must_not_be_inferred_from_not_evaluable"
        ) is True
    )

    mixed_ranking = evidence["mixed_ranking"]
    selected = _selected_candidate(mixed_ranking)
    selected_plan_id = str(selected["plan_id"]) if selected else None
    plan = None
    artifact = None
    quality_evidence: dict[str, dict[str, Any]] = {}
    benchmark: dict[str, Any] = {}
    if selected_plan_id:
        plan_path = root / "out/mixed_precision_search/plans" / selected_plan_id / "plan.json"
        artifact_path = root / "out/mixed_precision_search/exports" / selected_plan_id / "quantized_model_manifest.json"
        plan = _load(plan_path)
        artifact = _load(artifact_path)
        for suite in ("full_dev", "high_risk", "long_context"):
            quality_evidence[suite] = _load(
                root / "out/mixed_precision_search/evidence" / selected_plan_id / f"{suite}.json"
            )
        benchmark = _load(
            root / "out/mixed_precision_search/evidence" / selected_plan_id / "benchmark.json"
        )
        paths["selected_plan"] = plan_path
        paths["selected_artifact"] = artifact_path

    artifact_ready = bool(
        selected
        and plan
        and artifact
        and evidence["mixed_validation"].get("complete") is True
        and evidence["mixed_validation"].get("deployment_claims_allowed") is True
        and mixed_ranking.get("status") == "completed"
        and mixed_ranking.get("deployment_claims_allowed") is True
        and mixed_ranking.get("recommendation") == selected_plan_id
        and selected.get("quality_pass") is True
        and selected.get("performance_pass") is True
        and plan.get("runnable") is True
        and _plan_is_mixed_precision(plan)
        and artifact.get("status") == "completed"
        and artifact.get("backend_artifact_valid") is True
        and int(artifact.get("state_dict_bytes", 0)) > 0
        and isinstance(artifact.get("state_dict_sha256"), str)
        and len(artifact["state_dict_sha256"]) == 64
    )

    cases = benchmark.get("cases", [])
    completed_cases = [case for case in cases if case.get("status", "completed") == "completed"]
    regression_ready = bool(
        selected
        and _quality_suite_complete(quality_evidence.get("full_dev", {}), 883)
        and _quality_suite_complete(quality_evidence.get("high_risk", {}), 91)
        and _quality_suite_complete(quality_evidence.get("long_context", {}), 89)
        and len(cases) == 16
        and len(completed_cases) == 16
        and benchmark.get("benchmark_config", {}).get("measured_runs", 0) >= 2
        and selected.get("benchmark", {}).get("oom_case_count") == 0
    )

    checks = [
        CheckResult(
            "layer_sensitivity_ranking_generated",
            PASS if layer_ready else FAIL,
            f"28/28 pilot layers ranked; {len(final_layers)} pre-registered layers completed full-dev confirmation.",
        ),
        CheckResult(
            "module_sensitivity_ranking_generated",
            PASS if module_ready else FAIL,
            f"Final ranking contains {len(module_recovery)} recovery and {len(module_special)} special-module candidates.",
        ),
        CheckResult(
            "financial_business_degradation_localized",
            PASS if localization_ready else FAIL,
            f"Localized negative business-category transitions for: {', '.join(row['plan_id'] for row in localized) or 'none'}.",
        ),
        CheckResult(
            "deployable_mixed_precision_configuration_generated",
            PASS if artifact_ready else FAIL,
            f"Selected plan {selected_plan_id or 'none'} combines low-bit Linear weights with BF16 special modules and passed final gates.",
        ),
        CheckResult(
            "mixed_precision_business_regression_and_benchmark_completed",
            PASS if regression_ready else FAIL,
            "Selected plan has 883 full-dev, 91 high-risk, 89 long-context results and a complete repeated 16-case benchmark.",
        ),
    ]
    deliverable_paths = {
        "sensitive_layer_ranking": root / "out/layer_sensitivity/final_evaluation/ranking.json",
        "business_error_classification_report": root / "out/financial_error_localization/error_statistics.json",
        "mixed_precision_configuration": (
            root / "out/mixed_precision_search/plans" / (selected_plan_id or "missing") / "plan.json"
        ),
        "degradation_localization_script": root / "scripts/run_financial_error_localization.py",
        "degradation_analysis_report": root / "out/financial_error_localization/report.md",
    }
    deliverables = {name: path.is_file() and path.stat().st_size > 0 for name, path in deliverable_paths.items()}
    hash_paths = {**paths, **{f"deliverable:{name}": path for name, path in deliverable_paths.items()}}
    evidence_sha256 = {
        name: _sha256(path) for name, path in hash_paths.items() if path.is_file()
    }
    return Stage7Report(
        scope="degradation_localization_and_mixed_precision",
        checks=checks,
        deliverables=deliverables,
        selected_deployment_plan=selected_plan_id,
        claim_boundaries=[
            "Layer full-dev ranking covers the pre-registered candidate subset; the 28-layer ranking is pilot evidence.",
            "Module ranking covers pre-registered promoted candidates, not every module on full-dev.",
            "The selected all_linear_int4 plan is mixed precision at model scope: Linear INT4 with Embedding, Norm, and KV Cache BF16.",
            "Financial localization is complete for the active L2 FinQA scenario; L1/L3 and annotation-dependent safety categories remain not_evaluable.",
        ],
        evidence_sha256=evidence_sha256,
    )


def _markdown(report: Stage7Report) -> str:
    lines = [
        "# 第 7 章阶段验收报告",
        "",
        f"状态：`{'pass' if report.complete else 'fail'}`；选择的部署计划：`{report.selected_deployment_plan or 'none'}`。",
        "",
        "## 验收项",
        "",
        "| 验收项 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for check in report.checks:
        lines.append(f"| `{check.name}` | {check.status} | {check.detail} |")
    lines.extend(["", "## 交付物", ""])
    for name, ready in report.deliverables.items():
        lines.append(f"- [{'x' if ready else ' '}] `{name}`")
    lines.extend(["", "## 结论边界", ""])
    lines.extend(f"- {boundary}" for boundary in report.claim_boundaries)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="out/stage7/validation.json")
    parser.add_argument("--report", default="out/stage7/report.md")
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    report = validate_stage7(root)
    payload = report.to_dict()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
