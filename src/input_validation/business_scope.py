"""Validate the approved multi-scenario business scope portfolio."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, PASS, CheckResult


@dataclass
class BusinessScopeReport:
    config: str
    primary_scenario_id: str
    scenario_ids: list[str]
    active_scenario_ids: list[str]
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "primary_scenario_id": self.primary_scenario_id,
            "scenario_ids": self.scenario_ids,
            "active_scenario_ids": self.active_scenario_ids,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_text_list(value: Any, minimum: int = 1) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= minimum
        and all(_has_text(item) for item in value)
    )


def validate_business_scope(config_file: str | Path) -> BusinessScopeReport:
    config_path = Path(config_file).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid business scope config: {exc}") from exc

    portfolio = config.get("portfolio", {})
    scenarios = config.get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    scenario_ids = [
        scenario.get("scenario_id", "")
        for scenario in scenarios
        if isinstance(scenario, dict)
    ]
    active_scenario_ids = [
        scenario.get("scenario_id", "")
        for scenario in scenarios
        if isinstance(scenario, dict) and scenario.get("status") == "active"
    ]
    primary_scenario_id = portfolio.get("primary_scenario_id", "")
    primary = next(
        (
            scenario
            for scenario in scenarios
            if scenario.get("scenario_id") == primary_scenario_id
        ),
        {},
    )
    scenario_ready = (
        config.get("schema_version") == 2
        and len(scenarios) >= 3
        and len(scenario_ids) == len(set(scenario_ids))
        and all(
            _has_text(scenario.get("scenario_id"))
            and _has_text(scenario.get("name"))
            and _has_text(scenario.get("business_goal"))
            and _has_text_list(scenario.get("intended_users"))
            and scenario.get("status") in {"active", "planned"}
            for scenario in scenarios
        )
        and primary.get("status") == "active"
        and active_scenario_ids == [primary_scenario_id]
        and all(
            scenario.get("dataset_binding", {}).get("status")
            == ("ready" if scenario.get("status") == "active" else "pending")
            for scenario in scenarios
        )
        and portfolio.get("approval", {}).get("status")
        == "confirmed_for_prototype"
    )

    risk_levels = {scenario.get("risk_level") for scenario in scenarios}
    risk_limits = {"L1": 1.0, "L2": 0.5, "L3": 0.0}
    risk_ready = risk_levels == set(risk_limits)
    for scenario in scenarios:
        policy = scenario.get("compression_policy", {})
        risk_level = scenario.get("risk_level")
        allowed_quantization = policy.get("allowed_quantization", [])
        if risk_level == "L1":
            risk_ready = risk_ready and "INT4" in allowed_quantization
        elif risk_level == "L2":
            risk_ready = risk_ready and policy.get("mixed_precision_required") is True
        elif risk_level == "L3":
            risk_ready = (
                risk_ready
                and "INT4" not in allowed_quantization
                and "FP16" in allowed_quantization
                and policy.get("aggressive_pruning_allowed") is False
            )

    outputs_ready = bool(scenarios) and all(
        _has_text_list(scenario.get("allowed_outputs"), minimum=3)
        for scenario in scenarios
    )
    review_ready = bool(scenarios) and all(
        scenario.get("human_review", {}).get("required") is True
        and _has_text_list(
            scenario.get("human_review", {}).get("required_when"), minimum=3
        )
        for scenario in scenarios
    )

    metrics_ready = bool(scenarios)
    for scenario in scenarios:
        gates = scenario.get("quality_gates", {})
        maximum_drop = gates.get("max_drop_from_fp16_percentage_points")
        metrics_ready = (
            metrics_ready
            and _has_text(gates.get("primary_metric"))
            and isinstance(maximum_drop, (int, float))
            and maximum_drop <= risk_limits.get(scenario.get("risk_level"), -1)
            and gates.get("max_new_critical_errors") == 0
            and gates.get("require_input_context") is True
        )
    exclusions_ready = (
        _has_text_list(config.get("global_out_of_scope"), minimum=5)
        and all(
            _has_text_list(scenario.get("out_of_scope"), minimum=3)
            for scenario in scenarios
        )
    )

    checks = [
        CheckResult(
            "first_business_scenario_confirmed",
            PASS if scenario_ready else FAIL,
            "L1/L2/L3 scenario portfolio, primary scenario, and approval are recorded.",
        ),
        CheckResult(
            "business_risk_level_confirmed",
            PASS if risk_ready else FAIL,
            "Each scenario has a risk-aligned compression policy.",
        ),
        CheckResult(
            "allowed_model_outputs_confirmed",
            PASS if outputs_ready else FAIL,
            "Allowed outputs and abstention behavior are defined per scenario.",
        ),
        CheckResult(
            "mandatory_human_review_confirmed",
            PASS if review_ready else FAIL,
            "Mandatory human-review conditions are defined per scenario.",
        ),
        CheckResult(
            "quality_metrics_and_thresholds_confirmed",
            PASS if metrics_ready else FAIL,
            "Risk-specific quality, regression, and grounding gates are defined.",
        ),
        CheckResult(
            "out_of_scope_functions_confirmed",
            PASS if exclusions_ready else FAIL,
            "Global and scenario-specific exclusions are explicitly listed.",
        ),
    ]
    return BusinessScopeReport(
        config=str(config_path),
        primary_scenario_id=str(primary_scenario_id),
        scenario_ids=scenario_ids,
        active_scenario_ids=active_scenario_ids,
        checks=checks,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = validate_business_scope(args.config)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
