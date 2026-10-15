"""Locate TODO 7.3 financial error categories in frozen prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ERROR_TYPES = (
    "amount_error",
    "rate_error",
    "date_or_term_error",
    "clause_omission",
    "citation_error",
    "json_format_error",
    "improper_financial_advice",
    "should_refuse_but_answered",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _indexed(rows: list[dict[str, Any]], name: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "source_index" not in row:
            raise ValueError(f"{name} row has no source_index")
        source_index = int(row["source_index"])
        if source_index in indexed:
            raise ValueError(f"{name} has duplicate source_index={source_index}")
        indexed[source_index] = row
    return indexed


def _structural_citation_error(row: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    evidence = row.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return True
    return any(not isinstance(item, str) or not pattern.fullmatch(item.strip()) for item in evidence)


def _json_format_error(row: dict[str, Any]) -> bool:
    return not bool(row.get("parsed")) or not bool(row.get("structured_output")) or bool(row.get("validation_error"))


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _metric(status: str, sample_count: int, error_count: int, **extra: Any) -> dict[str, Any]:
    result = {
        "status": status,
        "sample_count": sample_count,
        "error_count": error_count,
        "error_rate": (error_count / sample_count) if sample_count else None,
    }
    result.update(extra)
    return result


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 7.3 金融错误定位报告",
        "",
        f"状态：`{report['status']}`；基线：`{report['baseline_plan_id']}`；完整 dev 样本：{report['sample_count']}。",
        "",
        "## 当前可评测业务场景",
        "",
        "| 场景 | 状态 | 样本数 | 说明 |",
        "|---|---|---:|---|",
    ]
    for row in report["scenario_summary"]:
        lines.append(f"| `{row['scenario_id']}` | {row['evaluation_status']} | {row['sample_count']} | {row['reason']} |")
    lines.extend([
        "",
        "## 错误统计",
        "",
        "金额、利率和日期/期限允许重叠，不能横向相加。引用错误是结构代理：只验证 evidence 非空且引用形态合法，不能证明语义引用正确。",
        "",
        "| 计划 | 金额 | 利率/比例 | 日期/期限 | JSON | 引用结构 | 条款遗漏 | 不当建议 | 应拒答误答 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    labels = {
        "amount_error": "amount",
        "rate_error": "rate",
        "date_or_term_error": "date",
        "json_format_error": "json",
        "citation_error": "citation",
        "clause_omission": "clause",
        "improper_financial_advice": "advice",
        "should_refuse_but_answered": "refusal",
    }
    for plan in report["plans"]:
        values = []
        for error_type in labels:
            metric = plan["errors"][error_type]
            values.append(
                f"{metric['error_count']}/{metric['sample_count']}"
                if metric["status"] != "not_evaluable" else "N/A"
            )
        lines.append(f"| `{plan['plan_id']}` | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## 相对基线的退化",
        "",
        "| 计划 | 恢复 | 退化 | 净正确数 | 金额净变化 | 利率/比例净变化 | 日期/期限净变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["degradation"]:
        lines.append(
            f"| `{row['plan_id']}` | {row['recovered_count']} | {row['regressed_count']} | "
            f"{row['net_correct_gain']:+d} | {row['business_net_change']['amount_error']:+d} | "
            f"{row['business_net_change']['rate_error']:+d} | {row['business_net_change']['date_or_term_error']:+d} |"
        )
    lines.extend([
        "",
        "## 覆盖边界",
        "",
        "当前冻结资产只覆盖 L2 内部财报数值问答。条款遗漏、不当金融建议和应拒答误答缺少带对应标注的样本，统计状态为 `not_evaluable`，不是零错误。引用统计仅为结构代理。L1/L3 场景须在数据接入后沿用同一接口重新生成报告。",
        "",
    ])
    return "\n".join(lines)


def analyze_financial_errors(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = _resolve(root, config["paths"]["output_root"])
    labels_path = _resolve(root, config["paths"]["business_labels"])
    scope_path = _resolve(root, config["paths"]["business_scope"])
    labels = _indexed(_read_jsonl(labels_path), "business labels")
    scope = _read_json(scope_path)
    plan_specs = config.get("plans", [])
    if not plan_specs:
        raise ValueError("financial_error_localization.plans must not be empty")
    plan_ids = [str(row["plan_id"]) for row in plan_specs]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("Plan IDs must be unique")
    baseline_id = str(config["baseline_plan_id"])
    if baseline_id not in plan_ids:
        raise ValueError("baseline_plan_id is not present in plans")

    taxonomy = config["taxonomy"]
    if set(taxonomy) != set(ERROR_TYPES):
        raise ValueError("Taxonomy must define every TODO 7.3 error type exactly once")
    citation_pattern = re.compile(taxonomy["citation_error"]["accepted_reference_pattern"], re.IGNORECASE)
    advice_patterns = [re.compile(value, re.IGNORECASE) for value in taxonomy["improper_financial_advice"]["output_patterns"]]
    refusal_patterns = [re.compile(value, re.IGNORECASE) for value in taxonomy["should_refuse_but_answered"]["refusal_patterns"]]

    predictions: dict[str, dict[int, dict[str, Any]]] = {}
    input_artifacts = {
        str(labels_path): _sha256(labels_path),
        str(scope_path): _sha256(scope_path),
    }
    expected_indices = set(labels)
    for spec in plan_specs:
        plan_id = str(spec["plan_id"])
        path = _resolve(root, spec["predictions"])
        rows = _indexed(_read_jsonl(path), plan_id)
        if set(rows) != expected_indices:
            missing = sorted(expected_indices - set(rows))[:10]
            extra = sorted(set(rows) - expected_indices)[:10]
            raise ValueError(f"{plan_id} source indices differ from frozen labels; missing={missing}, extra={extra}")
        for index, row in rows.items():
            if str(row.get("question", "")) != str(labels[index].get("question", "")):
                raise ValueError(f"{plan_id} question mismatch at source_index={index}")
        predictions[plan_id] = rows
        input_artifacts[str(path)] = _sha256(path)

    baseline = predictions[baseline_id]
    sample_rows: list[dict[str, Any]] = []
    plan_summaries = []
    annotations = {
        "clause_omission": taxonomy["clause_omission"]["requires_annotation"],
        "improper_financial_advice": taxonomy["improper_financial_advice"]["requires_annotation"],
        "should_refuse_but_answered": taxonomy["should_refuse_but_answered"]["requires_annotation"],
    }
    for spec in plan_specs:
        plan_id = str(spec["plan_id"])
        rows = predictions[plan_id]
        counters: Counter[str] = Counter()
        eligible: Counter[str] = Counter()
        for source_index in sorted(rows):
            row = rows[source_index]
            label_row = labels[source_index]
            business_labels = set(map(str, label_row.get("labels", [])))
            flags: dict[str, bool | None] = {}
            for error_type in ("amount_error", "rate_error", "date_or_term_error"):
                business_label = taxonomy[error_type]["business_label"]
                is_eligible = business_label in business_labels
                flags[error_type] = (not bool(row.get("correct"))) if is_eligible else None
            flags["citation_error"] = _structural_citation_error(row, citation_pattern)
            flags["json_format_error"] = _json_format_error(row)
            raw = str(row.get("raw_prediction", ""))
            flags["improper_financial_advice"] = (
                _matches_any(raw, advice_patterns)
                if bool(label_row.get(annotations["improper_financial_advice"])) else None
            )
            flags["should_refuse_but_answered"] = (
                not _matches_any(raw, refusal_patterns)
                if bool(label_row.get(annotations["should_refuse_but_answered"])) else None
            )
            flags["clause_omission"] = (
                not bool(row.get("evidence"))
                if bool(label_row.get(annotations["clause_omission"])) else None
            )
            for error_type, flag in flags.items():
                if flag is not None:
                    eligible[error_type] += 1
                    counters[error_type] += int(flag)
            baseline_correct = bool(baseline[source_index].get("correct"))
            candidate_correct = bool(row.get("correct"))
            transition = (
                "recovered" if not baseline_correct and candidate_correct else
                "regressed" if baseline_correct and not candidate_correct else
                "stable_correct" if baseline_correct else "stable_incorrect"
            )
            sample_rows.append({
                "scenario_id": str(spec["scenario_id"]),
                "plan_id": plan_id,
                "source_index": source_index,
                "question": row.get("question"),
                "business_labels": sorted(business_labels),
                "correct": candidate_correct,
                "parsed": bool(row.get("parsed")),
                "structured_output": bool(row.get("structured_output")),
                "transition_vs_baseline": transition,
                "errors": flags,
            })
        errors = {}
        for error_type in ERROR_TYPES:
            count = int(eligible[error_type])
            if count == 0 and error_type in annotations:
                errors[error_type] = _metric(
                    "not_evaluable", 0, 0,
                    reason=f"No frozen rows carry required annotation '{annotations[error_type]}'.",
                )
            else:
                errors[error_type] = _metric(
                    "evaluated" if error_type != "citation_error" else "proxy_only",
                    count, int(counters[error_type]),
                    **({"limitation": "Structural evidence validation only; semantic citation correctness is not observable."}
                       if error_type == "citation_error" else {}),
                )
        plan_summaries.append({
            "plan_id": plan_id,
            "scenario_id": str(spec["scenario_id"]),
            "sample_count": len(rows),
            "correct_count": sum(bool(row.get("correct")) for row in rows.values()),
            "errors": errors,
        })

    degradation = []
    for spec in plan_specs:
        plan_id = str(spec["plan_id"])
        if plan_id == baseline_id:
            continue
        transitions = Counter()
        label_transitions: dict[str, Counter[str]] = defaultdict(Counter)
        for source_index, row in predictions[plan_id].items():
            before = bool(baseline[source_index].get("correct"))
            after = bool(row.get("correct"))
            transition = "recovered" if not before and after else "regressed" if before and not after else "unchanged"
            transitions[transition] += 1
            for error_type in ("amount_error", "rate_error", "date_or_term_error"):
                label = taxonomy[error_type]["business_label"]
                if label in labels[source_index].get("labels", []):
                    label_transitions[error_type][transition] += 1
        degradation.append({
            "plan_id": plan_id,
            "baseline_plan_id": baseline_id,
            "recovered_count": transitions["recovered"],
            "regressed_count": transitions["regressed"],
            "unchanged_count": transitions["unchanged"],
            "net_correct_gain": transitions["recovered"] - transitions["regressed"],
            "business_net_change": {
                error_type: counts["recovered"] - counts["regressed"]
                for error_type, counts in label_transitions.items()
            },
        })

    plan_scenarios = {str(row["scenario_id"]) for row in plan_specs}
    scenario_summary = []
    for scenario in scope.get("scenarios", []):
        scenario_id = str(scenario["scenario_id"])
        covered = scenario_id in plan_scenarios
        scenario_summary.append({
            "scenario_id": scenario_id,
            "name": scenario.get("name"),
            "risk_level": scenario.get("risk_level"),
            "scope_status": scenario.get("status"),
            "evaluation_status": "evaluated" if covered else "not_evaluable",
            "sample_count": len(expected_indices) if covered else 0,
            "reason": "Frozen full-dev predictions available." if covered else "No dataset/prediction binding is available for this planned scenario.",
        })

    report = {
        "schema_version": 1,
        "stage": "7.3",
        "status": "completed_with_declared_coverage_gaps",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_plan_id": baseline_id,
        "sample_count": len(expected_indices),
        "plans": plan_summaries,
        "degradation": degradation,
        "scenario_summary": scenario_summary,
        "coverage": {
            "evaluated_scenarios": sorted(plan_scenarios),
            "not_evaluable_error_types": sorted(annotations),
            "proxy_error_types": ["citation_error"],
            "zero_errors_must_not_be_inferred_from_not_evaluable": True,
        },
    }
    validation = {
        "status": "pass",
        "stage": "7.3",
        "plan_count": len(plan_specs),
        "sample_count_per_plan": len(expected_indices),
        "aligned_source_indices": True,
        "all_error_types_reported": all(set(row["errors"]) == set(ERROR_TYPES) for row in plan_summaries),
        "coverage_gaps_declared": True,
    }
    _write_jsonl(output / "sample_errors.jsonl", sample_rows)
    _write_json(output / "error_statistics.json", report)
    (output / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    _write_json(output / "validation.json", validation)
    manifest = {
        "schema_version": 1,
        "stage": "7.3",
        "status": "completed",
        "created_at": report["created_at"],
        "inputs": input_artifacts,
        "outputs": {
            name: _sha256(output / name)
            for name in ("sample_errors.jsonl", "error_statistics.json", "report.md", "validation.json")
        },
    }
    _write_json(output / "experiment_manifest.json", manifest)
    return report


def validate_financial_error_outputs(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = _resolve(root, config["paths"]["output_root"])
    manifest = _read_json(output / "experiment_manifest.json")
    mismatches = []
    for raw_path, expected in manifest.get("inputs", {}).items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(f"input:{path}")
    for name, expected in manifest.get("outputs", {}).items():
        path = output / name
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(f"output:{name}")
    validation = _read_json(output / "validation.json")
    if validation.get("status") != "pass":
        mismatches.append("validation.status")
    return {"status": "pass" if not mismatches else "fail", "mismatches": mismatches}
