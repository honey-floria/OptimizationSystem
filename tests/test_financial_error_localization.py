import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.financial_error_localization import (
    ERROR_TYPES,
    analyze_financial_errors,
    validate_financial_error_outputs,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> dict:
    labels = [
        {"source_index": 0, "question": "amount in 2020", "labels": ["amount", "date_or_term"]},
        {"source_index": 1, "question": "interest rate", "labels": ["rate_or_ratio"]},
    ]
    base = [
        {"source_index": 0, "question": "amount in 2020", "correct": True, "parsed": True, "structured_output": True, "evidence": ["table_1"], "raw_prediction": "{}"},
        {"source_index": 1, "question": "interest rate", "correct": False, "parsed": False, "structured_output": False, "evidence": [], "raw_prediction": "bad"},
    ]
    candidate = [
        {"source_index": 0, "question": "amount in 2020", "correct": False, "parsed": True, "structured_output": True, "evidence": ["table_1"], "raw_prediction": "{}"},
        {"source_index": 1, "question": "interest rate", "correct": True, "parsed": True, "structured_output": True, "evidence": ["table_2"], "raw_prediction": "{}"},
    ]
    _write_jsonl(tmp_path / "labels.jsonl", labels)
    _write_jsonl(tmp_path / "base.jsonl", base)
    _write_jsonl(tmp_path / "candidate.jsonl", candidate)
    _write_json(tmp_path / "scope.json", {"scenarios": [
        {"scenario_id": "active", "name": "Active", "risk_level": "L2", "status": "active"},
        {"scenario_id": "planned", "name": "Planned", "risk_level": "L3", "status": "planned"},
    ]})
    return {
        "paths": {"business_scope": "scope.json", "business_labels": "labels.jsonl", "output_root": "out"},
        "baseline_plan_id": "base",
        "plans": [
            {"plan_id": "base", "scenario_id": "active", "predictions": "base.jsonl"},
            {"plan_id": "candidate", "scenario_id": "active", "predictions": "candidate.jsonl"},
        ],
        "taxonomy": {
            "amount_error": {"business_label": "amount"},
            "rate_error": {"business_label": "rate_or_ratio"},
            "date_or_term_error": {"business_label": "date_or_term"},
            "clause_omission": {"requires_annotation": "clause_required"},
            "citation_error": {"mode": "structural_proxy", "accepted_reference_pattern": "^table_[0-9]+$"},
            "json_format_error": {"mode": "prediction_contract"},
            "improper_financial_advice": {"requires_annotation": "advice_prohibited", "output_patterns": ["buy"]},
            "should_refuse_but_answered": {"requires_annotation": "should_refuse", "refusal_patterns": ["cannot"]},
        },
    }


class FinancialErrorLocalizationTest(unittest.TestCase):
    def test_analyze_reports_errors_degradation_and_coverage(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            config = _fixture(root)
            report = analyze_financial_errors(root, config)
            self.assertEqual(report["status"], "completed_with_declared_coverage_gaps")
            self.assertEqual(report["degradation"][0]["recovered_count"], 1)
            self.assertEqual(report["degradation"][0]["regressed_count"], 1)
            self.assertEqual(report["degradation"][0]["business_net_change"], {
                "amount_error": -1, "date_or_term_error": -1, "rate_error": 1,
            })
            candidate = next(row for row in report["plans"] if row["plan_id"] == "candidate")
            self.assertEqual(set(candidate["errors"]), set(ERROR_TYPES))
            self.assertEqual(candidate["errors"]["amount_error"]["error_count"], 1)
            self.assertEqual(candidate["errors"]["clause_omission"]["status"], "not_evaluable")
            self.assertEqual(report["scenario_summary"][1]["evaluation_status"], "not_evaluable")
            self.assertEqual(validate_financial_error_outputs(root, config)["status"], "pass")
            (root / "candidate.jsonl").write_text("{}\n")
            self.assertEqual(validate_financial_error_outputs(root, config)["status"], "fail")

    def test_analyze_rejects_unaligned_predictions(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            config = _fixture(root)
            _write_jsonl(root / "candidate.jsonl", [])
            with self.assertRaisesRegex(ValueError, "source indices differ"):
                analyze_financial_errors(root, config)


if __name__ == "__main__":
    unittest.main()
