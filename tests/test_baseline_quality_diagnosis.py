import unittest
import json
import tempfile
from decimal import Decimal
from pathlib import Path

from src.evaluation.baseline_quality_diagnosis import (
    analysis_dimensions,
    build_diagnosis,
    classify_error,
    classify_question,
    merge_manual_reviews,
    numeric_candidates,
    validate_diagnosis,
)


class BaselineQualityDiagnosisTest(unittest.TestCase):
    def test_extracts_all_numeric_candidates_with_units(self):
        self.assertEqual(
            numeric_candidates("$1,200 divided by 50 gives 24.0%"),
            [Decimal("1200"), Decimal("50"), Decimal("0.240")],
        )

    def test_flags_first_number_scoring_bias(self):
        row = {
            "prediction": "555.3 - 536.7 = 18.6 million",
            "reference": "18.6",
            "correct": False,
        }
        self.assertEqual(
            classify_error(row, Decimal("0.0001")),
            "answer_extraction_false_negative_candidate",
        )

    def test_classifies_question_types(self):
        self.assertEqual(
            classify_question("what percentage of the total was leased?"),
            "percentage_or_ratio",
        )
        self.assertEqual(
            classify_question("what was the change from 2016 to 2017?"),
            "change_or_difference",
        )
        self.assertEqual(
            analysis_dimensions("what was the percentage change from 2016 to 2017?"),
            ["multi_step", "percentage", "time_series"],
        )

    def test_project_report_covers_frozen_baseline(self):
        root = Path(__file__).resolve().parents[1]
        report = build_diagnosis(root, "configs/baseline_quality_diagnosis.json")
        validation = validate_diagnosis(report)
        self.assertEqual(report["coverage"]["diagnosed_rows"], 883)
        self.assertEqual(report["coverage"]["manual_review_sample_size"], 50)
        self.assertEqual(
            set(report["required_dimension_statistics"]),
            {"single_step", "multi_step", "comparison", "percentage", "time_series"},
        )
        self.assertFalse(validation["complete"])
        self.assertEqual(validation["checks"][-1]["status"], "not_run")

    def test_preserves_completed_manual_review(self):
        root = Path(__file__).resolve().parents[1]
        report = build_diagnosis(root, "configs/baseline_quality_diagnosis.json")
        reviewed = {**report["manual_review_rows"][0]}
        reviewed["manual_review"] = {
            "status": "completed",
            "prediction_correct": False,
            "error_type": "calculation_error",
            "notes": "checked",
        }
        with tempfile.TemporaryDirectory() as directory:
            review_path = Path(directory) / "manual_review.jsonl"
            review_path.write_text(json.dumps(reviewed) + "\n", encoding="utf-8")
            merge_manual_reviews(report, review_path)
        self.assertEqual(report["coverage"]["manual_review_completed"], 1)
        self.assertEqual(
            report["manual_review_rows"][0]["manual_review"]["notes"], "checked"
        )

    def test_manual_review_json_export_is_single_json_array(self):
        root = Path(__file__).resolve().parents[1]
        report = build_diagnosis(root, "configs/baseline_quality_diagnosis.json")
        self.assertIsInstance(report["manual_review_rows"], list)


if __name__ == "__main__":
    unittest.main()
