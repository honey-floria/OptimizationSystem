import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.baseline_quality import validate_quality_report
from src.input_validation.model_input import FAIL, PASS


class BaselineQualityValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.predictions_path = self.root / "baseline_predictions.jsonl"
        rows = [
            {"index": 0, "prediction": "10", "reference": "10"},
            {"index": 1, "prediction": "20", "reference": "20"},
        ]
        self.predictions_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def report(self):
        return {
            "model_metadata": {
                "model_id": "test/model",
                "version": "1",
                "commit": "abc",
            },
            "environment": {"dtype": "float16"},
            "dataset": {
                "role": "evaluation_dev",
                "version": "main",
                "sha256": "dataset-hash",
                "expected_size": 2,
                "evaluated_size": 2,
            },
            "metrics": {
                "total": 2,
                "parse_rate": 1.0,
                "numeric_accuracy": 1.0,
            },
            "quality_gate": {"parse_rate_passed": True},
            "predictions": {
                "file": self.predictions_path.name,
                "row_count": 2,
                "sha256": hashlib.sha256(
                    self.predictions_path.read_bytes()
                ).hexdigest(),
            },
        }

    def validate(self, payload):
        report_path = self.root / "report.json"
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return validate_quality_report(report_path)

    def test_complete_quality_report_passes_five_checks(self):
        report = self.validate(self.report())

        self.assertTrue(report.complete)
        self.assertEqual([check.status for check in report.checks], [PASS] * 5)

    def test_parse_gate_failure_is_reported(self):
        payload = self.report()
        payload["quality_gate"]["parse_rate_passed"] = False

        report = self.validate(payload)

        self.assertEqual(report.checks[2].status, FAIL)
