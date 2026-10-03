import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.finqa_assets import dataset_hash
from src.input_validation.dataset_input import validate_dataset_input
from src.input_validation.model_input import FAIL, PASS


class DatasetInputValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.datasets = {
            "calibration": [self.finqa_row("calibration question")],
            "evaluation_dev": [self.finqa_row("development question")],
            "evaluation_test": [self.finqa_row("test question")],
            "high_risk_regression": [
                {"question": "regression question", "answer": "10"}
            ],
        }
        self.manifest_path = self.root / "manifest.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def finqa_row(question):
        return {
            "pre_text": ["before"],
            "post_text": ["after"],
            "table": [["value", "10"]],
            "qa": {"question": question},
        }

    def write_manifest(self):
        assets = {}
        for role, rows in self.datasets.items():
            asset_path = self.root / role
            asset_path.mkdir(exist_ok=True)
            assets[role] = {
                "path": role,
                "version": "test-v1",
                "size": len(rows),
                "sha256": dataset_hash(rows),
                "required_fields": (
                    ["question", "answer"]
                    if role == "high_risk_regression"
                    else ["pre_text", "post_text", "table", "qa.question"]
                ),
            }
        manifest = {
            "schema_version": 1,
            "assets": assets,
            "metric_interface": {
                "callable": "src.evaluation.finqa_metrics:evaluate_numeric_answers"
            },
            "regression_policy": {
                "callable": "src.evaluation.finqa_metrics:classify_numeric_regression",
                "critical_error_types": ["numeric_answer_mismatch"],
                "new_regression_definition": "baseline_correct_and_candidate_incorrect",
            },
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_load_from_disk(self, path):
        return self.datasets[Path(path).name]

    def validate(self):
        with patch(
            "src.input_validation.dataset_input._require_datasets",
            return_value=self.fake_load_from_disk,
        ):
            return validate_dataset_input(self.manifest_path)

    def test_all_nine_checks_pass(self):
        self.write_manifest()
        report = self.validate()

        self.assertTrue(report.complete)
        self.assertEqual(len(report.checks), 9)
        self.assertEqual([check.status for check in report.checks], [PASS] * 9)

    def test_overlap_fails_isolation_check(self):
        self.datasets["evaluation_test"] = [self.finqa_row("calibration question")]
        self.write_manifest()
        report = self.validate()

        self.assertFalse(report.complete)
        self.assertEqual(report.checks[4].status, FAIL)


if __name__ == "__main__":
    unittest.main()
