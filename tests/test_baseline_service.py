import json
import tempfile
import unittest
from pathlib import Path

from src.baseline.service import (
    InferenceRequest,
    _resolve_model_path,
    validate_baseline_evidence,
)
from src.input_validation.model_input import FAIL, PASS


class BaselineServiceTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.evidence_path = Path(self.temporary_directory.name) / "evidence.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def evidence(self):
        return {
            "model_loaded": True,
            "config": {
                "dtype": "float16",
                "prompt_template": "finqa-v1",
                "generation": {"max_new_tokens": 32, "do_sample": False},
                "deterministic": True,
                "seed": 42,
            },
            "single_request": {"success": True},
            "batch_request": {"success": True, "result_count": 2},
            "deterministic_replay_match": True,
            "request_logs": {
                "record_count": 4,
                "all_success": True,
                "required_fields_present": True,
            },
            "model_metadata": {
                "model_id": "test/model",
                "version": "1",
                "commit": "abc",
            },
            "environment": {
                "python": "3.13",
                "torch": "2.11",
                "transformers": "5.18",
                "cuda": "13.0",
                "gpu": "T4",
                "device": "cuda:0",
            },
        }

    def validate(self, evidence):
        self.evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return validate_baseline_evidence(self.evidence_path)

    def test_complete_evidence_passes_seven_checks(self):
        report = self.validate(self.evidence())

        self.assertTrue(report.complete)
        self.assertEqual([check.status for check in report.checks], [PASS] * 7)

    def test_batch_requires_two_results(self):
        evidence = self.evidence()
        evidence["batch_request"]["result_count"] = 1
        report = self.validate(evidence)

        self.assertEqual(report.checks[3].status, FAIL)

    def test_finqa_request_uses_unified_prompt(self):
        request = InferenceRequest.from_finqa_row(
            {
                "pre_text": ["before"],
                "post_text": ["after"],
                "table": [["value", "10"]],
                "qa": {"question": "What is the value?"},
            },
            request_id="request-1",
        )

        self.assertEqual(request.request_id, "request-1")
        self.assertIn("What is the value?", request.render_prompt())

    def test_relative_model_path_resolves_from_manifest(self):
        model_path = Path(self.temporary_directory.name) / "model"
        model_path.mkdir()
        manifest_path = Path(self.temporary_directory.name) / "manifest.json"

        resolved = _resolve_model_path(
            manifest_path, {"model_path": "model"}
        )

        self.assertEqual(resolved, model_path.resolve())

    def test_missing_model_path_has_clear_error(self):
        manifest_path = Path(self.temporary_directory.name) / "manifest.json"

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            _resolve_model_path(
                manifest_path, {"model_path": "missing-model"}
            )


if __name__ == "__main__":
    unittest.main()
