import json
import tempfile
import unittest
from pathlib import Path

from src.quantization.framework import (
    build_quantization_config,
    validate_quantization_evidence,
)
from src.input_validation.model_input import FAIL, NOT_RUN, PASS


class QuantizationFrameworkTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_awq_config_requires_supported_group_size(self):
        with self.assertRaises(ValueError):
            build_quantization_config(
                {
                    "plan_id": "bad",
                    "method": "awq",
                    "bits": 4,
                    "weight_bits": 4,
                    "activation_bits": 16,
                    "group_size": 48,
                    "backend_package": "autoawq",
                },
                {"path": "calibration", "version": "main", "size": 1, "sha256": "x"},
                self.root,
            )

    def test_framework_evidence_requires_three_methods(self):
        evidence_path = self.root / "evidence.json"
        plan = {
            "plan": {"method": "awq"},
            "calibration": {"sha256": "x"},
            "export_contract": {
                "manifest_file": "quantized_model_manifest.json",
                "config_file": "quantization_config.json",
            },
            "logging_contract": {"format": "jsonl", "required_fields": [1, 2, 3, 4, 5]},
        }
        evidence_path.write_text(
            json.dumps(
                {
                    "plans": [plan, {**plan, "plan": {"method": "gptq"}}, {**plan, "plan": {"method": "smoothquant"}}],
                    "backend_packages": {"torch": "2", "transformers": "5"},
                    "status": "framework_ready_not_quantized",
                }
            ),
            encoding="utf-8",
        )
        report = validate_quantization_evidence(evidence_path)
        self.assertEqual([check.status for check in report.checks], [PASS, PASS, PASS, PASS, NOT_RUN])
        self.assertFalse(report.complete)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            build_quantization_config(
                {
                    "plan_id": "bad",
                    "method": "int4-custom",
                    "bits": 4,
                    "weight_bits": 4,
                    "activation_bits": 16,
                    "group_size": 128,
                    "backend_package": "custom",
                },
                {},
                self.root,
            )


if __name__ == "__main__":
    unittest.main()
