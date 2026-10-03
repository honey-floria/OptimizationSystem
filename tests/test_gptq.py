import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.model_input import NOT_RUN, PASS
from src.quantization.gptq import (
    REQUIRED_GROUP_SIZES,
    build_gptq_plans,
    collect_gptq_evidence,
    validate_gptq_evidence,
)


class GPTQTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = self.root / "gptq.json"
        self.calibration = self.root / "dataset.json"
        self.model = self.root / "model.json"
        self.config.write_text(json.dumps({
            "plan_id": "gptq-int4-w4a16", "method": "gptq", "bits": 4,
            "weight_bits": 4, "activation_bits": 16,
            "group_sizes": [32, 64, 128], "backend_package": "gptqmodel",
        }), encoding="utf-8")
        self.calibration.write_text(json.dumps({"assets": {"calibration": {
            "path": "calibration", "version": "main", "size": 1024,
            "sha256": "abc",
        }}}), encoding="utf-8")
        self.model.write_text(json.dumps({"model_path": "/tmp/model"}), encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_builds_awq_comparable_matrix(self):
        plans = build_gptq_plans(json.loads(self.config.read_text()))
        self.assertEqual([plan.group_size for plan in plans], list(REQUIRED_GROUP_SIZES))
        self.assertTrue(all(plan.bits == 4 for plan in plans))

    def test_rejects_incomplete_matrix(self):
        with self.assertRaises(ValueError):
            build_gptq_plans({"plan_id": "gptq", "method": "gptq", "bits": 4, "group_sizes": [128]})

    def test_missing_artifacts_are_not_run(self):
        output = self.root / "out"
        evidence = collect_gptq_evidence(self.config, self.calibration, self.model, output)
        evidence_path = output / "evidence.json"
        output.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        report = validate_gptq_evidence(evidence_path)
        self.assertEqual(report.checks[0].status, PASS)
        self.assertEqual(report.checks[2].status, NOT_RUN)
        self.assertFalse(report.complete)


if __name__ == "__main__":
    unittest.main()
