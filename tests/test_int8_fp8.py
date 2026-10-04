import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.model_input import NOT_RUN, PASS
from src.quantization.int8_fp8 import (
    check_structured_json,
    collect_int8_fp8_evidence,
    validate_int8_fp8_evidence,
)


class Int8Fp8Test(unittest.TestCase):
    def test_structured_json_contract(self):
        self.assertTrue(check_structured_json('{"answer": 1, "unit": "%"}', ["answer", "unit"])["passed"])
        self.assertFalse(check_structured_json("not json", ["answer"])["passed"])
    def test_contract_and_capability_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            calibration = root / "calibration.json"
            config.write_text(json.dumps({
                "plan_id": "smoothquant-int8-w8a8",
                "weight_bits": 8,
                "activation_bits": 8,
            }), encoding="utf-8")
            calibration.write_text(json.dumps({"assets": {"calibration": {"sha256": "x"}}}), encoding="utf-8")
            output = root / "out"
            evidence = collect_int8_fp8_evidence(config, calibration, output)
            evidence_path = output / "evidence.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            report = validate_int8_fp8_evidence(evidence_path)
            self.assertEqual(report.checks[0].status, PASS)
            self.assertEqual(report.checks[-1].status, NOT_RUN)
            self.assertFalse(report.complete)


if __name__ == "__main__":
    unittest.main()
