import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.model_input import NOT_RUN, PASS
from src.quantization.int8_fp8 import (
    build_smoothquant_plan,
    check_structured_json,
    collect_int8_fp8_evidence,
    evaluate_int8_outputs,
    validate_int8_evaluation,
    validate_int8_fp8_evidence,
)


class Int8Fp8Test(unittest.TestCase):
    def test_builds_w8a8_smoothquant_plan(self):
        plan = build_smoothquant_plan({
            "plan_id": "smoothquant-int8-w8a8",
            "method": "smoothquant",
            "weight_bits": 8,
            "activation_bits": 8,
            "alpha": 0.5,
            "quantization_granularity": "per_channel_weight_per_tensor_activation",
        })
        self.assertEqual(plan.weight_bits, 8)
        self.assertEqual(plan.activation_bits, 8)
        self.assertEqual(plan.alpha, 0.5)

    def test_rejects_non_w8a8_plan(self):
        with self.assertRaises(ValueError):
            build_smoothquant_plan({
                "plan_id": "invalid",
                "method": "smoothquant",
                "weight_bits": 4,
                "activation_bits": 8,
                "alpha": 0.5,
                "quantization_granularity": "groupwise",
            })

    def test_structured_json_contract(self):
        self.assertTrue(check_structured_json('{"answer": 1, "unit": "%"}', ["answer", "unit"])["passed"])
        self.assertFalse(check_structured_json("not json", ["answer"])["passed"])

    def test_structured_json_must_be_object(self):
        self.assertFalse(check_structured_json("[]", ["answer"])["passed"])

    def test_contract_and_capability_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            calibration = root / "calibration.json"
            config.write_text(json.dumps({
                "plan_id": "smoothquant-int8-w8a8",
                "method": "smoothquant",
                "weight_bits": 8,
                "activation_bits": 8,
                "alpha": 0.5,
                "quantization_granularity": "per_channel_weight_per_tensor_activation",
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

    def test_legacy_dtype_only_fp8_evidence_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text(json.dumps({
                "schema_version": 1,
                "calibration_asset": {"sha256": "x"},
                "plans": [{
                    "method": "smoothquant",
                    "weight_bits": 8,
                    "activation_bits": 8,
                    "export_contract": {
                        "config_file": "quantization_config.json",
                        "manifest_file": "quantized_model_manifest.json",
                        "event_log_file": "quantization_events.jsonl",
                    },
                }],
                "hardware_capabilities": {"nvidia_fp8": True},
            }), encoding="utf-8")
            report = validate_int8_fp8_evidence(evidence_path)
            statuses = {check.name: check.status for check in report.checks}
            self.assertEqual(statuses["nvidia_fp8_capability"], NOT_RUN)

    def test_evaluation_records_all_three_quality_views(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = evaluate_int8_outputs(
                predictions=["10", "bad"],
                references=["10", "20"],
                long_context_results=[{"passed": True, "baseline_passed": True}],
                structured_results=[
                    {"passed": True, "baseline_passed": True},
                    {"passed": False, "baseline_passed": False},
                ],
                baseline_numeric={"numeric_accuracy": 0.25},
            )
            evaluation_path = Path(directory) / "evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
            report = validate_int8_evaluation(evaluation_path)
            self.assertTrue(report.complete)
            self.assertEqual(evaluation["numeric_comparison"]["int8_numeric_accuracy"], 0.5)
            self.assertEqual(evaluation["long_context"]["pass_rate"], 1.0)
            self.assertEqual(evaluation["structured_output"]["pass_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
