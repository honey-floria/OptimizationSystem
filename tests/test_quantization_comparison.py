import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.model_input import NOT_RUN, PASS
from src.quantization.comparison import (
    build_quantization_comparison,
    render_markdown,
    validate_quantization_comparison,
)


class QuantizationComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = build_quantization_comparison(
            cls.root,
            cls.root / "configs/quantization_comparison.json",
        )

    def test_builds_all_method_rows_without_faking_missing_results(self):
        self.assertEqual(
            self.report["coverage"]["methods"],
            ["awq", "fp16", "fp8", "gptq", "int8"],
        )
        self.assertEqual(self.report["coverage"]["quality_missing"], [])
        self.assertIn(
            "smoothquant-int8-w8a8",
            self.report["coverage"]["performance_missing"],
        )
        self.assertEqual(
            self.report["coverage"]["performance_comparable_to_fp16"], []
        )

    def test_marks_int8_structured_regression_as_high_risk(self):
        int8 = next(
            candidate for candidate in self.report["candidates"]
            if candidate["method"] == "int8"
        )
        self.assertIn(
            "structured_output_regression",
            int8["risk_assessment"]["reasons"],
        )
        self.assertFalse(int8["risk_assessment"]["l3_candidate"])

    def test_validation_passes_available_quality_and_risk_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "comparison_report.json"
            report_path.write_text(json.dumps(self.report), encoding="utf-8")
            validation = validate_quantization_comparison(report_path)
        statuses = {check.name: check.status for check in validation.checks}
        self.assertEqual(statuses["fp16_int8_int4_fp8_comparison_table"], PASS)
        self.assertEqual(statuses["quality_changes_recorded"], PASS)
        self.assertEqual(statuses["high_risk_unsuitable_plans_marked"], PASS)
        self.assertEqual(statuses["unit_token_cost_recorded"], NOT_RUN)

    def test_markdown_calls_out_incomparable_performance(self):
        markdown = render_markdown(self.report)
        self.assertIn("Performance deltas against FP16 are intentionally blank", markdown)


if __name__ == "__main__":
    unittest.main()
