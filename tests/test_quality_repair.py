import json
import unittest
from decimal import Decimal

from src.evaluation.quality_repair import (
    build_few_shot_suffix,
    extract_final_numeric,
    parse_structured_output,
    quality_repair_schema,
    repair_with_calculator,
    safe_calculate,
)


class QualityRepairTest(unittest.TestCase):
    def test_safe_calculator_supports_decimal_arithmetic(self):
        self.assertEqual(safe_calculate("290.6 - 303.1"), Decimal("-12.5"))
        self.assertEqual(safe_calculate("(62 / 62) * 100"), Decimal("100"))

    def test_safe_calculator_rejects_code(self):
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('echo bad')")

    def test_structured_output_is_parsed(self):
        text = json.dumps(
            {
                "evidence": ["2016 diluted shares", "2017 diluted shares"],
                "formula": "290.6 - 303.1",
                "value": -12.5,
                "unit": "million",
            }
        )
        parsed = parse_structured_output(text)
        self.assertEqual(parsed["normalized_value"], "-12.5")
        self.assertEqual(repair_with_calculator(parsed)["normalized_value"], "-12.5")

    def test_structured_output_supports_nested_evidence(self):
        parsed = parse_structured_output(
            '{"evidence":[["2016", "303.1"]], "formula":"290.6 - 303.1", "value":-12.5, "unit":"million"}'
        )
        self.assertEqual(parsed["normalized_value"], "-12.5")

    def test_final_answer_marker_beats_first_number(self):
        parsed = extract_final_numeric("2016 value 303.1; 2017 value 290.6. Final answer: -12.5 million")
        self.assertEqual(parsed["normalized_value"], "-12.5")

    def test_few_shot_uses_only_passed_examples(self):
        suffix = build_few_shot_suffix([{"question": "q", "answer": "1"}])
        self.assertIn("Example final answer: 1", suffix)
        self.assertNotIn("quality-dev", suffix)

    def test_schema_is_strict(self):
        schema = quality_repair_schema()
        self.assertEqual(schema["required"], ["evidence", "formula", "value", "unit"])
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
