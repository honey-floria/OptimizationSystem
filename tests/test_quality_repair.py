import json
import unittest
from decimal import Decimal

from src.evaluation.quality_repair import (
    add_stable_table_ids,
    build_few_shot_suffix,
    materialize_cell_ids_operation,
    parse_validated_operation_output,
    parse_cell_ids_operation_output,
    parse_evidence_operation_output,
    extract_final_numeric,
    parse_structured_output,
    quality_repair_schema,
    repair_with_calculator,
    repair_with_evidence_operation,
    safe_calculate,
    validate_cell_ids_operation,
    validate_evidence_operation,
    question_operation_hint,
    numeric_cell_catalog,
    validate_question_operation,
    _select_evaluation_rows,
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

    def test_evidence_operation_uses_decimal_calculation(self):
        parsed = parse_evidence_operation_output(
            '{"evidence":[{"row":"shares","column":"2016","value": "303.1"}],'
            '"operands":[290.6,303.1],"operation":"subtract","unit":"million"}'
        )
        repaired = repair_with_evidence_operation(parsed)
        self.assertEqual(repaired["normalized_value"], "-12.5")
        self.assertTrue(repaired["calculator_used"])
        self.assertEqual(repaired["operands"], ["290.6", "303.1"])
        self.assertEqual(repaired["operation"], "subtract")

    def test_evidence_operation_percent_change_returns_fraction(self):
        parsed = parse_evidence_operation_output(
            '{"evidence":["new","old"],"operands":[125,100],'
            '"operation":"percent_change","unit":"percent"}'
        )
        repaired = repair_with_evidence_operation(parsed)
        self.assertEqual(repaired["normalized_value"], "0.25")

    def test_evidence_operation_validates_stable_cell_ids(self):
        row = {"table": [["metric", "303.1"], ["metric", "290.6"]]}
        enriched = add_stable_table_ids(row)
        self.assertIn("[cell_id=r0c1", enriched["table"][0][1])
        self.assertIn("column=303.1] 303.1", enriched["table"][0][1])
        parsed = parse_evidence_operation_output(
            '{"evidence":[{"cell_id":"r0c1","value":"303.1"},'
            '{"cell_id":"r1c1","value":"290.6"}],'
            '"operands":[290.6,303.1],"operation":"subtract","unit":"million"}'
        )
        self.assertIsNone(validate_evidence_operation(parsed, row))

    def test_evidence_operation_rejects_wrong_cell_value_and_arity(self):
        row = {"table": [["metric", "303.1"], ["metric", "290.6"]]}
        wrong_value = parse_evidence_operation_output(
            '{"evidence":[{"cell_id":"r0c1","value":"999"},'
            '{"cell_id":"r1c1","value":"290.6"}],'
            '"operands":[999,290.6],"operation":"subtract","unit":"million"}'
        )
        self.assertIn("does not match", validate_evidence_operation(wrong_value, row))
        wrong_arity = parse_evidence_operation_output(
            '{"evidence":[{"cell_id":"r0c1","value":"303.1"}],'
            '"operands":[303.1],"operation":"subtract","unit":"million"}'
        )
        self.assertIn("exactly two", validate_evidence_operation(wrong_arity, row))

    def test_cell_ids_operation_resolves_values_from_table(self):
        row = {"table": [["metric", "303.1"], ["metric", "290.6"]]}
        parsed = parse_cell_ids_operation_output(
            '{"cell_ids":["r1c1","r0c1"],"operation":"subtract",'
            '"unit":"million"}'
        )
        self.assertIsNone(validate_cell_ids_operation(parsed, row))
        materialized = materialize_cell_ids_operation(parsed, row)
        repaired = repair_with_evidence_operation(materialized)
        self.assertEqual(repaired["operands"], ["290.6", "303.1"])
        self.assertEqual(repaired["normalized_value"], "-12.5")

    def test_cell_ids_operation_rejects_non_numeric_cell(self):
        row = {"table": [["metric", "303.1"], ["label", "shares"]]}
        parsed = parse_cell_ids_operation_output(
            '{"cell_ids":["r0c1","r1c1"],"operation":"subtract",'
            '"unit":"million"}'
        )
        self.assertIn("not numeric", validate_cell_ids_operation(parsed, row))

    def test_validated_parser_skips_invalid_candidate(self):
        row = {"table": [["metric", "303.1"], ["metric", "290.6"]]}
        text = (
            '{"cell_ids":["r9c9","r0c1"],"operation":"subtract","unit":"million"}\n'
            '{"cell_ids":["r1c1","r0c1"],"operation":"subtract","unit":"million"}'
        )
        parsed, error = parse_validated_operation_output(
            text, row, "cell_ids_operation"
        )
        self.assertIsNone(error)
        self.assertEqual(parsed["cell_ids"], ["r1c1", "r0c1"])

    def test_question_routing_and_numeric_catalog(self):
        hint = question_operation_hint("what percentage increased from 2010 to 2011?")
        self.assertIn("percent_change", hint)
        self.assertIn("2010", hint)
        catalog = numeric_cell_catalog({"table": [["metric", "1.5"], ["label", "n/a"]]})
        self.assertIn("r0c1=1.5", catalog)
        self.assertNotIn("r1c1=n/a", catalog)

    def test_question_operation_validation_rejects_mismatched_family(self):
        self.assertIsNone(
            validate_question_operation(
                {"operation": "divide"},
                "what percentage of inventory was sold?",
            )
        )
        self.assertIn(
            "conflicts",
            validate_question_operation(
                {"operation": "add"},
                "what percentage of inventory was sold?",
            ),
        )

    def test_few_shot_uses_only_passed_examples(self):
        suffix = build_few_shot_suffix([{"question": "q", "answer": "1"}])
        self.assertIn("Example final answer: 1", suffix)
        self.assertNotIn("quality-dev", suffix)

    def test_schema_is_strict(self):
        schema = quality_repair_schema()
        self.assertEqual(schema["required"], ["evidence", "formula", "value", "unit"])
        self.assertFalse(schema["additionalProperties"])

    def test_pilot_sample_is_reproducible(self):
        dataset = [{"index": index} for index in range(883)]
        config = {"evaluation_size": 80, "evaluation_seed": 20261004}
        selected, indices, seed = _select_evaluation_rows(dataset, config)
        repeated, repeated_indices, repeated_seed = _select_evaluation_rows(dataset, config)
        self.assertEqual(len(selected), 80)
        self.assertEqual(seed, repeated_seed)
        self.assertEqual(indices, repeated_indices)
        self.assertEqual(selected, repeated)
        self.assertEqual([row["index"] for row in selected], indices)


if __name__ == "__main__":
    unittest.main()
