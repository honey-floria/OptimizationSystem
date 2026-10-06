import json
import unittest
from decimal import Decimal

from src.evaluation.finqa_metrics import evaluate_numeric_answers
from src.evaluation.quality_repair import (
    add_stable_table_ids,
    build_few_shot_suffix,
    materialize_cell_ids_operation,
    materialize_steps_operation,
    parse_validated_operation_output,
    parse_cell_ids_operation_output,
    parse_steps_operation_output,
    parse_candidate_cell_ids_operation_output,
    parse_evidence_operation_output,
    extract_final_numeric,
    parse_structured_output,
    quality_repair_schema,
    repair_with_calculator,
    repair_with_evidence_operation,
    safe_calculate,
    validate_cell_ids_operation,
    validate_steps_operation,
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

    def test_structured_output_accepts_ratio_times_unit(self):
        parsed = parse_structured_output(
            '{"evidence":["sales", "operating income"],'
            '"formula":"9.4", "value":9.4, "unit":"times"}'
        )
        self.assertEqual(parsed["normalized_value"], "9.4")

    def test_structured_output_accepts_mmboe_unit(self):
        parsed = parse_structured_output(
            '{"evidence":["production"],"formula":"12.5",'
            '"value":12.5,"unit":"mmboe"}'
        )
        self.assertEqual(parsed["normalized_value"], "12.5")

    def test_structured_output_normalizes_percent_unit_for_scoring(self):
        parsed = parse_structured_output(
            '{"evidence":["operating profit"],"formula":"divide(20,1063)",'
            '"value":1.9,"unit":"percent"}'
        )
        self.assertEqual(parsed["normalized_value"], "1.9%")
        metrics = evaluate_numeric_answers(
            [parsed["normalized_value"]],
            ["1.9%"],
        )
        self.assertEqual(metrics["correct"], 1)

    def test_structured_output_repairs_stray_quote_after_numeric_value(self):
        parsed = parse_structured_output(
            '{"evidence":["change"],"formula":"-2",'
            '"value":-2", "unit":"million"}'
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["normalized_value"], "-2")

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

    def test_deterministic_calculator_supports_average_and_interest(self):
        average = repair_with_evidence_operation(
            {
                "operands": ["121.1", "97.5", "132.4"],
                "operation": "average",
                "unit": "dollars",
            }
        )
        self.assertEqual(average["normalized_value"], "117.0")
        interest = repair_with_evidence_operation(
            {
                "operands": ["750", "0.01375"],
                "operation": "multiply",
                "unit": "million",
            }
        )
        self.assertEqual(interest["normalized_value"], "10.31250")

    def test_ratio_operation_accepts_times_unit(self):
        row = {"table": [["sales", "940"], ["operating income", "100"]]}
        parsed = parse_evidence_operation_output(
            '{"evidence":[{"cell_id":"r0c1","value":"940"},'
            '{"cell_id":"r1c1","value":"100"}],'
            '"operands":[940,100],"operation":"ratio","unit":"times"}'
        )
        self.assertIsNone(validate_evidence_operation(parsed, row))
        repaired = repair_with_evidence_operation(parsed)
        self.assertEqual(repaired["normalized_value"], "9.4")

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

    def test_steps_operation_resolves_prior_results(self):
        row = {"table": [["metric", "100", "120"], ["base", "10", "20"]]}
        parsed = parse_steps_operation_output(
            '{"steps":[{"operation":"subtract","operands":["r0c2","r0c1"]},'
            '{"operation":"divide","operands":["step0","r1c1"]}],'
            '"unit":"percent"}'
        )
        self.assertIsNotNone(parsed)
        self.assertIsNone(validate_steps_operation(parsed, row))
        repaired = materialize_steps_operation(parsed, row)
        self.assertEqual(repaired["normalized_value"], "2")
        self.assertTrue(repaired["calculator_used"])

    def test_steps_operation_accepts_numeric_constants_and_change_chain(self):
        row = {"table": [["metric", "100", "120"]]}
        parsed = parse_steps_operation_output(
            '{"steps":[{"operation":"subtract","operands":["r0c2","r0c1"]},'
            '{"operation":"divide","operands":["step0","r0c1"]}],'
            '"unit":"percent"}'
        )
        self.assertIsNone(validate_steps_operation(parsed, row))
        self.assertIsNone(
            validate_question_operation(
                parsed,
                "what was the percentage change from 2010 to 2011?",
            )
        )
        constant_steps = parse_steps_operation_output(
            '{"steps":[{"operation":"multiply","operands":["r0c2","0.05"]}],'
            '"unit":"million"}'
        )
        self.assertIsNone(validate_steps_operation(constant_steps, row))

    def test_candidate_operation_selects_valid_deterministic_candidate(self):
        row = {"table": [["metric", "2010", "2011"], ["sales", "100", "120"]]}
        parsed, error = parse_candidate_cell_ids_operation_output(
            '{"candidates":['
            '{"cell_ids":["r1c0","r9c9"],"operation":"subtract","unit":"million"},'
            '{"cell_ids":["r1c2","r1c1"],"operation":"subtract","unit":"million"}'
            ']}',
            row,
            "what was the change from 2010 to 2011?",
        )
        self.assertIsNone(error)
        self.assertEqual(parsed["cell_ids"], ["r1c2", "r1c1"])
        self.assertEqual(parsed["normalized_value"], "20")

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

    def test_question_routing_prioritizes_change_over_entity_average(self):
        self.assertIn(
            "percent_change",
            question_operation_hint("what was the percentage change from 2012 to 2013?"),
        )
        self.assertIn(
            "subtract",
            question_operation_hint("by how much did the average price increase?"),
        )
        self.assertIsNone(
            validate_question_operation(
                {"operation": "percent_change"},
                "what was the percentage change from 2012 to 2013?",
            )
        )

    def test_question_routing_handles_growth_and_interest(self):
        self.assertIn(
            "percent_change",
            question_operation_hint("what was the growth rate in sales from 2012 to 2013?"),
        )
        self.assertIn(
            "multiply",
            question_operation_hint("what is the annual interest expense for the note?"),
        )
        self.assertIsNone(
            validate_question_operation(
                {"operation": "multiply"},
                "what is the annual interest expense for the note?",
            )
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
