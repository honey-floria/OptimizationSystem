import unittest

from src.evaluation.finqa_metrics import (
    classify_numeric_regression,
    evaluate_numeric_answers,
    parse_numeric_answer,
)


class FinQAMetricsTest(unittest.TestCase):
    def test_parses_currency_commas_and_percent(self):
        self.assertEqual(str(parse_numeric_answer("$1,234.50")), "1234.50")
        self.assertEqual(str(parse_numeric_answer("12.5%")), "0.125")

    def test_numeric_accuracy(self):
        result = evaluate_numeric_answers(
            ["$100.00", "12.5%", "unknown"],
            ["100", "0.125", "5"],
        )
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["parsed"], 2)

    def test_new_regression_requires_correct_baseline(self):
        result = classify_numeric_regression("100", "99", "100")
        self.assertTrue(result["new_regression"])
        self.assertEqual(result["critical_error_type"], "numeric_answer_mismatch")


if __name__ == "__main__":
    unittest.main()
