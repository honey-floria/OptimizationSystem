import unittest

from scripts.train_finqa_qlora import build_training_examples


class FinqaQloraDataTest(unittest.TestCase):
    def test_build_training_examples_uses_numeric_program_rows(self):
        rows = [
            {
                "table": [["metric", "2010", "2011"], ["sales", "100", "120"]],
                "qa": {
                    "question": "what was the change?",
                    "answer": "20",
                    "program": "subtract(120, 100)",
                    "gold_inds": ["sales", "2010", "2011"],
                },
            },
            {
                "table": [["metric", "2010"], ["sales", "100"]],
                "qa": {"question": "is it higher?", "answer": "yes"},
            },
        ]
        examples = build_training_examples(rows)
        self.assertEqual(len(examples), 1)
        self.assertIn('"value":20.0', examples[0]["target"])
        self.assertIn("subtract(120, 100)", examples[0]["target"])
        self.assertIn("r1c1=100", examples[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
