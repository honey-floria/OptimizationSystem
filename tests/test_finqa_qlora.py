import unittest

from pathlib import Path

from scripts.train_finqa_qlora import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    _parse_args,
    build_training_examples,
)


class FinqaQloraDataTest(unittest.TestCase):
    def test_default_paths_follow_qlora_storage_contract(self):
        args = _parse_args([])
        self.assertEqual(args.model_path, DEFAULT_BASE_MODEL_PATH)
        self.assertEqual(args.train_path, DEFAULT_TRAIN_PATH)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)
        self.assertEqual(args.artifact_dir, DEFAULT_ARTIFACT_DIR)
        self.assertEqual(
            DEFAULT_BASE_MODEL_PATH,
            PROJECT_ROOT / "models" / "data" / "Qwen2.5-7B-Instruct",
        )
        self.assertEqual(
            DEFAULT_ARTIFACT_DIR,
            PROJECT_ROOT / "models" / "data" / "finqa_qlora_7b_v2",
        )
        self.assertEqual(
            DEFAULT_TRAIN_PATH,
            PROJECT_ROOT / "datasets" / "raw_finqa" / "train.json",
        )
        self.assertEqual(DEFAULT_OUTPUT_DIR, PROJECT_ROOT / "out" / "finqa_qlora_7b_v2")
        self.assertIsInstance(args.model_path, Path)

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
        self.assertIn("Financial table:", examples[0]["prompt"])
        self.assertIn("sales | 100 | 120", examples[0]["prompt"])
        self.assertIn("Return exactly one JSON object", examples[0]["prompt"])

    def test_build_training_examples_compacts_mapping_evidence(self):
        rows = [
            {
                "table": [["metric", "2010"], ["sales", "100"]],
                "qa": {
                    "question": "what were sales?",
                    "answer": "100",
                    "program": "table_lookup(100)",
                    "gold_inds": {
                        "table_1": "sales in 2010 was 100 with a very long evidence string",
                        "text_2": "another long evidence string",
                    },
                },
            }
        ]
        target = build_training_examples(rows)[0]["target"]
        self.assertIn('"evidence":["table_1","text_2"]', target)
        self.assertNotIn("very long evidence string", target)

    def test_resume_checkpoint_argument_is_available(self):
        args = _parse_args(
            [
                "--resume-from-checkpoint",
                "models/data/finqa_qlora_7b_v2/checkpoint-100",
            ]
        )
        self.assertEqual(
            args.resume_from_checkpoint,
            Path("models/data/finqa_qlora_7b_v2/checkpoint-100"),
        )


if __name__ == "__main__":
    unittest.main()
