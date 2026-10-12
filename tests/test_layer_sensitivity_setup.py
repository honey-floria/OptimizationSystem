import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.prepare_layer_sensitivity import _parse_args
from src.quantization.layer_sensitivity_setup import (
    build_business_labels,
    canonical_json_sha256,
    classify_business_question,
    compare_pilot_reports,
    discover_transformer_blocks,
    freeze_evaluation_contract,
    hash_directory,
    long_context_indices_from_log,
    tokenizer_identity,
    validate_pilot_indices,
    validate_precision_contract,
)


class LayerSensitivitySetupTest(unittest.TestCase):
    def test_precision_contract_accepts_only_w4a16_g32_and_w8a16(self):
        contract = {
            "int4": {"weight_bits": 4, "activation_bits": 16, "group_size": 32},
            "int8": {"weight_bits": 8, "activation_bits": 16},
            "forbidden_direct_comparison": "awq_w4a16_vs_smoothquant_w8a8",
        }
        validate_precision_contract(contract)
        contract["int8"]["activation_bits"] = 8
        with self.assertRaisesRegex(ValueError, "W8A16"):
            validate_precision_contract(contract)

    def test_discovers_qwen_blocks_without_hard_coded_count(self):
        blocks = [object() for _ in range(7)]
        model = SimpleNamespace(model=SimpleNamespace(layers=blocks))
        path, discovered = discover_transformer_blocks(model)
        self.assertEqual(path, "model.layers")
        self.assertIs(discovered, blocks)
        self.assertEqual(len(discovered), 7)

    def test_discovery_rejects_unknown_architecture(self):
        with self.assertRaisesRegex(ValueError, "Unable to discover"):
            discover_transformer_blocks(SimpleNamespace())

    def test_directory_and_tokenizer_identities_are_stable(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "weights.bin").write_bytes(b"weights")
            first = hash_directory(root)
            second = hash_directory(root)
            self.assertEqual(first["tree_sha256"], second["tree_sha256"])
            tokenizer = tokenizer_identity(first)
            self.assertEqual(tokenizer["file_count"], 1)
            self.assertEqual(tokenizer["files"][0]["path"], "tokenizer.json")

    def test_validates_frozen_pilot_index_hash(self):
        indices = [1, 4, 9]
        expected = hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        validate_pilot_indices(indices, 3, expected)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_pilot_indices(indices, 3, "0" * 64)

    def test_business_taxonomy_is_multi_label_and_detects_years(self):
        taxonomy = {
            "categories": {
                "amount": ["revenue"],
                "rate_or_ratio": ["percentage"],
                "date_or_term": ["year"],
            }
        }
        labels = classify_business_question(
            "What percentage of revenue changed from 2019 to 2020?", taxonomy
        )
        self.assertEqual(labels, ["amount", "rate_or_ratio", "date_or_term"])
        records = build_business_labels(
            [{"qa": {"question": "What was revenue?", "answer": "1"}}], taxonomy
        )
        self.assertEqual(records[0]["labels"], ["amount"])
        self.assertEqual(len(records[0]["row_sha256"]), 64)

    def test_long_context_list_uses_first_primary_request_and_ceil(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            log_path = Path(temporary_dir) / "quality.jsonl"
            rows = [
                {"request_id": "quality-000001", "input_tokens": 10},
                {"request_id": "quality-000002", "input_tokens": 30},
                {"request_id": "quality-000002", "input_tokens": 999},
                {"request_id": "quality-000003", "input_tokens": 20},
            ]
            log_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(long_context_indices_from_log(log_path, 3, 0.34), [1, 2])

    def test_freezes_exact_prompt_generation_and_scorer_contract(self):
        expected = {
            "variant": "structured_json",
            "generation": {
                "max_new_tokens": 256,
                "do_sample": False,
                "num_beams": 1,
            },
            "tolerance": 0.0001,
            "scorer": "module:score",
        }
        quality = {
            "tolerance": 0.0001,
            "generation": expected["generation"],
            "variants": [
                {
                    "name": "structured_json",
                    "prompt_suffix": "Return one JSON object.",
                }
            ],
        }
        baseline = {"seed": 42, "deterministic": True, "max_input_tokens": 4096}
        frozen = freeze_evaluation_contract(quality, baseline, expected)
        self.assertEqual(frozen["variant"]["prompt_suffix"], "Return one JSON object.")
        self.assertEqual(frozen["generation"]["max_new_tokens"], 256)
        self.assertEqual(frozen["scorer"], "module:score")
        quality["generation"] = {**expected["generation"], "do_sample": True}
        with self.assertRaisesRegex(ValueError, "generation settings"):
            freeze_evaluation_contract(quality, baseline, expected)

    def test_pilot_comparison_requires_aligned_samples(self):
        report = {
            "dataset": {
                "sample": {"indices": [2, 5], "indices_sha256": "sample-hash"}
            },
            "variants": [
                {
                    "variant": "structured_json",
                    "metrics": {"correct": 1, "total": 2},
                }
            ],
        }
        reference = [
            {"source_index": 2, "correct": False, "raw_prediction": "1"},
            {"source_index": 5, "correct": True, "raw_prediction": "2"},
        ]
        candidate = [
            {"source_index": 2, "correct": True, "raw_prediction": "3"},
            {"source_index": 5, "correct": False, "raw_prediction": "4"},
        ]
        result = compare_pilot_reports(report, report, reference, candidate)
        self.assertEqual(result["transitions"]["recovered"], 1)
        self.assertEqual(result["transitions"]["regressed"], 1)
        self.assertEqual(result["changed_raw_predictions"], 2)

    def test_cli_has_separate_non_overwriting_steps(self):
        merge = _parse_args(["merge"])
        compare = _parse_args(
            [
                "compare-pilot",
                "--reference-report",
                "reference.json",
                "--candidate-report",
                "candidate.json",
            ]
        )
        self.assertEqual(merge.command, "merge")
        self.assertEqual(compare.command, "compare-pilot")

    def test_canonical_hash_ignores_dictionary_order(self):
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": 2}),
            canonical_json_sha256({"b": 2, "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
