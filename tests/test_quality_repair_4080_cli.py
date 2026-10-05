import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.run_quality_repair_4080 import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    _parse_args,
    _build_quantized_manifest,
    _resolve_model_path,
)


class QualityRepair4080CliTest(unittest.TestCase):
    def test_default_paths_follow_qlora_storage_contract(self):
        args = _parse_args([])
        self.assertEqual(args.model_path, DEFAULT_MODEL_PATH)
        self.assertEqual(args.dataset_dir, DEFAULT_DATASET_DIR)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)
        self.assertEqual(DEFAULT_MODEL_PATH, PROJECT_ROOT / "models" / "data" / "Qwen2.5-7B-Instruct")
        self.assertEqual(DEFAULT_DATASET_DIR, PROJECT_ROOT / "datasets")
        self.assertEqual(DEFAULT_OUTPUT_DIR, PROJECT_ROOT / "out" / "quality_repair_4080")

    def test_quantized_mode_does_not_inject_default_base_model(self):
        args = _parse_args(
            [
                "--quantization",
                "gptq",
                "--quantized-model-path",
                "models/data/Qwen2.5-7B-Instruct-GPTQ-Int4",
            ]
        )
        self.assertIsNone(args.model_path)

    def test_builds_quantized_manifest_with_absolute_model_path(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path = Path(temporary_dir) / "model"
            manifest = _build_quantized_manifest(
                "Qwen/Qwen2.5-7B-Instruct-AWQ", model_path, "awq"
            )
            self.assertEqual(manifest["format"], "awq")
            self.assertEqual(manifest["model_path"], str(model_path.resolve()))
            self.assertEqual(manifest["plan"]["weight_bits"], 4)

    def test_resolves_model_path_from_quantized_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path = Path(temporary_dir) / "model"
            args = argparse.Namespace(
                model_path=None,
                quantized_model_path=None,
            )
            resolved = _resolve_model_path(
                args, {"model_path": str(model_path)}
            )
            self.assertEqual(resolved, model_path.resolve())


if __name__ == "__main__":
    unittest.main()
