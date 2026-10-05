import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.run_quality_repair_4080 import (
    _build_quantized_manifest,
    _resolve_model_path,
)


class QualityRepair4080CliTest(unittest.TestCase):
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
