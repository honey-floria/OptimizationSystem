import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.model_input import FAIL, NOT_RUN, PASS, validate_model_input


class ModelInputValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model_path = self.root / "model"
        self.model_path.mkdir()
        self.weight_bytes = b"test model weights"
        (self.model_path / "model.safetensors").write_bytes(self.weight_bytes)
        (self.model_path / "config.json").write_text("{}", encoding="utf-8")
        (self.model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.model_path / "LICENSE").write_text("test license", encoding="utf-8")
        self.manifest_path = self.root / "model_manifest.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_manifest(self, **updates):
        digest = hashlib.sha256(self.weight_bytes).hexdigest()
        manifest = {
            "model_id": "test/model",
            "model_path": "model",
            "version": "1.0.0",
            "commit": "abc123",
            "weight_files": ["model.safetensors"],
            "config_file": "config.json",
            "tokenizer_files": ["tokenizer.json"],
            "expected_sha256": {"model.safetensors": digest},
            "license": {
                "name": "test",
                "file": "LICENSE",
                "internal_use_approved": True,
                "usage_scope": "internal evaluation",
            },
            "runtime": {
                "framework": "transformers",
                "dtype": "bfloat16",
            },
        }
        manifest.update(updates)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_static_validation_records_four_passes_and_two_not_run(self):
        self.write_manifest()
        report = validate_model_input(self.manifest_path)

        self.assertTrue(report.validation_passed)
        self.assertFalse(report.complete)
        self.assertEqual([check.status for check in report.checks[:4]], [PASS] * 4)
        self.assertEqual([check.status for check in report.checks[4:]], [NOT_RUN] * 2)
        self.assertEqual(
            report.weight_sha256["model.safetensors"],
            hashlib.sha256(self.weight_bytes).hexdigest(),
        )

    def test_hash_mismatch_fails_identity_check(self):
        self.write_manifest(expected_sha256={"model.safetensors": "0" * 64})
        report = validate_model_input(self.manifest_path)

        self.assertFalse(report.validation_passed)
        self.assertEqual(report.checks[3].status, FAIL)

    def test_missing_tokenizer_fails_config_check(self):
        self.write_manifest(tokenizer_files=["missing.json"])
        report = validate_model_input(self.manifest_path, runtime_check=True)

        self.assertFalse(report.validation_passed)
        self.assertEqual(report.checks[1].status, FAIL)
        self.assertEqual([check.status for check in report.checks[4:]], [NOT_RUN] * 2)

    def test_unknown_expected_hash_file_fails_identity_check(self):
        self.write_manifest(expected_sha256={"unknown.safetensors": "0" * 64})
        report = validate_model_input(self.manifest_path)

        self.assertEqual(report.checks[3].status, FAIL)


if __name__ == "__main__":
    unittest.main()
