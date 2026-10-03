import json
import tempfile
import unittest
from pathlib import Path

from src.environment_lock.manager import validate_environment_lock
from src.input_validation.model_input import NOT_RUN, PASS


class EnvironmentLockTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.freeze_path = self.root / "requirements.freeze.txt"
        self.freeze_path.write_text(
            "torch==2.11.0\ntransformers==5.18.0\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_json(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def nvidia_lock(self):
        import hashlib

        return {
            "target": "nvidia",
            "status": "locked",
            "profile_id": "test-nvidia",
            "runtime": {"python": "3.13.15"},
            "hardware": {
                "devices": [
                    {
                        "name": "T4",
                        "memory_mib": 15360,
                        "driver_version": "580.82.07",
                    }
                ]
            },
            "software": {"cuda_version": "13.0"},
            "frameworks": {
                "torch": "2.11.0",
                "transformers": "5.18.0",
            },
            "inference": {
                "framework": "transformers",
                "version": "5.18.0",
            },
            "dependencies": {
                "sha256": hashlib.sha256(
                    self.freeze_path.read_bytes()
                ).hexdigest()
            },
        }

    def test_nvidia_lock_passes_five_available_checks(self):
        nvidia_path = self.write_json("nvidia.json", self.nvidia_lock())
        ascend_path = self.write_json(
            "ascend.json",
            {"target": "ascend", "status": "pending_real_hardware"},
        )

        report = validate_environment_lock(
            nvidia_path, self.freeze_path, ascend_path
        )

        self.assertEqual(
            [check.status for check in report.checks],
            [PASS, NOT_RUN, PASS, NOT_RUN, PASS, NOT_RUN, PASS, PASS],
        )
        self.assertFalse(report.complete)

    def test_built_container_and_ascend_complete_all_checks(self):
        nvidia_path = self.write_json("nvidia.json", self.nvidia_lock())
        ascend_path = self.write_json(
            "ascend.json",
            {
                "target": "ascend",
                "status": "locked",
                "profile_id": "test-ascend",
                "software": {"cann": "8.0"},
            },
        )
        container_path = self.write_json(
            "container.json",
            {
                "build_status": "passed",
                "image": "test/image:1",
                "image_digest": "sha256:abc",
            },
        )

        report = validate_environment_lock(
            nvidia_path, self.freeze_path, ascend_path, container_path
        )

        self.assertTrue(report.complete)
        self.assertEqual([check.status for check in report.checks], [PASS] * 8)


if __name__ == "__main__":
    unittest.main()
