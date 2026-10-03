import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.hardware_input import validate_hardware_inputs
from src.input_validation.model_input import NOT_RUN, PASS


class HardwareInputValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_manifest(self, target):
        is_nvidia = target == "nvidia"
        payload = {
            "schema_version": 1,
            "target": target,
            "captured_at": "2026-01-01T00:00:00+00:00",
            "environment_id": f"test-{target}",
            "system": {"container_version": "test-container"},
            "accelerator": {
                "available": True,
                "count": 1,
                "devices": [
                    {"name": "T4" if is_nvidia else "Ascend 910B", "memory_mib": 16000}
                ],
            },
            "software": (
                {"cuda_version": "12.1", "driver_version": "535"}
                if is_nvidia
                else {
                    "cann_version": "8.0",
                    "driver_version": "24.1",
                    "firmware_version": "7.1",
                }
            ),
            "frameworks": (
                {"torch": "2.5", "transformers": "4.50"}
                if is_nvidia
                else {"torch_npu": "2.5"}
            ),
            "communication": {
                "mode": "single_gpu",
                "topology_recorded": True,
                "network_host": "test-host",
            },
        }
        path = self.root / f"{target}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_both_platform_manifests_pass_all_checks(self):
        report = validate_hardware_inputs(
            self.write_manifest("nvidia"), self.write_manifest("ascend")
        )

        self.assertTrue(report.complete)
        self.assertEqual([check.status for check in report.checks], [PASS] * 7)

    def test_nvidia_only_leaves_ascend_checks_not_run(self):
        report = validate_hardware_inputs(self.write_manifest("nvidia"))

        self.assertFalse(report.complete)
        self.assertEqual(report.checks[0].status, PASS)
        self.assertEqual(report.checks[1].status, PASS)
        self.assertEqual(report.checks[2].status, NOT_RUN)
        self.assertEqual(report.checks[3].status, NOT_RUN)


if __name__ == "__main__":
    unittest.main()
