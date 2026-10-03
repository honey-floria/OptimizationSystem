import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.stage5_acceptance import validate_stage5
from src.input_validation.model_input import PASS


class Stage5AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_complete_nvidia_stage_passes(self):
        complete = self.write("complete.json", {"complete": True})
        benchmark = self.write(
            "benchmark.json",
            {"benchmark_config": {"measured_runs": 5}, "cases": [{}, {}]},
        )
        environment = self.write(
            "environment.json",
            {
                "checks": [
                    {"name": name, "status": "pass"}
                    for name in (
                        "nvidia_environment_file_created",
                        "python_torch_transformers_versions_pinned",
                        "inference_framework_version_pinned",
                        "dependency_lock_saved",
                        "hardware_and_driver_recorded",
                    )
                ]
                + [{"name": "ascend_environment_file_created", "status": "not_run"}]
            },
        )
        deliverable = self.root / "deliverable.md"
        deliverable.write_text("ready", encoding="utf-8")

        report = validate_stage5(
            complete,
            complete,
            complete,
            benchmark,
            complete,
            environment,
            complete,
            complete,
            {"test": deliverable},
        )

        self.assertTrue(report.complete)
        self.assertEqual([check.status for check in report.checks], [PASS] * 5)
