import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark.runner import (
    build_cases,
    percentile,
    validate_benchmark_report,
)
from src.input_validation.model_input import FAIL, PASS


class BenchmarkRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.report_path = Path(self.temporary_directory.name) / "report.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def complete_report(self):
        cases = []
        for batch_size in (1, 2):
            for input_tokens in (128, 512):
                for output_tokens in (16, 32):
                    sample = {
                        "ttft_ms": 10.0,
                        "time_per_output_token_ms": 2.0,
                        "end_to_end_latency_ms": 40.0,
                        "output_tokens_per_second": 100.0,
                        "requests_per_second": 5.0,
                    }
                    cases.append(
                        {
                            "batch_size": batch_size,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "samples": [sample, sample],
                            "latency_ms": {
                                "end_to_end": {
                                    "p50": 40.0,
                                    "p95": 40.0,
                                    "p99": 40.0,
                                }
                            },
                            "memory": {
                                "model_allocated_mib": 1000.0,
                                "peak_allocated_mib": 1100.0,
                                "kv_cache_mib": 10.0,
                                "estimated_kv_cache_mib": 10.0,
                            },
                        }
                    )
        return {
            "command": "python -m src.benchmark.runner run",
            "baseline_config": {"dtype": "float16"},
            "model_metadata": {"model_id": "test/model", "commit": "abc"},
            "environment": {
                "torch": "2.11",
                "transformers": "5.18",
                "cuda": "13.0",
                "gpu": "T4",
                "device": "cuda:0",
            },
            "cases": cases,
        }

    def validate(self, report):
        self.report_path.write_text(json.dumps(report), encoding="utf-8")
        return validate_benchmark_report(self.report_path)

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([10, 20, 30], 0.50), 20)
        self.assertEqual(percentile([10, 20], 0.95), 19.5)

    def test_build_cases_creates_cartesian_matrix(self):
        cases = build_cases(
            {
                "matrix": {
                    "batch_sizes": [1, 2],
                    "input_lengths": [128],
                    "output_lengths": [16, 32],
                }
            }
        )

        self.assertEqual(len(cases), 4)

    def test_complete_report_passes_eleven_checks(self):
        report = self.validate(self.complete_report())

        self.assertTrue(report.complete)
        self.assertEqual([check.status for check in report.checks], [PASS] * 11)

    def test_single_batch_size_fails_matrix_check(self):
        payload = self.complete_report()
        payload["cases"] = [
            case for case in payload["cases"] if case["batch_size"] == 1
        ]

        report = self.validate(payload)

        self.assertEqual(report.checks[8].status, FAIL)


if __name__ == "__main__":
    unittest.main()
