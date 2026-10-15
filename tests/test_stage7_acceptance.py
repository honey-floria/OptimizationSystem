import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.model_input import FAIL, PASS
from src.quantization.stage7_acceptance import validate_stage7


def _write(root: Path, relative: str, value: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(root: Path) -> None:
    preview = [{"plan": {"layer_index": index}} for index in range(28)]
    _write(root, "out/layer_sensitivity/matrix/matrix_summary.json", {"status": "complete", "pending": [], "ranking_preview": preview})
    _write(root, "out/layer_sensitivity/final_evaluation/validation.json", {"status": "pass"})
    _write(root, "out/layer_sensitivity/final_evaluation/ranking.json", {"status": "candidate_subset_final", "ranking": [{}]})
    _write(root, "out/module_sensitivity/final_evaluation/validation.json", {"status": "pass", "pending": {"full_dev": [], "high_risk": [], "determinism": []}})
    _write(root, "out/module_sensitivity/final_evaluation/ranking.json", {"status": "candidate_subset_final", "recovery_ranking": [{}], "special_loss_benefit_ranking": [{}]})
    _write(root, "out/financial_error_localization/validation.json", {"status": "pass"})
    _write(root, "out/financial_error_localization/error_statistics.json", {
        "status": "completed_with_declared_coverage_gaps",
        "degradation": [{"plan_id": "mixed", "net_correct_gain": -1, "business_net_change": {"amount_error": -1}}],
        "coverage": {"zero_errors_must_not_be_inferred_from_not_evaluable": True},
    })
    _write(root, "out/mixed_precision_search/validation.json", {"complete": True, "deployment_claims_allowed": True})
    candidate = {"plan_id": "selected", "selected": True, "quality_pass": True, "performance_pass": True, "benchmark": {"oom_case_count": 0}}
    _write(root, "out/mixed_precision_search/ranking.json", {"status": "completed", "deployment_claims_allowed": True, "recommendation": "selected", "candidates": [candidate]})
    _write(root, "out/mixed_precision_search/plans/selected/plan.json", {"runnable": True, "default_linear_precision": "int4", "overrides": {}, "special_modules": {"embedding": "bf16"}})
    _write(root, "out/mixed_precision_search/exports/selected/quantized_model_manifest.json", {"status": "completed", "backend_artifact_valid": True, "state_dict_bytes": 10, "state_dict_sha256": "a" * 64})
    for suite, total in (("full_dev", 883), ("high_risk", 91), ("long_context", 89)):
        _write(root, f"out/mixed_precision_search/evidence/selected/{suite}.json", {"status": "completed", "metrics": {"total": total, "correct": 1, "parse_rate": 1.0}, "predictions_sha256": "b" * 64})
    _write(root, "out/mixed_precision_search/evidence/selected/benchmark.json", {"benchmark_config": {"measured_runs": 2}, "cases": [{} for _ in range(16)]})
    script = root / "scripts/run_financial_error_localization.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# test")
    (root / "out/financial_error_localization/report.md").write_text("report")


class Stage7AcceptanceTest(unittest.TestCase):
    def test_complete_stage_passes(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            _fixture(root)
            report = validate_stage7(root)
            self.assertTrue(report.complete)
            self.assertEqual([check.status for check in report.checks], [PASS] * 5)
            self.assertEqual(report.selected_deployment_plan, "selected")
            self.assertTrue(all(report.deliverables.values()))

    def test_deployment_claim_must_be_enabled(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            _fixture(root)
            _write(root, "out/mixed_precision_search/validation.json", {"complete": True, "deployment_claims_allowed": False})
            report = validate_stage7(root)
            check = next(item for item in report.checks if item.name == "deployable_mixed_precision_configuration_generated")
            self.assertEqual(check.status, FAIL)
            self.assertFalse(report.complete)


if __name__ == "__main__":
    unittest.main()
