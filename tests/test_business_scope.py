import json
import tempfile
import unittest
from pathlib import Path

from src.input_validation.business_scope import validate_business_scope
from src.input_validation.model_input import FAIL, PASS


class BusinessScopeValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary_directory.name) / "scope.json"
        self.scope = json.loads(
            Path("configs/business_scope.json").read_text(encoding="utf-8")
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def validate(self):
        self.config_path.write_text(json.dumps(self.scope), encoding="utf-8")
        return validate_business_scope(self.config_path)

    def test_approved_scope_passes_all_six_checks(self):
        report = self.validate()

        self.assertTrue(report.complete)
        self.assertEqual([check.status for check in report.checks], [PASS] * 6)

    def test_missing_human_review_conditions_fails(self):
        self.scope["scenarios"][1]["human_review"]["required_when"] = []
        report = self.validate()

        self.assertFalse(report.complete)
        self.assertEqual(report.checks[3].status, FAIL)

    def test_quality_drop_cannot_exceed_risk_limit(self):
        self.scope["scenarios"][1]["quality_gates"][
            "max_drop_from_fp16_percentage_points"
        ] = 1.0
        report = self.validate()

        self.assertEqual(report.checks[4].status, FAIL)

    def test_l3_cannot_allow_int4(self):
        self.scope["scenarios"][2]["compression_policy"][
            "allowed_quantization"
        ].append("INT4")
        report = self.validate()

        self.assertEqual(report.checks[1].status, FAIL)

    def test_scenario_ids_must_be_unique(self):
        self.scope["scenarios"][2]["scenario_id"] = self.scope["scenarios"][1][
            "scenario_id"
        ]
        report = self.validate()

        self.assertEqual(report.checks[0].status, FAIL)

    def test_planned_scenario_cannot_activate_without_dataset(self):
        self.scope["scenarios"][0]["status"] = "active"
        report = self.validate()

        self.assertEqual(report.checks[0].status, FAIL)


if __name__ == "__main__":
    unittest.main()
