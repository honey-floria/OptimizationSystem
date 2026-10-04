import unittest
from pathlib import Path

from src.input_validation.model_input import PASS
from src.quantization.stage6_acceptance import validate_stage6


class Stage6AcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = validate_stage6(cls.root)

    def test_stage6_evidence_is_complete(self):
        self.assertTrue(self.report.complete)
        self.assertEqual([check.status for check in self.report.checks], [PASS] * 5)
        self.assertTrue(all(self.report.deliverables.values()))

    def test_technical_stability_does_not_claim_production_readiness(self):
        self.assertGreaterEqual(len(self.report.technical_deployment_candidates), 1)
        self.assertIn("暂不部署", self.report.production_recommendation)


if __name__ == "__main__":
    unittest.main()
