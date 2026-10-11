import unittest

from src.data.finqa_assets import _enrich_regression_rows


class FakeDataset(list):
    def map(self, function):
        return FakeDataset([{**row, **function(row)} for row in self])


class FinQAAssetsTest(unittest.TestCase):
    def test_enriches_verified_rows_with_finqa_context(self):
        regression = FakeDataset([{"question": "What changed?", "answer": "10"}])
        source = FakeDataset(
            [
                {
                    "pre_text": ["before"],
                    "post_text": ["after"],
                    "table": [["year", "value"], ["2020", "10"]],
                    "qa": {"question": " what  changed? ", "answer": "9"},
                }
            ]
        )

        enriched = _enrich_regression_rows(regression, source)

        self.assertEqual(enriched[0]["pre_text"], ["before"])
        self.assertEqual(enriched[0]["table"][1], ["2020", "10"])
        self.assertEqual(enriched[0]["qa"]["answer"], "10")

    def test_rejects_verified_rows_without_source_context(self):
        with self.assertRaisesRegex(ValueError, "missing FinQA source context"):
            _enrich_regression_rows(
                FakeDataset([{"question": "missing", "answer": "10"}]),
                FakeDataset([]),
            )


if __name__ == "__main__":
    unittest.main()
