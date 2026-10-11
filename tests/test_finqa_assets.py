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
        with self.assertRaisesRegex(ValueError, "no unique FinQA source context"):
            _enrich_regression_rows(
                FakeDataset([{"question": "missing", "answer": "10"}]),
                FakeDataset([]),
            )

    def test_matches_verified_question_with_rewritten_ratio_wording(self):
        regression = FakeDataset(
            [
                {
                    "question": "What decimal proportion of total purchase price is related to stock awards?",
                    "answer": "0.03",
                }
            ]
        )
        source = FakeDataset(
            [
                {
                    "pre_text": ["purchase context"],
                    "post_text": [],
                    "table": [["purchase price", "stock awards"]],
                    "qa": {
                        "question": "What percentage of total purchase price is related to stock awards?",
                        "answer": "3%",
                    },
                }
            ]
        )

        enriched = _enrich_regression_rows(regression, source)

        self.assertEqual(enriched[0]["pre_text"], ["purchase context"])
        self.assertEqual(enriched[0]["qa"]["answer"], "0.03")

    def test_matches_verified_proportion_to_finqa_portion_wording(self):
        regression = FakeDataset(
            [
                {
                    "question": "What decimal proportion of total purchase price is related to stock awards?",
                    "answer": "0.03",
                }
            ]
        )
        source = FakeDataset(
            [
                {
                    "pre_text": ["purchase context"],
                    "post_text": [],
                    "table": [["purchase price", "stock awards"]],
                    "qa": {
                        "question": "What portion of total purchase price is related to stock awards?",
                        "answer": "2.9%",
                    },
                }
            ]
        )

        enriched = _enrich_regression_rows(regression, source)

        self.assertEqual(enriched[0]["pre_text"], ["purchase context"])
        self.assertEqual(enriched[0]["qa"]["answer"], "0.03")

    def test_rejects_ambiguous_rewritten_question(self):
        regression = FakeDataset(
            [{"question": "What decimal proportion of revenue?", "answer": "0.5"}]
        )
        source = FakeDataset(
            [
                {"qa": {"question": "What percentage of revenue?"}},
                {"qa": {"question": "What ratio of revenue?"}},
            ]
        )

        with self.assertRaisesRegex(ValueError, "no unique FinQA source context"):
            _enrich_regression_rows(regression, source)


if __name__ == "__main__":
    unittest.main()
