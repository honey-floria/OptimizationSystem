import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_layer_sensitivity import _parse_args
from src.quantization.layer_sensitivity_final import (
    _group_metrics,
    _protection_recommendation,
    _ranking_key,
    finalize_layer_sensitivity,
    paired_bootstrap_interval,
)
from src.quantization.layer_sensitivity_setup import (
    canonical_json_sha256,
    sha256_file,
    write_json,
    write_jsonl,
)


def _row(index, correct, *, parsed=True, structured=True):
    return {
        "source_index": index,
        "question": f"q{index}",
        "correct": correct,
        "parsed": parsed,
        "structured_output": structured,
        "raw_prediction": str(correct),
    }


def _write_prediction_run(directory, rows, *, suite=None, plan=None, role="evaluation_dev"):
    prediction_path = directory / "structured_json" / "predictions.jsonl"
    write_jsonl(prediction_path, rows)
    prediction_sha = sha256_file(prediction_path)
    dataset = {"role": role, "size": len(rows), "source_size": len(rows)}
    if suite == "determinism":
        dataset["sample"] = {"indices_sha256": "pilot"}
    report = {
        "dataset": dataset,
        "variants": [
            {
                "variant": "structured_json",
                "metrics": {
                    "total": len(rows),
                    "parsed": sum(bool(row["parsed"]) for row in rows),
                    "correct": sum(bool(row["correct"]) for row in rows),
                    "parse_rate": 1.0,
                    "numeric_accuracy": 0.5,
                },
                "structured_output_rate": 1.0,
                "format_retry_count": 0,
                "predictions_file": "structured_json/predictions.jsonl",
                "predictions_sha256": prediction_sha,
            }
        ],
    }
    write_json(directory / "comparison_report.json", report)
    if suite is not None:
        write_json(
            directory / "result.json",
            {
                "status": "completed",
                "suite": suite,
                "plan": plan,
                "predictions_sha256": prediction_sha,
                "comparison_report_sha256": sha256_file(
                    directory / "comparison_report.json"
                ),
            },
        )


class LayerSensitivityFinalTest(unittest.TestCase):
    def test_group_metrics_reports_counts_and_rates(self):
        baseline = [_row(0, False), _row(1, True), _row(2, False)]
        candidate = [
            _row(0, True),
            _row(1, False),
            _row(2, False, parsed=False, structured=False),
        ]
        metrics = _group_metrics(baseline, candidate, {0, 1, 2})
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["baseline_correct"], 1)
        self.assertEqual(metrics["candidate_correct"], 1)
        self.assertEqual(metrics["recovered"], 1)
        self.assertEqual(metrics["regressed"], 1)
        self.assertEqual(metrics["candidate_parsed"], 2)
        self.assertEqual(metrics["candidate_structured"], 2)

    def test_paired_bootstrap_is_zero_for_identical_predictions(self):
        rows = [_row(0, False), _row(1, True), _row(2, True)]
        interval = paired_bootstrap_interval(
            rows, rows, iterations=100, confidence=0.95, seed=7
        )
        self.assertEqual(interval["point_estimate"], 0.0)
        self.assertEqual(interval["lower"], 0.0)
        self.assertEqual(interval["upper"], 0.0)

    def test_ranking_key_uses_declared_quality_first_order(self):
        def candidate(layer, net, recovered, high, long, structured, error):
            return {
                "layer_index": layer,
                "full_dev": {"net_correct_gain": net, "recovered": recovered},
                "high_risk": {"recovered": high},
                "long_context": {"recovered": long},
                "structured_output_transitions": {"net": structured},
                "quantization_error_recovery": error,
            }

        rows = [
            candidate(2, 1, 1, 0, 0, 0, 1.0),
            candidate(1, 1, 2, 0, 0, 0, 0.0),
            candidate(0, 2, 0, 0, 0, 0, 0.0),
        ]
        self.assertEqual(
            [row["layer_index"] for row in sorted(rows, key=_ranking_key)],
            [0, 1, 2],
        )

    def test_protection_policy_filters_regressions(self):
        policy = {
            "max_layers": 2,
            "require_positive_full_dev_gain": True,
            "require_nonnegative_high_risk_gain": True,
            "require_nonnegative_long_context_gain": True,
        }
        ranking = [
            {
                "layer_index": 3,
                "full_dev": {"net_correct_gain": 2},
                "high_risk": {"net_correct_gain": -1},
                "long_context": {"net_correct_gain": 1},
            },
            {
                "layer_index": 4,
                "full_dev": {"net_correct_gain": 1},
                "high_risk": {"net_correct_gain": 0},
                "long_context": {"net_correct_gain": 0},
            },
            {
                "layer_index": 5,
                "full_dev": {"net_correct_gain": 1},
                "high_risk": {"net_correct_gain": 1},
                "long_context": {"net_correct_gain": 1},
            },
        ]
        result = _protection_recommendation(ranking, policy)
        self.assertEqual(result["recommended_protected_layers"], [4, 5])

    def test_cli_exposes_final_suite_commands(self):
        self.assertEqual(_parse_args(["prepare-final"]).command, "prepare-final")
        pending = _parse_args(["pending-final", "--suite", "full_dev"])
        self.assertEqual(pending.suite, "full_dev")
        run = _parse_args(
            ["run-final-plan", "--suite", "high_risk", "--plan-id", "all_int4"]
        )
        self.assertEqual(run.suite, "high_risk")
        self.assertEqual(run.plan_id, "all_int4")
        self.assertEqual(_parse_args(["finalize"]).command, "finalize")
        self.assertEqual(
            _parse_args(["audit-determinism"]).command, "audit-determinism"
        )

    def test_finalize_builds_complete_artifact_set(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            matrix = root / "matrix"
            setup = root / "setup"
            final = root / "final"
            global_plans = [
                {"plan_id": "all_int4", "kind": "all_int4", "runnable": True},
                {"plan_id": "all_int8", "kind": "all_int8", "runnable": True},
            ]
            layer_plans = [
                {
                    "plan_id": f"layer_{index:02d}_int8",
                    "kind": "layer_int8",
                    "runnable": True,
                    "layer_index": index,
                }
                for index in range(28)
            ]
            plans = [*global_plans, *layer_plans]
            write_json(
                matrix / "experiment_manifest.json",
                {"status": "matrix_prepared", "plans": plans},
            )
            write_json(
                matrix / "matrix_summary.json",
                {"all_int4": {"weighted_nmse": 1.0}},
            )
            for index, plan in enumerate(layer_plans):
                write_json(
                    matrix / plan["plan_id"] / "result.json",
                    {"weighted_nmse": 0.9 - index / 1000},
                )

            full_baseline = [_row(index, index % 2 == 0) for index in range(883)]
            risk_baseline = [_row(index, index % 2 == 0) for index in range(91)]
            pilot = full_baseline[:80]
            for plan_id in ("all_int4", "layer_22_int8"):
                _write_prediction_run(matrix / plan_id, pilot)

            for suite, baseline, role in (
                ("full_dev", full_baseline, "evaluation_dev"),
                ("high_risk", risk_baseline, "high_risk_regression"),
            ):
                for plan in plans:
                    rows = [dict(row) for row in baseline]
                    if plan["plan_id"] == "layer_10_int8":
                        rows[1]["correct"] = True
                        rows[1]["raw_prediction"] = "recovered"
                    _write_prediction_run(
                        final / suite / plan["plan_id"],
                        rows,
                        suite=suite,
                        plan=plan,
                        role=role,
                    )
            for plan_id in ("all_int4", "layer_22_int8"):
                plan = next(row for row in plans if row["plan_id"] == plan_id)
                _write_prediction_run(
                    final / "determinism" / plan_id,
                    pilot,
                    suite="determinism",
                    plan=plan,
                )

            labels = [
                {
                    "source_index": index,
                    "labels": ["amount", "rate_or_ratio", "date_or_term"],
                }
                for index in range(883)
            ]
            write_jsonl(setup / "business_labels.jsonl", labels)
            write_json(
                setup / "frozen_subsets.json",
                {"long_context": [{"index": index} for index in range(89)]},
            )
            candidate_plan_ids = [
                "layer_10_int8",
                "layer_22_int8",
                "layer_23_int8",
            ]
            suites = {
                "full_dev": {
                    "dataset_role": "evaluation_dev",
                    "expected_size": 883,
                    "frozen_subset": "evaluation_dev",
                    "plans": ["all_int4", "all_int8", *candidate_plan_ids],
                    "plan_ids": ["all_int4", "all_int8", *candidate_plan_ids],
                },
                "high_risk": {
                    "dataset_role": "high_risk_regression",
                    "expected_size": 91,
                    "frozen_subset": "contextual_high_risk",
                    "plans": ["all_int4", *candidate_plan_ids],
                    "plan_ids": ["all_int4", *candidate_plan_ids],
                },
                "determinism": {
                    "dataset_role": "evaluation_dev",
                    "pilot_size": 80,
                    "pilot_seed": 1,
                    "expected_indices_sha256": "pilot",
                    "plans": ["all_int4", "layer_22_int8"],
                    "plan_ids": ["all_int4", "layer_22_int8"],
                },
            }
            bootstrap = {"iterations": 20, "confidence": 0.95, "seed": 9}
            tie_break = [
                "net_correct_gain_desc",
                "recovered_count_desc",
                "high_risk_recovered_desc",
                "long_context_recovered_desc",
                "structured_output_recovery_desc",
                "quantization_error_recovery_desc",
                "layer_index_asc",
            ]
            policy = {
                "max_layers": 3,
                "require_positive_full_dev_gain": True,
                "require_nonnegative_high_risk_gain": True,
                "require_nonnegative_long_context_gain": True,
            }
            analysis_scope = {
                "mode": "candidate_subset",
                "candidate_layer_indices": [10, 22, 23],
                "candidate_plan_ids": candidate_plan_ids,
                "candidate_layer_count": 3,
                "total_model_layers": 28,
                "complete_layer_coverage": False,
                "full_layer_ranking_claim_allowed": False,
                "selection_basis": "pilot top three",
            }
            contract = {
                "analysis_scope": analysis_scope,
                "suites": suites,
                "runtime_contract": {
                    "evaluation": {},
                    "precision": {},
                    "backend": {},
                },
                "bootstrap": bootstrap,
                "ranking_tie_break": tie_break,
                "protection_policy": policy,
            }
            write_json(
                final / "experiment_manifest.json",
                {
                    "status": "prepared",
                    **contract,
                    "contract_sha256": canonical_json_sha256(contract),
                },
            )
            config = {
                "paths": {"output_root": "matrix", "setup_dir": "setup"},
                "evaluation": {},
                "precision": {},
                "backend": {},
                "final_evaluation": {
                    "output_root": "final",
                    "analysis_scope": {
                        "mode": "candidate_subset",
                        "candidate_layer_indices": [10, 22, 23],
                        "selection_basis": "pilot top three",
                        "full_layer_ranking_claim_allowed": False,
                    },
                    "suites": {
                        name: {key: value for key, value in suite.items() if key != "plan_ids"}
                        for name, suite in suites.items()
                    },
                    "bootstrap": bootstrap,
                    "ranking_tie_break": tie_break,
                    "protection_policy": policy,
                },
            }
            result = finalize_layer_sensitivity(root, config)
            self.assertEqual(result["validation"]["status"], "pass")
            self.assertEqual(result["validation"]["ranked_layers"], 3)
            self.assertFalse(
                result["validation"]["analysis_scope"][
                    "full_layer_ranking_claim_allowed"
                ]
            )
            self.assertEqual(result["ranking"]["status"], "candidate_subset_final")
            self.assertEqual(result["ranking"]["ranking"][0]["layer_index"], 10)
            self.assertEqual(
                result["ranking"]["recommendation"]["recommended_protected_layers"],
                [10],
            )
            for name in (
                "experiment_manifest.json",
                "layer_metrics.jsonl",
                "sample_transitions.jsonl",
                "ranking.json",
                "ranking.csv",
                "report.md",
                "validation.json",
            ):
                self.assertTrue((final / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
