import tempfile
import unittest
from pathlib import Path

from scripts.run_layer_sensitivity import _parse_args
from src.quantization.layer_sensitivity import (
    ERROR_METRICS_SCHEMA_VERSION,
    _stable_error_statistics,
    aggregate_error_metrics,
    apply_plan_with_callback,
    build_plan_matrix,
    pending_plan_ids,
    paired_prediction_transitions,
    plan_precision_vector,
    prepare_matrix,
    rank_completed_results,
    validate_plan_isolation,
    validate_plan_matrix,
    validate_quantization_error_report,
)
from src.quantization.layer_sensitivity_setup import write_json


class LayerSensitivityMatrixTest(unittest.TestCase):
    class _FakeScalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class _FakeTensor:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(self.values),)

        def reshape(self, *_shape):
            return self

        def to(self, dtype=None):
            return self

        def numel(self):
            return len(self.values)

        def __getitem__(self, item):
            values = self.values[item]
            return LayerSensitivityMatrixTest._FakeTensor(
                values if isinstance(values, list) else [values]
            )

        def __sub__(self, other):
            return LayerSensitivityMatrixTest._FakeTensor(
                left - right for left, right in zip(self.values, other.values)
            )

        def __mul__(self, other):
            return LayerSensitivityMatrixTest._FakeTensor(
                left * right for left, right in zip(self.values, other.values)
            )

        def abs(self):
            return LayerSensitivityMatrixTest._FakeTensor(abs(value) for value in self.values)

        def max(self):
            return LayerSensitivityMatrixTest._FakeScalar(max(self.values))

    class _FakeTorch:
        float64 = "float64"

        @staticmethod
        def sum(tensor, dtype=None):
            return LayerSensitivityMatrixTest._FakeScalar(sum(tensor.values))

    def test_builds_two_references_two_global_and_one_plan_per_layer(self):
        plans = build_plan_matrix(28)
        self.assertEqual(len(plans), 32)
        self.assertEqual(sum(plan["runnable"] for plan in plans), 30)
        self.assertEqual(plans[-1]["plan_id"], "layer_27_int8")
        validate_plan_matrix(plans, 28)
        validate_plan_isolation(plans, 28)

    def test_single_layer_plan_changes_only_its_target(self):
        plans = build_plan_matrix(4)
        plan = next(item for item in plans if item["plan_id"] == "layer_02_int8")
        self.assertEqual(
            plan_precision_vector(plan, 4), ["int4", "int4", "int8", "int4"]
        )
        calls = []
        apply_plan_with_callback(
            ["a", "b", "c", "d"],
            plan,
            lambda index, block, precision: calls.append((index, block, precision)),
        )
        self.assertEqual(calls[2], (2, "c", "int8"))
        self.assertEqual(sum(item[2] == "int8" for item in calls), 1)

    def test_rejects_missing_layer_plan(self):
        plans = build_plan_matrix(3)
        plans = [plan for plan in plans if plan["plan_id"] != "layer_01_int8"]
        with self.assertRaisesRegex(ValueError, "cover every"):
            validate_plan_matrix(plans, 3)

    def test_aggregates_errors_by_numel_and_uses_max_absolute_error(self):
        aggregate = aggregate_error_metrics(
            [
                {
                    "numel": 1,
                    "mse": 1.0,
                    "nmse": 2.0,
                    "cosine_similarity": 0.5,
                    "max_abs_error": 3.0,
                    "saturation_rate": 0.1,
                },
                {
                    "numel": 3,
                    "mse": 5.0,
                    "nmse": 6.0,
                    "cosine_similarity": 0.9,
                    "max_abs_error": 2.0,
                    "saturation_rate": 0.3,
                },
            ]
        )
        self.assertEqual(aggregate["numel"], 4)
        self.assertEqual(aggregate["mse"], 4.0)
        self.assertEqual(aggregate["nmse"], 5.0)
        self.assertEqual(aggregate["max_abs_error"], 3.0)

    def test_quantization_error_validation_rejects_invalid_cosine(self):
        metric = {
            "numel": 4,
            "mse": 0.1,
            "nmse": 0.2,
            "cosine_similarity": 0.9,
            "max_abs_error": 0.3,
            "saturation_rate": 0.1,
        }
        report = {
            "error_metrics_schema_version": ERROR_METRICS_SCHEMA_VERSION,
            "modules": [{**metric, "module": "q_proj"}],
            "layers": [{**metric, "layer_index": 0}],
            "aggregate": metric,
        }
        validate_quantization_error_report(report)
        report["modules"][0]["cosine_similarity"] = 1.0001
        with self.assertRaisesRegex(ValueError, "cosine similarity"):
            validate_quantization_error_report(report)

    def test_error_statistics_use_chunked_stable_accumulation(self):
        source = self._FakeTensor([1.0, 2.0, 3.0, 4.0])
        measured = self._FakeTensor([1.0, 2.0, 3.0, 3.0])
        metrics = _stable_error_statistics(
            source,
            measured,
            torch_module=self._FakeTorch,
            chunk_size=2,
        )
        self.assertAlmostEqual(metrics["source_power"], 7.5)
        self.assertAlmostEqual(metrics["mse"], 0.25)
        self.assertAlmostEqual(metrics["max_abs_error"], 1.0)
        self.assertGreaterEqual(metrics["cosine_similarity"], -1.0)
        self.assertLessEqual(metrics["cosine_similarity"], 1.0)

    def test_quantization_error_validation_rejects_obsolete_schema(self):
        with self.assertRaisesRegex(ValueError, "obsolete"):
            validate_quantization_error_report(
                {"error_metrics_schema_version": 1, "modules": [], "layers": []}
            )

    def test_pending_skips_only_completed_results(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output = root / "out"
            plans = build_plan_matrix(2)
            write_json(output / "experiment_manifest.json", {"status": "matrix_prepared", "plans": plans})
            completed = output / "all_int4"
            completed.mkdir()
            write_json(completed / "result.json", {"status": "completed"})
            failed = output / "all_int8"
            failed.mkdir()
            write_json(failed / "result.json", {"status": "failed"})
            config = {"paths": {"output_root": "out"}}
            pending = pending_plan_ids(root, config)
            self.assertNotIn("all_int4", pending)
            self.assertIn("all_int8", pending)
            self.assertIn("layer_00_int8", pending)

    def test_prepare_matrix_validates_identities_and_registers_references(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_json(
                root / "merged.json",
                {
                    "status": "merged",
                    "merged_model": {"tree_sha256": "merged", "path": "/model"},
                    "adapter": {"tree_sha256": "adapter"},
                    "transformer_blocks": {"path": "model.layers", "count": 2},
                },
            )
            write_json(
                root / "freeze.json",
                {
                    "status": "inputs_frozen",
                    "frozen_adapter_record": {"adapter_weights_sha256": "weights"},
                },
            )
            common_report = {
                "dataset": {"sample": {"indices_sha256": "sample"}},
                "variants": [
                    {
                        "variant": "structured_json",
                        "metrics": {"total": 2, "correct": 1},
                        "predictions_sha256": "predictions",
                        "structured_output_rate": 1.0,
                    }
                ],
            }
            write_json(
                root / "fp.json",
                {
                    **common_report,
                    "model": {
                        "quantization": "none",
                        "adapter_path": None,
                        "load_in_4bit": False,
                    },
                },
            )
            write_json(
                root / "nf4.json",
                {
                    **common_report,
                    "model": {
                        "quantization": "none",
                        "adapter_path": "/adapter",
                        "load_in_4bit": True,
                    },
                },
            )
            config = {
                "paths": {
                    "merged_manifest": "merged.json",
                    "input_freeze": "freeze.json",
                    "output_root": "matrix",
                    "ref_fp_report": "fp.json",
                    "ref_nf4_report": "nf4.json",
                },
                "expected_identity": {
                    "merged_tree_sha256": "merged",
                    "adapter_tree_sha256": "adapter",
                    "adapter_weights_sha256": "weights",
                    "pilot_indices_sha256": "sample",
                    "block_path": "model.layers",
                    "block_count": 2,
                },
                "precision": {"int4": {}, "int8": {}},
                "backend": {"name": "symmetric_fake_quant"},
            }
            experiment = prepare_matrix(project_root=root, config=config)
            self.assertEqual(experiment["plan_count"], 6)
            self.assertEqual(experiment["runnable_plan_count"], 4)
            self.assertEqual(
                experiment["references"]["ref_nf4"]["normalized_runtime_quantization"],
                "bitsandbytes-nf4",
            )

    def test_ranking_excludes_failed_and_applies_stable_tie_break(self):
        baseline = {"metrics": {"correct": 10}, "weighted_nmse": 0.5}
        results = [
            {
                "status": "completed",
                "plan": {"kind": "layer_int8", "layer_index": 2},
                "metrics": {"correct": 11, "numeric_accuracy": 0.55},
                "transitions": {"recovered": 2, "regressed": 1},
                "weighted_nmse": 0.3,
            },
            {
                "status": "completed",
                "plan": {"kind": "layer_int8", "layer_index": 1},
                "metrics": {"correct": 11, "numeric_accuracy": 0.55},
                "transitions": {"recovered": 2, "regressed": 1},
                "weighted_nmse": 0.2,
            },
            {
                "status": "failed",
                "plan": {"kind": "layer_int8", "layer_index": 0},
                "metrics": {"correct": 99, "numeric_accuracy": 1.0},
            },
        ]
        ranked = rank_completed_results(results, baseline)
        self.assertEqual([item["plan"]["layer_index"] for item in ranked], [1, 2])
        self.assertAlmostEqual(ranked[0]["quantization_error_recovery"], 0.3)
        self.assertEqual([item["rank"] for item in ranked], [1, 2])

    def test_pairs_candidate_predictions_against_all_int4(self):
        baseline = [
            {"source_index": 1, "correct": False, "raw_prediction": "a"},
            {"source_index": 2, "correct": True, "raw_prediction": "b"},
            {"source_index": 3, "correct": True, "raw_prediction": "c"},
            {"source_index": 4, "correct": False, "raw_prediction": "d"},
        ]
        candidate = [
            {"source_index": 1, "correct": True, "raw_prediction": "A"},
            {"source_index": 2, "correct": False, "raw_prediction": "B"},
            {"source_index": 3, "correct": True, "raw_prediction": "c"},
            {"source_index": 4, "correct": False, "raw_prediction": "d"},
        ]
        transitions = paired_prediction_transitions(baseline, candidate)
        self.assertEqual(transitions["recovered"], 1)
        self.assertEqual(transitions["regressed"], 1)
        self.assertEqual(transitions["stable_correct"], 1)
        self.assertEqual(transitions["stable_wrong"], 1)
        self.assertEqual(transitions["changed_raw_predictions"], 2)

    def test_cli_exposes_prepare_pending_summary_and_single_plan_execution(self):
        self.assertEqual(_parse_args(["prepare"]).command, "prepare")
        self.assertEqual(_parse_args(["pending"]).command, "pending")
        self.assertEqual(_parse_args(["summarize"]).command, "summarize")
        self.assertEqual(_parse_args(["audit-errors"]).command, "audit-errors")
        args = _parse_args(["run-plan", "--plan-id", "layer_03_int8"])
        self.assertEqual(args.plan_id, "layer_03_int8")
        repair = _parse_args(["recompute-error", "--plan-id", "all_int4"])
        self.assertEqual(repair.command, "recompute-error")
        self.assertEqual(repair.plan_id, "all_int4")


if __name__ == "__main__":
    unittest.main()
