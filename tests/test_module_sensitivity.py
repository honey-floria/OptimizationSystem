import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_module_sensitivity import _parse_args, _workflow_config
from src.quantization.layer_sensitivity_setup import (
    canonical_json_sha256,
    sha256_file,
    write_json,
)
from src.quantization.module_sensitivity import (
    ATTENTION_MODULES,
    MLP_MODULES,
    discover_module_inventory,
    inspect_kv_cache_capability,
    inspect_model_contract,
    prepare_contract,
    validate_config,
    validate_upstream_contract,
    weights_are_tied,
)
from src.quantization.module_sensitivity_experiments import (
    audit_combined_all_int8_equivalence,
    build_experiment_plans,
    effective_weight_assignment,
    pending_plan_ids,
    prepare_experiment_matrix,
    validate_experiment_plans,
    validate_plan_isolation,
)
from src.quantization.module_sensitivity_final import (
    _validate_dataset_against_freeze,
    select_advancement,
)


class _Weight:
    def __init__(self, shape, pointer):
        self.shape = shape
        self.dtype = "float16"
        self._pointer = pointer

    def numel(self):
        total = 1
        for value in self.shape:
            total *= value
        return total

    def data_ptr(self):
        return self._pointer


class _Module:
    def __init__(self, shape=(4, 4), pointer=1):
        self.weight = _Weight(shape, pointer)


class _Block:
    def __init__(self, pointer):
        self.self_attn = SimpleNamespace(
            q_proj=_Module(pointer=pointer + 1),
            k_proj=_Module(pointer=pointer + 2),
            v_proj=_Module(pointer=pointer + 3),
            o_proj=_Module(pointer=pointer + 4),
        )
        self.mlp = SimpleNamespace(
            gate_proj=_Module(pointer=pointer + 5),
            up_proj=_Module(pointer=pointer + 6),
            down_proj=_Module(pointer=pointer + 7),
        )
        self.input_layernorm = _Module(shape=(4,), pointer=pointer + 8)
        self.post_attention_layernorm = _Module(shape=(4,), pointer=pointer + 9)


class _FakeModel:
    def __init__(self, block_count=2, tied=False):
        self.model = SimpleNamespace(
            layers=[_Block(index * 20) for index in range(block_count)],
            norm=_Module(shape=(4,), pointer=999),
            embed_tokens=_Module(shape=(8, 4), pointer=1000),
        )
        output_pointer = 1000 if tied else 1001
        self.lm_head = _Module(shape=(8, 4), pointer=output_pointer)
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, use_cache=None, past_key_values=None):
        return input_ids, use_cache, past_key_values

    def named_modules(self, remove_duplicate=True):
        rows = [
            ("", self),
            ("model.norm", self.model.norm),
            ("model.embed_tokens", self.model.embed_tokens),
            ("lm_head", self.lm_head),
        ]
        for index, block in enumerate(self.model.layers):
            prefix = f"model.layers.{index}"
            rows.extend(
                [
                    (f"{prefix}.input_layernorm", block.input_layernorm),
                    (f"{prefix}.post_attention_layernorm", block.post_attention_layernorm),
                ]
            )
        return iter(rows)


class _CacheUtils:
    class QuantizedCache:
        pass

    class QuantizedCacheConfig:
        pass


class _Transformers:
    __version__ = "test"
    cache_utils = _CacheUtils


def _base_config(block_count=2):
    return {
        "schema_version": 1,
        "stage": "7.2.1",
        "experiment_name": "test-module-sensitivity",
        "paths": {
            "merged_manifest": "merged.json",
            "input_freeze": "freeze.json",
            "layer_matrix_manifest": "matrix.json",
            "all_int4_plan": "plan.json",
            "all_int4_result": "result.json",
            "all_int4_report": "report.json",
            "all_int8_error": "all_int8_error.json",
            "dataset_dir": "dataset",
            "baseline_config": "baseline.json",
            "quality_config": "quality.json",
            "output_root": "out",
            "experiment_output_root": "experiments",
        },
        "expected_identity": {
            "adapter_weights_sha256": "adapter-weights",
            "adapter_tree_sha256": "adapter-tree",
            "merged_tree_sha256": "merged-tree",
            "tokenizer_tree_sha256": "tokenizer-tree",
            "pilot_indices_sha256": "pilot-indices",
            "calibration_subset_sha256": "calibration-subset",
            "evaluation_dev_subset_sha256": "dev-subset",
            "high_risk_subset_sha256": "risk-subset",
            "long_context_subset_sha256": "long-subset",
            "block_path": "model.layers",
            "block_count": block_count,
        },
        "evaluation_contract": {
            "variant": "structured_json",
            "pilot_size": 80,
            "pilot_seed": 1,
            "dev_size": 883,
            "calibration_size": 1024,
            "high_risk_size": 91,
            "long_context_fraction": 0.1,
            "tolerance": 0.0001,
            "generation": {"max_new_tokens": 256, "do_sample": False, "num_beams": 1},
            "scorer": "module:score",
        },
        "module_groups": {
            "attention": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
            "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            "block_norms": ["input_layernorm", "post_attention_layernorm"],
        },
        "precision": {
            "linear_int4": {
                "weight_bits": 4,
                "activation_bits": 16,
                "group_size": 32,
                "axis": "input_dimension",
                "scheme": "symmetric_groupwise",
            },
            "linear_int8": {
                "weight_bits": 8,
                "activation_bits": 16,
                "group_size": None,
                "axis": "output_channel",
                "scheme": "symmetric_per_output_channel",
            },
            "embedding": {
                "allowed": ["fp16", "int8", "int4"],
                "int4": {"group_size": 32, "axis": "embedding_dimension", "scheme": "symmetric_groupwise", "zero_point": 0},
                "int8": {"group_size": None, "axis": "embedding_row", "scheme": "symmetric_per_row", "zero_point": 0},
            },
            "lm_head": {
                "allowed": ["fp16", "int8", "int4"],
                "int4": {"group_size": 32, "axis": "input_dimension", "scheme": "symmetric_groupwise", "zero_point": 0},
                "int8": {"group_size": None, "axis": "output_channel", "scheme": "symmetric_per_output_channel", "zero_point": 0},
            },
            "norm": {
                "allowed": ["fp16", "int8", "int4"],
                "int4": {"group_size": None, "axis": None, "scheme": "symmetric_per_tensor", "zero_point": 0},
                "int8": {"group_size": None, "axis": None, "scheme": "symmetric_per_tensor", "zero_point": 0},
            },
            "shared_weight_policy": "joint_plan_only",
        },
        "weight_backend": {"deployment_performance_claims_allowed": False},
        "kv_cache": {
            "requested_precisions": ["fp16", "int8", "int4"],
            "cache_tensors": ["key", "value"],
            "phases": ["prefill", "decode"],
            "quantize_at": "cache_write",
            "dequantize_at": "attention_read",
            "scale_mode": "backend_native_dynamic",
            "granularity": "record_from_runtime_backend",
            "cache_layout": "record_from_runtime_backend",
            "weight_fake_quant_allowed": False,
            "sensitivity_qdq_hook_allowed": True,
            "real_backend": {
                "runtime_arguments": {
                    "int8": {
                        "cache_implementation": "quantized",
                        "cache_config": {"backend": "hqq", "nbits": 8, "axis_key": 1, "axis_value": 1},
                    },
                    "int4": {
                        "cache_implementation": "quantized",
                        "cache_config": {"backend": "hqq", "nbits": 4, "axis_key": 1, "axis_value": 1},
                    },
                },
                "supports_real_int8": False,
                "supports_real_int4": False,
                "performance_claims_allowed_before_runtime_probe": False,
            },
        },
        "runtime_inspection": {
            "device_map": "cpu",
            "dtype": "float16",
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "pilot_evaluation": {
            "dataset_role": "evaluation_dev",
            "variant": "structured_json",
            "pilot_size": 80,
            "pilot_seed": 1,
            "batch_size": 1,
            "max_new_tokens": 256,
            "device": "cuda:0",
            "dtype": "float16",
        },
    }


def _write_upstream(root, config):
    evaluation = config["evaluation_contract"]
    expected = config["expected_identity"]
    write_json(
        root / "merged.json",
        {
            "status": "merged",
            "merged_model": {"tree_sha256": "merged-tree", "path": "/server/model"},
            "adapter": {"tree_sha256": "adapter-tree"},
            "transformer_blocks": {"path": "model.layers", "count": expected["block_count"]},
        },
    )
    write_json(
        root / "freeze.json",
        {
            "status": "inputs_frozen",
            "frozen_adapter_record": {"adapter_weights_sha256": "adapter-weights"},
            "identities": {"tokenizer": {"tree_sha256": "tokenizer-tree"}},
            "subset_hashes": {
                "calibration": "calibration-subset",
                "evaluation_dev": "dev-subset",
                "contextual_high_risk": "risk-subset",
                "long_context": "long-subset",
            },
            "evaluation_contract": evaluation,
        },
    )
    write_json(
        root / "matrix.json",
        {
            "status": "matrix_prepared",
            "block_path": "model.layers",
            "block_count": expected["block_count"],
        },
    )
    plan = {
        "schema_version": 1,
        "stage": "7.1.2",
        "plan": {
            "plan_id": "all_int4",
            "kind": "all_int4",
            "runnable": True,
            "default_precision": "int4",
            "overrides": {},
        },
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    write_json(root / "plan.json", plan)
    write_json(
        root / "result.json",
        {
            "status": "completed",
            "plan": plan["plan"],
            "sample_indices_sha256": "pilot-indices",
            "predictions_sha256": "predictions",
            "metrics": {"total": 80, "correct": 36},
        },
    )
    write_json(
        root / "report.json",
        {
            "dataset": {"sample": {"indices_sha256": "pilot-indices"}},
            "variants": [
                {
                    "variant": "structured_json",
                    "predictions_sha256": "predictions",
                    "metrics": {"total": 80, "correct": 36},
                }
            ],
        },
    )
    write_json(
        root / "all_int8_error.json",
        {"plan_id": "all_int8", "block_count": config["expected_identity"]["block_count"], "modules": []},
    )


class ModuleSensitivityContractTest(unittest.TestCase):
    def test_discovers_complete_attention_mlp_norm_and_special_modules(self):
        model = _FakeModel(block_count=2, tied=True)
        inventory = discover_module_inventory(model, _base_config()["module_groups"])
        self.assertEqual(inventory["block_path"], "model.layers")
        self.assertEqual(inventory["counts"]["attention"], 8)
        self.assertEqual(inventory["counts"]["mlp"], 6)
        self.assertEqual(inventory["counts"]["norm"], 5)
        self.assertTrue(inventory["embedding_lm_head_tied"])
        self.assertEqual(inventory["shared_weight_policy"], "joint_plan_only")

    def test_missing_projection_fails_instead_of_silent_fallback(self):
        model = _FakeModel()
        del model.model.layers[1].self_attn.q_proj
        with self.assertRaisesRegex(ValueError, "Required module path is missing"):
            discover_module_inventory(model, _base_config()["module_groups"])

    def test_tied_weight_detection_uses_shared_storage(self):
        model = _FakeModel(tied=True)
        self.assertTrue(weights_are_tied(model.get_input_embeddings(), model.get_output_embeddings()))
        model = _FakeModel(tied=False)
        self.assertFalse(weights_are_tied(model.get_input_embeddings(), model.get_output_embeddings()))

    def test_kv_cache_is_sensitivity_only_until_real_runtime_probe(self):
        config = _base_config()
        audit = inspect_kv_cache_capability(
            _FakeModel(), config["kv_cache"], _Transformers
        )
        self.assertEqual(audit["status"], "sensitivity_only")
        self.assertTrue(audit["transformers_quantized_cache_api"]["QuantizedCache"])
        self.assertFalse(audit["deployment_performance_claims_allowed"])
        self.assertFalse(audit["runtime_probe_completed"])

    def test_config_rejects_kv_cache_weight_fake_quant(self):
        config = _base_config()
        config["kv_cache"]["weight_fake_quant_allowed"] = True
        with self.assertRaisesRegex(ValueError, "weight fake-quant"):
            validate_config(config)

    def test_prepares_frozen_contract_from_7_1_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _base_config()
            _write_upstream(root, config)
            upstream = validate_upstream_contract(root, config)
            self.assertEqual(upstream["status"], "pass")
            self.assertEqual(upstream["all_int4_reference"]["predictions_sha256"], "predictions")
            contract = prepare_contract(root, config)
            self.assertEqual(contract["status"], "awaiting_model_inspection")
            self.assertTrue((root / "out" / "contract.json").is_file())

    def test_model_inspection_writes_pass_validation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _base_config()
            _write_upstream(root, config)
            result = inspect_model_contract(
                root,
                config,
                model=_FakeModel(block_count=2),
                transformers_module=_Transformers,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["attention_module_count"], 8)
            self.assertEqual(result["mlp_module_count"], 6)
            self.assertTrue((root / "out" / "module_inventory.json").is_file())
            self.assertTrue((root / "out" / "kv_cache_capability.json").is_file())

    def test_cli_separates_contract_freeze_from_model_inspection(self):
        self.assertEqual(_parse_args(["prepare"]).command, "prepare")
        args = _parse_args(["inspect-model", "--model-path", "/server/model"])
        self.assertEqual(args.command, "inspect-model")
        self.assertEqual(args.model_path, Path("/server/model"))

    def test_builds_complete_tied_module_and_special_plan_matrix(self):
        plans = build_experiment_plans(28, True)
        self.assertEqual(len(plans), 69)
        self.assertEqual(sum(bool(plan["runnable"]) for plan in plans), 68)
        validate_experiment_plans(plans, 28, True)
        validate_plan_isolation(plans, 28, True)
        kinds = {plan["kind"] for plan in plans}
        self.assertIn("shared_embedding_lm_head_quantized", kinds)
        self.assertNotIn("embedding_quantized", kinds)

    def test_untied_matrix_uses_independent_embedding_and_lm_head_plans(self):
        plans = build_experiment_plans(2, False)
        self.assertEqual(len(plans), 19)
        validate_plan_isolation(plans, 2, False)
        kinds = {plan["kind"] for plan in plans}
        self.assertIn("embedding_quantized", kinds)
        self.assertIn("lm_head_quantized", kinds)
        self.assertNotIn("shared_embedding_lm_head_quantized", kinds)

    def test_layer_module_assignment_changes_only_the_target_family(self):
        plans = build_experiment_plans(2, False)
        baseline = effective_weight_assignment(plans[0], 2, False)
        attention = next(plan for plan in plans if plan["plan_id"] == "layer_01_attn_int8")
        assignment = effective_weight_assignment(attention, 2, False)
        changed = {key for key in baseline if baseline[key] != assignment[key]}
        self.assertEqual(
            changed,
            {
                "model.layers.1.self_attn.q_proj",
                "model.layers.1.self_attn.k_proj",
                "model.layers.1.self_attn.v_proj",
                "model.layers.1.self_attn.o_proj",
            },
        )

    def test_candidate_mixed_plan_changes_only_declared_families(self):
        candidates = [
            {
                "plan_id": "candidate_l02_all_l01_mlp_int8",
                "int8_attention_layers": [2],
                "int8_mlp_layers": [1, 2],
                "selection_basis": "test",
            }
        ]
        plans = build_experiment_plans(3, False, candidates)
        validate_experiment_plans(plans, 3, False)
        validate_plan_isolation(plans, 3, False)
        candidate = next(plan for plan in plans if plan["kind"] == "candidate_mixed_int8")
        baseline = effective_weight_assignment(plans[0], 3, False)
        assignment = effective_weight_assignment(candidate, 3, False)
        changed = {key for key in baseline if baseline[key] != assignment[key]}
        self.assertEqual(len(changed), len(ATTENTION_MODULES) + 2 * len(MLP_MODULES))
        self.assertTrue(
            all(
                assignment[f"model.layers.2.{name}"] == "int8"
                for name in ATTENTION_MODULES
            )
        )
        self.assertTrue(
            all(
                assignment[f"model.layers.1.{name}"] == "int8"
                for name in MLP_MODULES
            )
        )
        self.assertTrue(
            all(
                assignment[f"model.layers.2.{name}"] == "int8"
                for name in MLP_MODULES
            )
        )

    def test_candidate_combines_layer_and_special_module_precisions(self):
        candidates = [{
            "plan_id": "candidate_l01_mlp_lm_head_embedding_int8",
            "int8_attention_layers": [],
            "int8_mlp_layers": [1],
            "embedding_precision": "int8",
            "lm_head_precision": "int8",
        }]
        plans = build_experiment_plans(2, False, candidates)
        validate_experiment_plans(plans, 2, False)
        validate_plan_isolation(plans, 2, False)
        candidate = next(plan for plan in plans if plan["plan_id"] == candidates[0]["plan_id"])
        assignment = effective_weight_assignment(candidate, 2, False)
        self.assertEqual(assignment["embedding"], "int8")
        self.assertEqual(assignment["lm_head"], "int8")
        self.assertTrue(
            all(assignment[f"model.layers.1.{name}"] == "int8" for name in MLP_MODULES)
        )

    def test_combination_workflow_uses_isolated_outputs_and_two_candidates(self):
        config = {
            "paths": {"experiment_output_root": "pilot"},
            "candidate_recovery_plans": [],
            "final_evaluation": {
                "output_root": "final",
                "advancement_output": "final/advancement.json",
                "required_special_pilot_plans": ["embedding_int8"],
                "require_complete_pilot_matrix": True,
                "recovery_candidate_scope": "explicit_pilot_shortlist",
                "recovery_candidate_plans": ["layer_10_mlp_int8"],
                "determinism_sentinel_plans": ["embedding_int8"],
                "advancement_rules": {"recovery_min_net_correct_gain": 1},
            },
            "combination_evaluation": {
                "pilot_output_root": "combo/pilot",
                "output_root": "combo/final",
                "advancement_output": "combo/final/advancement.json",
                "pilot_min_net_correct_gain": -80,
                "determinism_sentinel_plans": ["candidate_a", "candidate_b"],
                "plans": [{"plan_id": "candidate_a"}, {"plan_id": "candidate_b"}],
            },
        }
        combined = _workflow_config(config, "combination")
        self.assertEqual(combined["paths"]["experiment_output_root"], "combo/pilot")
        self.assertEqual(combined["final_evaluation"]["output_root"], "combo/final")
        self.assertEqual(
            combined["final_evaluation"]["recovery_candidate_plans"],
            ["candidate_a", "candidate_b"],
        )
        self.assertEqual(config["paths"]["experiment_output_root"], "pilot")

    def test_prepares_experiment_matrix_only_after_real_inspection_validation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _base_config(block_count=2)
            _write_upstream(root, config)
            contract = prepare_contract(root, config)
            inventory = discover_module_inventory(
                _FakeModel(block_count=2), config["module_groups"]
            )
            write_json(root / "out" / "module_inventory.json", inventory)
            write_json(root / "out" / "kv_cache_capability.json", {"status": "sensitivity_only"})
            write_json(
                root / "out" / "validation.json",
                {
                    "status": "pass",
                    "contract_sha256": contract["contract_sha256"],
                    "module_inventory_sha256": sha256_file(root / "out" / "module_inventory.json"),
                },
            )
            manifest = prepare_experiment_matrix(root, config)
            self.assertEqual(manifest["status"], "matrix_prepared")
            self.assertEqual(manifest["plan_count"], 19)
            self.assertTrue((root / "experiments" / "layer_00_attn_int8" / "plan.json").is_file())

    def test_pending_treats_unsupported_kv_cache_as_terminal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _base_config(block_count=1)
            output = root / "experiments"
            plans = build_experiment_plans(1, False)
            write_json(output / "experiment_manifest.json", {"status": "matrix_prepared", "plans": plans})
            write_json(output / "kv_cache_int8" / "result.json", {"status": "unsupported"})
            pending = pending_plan_ids(root, config)
            self.assertNotIn("kv_cache_int8", pending)
            self.assertIn("kv_cache_int4", pending)

    def test_cli_exposes_experiment_matrix_and_plan_execution(self):
        self.assertEqual(_parse_args(["prepare-experiments"]).command, "prepare-experiments")
        self.assertEqual(_parse_args(["pending"]).command, "pending")
        args = _parse_args(["run-plan", "--plan-id", "attn_all_int8"])
        self.assertEqual(args.command, "run-plan")
        self.assertEqual(args.plan_id, "attn_all_int8")
        probe = _parse_args(["probe-kv-cache", "--precision", "int4"])
        self.assertEqual(probe.command, "probe-kv-cache")
        self.assertEqual(probe.precision, "int4")

    def test_cli_exposes_advancement_final_suites_and_finalize(self):
        self.assertEqual(_parse_args(["select-advancement"]).command, "select-advancement")
        self.assertEqual(_parse_args(["prepare-final"]).command, "prepare-final")
        pending = _parse_args(["pending-final", "--suite", "high_risk"])
        self.assertEqual(pending.suite, "high_risk")
        run = _parse_args(
            ["run-final-plan", "--suite", "full_dev", "--plan-id", "embedding_int8"]
        )
        self.assertEqual(run.plan_id, "embedding_int8")
        self.assertEqual(_parse_args(["finalize"]).command, "finalize")
        combination = _parse_args(["--workflow", "combination", "prepare-experiments"])
        self.assertEqual(combination.workflow, "combination")

    def test_advancement_is_frozen_from_pilot_results(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _base_config()
            config["final_evaluation"] = {
                "output_root": "final",
                "advancement_output": "final/advancement.json",
                "reference_baselines": {"full_dev": "full", "high_risk": "risk"},
                "required_special_pilot_plans": [
                    "embedding_int8", "kv_cache_fp16_ref", "kv_cache_int8"
                ],
                "recovery_candidate_plans": ["layer_01_attn_int8"],
                "determinism_sentinel_plans": [],
                "advancement_rules": {
                    "recovery_min_net_correct_gain": 1,
                    "special_min_net_correct_change": 0,
                    "require_positive_storage_savings": True,
                    "kv_cache_require_low_bit_residency": True,
                },
                "suites": {
                    "full_dev": {}, "high_risk": {}, "determinism": {}
                },
                "bootstrap": {"iterations": 10, "confidence": 0.95, "seed": 1},
                "quality_gates": {},
            }
            results = {
                "layer_01_attn_int8": {
                    "status": "completed",
                    "plan": {"kind": "layer_attention_int8"},
                    "transitions_vs_all_int4": {"net_correct_gain": 2},
                },
                "embedding_int8": {
                    "status": "completed",
                    "plan": {"kind": "embedding_quantized"},
                    "transitions_vs_all_int4": {"net_correct_change": 0},
                    "special_storage": {"savings_bytes": 128},
                },
                "kv_cache_fp16_ref": {
                    "status": "completed",
                    "plan": {"kind": "kv_cache_reference"},
                    "transitions_vs_all_int4": {"net_correct_change": 0},
                    "kv_cache_benchmark": {"cache_tensor_bytes": 1024},
                },
                "kv_cache_int8": {
                    "status": "unsupported",
                    "plan": {"kind": "kv_cache_quantized"},
                    "reason": "backend unavailable",
                },
            }
            for plan_id, result in results.items():
                write_json(root / "experiments" / plan_id / "result.json", result)
            frozen = select_advancement(root, config)
            self.assertEqual(frozen["selected_recovery_plans"], ["layer_01_attn_int8"])
            self.assertEqual(frozen["selected_special_plans"], ["embedding_int8"])
            self.assertIn("kv_cache_int8", frozen["excluded_plans"])
            self.assertEqual(
                frozen["selection_sha256"],
                canonical_json_sha256(
                    {key: value for key, value in frozen.items() if key != "selection_sha256"}
                ),
            )

    def test_advancement_rejects_an_incomplete_pilot_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _base_config()
            config["final_evaluation"] = {
                "output_root": "final",
                "advancement_output": "final/advancement.json",
                "reference_baselines": {"full_dev": "full", "high_risk": "risk"},
                "required_special_pilot_plans": [],
                "recovery_candidate_plans": [],
                "determinism_sentinel_plans": [],
                "require_complete_pilot_matrix": True,
                "advancement_rules": {},
                "suites": {"full_dev": {}, "high_risk": {}, "determinism": {}},
                "bootstrap": {"iterations": 10, "confidence": 0.95, "seed": 1},
                "quality_gates": {},
            }
            write_json(
                root / "experiments" / "experiment_manifest.json",
                {
                    "status": "matrix_prepared",
                    "plans": [
                        {"plan_id": "layer_00_attn_int8", "runnable": True}
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "must be complete"):
                select_advancement(root, config)

    def test_final_dataset_validation_rejects_reordered_or_changed_rows(self):
        rows = [{"qa": {"question": "first"}}, {"qa": {"question": "second"}}]
        from src.quantization.layer_sensitivity_setup import build_sample_entries

        frozen = build_sample_entries(rows)
        expected_hash = canonical_json_sha256(frozen)
        _validate_dataset_against_freeze(
            rows, frozen_entries=frozen, expected_hash=expected_hash, suite="full_dev"
        )
        with self.assertRaisesRegex(ValueError, "frozen sample list"):
            _validate_dataset_against_freeze(
                list(reversed(rows)),
                frozen_entries=frozen,
                expected_hash=expected_hash,
                suite="full_dev",
            )

    def test_combined_attention_mlp_audit_matches_layer_all_int8(self):
        metric = {
            "bits": 8,
            "group_size": None,
            "numel": 4,
            "mse": 0.1,
            "nmse": 0.2,
            "cosine_similarity": 0.9,
            "max_abs_error": 0.3,
            "saturation_rate": 0.1,
        }
        module_report = {
            "modules": [
                {
                    **metric,
                    "layer_index": 0,
                    "module": "model.layers.0.self_attn.q_proj",
                    "family": "attention",
                }
            ]
        }
        layer_report = {
            "plan_id": "all_int8",
            "modules": [
                {**metric, "layer_index": 0, "module": "self_attn.q_proj"}
            ],
        }
        result = audit_combined_all_int8_equivalence(module_report, layer_report)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["matched_module_count"], 1)


if __name__ == "__main__":
    unittest.main()
