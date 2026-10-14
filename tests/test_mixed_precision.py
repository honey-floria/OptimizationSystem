import json
from pathlib import Path

import pytest

from src.quantization.mixed_precision import (
    build_candidates,
    candidate_by_id,
    finalize_search,
    pending_evidence,
    precision_for_module,
    prepare_search,
    record_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/mixed_precision_search.json").read_text())


def test_candidate_matrix_covers_required_74_variants():
    candidates = build_candidates(CONFIG)
    ids = {row.plan_id for row in candidates}
    assert {"all_linear_int4", "all_linear_int8", "mixed_l10_mlp_int8", "mixed_l10_mlp_bf16"} <= ids
    assert candidate_by_id(CONFIG, "mixed_l10_mlp_lm_head_int8").role == "primary"
    embedding = candidate_by_id(CONFIG, "mixed_l10_mlp_lm_head_embedding_int8")
    assert not embedding.runnable
    assert "Embedding" in embedding.unsupported_reason


def test_longest_prefix_module_precision():
    candidate = candidate_by_id(CONFIG, "mixed_l10_mlp_lm_head_int8")
    assert precision_for_module(candidate, "model.layers.10.mlp.gate_proj") == "int8"
    assert precision_for_module(candidate, "model.layers.9.mlp.gate_proj") == "int4"
    assert precision_for_module(candidate, "lm_head") == "int8"


def _config_for_tmp(tmp_path: Path) -> dict:
    value = json.loads(json.dumps(CONFIG))
    value["paths"]["output_root"] = "out"
    value["paths"]["source_model_manifest"] = "source.json"
    value["paths"]["module_policy"] = "policy.json"
    value["paths"]["combination_ranking"] = "combination.json"
    return value


def test_prepare_marks_unsupported_candidate(tmp_path):
    config = _config_for_tmp(tmp_path)
    manifest = prepare_search(tmp_path, config)
    assert manifest["status"] == "prepared"
    capability = json.loads((tmp_path / "out/plans/mixed_l10_mlp_lm_head_embedding_int8/capability.json").read_text())
    assert capability["status"] == "unsupported"
    assert capability["deployment_claims_allowed"] is False


def _quality(net: int = 0) -> dict:
    return {"metrics": {"total": 10, "correct": 5 + net, "parse_rate": 1.0}, "transitions_vs_all_int4": {"net_correct_gain": net}}


def _benchmark(memory: float = 100, latency: float = 50, throughput: float = 10) -> dict:
    return {"cases": [{"memory": {"peak_allocated_mib": memory}, "latency_ms": {"end_to_end": {"p95": latency}}, "throughput": {"mean_output_tokens_per_second": throughput}}]}


def test_finalize_requires_and_gates_all_evidence(tmp_path):
    config = _config_for_tmp(tmp_path)
    prepare_search(tmp_path, config)
    runnable = [row for row in build_candidates(config) if row.runnable]
    for candidate in runnable:
        for suite in ("full_dev", "high_risk", "long_context"):
            source = tmp_path / f"{candidate.plan_id}-{suite}.json"
            source.write_text(json.dumps(_quality(1 if candidate.role == "primary" else 0)))
            record_evidence(tmp_path, config, candidate.plan_id, suite, source)
        source = tmp_path / f"{candidate.plan_id}-benchmark.json"
        source.write_text(json.dumps(_benchmark()))
        record_evidence(tmp_path, config, candidate.plan_id, "benchmark", source)
    assert all(not value for value in pending_evidence(tmp_path, config).values())
    result = finalize_search(tmp_path, config)
    assert result["recommendation"] == "mixed_l10_mlp_lm_head_int8"
    unsupported = next(row for row in result["candidates"] if row["status"] == "unsupported")
    assert unsupported["selected"] is False


def test_finalize_rejects_missing_evidence(tmp_path):
    config = _config_for_tmp(tmp_path)
    prepare_search(tmp_path, config)
    with pytest.raises(ValueError, match="incomplete"):
        finalize_search(tmp_path, config)
