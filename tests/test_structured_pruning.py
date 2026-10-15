import json
from pathlib import Path

import pytest

from src.pruning.structured import (
    aligned_keep_size,
    build_candidates,
    feasible_gqa_shape,
    finalize,
    pending,
    prepare_experiment,
    record_quality_evidence,
    _version_key,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/structured_pruning.json").read_text())


def _config(tmp_path: Path) -> dict:
    value = json.loads(json.dumps(CONFIG))
    value["paths"]["output_root"] = "out"
    value["paths"]["source_model_manifest"] = "source.json"
    value["paths"]["baseline_quality_report"] = "baseline.json"
    return value


def test_candidate_matrix_has_four_required_rates():
    candidates = build_candidates(CONFIG)
    assert [row.target_rate for row in candidates] == [0.05, 0.10, 0.15, 0.20]
    assert all(row.attention and row.mlp for row in candidates)


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(0.05, (27, 3, 0.0625)), (0.10, (24, 4, 0.125)),
     (0.15, (24, 3, 0.15625)), (0.20, (21, 3, 0.25))],
)
def test_qwen_gqa_rates_choose_nearest_legal_shape(rate, expected):
    assert feasible_gqa_shape(28, 4, rate) == expected


def test_mlp_keep_size_is_aligned_and_tie_prefers_less_pruning():
    assert aligned_keep_size(18944, 0.05, 128) == 18048
    assert aligned_keep_size(1024, 0.125, 256) == 1024


def test_version_key_handles_release_suffixes():
    assert _version_key("5.18.0.dev0") >= _version_key("5.18.0")
    assert _version_key("4.57.1") < _version_key("5.18")


def test_prepare_and_pending(tmp_path):
    config = _config(tmp_path)
    result = prepare_experiment(tmp_path, config)
    assert result["status"] == "prepared"
    assert len(result["contract_sha256"]) == 64
    assert set(pending(tmp_path, config)) == {f"structured_{rate:02d}" for rate in (5, 10, 15, 20)}
    assert all(value == ["export", "quality"] for value in pending(tmp_path, config).values())


def test_finalize_records_first_quality_turning_point(tmp_path):
    config = _config(tmp_path)
    prepare_experiment(tmp_path, config)
    output = tmp_path / "out"
    for index, candidate in enumerate(build_candidates(config)):
        export = output / "exports" / candidate.plan_id
        export.mkdir(parents=True)
        (export / "pruning_manifest.json").write_text(json.dumps({
            "status": "completed",
            "realized_parameter_rate": candidate.target_rate,
            "structure_validation": {"status": "pass"},
            "standalone_reload": {"status": "pass"},
        }))
        source = tmp_path / f"{candidate.plan_id}.json"
        source.write_text(json.dumps({"metrics": {
            "accuracy": 0.50 - index * 0.01, "baseline_accuracy": 0.50,
        }}))
        record_quality_evidence(tmp_path, config, candidate.plan_id, source)
    result = finalize(tmp_path, config)
    assert result["quality_turning_point"] == "structured_15"


def test_layer_pruning_requires_explicit_indices():
    config = json.loads(json.dumps(CONFIG))
    config["candidates"].append({"plan_id": "bad", "target_rate": 0.1, "components": ["layers"]})
    with pytest.raises(ValueError, match="explicit layer_indices"):
        build_candidates(config)
