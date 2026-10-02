"""Load and prepare the public FinQA assets used by the optimization project."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINQA_DATASET = "dreamerdeo/finqa"
VERIFIED_DATASET = "Aiera/finqa-verified"


def _require_datasets():
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Install the dataset dependency with `pip install datasets`."
        ) from exc
    return load_dataset, load_from_disk


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def render_table(table: list[list[Any]] | None) -> str:
    if not table:
        return ""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def render_finqa_prompt(row: dict[str, Any]) -> str:
    pre_text = "\n".join(row.get("pre_text", []))
    post_text = "\n".join(row.get("post_text", []))
    table_text = render_table(row.get("table", []))
    return (
        "You are a financial analysis assistant.\n\n"
        f"Financial context:\n{pre_text}\n\n"
        f"Financial table:\n{table_text}\n\n"
        f"Additional context:\n{post_text}\n\n"
        f"Question:\n{row['question']}\n\nAnswer:"
    )


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class FinQAAssets:
    calibration: Any
    quality_dev: Any
    quality_test: Any
    high_risk_regression: Any
    calibration_prompts: list[str]
    manifest: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "calibration": len(self.calibration),
            "quality_dev": len(self.quality_dev),
            "quality_test": len(self.quality_test),
            "high_risk_regression": len(self.high_risk_regression),
            "manifest": self.manifest,
        }


def _remove_regression_overlap(quality_test: Any, regression: Any) -> Any:
    regression_questions = {
        normalize_question(row["question"])
        for row in regression
        if "question" in row
    }
    return quality_test.filter(
        lambda row: normalize_question(row["question"]) not in regression_questions
    )


def prepare_assets(
    output_dir: str | Path,
    calibration_size: int = 1024,
    seed: int = 42,
    finqa_revision: str | None = None,
    verified_revision: str | None = None,
) -> FinQAAssets:
    """Download, split, deduplicate, and persist the public evaluation assets."""

    load_dataset, _ = _require_datasets()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path / "hf_cache"

    finqa_kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
    verified_kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
    if finqa_revision:
        finqa_kwargs["revision"] = finqa_revision
    if verified_revision:
        verified_kwargs["revision"] = verified_revision

    finqa = load_dataset(FINQA_DATASET, **finqa_kwargs)
    verified = load_dataset(
        VERIFIED_DATASET,
        split="test",
        **verified_kwargs,
    )

    if calibration_size > len(finqa["train"]):
        raise ValueError(
            f"calibration_size={calibration_size} exceeds train size "
            f"{len(finqa['train'])}."
        )

    calibration = finqa["train"].shuffle(seed=seed).select(range(calibration_size))
    quality_dev = finqa["validation"]
    quality_test = _remove_regression_overlap(finqa["test"], verified)
    prompts = [render_finqa_prompt(row) for row in calibration]

    calibration_dir = output_path / f"calibration_finqa_{calibration_size}"
    calibration.save_to_disk(str(calibration_dir))
    quality_dev.save_to_disk(str(output_path / "quality_dev_finqa"))
    quality_test.save_to_disk(str(output_path / "quality_test_finqa"))
    verified.save_to_disk(str(output_path / "regression_finqa_verified"))

    manifest = {
        "finqa_dataset": FINQA_DATASET,
        "verified_dataset": VERIFIED_DATASET,
        "finqa_revision": finqa_revision or "default",
        "verified_revision": verified_revision or "default",
        "seed": seed,
        "calibration_size": calibration_size,
        "sizes": {
            "calibration": len(calibration),
            "quality_dev": len(quality_dev),
            "quality_test": len(quality_test),
            "high_risk_regression": len(verified),
        },
        "hashes": {
            "calibration": _dataset_hash(list(calibration)),
            "quality_dev": _dataset_hash(list(quality_dev)),
            "quality_test": _dataset_hash(list(quality_test)),
            "high_risk_regression": _dataset_hash(list(verified)),
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return FinQAAssets(
        calibration=calibration,
        quality_dev=quality_dev,
        quality_test=quality_test,
        high_risk_regression=verified,
        calibration_prompts=prompts,
        manifest=manifest,
    )


def load_assets(output_dir: str | Path) -> FinQAAssets:
    """Load assets previously created by :func:`prepare_assets`."""

    _, load_from_disk = _require_datasets()
    output_path = Path(output_dir)
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    calibration_size = manifest["sizes"]["calibration"]
    calibration = load_from_disk(
        str(output_path / f"calibration_finqa_{calibration_size}")
    )
    quality_dev = load_from_disk(str(output_path / "quality_dev_finqa"))
    quality_test = load_from_disk(str(output_path / "quality_test_finqa"))
    verified = load_from_disk(str(output_path / "regression_finqa_verified"))
    prompts = [render_finqa_prompt(row) for row in calibration]
    return FinQAAssets(
        calibration=calibration,
        quality_dev=quality_dev,
        quality_test=quality_test,
        high_risk_regression=verified,
        calibration_prompts=prompts,
        manifest=manifest,
    )
