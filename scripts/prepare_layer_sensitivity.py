#!/usr/bin/env python3
"""Run the merge, input-freeze, and pilot-comparison steps for TODO 7.1.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.layer_sensitivity_setup import (
    compare_pilot_reports,
    freeze_inputs,
    load_json,
    merge_frozen_adapter,
    predictions_path,
    read_jsonl,
    resolve_project_path,
    write_json,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "layer_sensitivity_setup.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("merge", help="Merge the frozen adapter into the FP16/BF16 base.")
    subparsers.add_parser("freeze", help="Freeze model, dataset, prompt, scorer, and subset identities.")
    compare = subparsers.add_parser("compare-pilot", help="Compare merged-FP and frozen-NF4 pilot reports.")
    compare.add_argument("--reference-report", type=Path, required=True)
    compare.add_argument("--candidate-report", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.project_root.expanduser().resolve()
    config_path = resolve_project_path(root, args.config)
    config = load_json(config_path)
    setup_output = resolve_project_path(root, config["paths"]["setup_output"])
    if args.command == "merge":
        payload = merge_frozen_adapter(
            project_root=root, config=config, output_dir=setup_output
        )
    elif args.command == "freeze":
        from src.data.finqa_assets import load_assets

        assets = load_assets(resolve_project_path(root, config["paths"]["dataset_dir"]))
        payload = freeze_inputs(
            project_root=root,
            config=config,
            assets=assets,
            output_dir=setup_output,
        )
    else:
        reference_path = resolve_project_path(root, args.reference_report)
        candidate_path = resolve_project_path(root, args.candidate_report)
        reference = load_json(reference_path)
        candidate = load_json(candidate_path)
        payload = compare_pilot_reports(
            reference,
            candidate,
            read_jsonl(predictions_path(reference_path, reference)),
            read_jsonl(predictions_path(candidate_path, candidate)),
        )
        output = (
            resolve_project_path(root, args.output)
            if args.output
            else setup_output / "merged_vs_nf4_pilot.json"
        )
        write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
