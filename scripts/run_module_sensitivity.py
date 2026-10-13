#!/usr/bin/env python3
"""Run the frozen TODO 7.2 module-sensitivity workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.layer_sensitivity_setup import load_json, resolve_project_path
from src.quantization.module_sensitivity import (
    inspect_model_contract,
    prepare_contract,
    probe_kv_cache_runtime,
)
from src.quantization.module_sensitivity_experiments import (
    pending_plan_ids,
    prepare_experiment_matrix,
    run_experiment_plan,
)
from src.quantization.module_sensitivity_final import (
    finalize_module_sensitivity,
    pending_final_plan_ids,
    prepare_final_evaluation,
    run_final_plan,
    select_advancement,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "module_sensitivity.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare",
        help="Validate and freeze 7.1 identities plus the ALL-I4 reference.",
    )
    inspect_parser = subparsers.add_parser(
        "inspect-model",
        help="Load the merged model and write the module/KV Cache capability audit.",
    )
    inspect_parser.add_argument(
        "--model-path",
        type=Path,
        help="Override the server-local merged model path recorded by 7.1.1.",
    )
    probe_parser = subparsers.add_parser(
        "probe-kv-cache",
        help="Run short real-generation probes for configured INT8/INT4 KV Cache backends.",
    )
    probe_parser.add_argument(
        "--precision", choices=["all", "int8", "int4"], default="all"
    )
    probe_parser.add_argument("--model-path", type=Path)
    subparsers.add_parser(
        "prepare-experiments",
        help="Build the isolated TODO 7.2.2/7.2.3 pilot matrix after inspection passes.",
    )
    subparsers.add_parser(
        "pending",
        help="List module and special-component plans without a terminal result.",
    )
    run_parser = subparsers.add_parser(
        "run-plan",
        help="Run one isolated module or special-component pilot plan.",
    )
    run_parser.add_argument("--plan-id", required=True)
    run_parser.add_argument(
        "--model-path",
        type=Path,
        help="Override the server-local merged model path recorded by 7.1.1.",
    )
    subparsers.add_parser(
        "select-advancement",
        help="Apply and freeze the pre-registered pilot-to-full-dev promotion rules.",
    )
    subparsers.add_parser(
        "prepare-final",
        help="Freeze promoted plans, datasets, references, and final evaluation suites.",
    )
    pending_final_parser = subparsers.add_parser(
        "pending-final",
        help="List final-evaluation plans without a completed result.",
    )
    pending_final_parser.add_argument(
        "--suite", choices=["full_dev", "high_risk", "determinism"]
    )
    final_plan_parser = subparsers.add_parser(
        "run-final-plan",
        help="Run one frozen promoted plan on a final-evaluation suite.",
    )
    final_plan_parser.add_argument(
        "--suite", required=True, choices=["full_dev", "high_risk", "determinism"]
    )
    final_plan_parser.add_argument("--plan-id", required=True)
    final_plan_parser.add_argument("--model-path", type=Path)
    subparsers.add_parser(
        "finalize",
        help="Audit final runs and emit rankings, report, and protection policy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.project_root.expanduser().resolve()
    config = load_json(resolve_project_path(root, args.config))
    if args.command == "prepare":
        payload = prepare_contract(root, config)
    elif args.command == "inspect-model":
        payload = inspect_model_contract(
            root,
            config,
            model_path=args.model_path,
        )
    elif args.command == "probe-kv-cache":
        payload = probe_kv_cache_runtime(
            root,
            config,
            precisions=(
                ["int8", "int4"] if args.precision == "all" else [args.precision]
            ),
            model_path=args.model_path,
        )
    elif args.command == "prepare-experiments":
        payload = prepare_experiment_matrix(root, config)
    elif args.command == "pending":
        payload = {"pending": pending_plan_ids(root, config)}
    elif args.command == "run-plan":
        payload = run_experiment_plan(
            root,
            config,
            args.plan_id,
            model_path=args.model_path,
        )
    elif args.command == "select-advancement":
        payload = select_advancement(root, config)
    elif args.command == "prepare-final":
        payload = prepare_final_evaluation(root, config)
    elif args.command == "pending-final":
        payload = {"pending": pending_final_plan_ids(root, config, args.suite)}
    elif args.command == "run-final-plan":
        payload = run_final_plan(
            root,
            config,
            args.suite,
            args.plan_id,
            model_path=args.model_path,
        )
    else:
        payload = finalize_module_sensitivity(root, config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
