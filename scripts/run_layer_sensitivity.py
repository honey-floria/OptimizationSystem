#!/usr/bin/env python3
"""Prepare, inspect, and run the TODO 7.1.2/7.1.3 sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.layer_sensitivity import (
    audit_activation_reports,
    audit_error_reports,
    pending_activation_plan_ids,
    pending_plan_ids,
    prepare_activation_reference,
    prepare_matrix,
    recompute_plan_error,
    run_activation_plan,
    run_plan,
    summarize_activation_reports,
    summarize_matrix,
)
from src.quantization.layer_sensitivity_setup import (
    load_json,
    resolve_project_path,
)
from src.quantization.layer_sensitivity_final import (
    audit_determinism,
    finalize_layer_sensitivity,
    pending_final_plan_ids,
    prepare_final_evaluation,
    run_final_plan,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "layer_sensitivity.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Validate 7.1.1 and create all plan contracts.")
    subparsers.add_parser("pending", help="List runnable plans without completed results.")
    subparsers.add_parser("summarize", help="Pair completed plans with ALL-I4 and build a preview ranking.")
    subparsers.add_parser(
        "audit-errors",
        help="Validate all error reports, schema versions, hashes, and cosine ranges.",
    )
    subparsers.add_parser(
        "prepare-activation-reference",
        help="Cache sampled FP16 block outputs on fixed calibration prompts.",
    )
    subparsers.add_parser(
        "pending-activations",
        help="List plans without a completed activation comparison.",
    )
    subparsers.add_parser(
        "audit-activations",
        help="Validate activation report schemas, hashes, and metric ranges.",
    )
    subparsers.add_parser(
        "summarize-activations",
        help="Summarize activation-error recovery relative to ALL-I4.",
    )
    subparsers.add_parser(
        "prepare-final",
        help="Freeze the full-dev, high-risk, and determinism suite contracts.",
    )
    pending_final = subparsers.add_parser(
        "pending-final", help="List incomplete plans in the remaining TODO 7.1 suites."
    )
    pending_final.add_argument(
        "--suite", choices=["full_dev", "high_risk", "determinism"]
    )
    final_run = subparsers.add_parser(
        "run-final-plan", help="Run one plan in a TODO 7.1 final evaluation suite."
    )
    final_run.add_argument(
        "--suite", required=True,
        choices=["full_dev", "high_risk", "determinism"],
    )
    final_run.add_argument("--plan-id", required=True)
    subparsers.add_parser(
        "finalize", help="Build final ranking, bootstrap, reports, and validation artifacts."
    )
    subparsers.add_parser(
        "audit-determinism",
        help="Compare repeated ALL-I4 and sentinel pilot predictions byte-for-byte.",
    )
    run = subparsers.add_parser("run-plan", help="Run one isolated fake-quant plan.")
    run.add_argument("--plan-id", required=True)
    repair = subparsers.add_parser(
        "recompute-error",
        help="Recompute one plan's error metrics without repeating quality inference.",
    )
    repair.add_argument("--plan-id", required=True)
    activation = subparsers.add_parser(
        "run-activation", help="Compare one plan with the cached FP16 activations."
    )
    activation.add_argument("--plan-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.project_root.expanduser().resolve()
    config = load_json(resolve_project_path(root, args.config))
    if args.command == "prepare":
        payload: object = prepare_matrix(project_root=root, config=config)
    elif args.command == "pending":
        payload = {"pending": pending_plan_ids(root, config)}
    elif args.command == "summarize":
        payload = summarize_matrix(root, config)
    elif args.command == "audit-errors":
        payload = audit_error_reports(root, config)
    elif args.command == "prepare-activation-reference":
        payload = prepare_activation_reference(project_root=root, config=config)
    elif args.command == "pending-activations":
        payload = {"pending": pending_activation_plan_ids(root, config)}
    elif args.command == "audit-activations":
        payload = audit_activation_reports(root, config)
    elif args.command == "summarize-activations":
        payload = summarize_activation_reports(root, config)
    elif args.command == "prepare-final":
        payload = prepare_final_evaluation(project_root=root, config=config)
    elif args.command == "pending-final":
        payload = {"pending": pending_final_plan_ids(root, config, args.suite)}
    elif args.command == "run-final-plan":
        payload = run_final_plan(
            project_root=root,
            config=config,
            suite=args.suite,
            plan_id=args.plan_id,
        )
    elif args.command == "finalize":
        payload = finalize_layer_sensitivity(root, config)
    elif args.command == "audit-determinism":
        payload = audit_determinism(root, config)
    elif args.command == "run-activation":
        payload = run_activation_plan(
            project_root=root, config=config, plan_id=args.plan_id
        )
    elif args.command == "recompute-error":
        payload = recompute_plan_error(
            project_root=root, config=config, plan_id=args.plan_id
        )
    else:
        payload = run_plan(
            project_root=root, config=config, plan_id=args.plan_id
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
