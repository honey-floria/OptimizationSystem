#!/usr/bin/env python3
"""Prepare, inspect, and run the TODO 7.1.2 sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.layer_sensitivity import (
    pending_plan_ids,
    prepare_matrix,
    run_plan,
    summarize_matrix,
)
from src.quantization.layer_sensitivity_setup import (
    load_json,
    resolve_project_path,
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
    run = subparsers.add_parser("run-plan", help="Run one isolated fake-quant plan.")
    run.add_argument("--plan-id", required=True)
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
    else:
        payload = run_plan(
            project_root=root, config=config, plan_id=args.plan_id
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
