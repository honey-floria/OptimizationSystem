#!/usr/bin/env python3
"""Prepare, export, audit, and finalize TODO 8.1 structured pruning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pruning.structured import (
    export_candidate, finalize, pending, prepare_experiment, record_quality_evidence,
    run_baseline_quality_evaluation, run_quality_evaluation,
)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/structured_pruning.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    evaluate_baseline = commands.add_parser("evaluate-baseline")
    evaluate_baseline.add_argument("--model-path", type=Path)
    evaluate_baseline.add_argument("--device", default="cuda:0")
    export = commands.add_parser("export")
    export.add_argument("--plan-id", required=True)
    export.add_argument("--model-path", type=Path)
    export.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--plan-id", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    record = commands.add_parser("record-quality")
    record.add_argument("--plan-id", required=True)
    record.add_argument("--source", type=Path, required=True)
    commands.add_parser("pending")
    commands.add_parser("finalize")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.command == "prepare":
        result = prepare_experiment(ROOT, config)
    elif args.command == "evaluate-baseline":
        result = run_baseline_quality_evaluation(
            ROOT, config, model_path=args.model_path, device=args.device,
        )
    elif args.command == "export":
        result = export_candidate(ROOT, config, args.plan_id, model_path=args.model_path, device=args.device)
    elif args.command == "evaluate":
        result = run_quality_evaluation(ROOT, config, args.plan_id, device=args.device)
    elif args.command == "record-quality":
        result = record_quality_evidence(ROOT, config, args.plan_id, args.source)
    elif args.command == "pending":
        result = pending(ROOT, config)
    else:
        result = finalize(ROOT, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
