#!/usr/bin/env python3
"""Prepare, export, and close TODO 7.4 mixed-precision search."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantization.mixed_precision import (
    export_candidate, finalize_search, pending_evidence, prepare_search, record_evidence,
    run_performance_benchmark, run_quality_evaluation,
)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mixed_precision_search.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    export = sub.add_parser("export")
    export.add_argument("--plan-id", required=True)
    export.add_argument("--model-path", type=Path)
    export.add_argument("--device", default="cuda:0")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--plan-id", required=True)
    evaluate.add_argument("--suite", choices=["full_dev", "high_risk", "long_context"], required=True)
    evaluate.add_argument("--device", default="cuda:0")
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--plan-id", required=True)
    benchmark.add_argument("--prompt-file", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda:0")
    record = sub.add_parser("record-evidence")
    record.add_argument("--plan-id", required=True)
    record.add_argument("--suite", choices=["full_dev", "high_risk", "long_context", "benchmark"], required=True)
    record.add_argument("--source", type=Path, required=True)
    sub.add_parser("pending")
    sub.add_parser("finalize")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_override = os.environ.get("MIXED_PRECISION_OUTPUT_ROOT", "").strip()
    if output_override:
        config["paths"]["output_root"] = str(Path(output_override).expanduser().resolve())
    if args.command == "prepare":
        result = prepare_search(ROOT, config)
    elif args.command == "export":
        result = export_candidate(ROOT, config, args.plan_id, model_path=args.model_path, device=args.device)
    elif args.command == "record-evidence":
        result = record_evidence(ROOT, config, args.plan_id, args.suite, args.source)
    elif args.command == "evaluate":
        result = run_quality_evaluation(ROOT, config, args.plan_id, args.suite, device=args.device)
    elif args.command == "benchmark":
        result = run_performance_benchmark(ROOT, config, args.plan_id, args.prompt_file, device=args.device)
    elif args.command == "pending":
        result = pending_evidence(ROOT, config)
    else:
        result = finalize_search(ROOT, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
