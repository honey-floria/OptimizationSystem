#!/usr/bin/env python3
"""Generate or validate the TODO 7.3 financial error localization report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.financial_error_localization import (
    analyze_financial_errors,
    validate_financial_error_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["analyze", "validate"])
    parser.add_argument("--config", type=Path, default=Path("configs/financial_error_localization.json"))
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = (
        analyze_financial_errors(ROOT, config)
        if args.command == "analyze"
        else validate_financial_error_outputs(ROOT, config)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
