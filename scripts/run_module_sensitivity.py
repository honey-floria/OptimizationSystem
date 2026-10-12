#!/usr/bin/env python3
"""Prepare and inspect the TODO 7.2.1 module-sensitivity contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.layer_sensitivity_setup import load_json, resolve_project_path
from src.quantization.module_sensitivity import inspect_model_contract, prepare_contract


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.project_root.expanduser().resolve()
    config = load_json(resolve_project_path(root, args.config))
    if args.command == "prepare":
        payload = prepare_contract(root, config)
    else:
        payload = inspect_model_contract(
            root,
            config,
            model_path=args.model_path,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
