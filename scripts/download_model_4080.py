#!/usr/bin/env python3
"""Download a Hugging Face causal language model for the 4080 runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "data" / "Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "请先安装 requirements-server.txt，或单独安装 huggingface_hub。"
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    download_args = {
        "repo_id": args.model_id,
        "local_dir": str(output_dir),
    }
    if args.revision:
        download_args["revision"] = args.revision
    token = os.environ.get("HF_TOKEN")
    if token:
        download_args["token"] = token

    print(f"模型：{args.model_id}")
    print(f"保存目录：{output_dir}")
    snapshot_download(**download_args)
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"下载完成但缺少 config.json：{config_path}")
    print(f"下载完成：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
