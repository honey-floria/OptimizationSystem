#!/usr/bin/env python3
"""Run quality-repair evaluation for FP16, AWQ, or GPTQ models on NVIDIA."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.service import BaselineService
from src.data.finqa_assets import load_assets
from src.evaluation.quality_repair import run_quality_repair_experiment
from src.quantization.awq import load_awq_service
from src.quantization.gptq import load_gptq_service


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 文件 {path}: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_manifest(model_id: str, model_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "model_path": str(model_path.resolve()),
        "version": model_id,
        "commit": "local-server",
        "weight_files": [],
        "config_file": "config.json",
        "tokenizer_files": [],
        "expected_sha256": {},
        "runtime": {
            "framework": "transformers",
            "model_class": "causal_lm",
            "dtype": "float16",
            "device": "cuda:0",
            "trust_remote_code": False,
        },
    }


def _build_quantized_manifest(
    model_id: str, model_path: Path, quantization: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": quantization,
        "model_path": str(model_path.resolve()),
        "source_model_path": str(model_path.resolve()),
        "required_files": ["config.json"],
        "plan": {
            "method": quantization,
            "bits": 4,
            "weight_bits": 4,
            "activation_bits": 16,
            "model_id": model_id,
        },
    }


def _resolve_model_path(
    args: argparse.Namespace, quantized_manifest: dict[str, Any] | None
) -> Path:
    candidate = args.quantized_model_path or args.model_path
    if candidate is None and quantized_manifest is not None:
        value = quantized_manifest.get("model_path")
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
    if candidate is None:
        raise ValueError(
            "必须提供 --model-path、--quantized-model-path 或包含 model_path 的 --quantized-manifest"
        )
    return candidate.expanduser().resolve()


def _build_quality_config(
    source: dict[str, Any],
    *,
    full: bool,
    limit: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    variants: list[str],
    progress_every_batches: int,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    if full:
        config.pop("evaluation_size", None)
        config.pop("evaluation_seed", None)
    else:
        if limit <= 0:
            raise ValueError("pilot 模式的 --limit 必须大于 0")
        config["evaluation_size"] = limit
        config["evaluation_seed"] = seed
    config["batch_size"] = batch_size
    config["show_progress"] = True
    config["progress_every_batches"] = progress_every_batches
    config["generation"]["max_new_tokens"] = max_new_tokens
    config["pilot_variants"] = variants
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="FP16/BF16 模型目录。")
    parser.add_argument(
        "--quantized-model-path",
        type=Path,
        help="AWQ/GPTQ 量化模型目录；与 --quantization 一起使用。",
    )
    parser.add_argument(
        "--quantized-manifest",
        type=Path,
        help="已有 quantized_model_manifest.json；不提供时由入口生成临时 manifest。",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "awq", "gptq"],
        default="none",
        help="模型加载方式，默认 none（Transformers FP16/BF16）。",
    )
    parser.add_argument("--model-id", default="local-model")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/quality_repair_4080.json"))
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--limit", type=int, default=80, help="Pilot 样本数；使用 --full 时忽略。")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--variants",
        default="structured_json,cell_ids_operation",
        help="逗号分隔的方案名，默认对照原始 JSON 和单元格运算；题型路由需显式加入。",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=10,
        help="每隔多少批输出一次进度；0 表示自动按约 10 个节点输出。",
    )
    parser.add_argument("--full", action="store_true", help="运行完整 883 条 dev，而不是 pilot。")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.quantization == "none" and args.quantized_model_path is not None:
        raise ValueError("--quantized-model-path 需要同时指定 --quantization awq 或 gptq")
    if args.quantization == "none" and args.quantized_manifest is not None:
        raise ValueError("--quantized-manifest 需要同时指定 --quantization awq 或 gptq")
    if args.quantization != "none" and args.model_path is not None and args.quantized_model_path is not None:
        raise ValueError("量化模式下只传 --quantized-model-path，不要同时传 --model-path")
    quantized_manifest = None
    quantized_manifest_path = None
    if args.quantized_manifest is not None:
        quantized_manifest_path = args.quantized_manifest.expanduser().resolve()
        quantized_manifest = _read_json(quantized_manifest_path)
    model_path = _resolve_model_path(args, quantized_manifest)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json：{model_path}")
    if not (dataset_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"数据目录缺少 manifest.json：{dataset_dir}。"
            "请传入由 prepare_assets() 生成的完整 FinQA assets 目录，"
            "而不是原始 JSON 文件或其父目录。"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_config = _read_json(args.baseline_config.expanduser().resolve())
    baseline_config["device"] = args.device
    baseline_config["dtype"] = args.dtype
    effective_baseline_path = output_dir / "baseline_config.effective.json"
    _write_json(effective_baseline_path, baseline_config)

    source_quality_config = _read_json(args.config.expanduser().resolve())
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    if not variants:
        raise ValueError("至少需要一个质量评测方案")
    quality_config = _build_quality_config(
        source_quality_config,
        full=args.full,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        variants=variants,
        progress_every_batches=args.progress_every_batches,
    )
    effective_quality_path = output_dir / "quality_repair.effective.json"
    _write_json(effective_quality_path, quality_config)

    model_manifest_path = output_dir / "model_manifest.json"
    model_manifest = _build_manifest(args.model_id, model_path)
    model_manifest["runtime"]["quantization"] = args.quantization
    _write_json(model_manifest_path, model_manifest)
    if args.quantization != "none":
        if quantized_manifest is None:
            quantized_manifest = _build_quantized_manifest(
                args.model_id, model_path, args.quantization
            )
            quantized_manifest_path = output_dir / "quantized_model_manifest.json"
            _write_json(quantized_manifest_path, quantized_manifest)
        elif quantized_manifest_path is None:
            raise AssertionError("quantized manifest path was not resolved")
    dataset_manifest = _read_json(dataset_dir / "manifest.json")
    assets = load_assets(dataset_dir)

    print(f"模型：{args.model_id}")
    print(f"模型目录：{model_path}")
    print(f"量化方式：{args.quantization}")
    print(f"数据目录：{dataset_dir}")
    print(f"输出目录：{output_dir}")
    print(f"模式：{'full-883' if args.full else f'pilot-{args.limit}'}")
    print(f"方案：{', '.join(variants)}")

    if args.quantization == "awq":
        service = load_awq_service(
            effective_baseline_path,
            quantized_manifest_path,
            model_manifest_path,
            output_dir / "quality_service.jsonl",
        )
    elif args.quantization == "gptq":
        service = load_gptq_service(
            effective_baseline_path,
            quantized_manifest_path,
            model_manifest_path,
            output_dir / "quality_service.jsonl",
        )
    else:
        service = BaselineService.from_local_model(
            effective_baseline_path,
            model_manifest_path,
            output_dir / "quality_service.jsonl",
        )
    report = run_quality_repair_experiment(
        service,
        assets.quality_dev,
        dataset_manifest,
        quality_config,
        output_dir,
        calibration=assets.calibration,
    )
    report["model"] = {
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quantized_manifest": (
            str(quantized_manifest_path) if quantized_manifest_path else None
        ),
    }
    _write_json(output_dir / "comparison_report.json", report)

    print("\n===== 结果摘要 =====")
    for variant in report["variants"]:
        metrics = variant["metrics"]
        print(
            f"{variant['variant']}: "
            f"parse={metrics['parse_rate']:.2%}, "
            f"accuracy={metrics['numeric_accuracy']:.2%}, "
            f"correct={metrics['correct']}/{metrics['total']}"
        )
    print(f"报告：{output_dir / 'comparison_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
