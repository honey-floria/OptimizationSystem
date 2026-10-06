#!/usr/bin/env python3
"""Train a QLoRA adapter on FinQA train without touching dev/test answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.finqa_assets import render_finqa_prompt

MODEL_DATA_DIR = PROJECT_ROOT / "models" / "data"
DEFAULT_BASE_MODEL_PATH = MODEL_DATA_DIR / "Qwen2.5-7B-Instruct"
DEFAULT_ARTIFACT_DIR = MODEL_DATA_DIR / "finqa_qlora_7b_v2"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "datasets" / "raw_finqa" / "train.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "finqa_qlora_7b_v2"

NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
STRUCTURED_JSON_PROMPT_SUFFIX = (
    "\nReturn exactly one JSON object and no other text. Use this schema: "
    '{"evidence":["table field and year"],"formula":"numeric arithmetic formula",'
    '"value":0,"unit":"million|percent|dollars|shares|times|multiple|mmboe|"}'
    ". Preserve the table column, row, year, and unit in evidence. "
    "Use times or multiple for a ratio, and mmboe when the table uses that unit. "
    "The value must be the final answer."
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 FinQA JSON：{path}: {exc}") from exc


def _load_train_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError(
                "读取 HuggingFace Arrow 数据集需要安装 datasets。"
            ) from exc
        dataset = load_from_disk(str(path))
        if hasattr(dataset, "keys") and not hasattr(dataset, "column_names"):
            raise ValueError("--train-path 必须指向单个 Dataset 目录，不能是 DatasetDict 根目录")
        return [dict(row) for row in dataset]
    return _rows_from_payload(_load_json(path))


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "train", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError("FinQA train 文件必须是对象数组，或包含 data/train/examples/rows 数组")


def _qa_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row.get("qa"), dict) and key in row["qa"]:
        return row["qa"][key]
    return row.get(key, default)


def _answer_parts(answer: Any) -> tuple[str, str, str] | None:
    answer_text = str(answer).strip()
    match = NUMBER_PATTERN.search(answer_text)
    if match is None:
        return None
    value = match.group(0).replace(",", "")
    lowered = answer_text.lower()
    if "%" in answer_text:
        unit = "percent"
    elif "mmboe" in lowered:
        unit = "mmboe"
    elif "billion" in lowered:
        unit = "billion"
    elif "million" in lowered:
        unit = "million"
    elif "thousand" in lowered:
        unit = "thousand"
    elif "dollar" in lowered or "$" in answer_text:
        unit = "dollars"
    else:
        unit = ""
    return value, unit, answer_text


def _compact_evidence(gold_inds: Any) -> list[str]:
    if isinstance(gold_inds, dict):
        return [str(key) for key in gold_inds]
    if not isinstance(gold_inds, list):
        gold_inds = [gold_inds]
    evidence = []
    for item in gold_inds:
        if isinstance(item, dict):
            evidence.extend(str(key) for key in item)
        else:
            text = str(item).strip()
            if text:
                evidence.append(text)
    return evidence


def build_training_examples(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert numeric FinQA train rows to the structured-json supervision contract."""

    examples: list[dict[str, str]] = []
    for row in rows:
        answer_parts = _answer_parts(_qa_value(row, "answer", ""))
        program = _qa_value(row, "program", "")
        if answer_parts is None or not str(program).strip():
            continue
        value, unit, answer_text = answer_parts
        target = {
            "evidence": _compact_evidence(_qa_value(row, "gold_inds", [])),
            "formula": str(program),
            "value": float(value),
            "unit": unit,
        }
        prompt = f"{render_finqa_prompt(row)}{STRUCTURED_JSON_PROMPT_SUFFIX}"
        examples.append(
            {
                "prompt": prompt,
                "target": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                "reference_answer": answer_text,
            }
        )
    if not examples:
        raise ValueError("FinQA train 中没有找到同时包含 numeric answer 和 program 的样本")
    return examples


class SupervisedDataset:
    def __init__(self, examples: list[dict[str, str]], tokenizer: Any, max_length: int):
        self.items = []
        for example in examples:
            prompt_tokens = tokenizer(
                example["prompt"], add_special_tokens=True, truncation=True, max_length=max_length
            )
            full_text = example["prompt"] + example["target"] + tokenizer.eos_token
            full_tokens = tokenizer(
                full_text, add_special_tokens=True, truncation=True, max_length=max_length
            )
            prompt_length = min(len(prompt_tokens["input_ids"]), len(full_tokens["input_ids"]))
            labels = list(full_tokens["input_ids"])
            labels[:prompt_length] = [-100] * prompt_length
            self.items.append(
                {
                    "input_ids": full_tokens["input_ids"],
                    "attention_mask": full_tokens["attention_mask"],
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CausalCollator:
    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path.parent if path.is_file() else path)).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_BASE_MODEL_PATH,
        help="未量化基座模型目录，默认 models/data/Qwen2.5-7B-Instruct。",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="写入 manifest 的模型标识；模型权重实际从 --model-path 读取。",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="FinQA train JSON 或 Dataset 目录，默认 datasets/raw_finqa/train.json。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="adapter、tokenizer 和 checkpoint 目录；默认 models/data/finqa_qlora_7b_v2。",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.eval_ratio < 0.5:
        raise ValueError("--eval-ratio 必须在 0 和 0.5 之间")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("--lora-r 和 --lora-alpha 必须大于 0")
    model_path = args.model_path.expanduser().resolve()
    train_path = args.train_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    model_data_root = MODEL_DATA_DIR.resolve()
    out_root = (PROJECT_ROOT / "out").resolve()
    if not _is_relative_to(model_path, model_data_root):
        raise ValueError(f"--model-path 必须位于项目 models/data/ 目录下：{model_data_root}")
    if not _is_relative_to(output_dir, out_root):
        raise ValueError(f"--output-dir 必须位于项目 out/ 目录下：{out_root}")
    if not _is_relative_to(artifact_dir, model_data_root):
        raise ValueError(f"--artifact-dir 必须位于项目 models/data/ 目录下：{model_data_root}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"基座模型目录不存在：{model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"基座模型目录缺少 config.json：{model_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"训练集路径不存在：{train_path}")
    rows = _load_train_rows(train_path)
    examples = build_training_examples(rows)
    random.Random(args.seed).shuffle(examples)
    split_index = max(1, int(len(examples) * (1.0 - args.eval_ratio)))
    train_examples = examples[:split_index]
    eval_examples = examples[split_index:]

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA 需要安装 torch、transformers、peft、bitsandbytes"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_dataset = SupervisedDataset(train_examples, tokenizer, args.max_length)
    eval_dataset = SupervisedDataset(eval_examples, tokenizer, args.max_length)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs = {
        "output_dir": str(artifact_dir),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "paged_adamw_8bit",
        "fp16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
    }
    try:
        training_args = TrainingArguments(
            evaluation_strategy="steps",
            **training_kwargs,
        )
    except TypeError as exc:
        if "evaluation_strategy" not in str(exc):
            raise
        training_args = TrainingArguments(
            eval_strategy="steps",
            **training_kwargs,
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))
    manifest = {
        "schema_version": 1,
        "method": "qlora",
        "supervision_contract": "structured-json-compact-evidence-v2",
        "base_model_id": args.model_id,
        "base_model_path": str(model_path),
        "train_path": str(train_path),
        "artifact_dir": str(artifact_dir),
        "train_sha256": _sha256(train_path),
        "source_rows": len(rows),
        "numeric_program_examples": len(examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "seed": args.seed,
        "max_length": args.max_length,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
