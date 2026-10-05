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


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 FinQA JSON：{path}: {exc}") from exc


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "train", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError("FinQA train 文件必须是对象数组，或包含 data/train/examples/rows 数组")


def _question(row: dict[str, Any]) -> str:
    if isinstance(row.get("qa"), dict):
        return str(row["qa"].get("question", ""))
    return str(row.get("question", ""))


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


def _table_text(table: Any) -> str:
    if not isinstance(table, list):
        return ""
    lines = []
    for row_index, table_row in enumerate(table):
        if not isinstance(table_row, list):
            continue
        cells = [f"r{row_index}c{column_index}={cell}" for column_index, cell in enumerate(table_row)]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def build_training_examples(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert numeric FinQA train rows to the structured-json supervision contract."""

    examples: list[dict[str, str]] = []
    for row in rows:
        answer_parts = _answer_parts(_qa_value(row, "answer", ""))
        program = _qa_value(row, "program", "")
        if answer_parts is None or not str(program).strip():
            continue
        value, unit, answer_text = answer_parts
        gold_inds = _qa_value(row, "gold_inds", [])
        if not isinstance(gold_inds, list):
            gold_inds = [str(gold_inds)]
        target = {
            "evidence": [str(item) for item in gold_inds],
            "formula": str(program),
            "value": float(value),
            "unit": unit,
        }
        prompt = (
            "You solve FinQA table questions. Return exactly one JSON object and no other text. "
            "Use keys evidence, formula, value, unit. Preserve the table evidence and operation. "
            "The value must be the final numeric answer.\n\n"
            f"Table:\n{_table_text(row.get('table', []))}\n\n"
            f"Question: {_question(row)}\n"
        )
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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.eval_ratio < 0.5:
        raise ValueError("--eval-ratio 必须在 0 和 0.5 之间")
    train_path = args.train_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = _rows_from_payload(_load_json(train_path))
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
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
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_dataset = SupervisedDataset(train_examples, tokenizer, args.max_length)
    eval_dataset = SupervisedDataset(eval_examples, tokenizer, args.max_length)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest = {
        "schema_version": 1,
        "method": "qlora",
        "base_model_id": args.model_id,
        "train_path": str(train_path),
        "train_sha256": _sha256(train_path),
        "source_rows": len(rows),
        "numeric_program_examples": len(examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "seed": args.seed,
        "max_length": args.max_length,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
