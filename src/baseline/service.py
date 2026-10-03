"""Deterministic FP16/BF16 baseline inference service."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.finqa_assets import render_finqa_prompt
from src.input_validation.model_input import FAIL, PASS, CheckResult


@dataclass
class InferenceRequest:
    question: str
    pre_text: list[str]
    post_text: list[str]
    table: list[list[Any]]
    request_id: str = ""

    @classmethod
    def from_finqa_row(
        cls, row: dict[str, Any], request_id: str = ""
    ) -> "InferenceRequest":
        qa = row.get("qa", {})
        question = row.get("question") or qa.get("question")
        if not question:
            raise ValueError("FinQA row has no question.")
        return cls(
            question=str(question),
            pre_text=[str(value) for value in row.get("pre_text", [])],
            post_text=[str(value) for value in row.get("post_text", [])],
            table=[list(table_row) for table_row in row.get("table", [])],
            request_id=request_id or str(uuid.uuid4()),
        )

    def render_prompt(self) -> str:
        return render_finqa_prompt(
            {
                "question": self.question,
                "pre_text": self.pre_text,
                "post_text": self.post_text,
                "table": self.table,
            }
        )


@dataclass
class BaselineEvidenceReport:
    evidence: str
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {file_path}: {exc}") from exc
    return file_path, payload


class BaselineService:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        config: dict[str, Any],
        model_metadata: dict[str, Any],
        log_file: str | Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.config = config
        self.model_metadata = model_metadata
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._configure_determinism()

    @classmethod
    def from_local_model(
        cls,
        config_file: str | Path,
        model_manifest_file: str | Path,
        log_file: str | Path,
    ) -> "BaselineService":
        _, config = _load_json(config_file)
        _, model_manifest = _load_json(model_manifest_file)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for baseline inference.") from exc

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(config.get("dtype"))
        if dtype is None:
            raise ValueError("Baseline dtype must be float16 or bfloat16.")
        model_path = Path(model_manifest["model_path"])
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=model_manifest.get("runtime", {}).get(
                "trust_remote_code", False
            ),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=model_manifest.get("runtime", {}).get(
                "trust_remote_code", False
            ),
            dtype=dtype,
        )
        model.to(config["device"])
        model.eval()
        metadata = {
            "model_id": model_manifest.get("model_id"),
            "version": model_manifest.get("version"),
            "commit": model_manifest.get("commit"),
            "weight_files": model_manifest.get("weight_files", []),
        }
        return cls(model, tokenizer, torch, config, metadata, log_file)

    def _configure_determinism(self) -> None:
        seed = int(self.config["seed"])
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.torch.use_deterministic_algorithms(
            bool(self.config.get("deterministic")), warn_only=True
        )
        if hasattr(self.torch.backends, "cudnn"):
            self.torch.backends.cudnn.deterministic = True
            self.torch.backends.cudnn.benchmark = False

    def environment_metadata(self) -> dict[str, Any]:
        cuda_name = None
        if self.torch.cuda.is_available():
            cuda_name = self.torch.cuda.get_device_name(0)
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": self.torch.__version__,
            "transformers": _package_version("transformers"),
            "cuda": self.torch.version.cuda,
            "gpu": cuda_name,
            "device": self.config["device"],
            "dtype": self.config["dtype"],
            "seed": self.config["seed"],
            "deterministic": self.config["deterministic"],
        }

    def generate_one(self, request: InferenceRequest) -> dict[str, Any]:
        return self.generate_batch([request])[0]

    def generate_batch(
        self, requests: list[InferenceRequest]
    ) -> list[dict[str, Any]]:
        if not requests:
            raise ValueError("At least one request is required.")
        prompts = [request.render_prompt() for request in requests]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.config["max_input_tokens"]),
        )
        encoded = {
            name: tensor.to(self.config["device"])
            for name, tensor in encoded.items()
        }
        if self.config["device"].startswith("cuda"):
            self.torch.cuda.synchronize()
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    **self.config["generation"],
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            if self.config["device"].startswith("cuda"):
                self.torch.cuda.synchronize()
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            for request in requests:
                self._append_log(
                    {
                        "request_id": request.request_id,
                        "success": False,
                        "batch_size": len(requests),
                        "latency_ms": latency_ms,
                        "error": repr(exc),
                    }
                )
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        input_width = encoded["input_ids"].shape[1]
        input_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        results = []
        for index, request in enumerate(requests):
            generated_tokens = outputs[index, input_width:]
            output_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            generated_token_ids = generated_tokens.tolist()
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id in generated_token_ids:
                output_token_count = generated_token_ids.index(eos_token_id) + 1
            else:
                output_token_count = len(generated_token_ids)
            record = {
                "request_id": request.request_id,
                "success": True,
                "started_at": started_at.isoformat(),
                "batch_size": len(requests),
                "input_characters": len(prompts[index]),
                "input_tokens": int(input_token_counts[index]),
                "output_tokens": output_token_count,
                "latency_ms": latency_ms,
                "output_tokens_per_second": (
                    output_token_count / (latency_ms / 1000) if latency_ms else 0.0
                ),
                "model_id": self.model_metadata.get("model_id"),
                "model_commit": self.model_metadata.get("commit"),
                "dtype": self.config["dtype"],
                "device": self.config["device"],
            }
            self._append_log(record)
            results.append(
                {
                    "request_id": request.request_id,
                    "success": True,
                    "output_text": output_text,
                    "metrics": record,
                }
            )
        return results

    def _append_log(self, record: dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_baseline_evidence(evidence_file: str | Path) -> BaselineEvidenceReport:
    evidence_path, evidence = _load_json(evidence_file)
    config = evidence.get("config", {})
    model_loaded = evidence.get("model_loaded") is True and config.get("dtype") in {
        "float16",
        "bfloat16",
    }
    template_ready = (
        config.get("prompt_template") == "finqa-v1"
        and isinstance(config.get("generation"), dict)
        and config["generation"].get("max_new_tokens", 0) > 0
    )
    single_ready = evidence.get("single_request", {}).get("success") is True
    batch_ready = (
        evidence.get("batch_request", {}).get("success") is True
        and evidence["batch_request"].get("result_count", 0) >= 2
    )
    deterministic_ready = (
        config.get("deterministic") is True
        and config.get("seed") is not None
        and evidence.get("deterministic_replay_match") is True
    )
    logs = evidence.get("request_logs", {})
    logs_ready = (
        logs.get("record_count", 0) >= 4
        and logs.get("all_success") is True
        and logs.get("required_fields_present") is True
    )
    model_metadata = evidence.get("model_metadata", {})
    environment = evidence.get("environment", {})
    metadata_ready = (
        all(model_metadata.get(field) for field in ("model_id", "version", "commit"))
        and all(
            environment.get(field)
            for field in ("python", "torch", "transformers", "cuda", "gpu", "device")
        )
    )
    checks = [
        CheckResult("fp16_or_bf16_baseline_service", PASS if model_loaded else FAIL, "Baseline model is loaded in FP16/BF16."),
        CheckResult("unified_prompt_and_generation", PASS if template_ready else FAIL, "Prompt template and generation parameters are versioned."),
        CheckResult("single_request_inference", PASS if single_ready else FAIL, "Single-request inference succeeded."),
        CheckResult("batch_inference", PASS if batch_ready else FAIL, "Batch inference returned at least two results."),
        CheckResult("deterministic_seed_configuration", PASS if deterministic_ready else FAIL, "Fixed seed replay produced identical output."),
        CheckResult("request_level_performance_logs", PASS if logs_ready else FAIL, "Request-level token and latency logs are complete."),
        CheckResult("model_and_environment_metadata", PASS if metadata_ready else FAIL, "Model version and runtime environment are recorded."),
    ]
    return BaselineEvidenceReport(str(evidence_path), checks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = validate_baseline_evidence(args.evidence)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
