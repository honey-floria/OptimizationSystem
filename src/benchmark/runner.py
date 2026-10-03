"""Unified FP16/BF16 benchmark runner and evidence validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.baseline.service import BaselineService
from src.input_validation.model_input import FAIL, PASS, CheckResult


MIB = 1024 * 1024


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    input_tokens: int
    output_tokens: int

    @property
    def case_id(self) -> str:
        return (
            f"bs{self.batch_size}-in{self.input_tokens}"
            f"-out{self.output_tokens}"
        )


@dataclass
class BenchmarkValidationReport:
    report: str
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


class _FirstTokenTimer:
    def __init__(self, torch_module: Any, started: float) -> None:
        self.torch = torch_module
        self.started = started
        self.skip_prompt = True
        self.ttft_ms: float | None = None

    def put(self, value: Any) -> None:
        if self.skip_prompt:
            self.skip_prompt = False
            return
        if self.ttft_ms is None:
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            self.ttft_ms = (time.perf_counter() - self.started) * 1000

    def end(self) -> None:
        return None


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one.")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {file_path}: {exc}") from exc
    return file_path, payload


def build_cases(config: dict[str, Any]) -> list[BenchmarkCase]:
    matrix = config.get("matrix", {})
    cases = [
        BenchmarkCase(int(batch_size), int(input_tokens), int(output_tokens))
        for batch_size in matrix.get("batch_sizes", [])
        for input_tokens in matrix.get("input_lengths", [])
        for output_tokens in matrix.get("output_lengths", [])
    ]
    if not cases or any(
        value <= 0
        for case in cases
        for value in (case.batch_size, case.input_tokens, case.output_tokens)
    ):
        raise ValueError("Benchmark matrix values must be positive integers.")
    return cases


def _fixed_length_input_ids(
    tokenizer: Any, prompt: str, target_length: int
) -> list[int]:
    token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if not token_ids:
        raise ValueError("Benchmark prompt produced no input tokens.")
    repeats = (target_length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[-target_length:]


def _estimated_kv_cache_mib(
    service: BaselineService, case: BenchmarkCase
) -> float:
    model_config = service.model.config
    hidden_size = int(model_config.hidden_size)
    attention_heads = int(model_config.num_attention_heads)
    key_value_heads = int(
        getattr(model_config, "num_key_value_heads", None) or attention_heads
    )
    head_dim = int(
        getattr(model_config, "head_dim", None)
        or hidden_size // attention_heads
    )
    layers = int(model_config.num_hidden_layers)
    element_size = next(service.model.parameters()).element_size()
    sequence_length = case.input_tokens + case.output_tokens
    cache_bytes = (
        case.batch_size
        * layers
        * 2
        * key_value_heads
        * head_dim
        * sequence_length
        * element_size
    )
    return cache_bytes / MIB


def _kv_cache_mib(cache: Any, torch_module: Any) -> float:
    if cache is None:
        return 0.0
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    tensors = []

    def collect(value: Any) -> None:
        if torch_module.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif hasattr(value, "layers"):
            collect(value.layers)
        else:
            for attribute in ("keys", "values", "key_cache", "value_cache"):
                if hasattr(value, attribute):
                    collect(getattr(value, attribute))

    collect(cache)
    seen = set()
    total_bytes = 0
    for tensor in tensors:
        identity = (tensor.device, tensor.data_ptr())
        if identity not in seen:
            seen.add(identity)
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / MIB


def _run_sample(
    service: BaselineService,
    prompt: str,
    case: BenchmarkCase,
    use_cache: bool,
) -> dict[str, Any]:
    torch = service.torch
    token_ids = _fixed_length_input_ids(
        service.tokenizer, prompt, case.input_tokens
    )
    input_ids = torch.tensor(
        [token_ids] * case.batch_size,
        dtype=torch.long,
        device=service.config["device"],
    )
    attention_mask = torch.ones_like(input_ids)
    if service.config["device"].startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    allocated_before = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    started = time.perf_counter()
    timer = _FirstTokenTimer(torch, started)
    with torch.no_grad():
        outputs = service.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=case.output_tokens,
            min_new_tokens=case.output_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=use_cache,
            pad_token_id=service.tokenizer.pad_token_id,
            streamer=timer,
            return_dict_in_generate=True,
        )
    if service.config["device"].startswith("cuda"):
        torch.cuda.synchronize()
    end_to_end_ms = (time.perf_counter() - started) * 1000
    sequences = outputs.sequences
    generated_tokens = int(sequences.shape[1] - input_ids.shape[1])
    total_output_tokens = generated_tokens * case.batch_size
    ttft_ms = timer.ttft_ms if timer.ttft_ms is not None else end_to_end_ms
    remaining_tokens = max(generated_tokens - 1, 1)
    per_token_ms = max(end_to_end_ms - ttft_ms, 0.0) / remaining_tokens
    peak_allocated = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    )
    seconds = end_to_end_ms / 1000
    return {
        "ttft_ms": ttft_ms,
        "time_per_output_token_ms": per_token_ms,
        "end_to_end_latency_ms": end_to_end_ms,
        "generated_tokens_per_request": generated_tokens,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": (
            total_output_tokens / seconds if seconds else 0.0
        ),
        "requests_per_second": case.batch_size / seconds if seconds else 0.0,
        "model_allocated_mib": allocated_before / MIB,
        "peak_allocated_mib": peak_allocated / MIB,
        "incremental_peak_mib": max(peak_allocated - allocated_before, 0) / MIB,
        "peak_reserved_mib": peak_reserved / MIB,
        "kv_cache_mib": _kv_cache_mib(outputs.past_key_values, torch),
        "estimated_kv_cache_mib": _estimated_kv_cache_mib(service, case),
    }


def run_benchmark(
    service: BaselineService,
    prompt: str,
    benchmark_config: dict[str, Any],
) -> dict[str, Any]:
    warmup_runs = int(benchmark_config.get("warmup_runs", 0))
    measured_runs = int(benchmark_config.get("measured_runs", 0))
    if warmup_runs < 0 or measured_runs < 2:
        raise ValueError("Use non-negative warmups and at least two measured runs.")
    cases = build_cases(benchmark_config)
    case_reports = []
    for case_index, case in enumerate(cases, start=1):
        print(
            f"Benchmark {case_index}/{len(cases)}: {case.case_id}",
            flush=True,
        )
        for _ in range(warmup_runs):
            _run_sample(
                service, prompt, case, bool(benchmark_config.get("use_cache", True))
            )
        samples = [
            _run_sample(
                service, prompt, case, bool(benchmark_config.get("use_cache", True))
            )
            for _ in range(measured_runs)
        ]
        case_reports.append(
            {
                "case_id": case.case_id,
                "batch_size": case.batch_size,
                "input_tokens": case.input_tokens,
                "output_tokens": case.output_tokens,
                "samples": samples,
                "latency_ms": {
                    "ttft": _distribution([row["ttft_ms"] for row in samples]),
                    "per_output_token": _distribution(
                        [row["time_per_output_token_ms"] for row in samples]
                    ),
                    "end_to_end": _distribution(
                        [row["end_to_end_latency_ms"] for row in samples]
                    ),
                },
                "throughput": {
                    "mean_output_tokens_per_second": mean(
                        [row["output_tokens_per_second"] for row in samples]
                    ),
                    "mean_requests_per_second": mean(
                        [row["requests_per_second"] for row in samples]
                    ),
                },
                "memory": {
                    "model_allocated_mib": max(
                        row["model_allocated_mib"] for row in samples
                    ),
                    "peak_allocated_mib": max(
                        row["peak_allocated_mib"] for row in samples
                    ),
                    "incremental_peak_mib": max(
                        row["incremental_peak_mib"] for row in samples
                    ),
                    "peak_reserved_mib": max(
                        row["peak_reserved_mib"] for row in samples
                    ),
                    "kv_cache_mib": max(
                        row["kv_cache_mib"] for row in samples
                    ),
                    "estimated_kv_cache_mib": max(
                        row["estimated_kv_cache_mib"] for row in samples
                    ),
                },
            }
        )
    return {
        "schema_version": 1,
        "benchmark_name": benchmark_config.get("benchmark_name"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "python -m src.benchmark.runner run",
        "model_metadata": service.model_metadata,
        "environment": service.environment_metadata(),
        "baseline_config": service.config,
        "benchmark_config": benchmark_config,
        "workload": {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "synthetic_length_control": True,
        },
        "cases": case_reports,
    }


def _all_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        sample
        for case in report.get("cases", [])
        for sample in case.get("samples", [])
    ]


def validate_benchmark_report(
    report_file: str | Path,
) -> BenchmarkValidationReport:
    report_path, report = _load_json(report_file)
    cases = report.get("cases", [])
    samples = _all_samples(report)
    command_ready = report.get("command") == "python -m src.benchmark.runner run"
    ttft_ready = bool(samples) and all(row.get("ttft_ms", 0) > 0 for row in samples)
    per_token_ready = bool(samples) and all(
        row.get("time_per_output_token_ms", 0) > 0 for row in samples
    )
    end_to_end_ready = bool(samples) and all(
        row.get("end_to_end_latency_ms", 0) > 0 for row in samples
    )
    tokens_ready = bool(samples) and all(
        row.get("output_tokens_per_second", 0) > 0 for row in samples
    )
    batch_throughput_ready = bool(samples) and all(
        row.get("requests_per_second", 0) > 0 for row in samples
    )
    percentile_ready = bool(cases) and all(
        0 < latency.get("p50", 0)
        <= latency.get("p95", 0)
        <= latency.get("p99", 0)
        for case in cases
        for latency in (case.get("latency_ms", {}).get("end_to_end", {}),)
    )
    memory_ready = bool(cases) and all(
        case.get("memory", {}).get("model_allocated_mib", 0) > 0
        and case.get("memory", {}).get("peak_allocated_mib", 0)
        >= case.get("memory", {}).get("model_allocated_mib", 0)
        and case.get("memory", {}).get("kv_cache_mib", 0) > 0
        for case in cases
    )
    batch_sizes = {case.get("batch_size") for case in cases}
    batch_matrix_ready = 1 in batch_sizes and any(size > 1 for size in batch_sizes)
    input_lengths = {case.get("input_tokens") for case in cases}
    output_lengths = {case.get("output_tokens") for case in cases}
    length_matrix_ready = len(input_lengths) >= 2 and len(output_lengths) >= 2
    baseline_config = report.get("baseline_config", {})
    model_metadata = report.get("model_metadata", {})
    environment = report.get("environment", {})
    baseline_report_ready = (
        baseline_config.get("dtype") in {"float16", "bfloat16"}
        and all(model_metadata.get(field) for field in ("model_id", "commit"))
        and all(
            environment.get(field)
            for field in ("torch", "transformers", "cuda", "gpu", "device")
        )
    )
    checks = [
        CheckResult("unified_benchmark_command", PASS if command_ready else FAIL, "Unified benchmark command is recorded."),
        CheckResult("ttft_measured", PASS if ttft_ready else FAIL, "TTFT is positive for every measured sample."),
        CheckResult("per_token_latency_measured", PASS if per_token_ready else FAIL, "Per-output-token latency is recorded."),
        CheckResult("end_to_end_latency_measured", PASS if end_to_end_ready else FAIL, "End-to-end latency is positive."),
        CheckResult("tokens_per_second_measured", PASS if tokens_ready else FAIL, "Output token throughput is positive."),
        CheckResult("batch_throughput_measured", PASS if batch_throughput_ready else FAIL, "Request throughput is positive."),
        CheckResult("latency_percentiles_measured", PASS if percentile_ready else FAIL, "P50, P95, and P99 end-to-end latency are recorded."),
        CheckResult("gpu_and_kv_cache_memory_measured", PASS if memory_ready else FAIL, "GPU peaks and actual KV cache tensor memory are recorded."),
        CheckResult("batch_size_matrix_measured", PASS if batch_matrix_ready else FAIL, "Batch size one and larger batches are covered."),
        CheckResult("input_output_length_matrix_measured", PASS if length_matrix_ready else FAIL, "Multiple input and output lengths are covered."),
        CheckResult("fp16_or_bf16_baseline_report_generated", PASS if baseline_report_ready else FAIL, "Baseline model and runtime metadata are complete."),
    ]
    return BenchmarkValidationReport(str(report_path), checks)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--baseline-config", required=True)
    run_parser.add_argument("--benchmark-config", required=True)
    run_parser.add_argument("--model-manifest", required=True)
    run_parser.add_argument("--prompt-file", required=True)
    run_parser.add_argument("--output-dir", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--report", required=True)
    validate_parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command_name == "validate":
        try:
            validation = validate_benchmark_report(args.report)
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        payload = validation.to_dict()
        if args.output:
            _write_json(Path(args.output).expanduser(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if validation.complete else 1

    _, benchmark_config = _load_json(args.benchmark_config)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    service = BaselineService.from_local_model(
        args.baseline_config,
        args.model_manifest,
        output_dir / "service.jsonl",
    )
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    report = run_benchmark(service, prompt, benchmark_config)
    report_path = output_dir / "baseline_fp16_report.json"
    _write_json(report_path, report)
    validation = validate_benchmark_report(report_path)
    _write_json(output_dir / "validation.json", validation.to_dict())
    shutil.copy2(args.benchmark_config, output_dir / "benchmark_config.json")
    print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
    return 0 if validation.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
