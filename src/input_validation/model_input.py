"""Validate model artifacts and optionally run a Transformers smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PASS = "pass"
FAIL = "fail"
NOT_RUN = "not_run"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass
class ValidationReport:
    manifest: str
    model_id: str
    checks: list[CheckResult]
    weight_sha256: dict[str, str]

    @property
    def validation_passed(self) -> bool:
        return all(check.status != FAIL for check in self.checks)

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "model_id": self.model_id,
            "validation_passed": self.validation_passed,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
            "weight_sha256": self.weight_sha256,
        }


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Model manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Model manifest must contain a JSON object.")
    return manifest


def _resolve(base_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _listed_files(model_path: Path, values: Any) -> list[Path]:
    if not isinstance(values, list):
        return []
    return [model_path / value for value in values if _nonempty_string(value)]


def _files_ready(paths: list[Path]) -> bool:
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _static_checks(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[list[CheckResult], dict[str, str], Path]:
    checks: list[CheckResult] = []
    model_value = manifest.get("model_path")
    model_path = (
        _resolve(manifest_path.parent, model_value)
        if _nonempty_string(model_value)
        else manifest_path.parent
    )

    weight_paths = _listed_files(model_path, manifest.get("weight_files"))
    weights_ready = model_path.is_dir() and _files_ready(weight_paths)
    checks.append(
        CheckResult(
            "base_model_weights_received",
            PASS if weights_ready else FAIL,
            f"Validated {len(weight_paths)} non-empty weight file(s)."
            if weights_ready
            else "model_path or non-empty weight_files are missing.",
        )
    )

    config_value = manifest.get("config_file")
    config_path = (
        model_path / config_value if _nonempty_string(config_value) else model_path
    )
    tokenizer_paths = _listed_files(model_path, manifest.get("tokenizer_files"))
    config_tokenizer_ready = (
        config_path.is_file()
        and config_path.stat().st_size > 0
        and _files_ready(tokenizer_paths)
    )
    checks.append(
        CheckResult(
            "config_and_tokenizer_received",
            PASS if config_tokenizer_ready else FAIL,
            "Config and tokenizer files are present and non-empty."
            if config_tokenizer_ready
            else "config_file or tokenizer_files are missing or empty.",
        )
    )

    license_info = manifest.get("license", {})
    license_file_value = license_info.get("file") if isinstance(license_info, dict) else None
    license_path = (
        model_path / license_file_value
        if _nonempty_string(license_file_value)
        else model_path
    )
    license_ready = (
        isinstance(license_info, dict)
        and _nonempty_string(license_info.get("name"))
        and license_info.get("internal_use_approved") is True
        and _nonempty_string(license_info.get("usage_scope"))
        and license_path.is_file()
        and license_path.stat().st_size > 0
    )
    checks.append(
        CheckResult(
            "license_and_internal_scope_confirmed",
            PASS if license_ready else FAIL,
            "License file, approval, and internal usage scope are recorded."
            if license_ready
            else "License metadata, approval, usage scope, or license file is missing.",
        )
    )

    weight_hashes = {
        str(path.relative_to(model_path)): _sha256(path)
        for path in weight_paths
        if path.is_file()
    }
    expected_hashes = manifest.get("expected_sha256", {})
    hashes_match = isinstance(expected_hashes, dict) and all(
        filename in weight_hashes and weight_hashes[filename] == expected_digest
        for filename, expected_digest in expected_hashes.items()
    )
    version_ready = all(
        _nonempty_string(manifest.get(field))
        for field in ("model_id", "version", "commit")
    )
    identity_ready = weights_ready and version_ready and hashes_match
    checks.append(
        CheckResult(
            "version_commit_and_weight_hash_recorded",
            PASS if identity_ready else FAIL,
            f"Recorded identity and {len(weight_hashes)} SHA-256 hash(es)."
            if identity_ready
            else "model_id/version/commit is missing or an expected hash mismatched.",
        )
    )
    return checks, weight_hashes, model_path


def _runtime_checks(
    manifest: dict[str, Any], model_path: Path
) -> list[CheckResult]:
    runtime = manifest.get("runtime", {})
    if not isinstance(runtime, dict) or runtime.get("framework") != "transformers":
        return [
            CheckResult(
                "fp16_or_bf16_model_load",
                FAIL,
                "runtime.framework must be 'transformers'.",
            ),
            CheckResult(
                "target_framework_inference",
                NOT_RUN,
                "Inference was skipped because the framework is unsupported.",
            ),
        ]

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        return [
            CheckResult("fp16_or_bf16_model_load", FAIL, str(exc)),
            CheckResult(
                "target_framework_inference",
                NOT_RUN,
                "Inference was skipped because runtime dependencies are missing.",
            ),
        ]

    dtype_name = runtime.get("dtype")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_name)
    if dtype is None:
        return [
            CheckResult(
                "fp16_or_bf16_model_load",
                FAIL,
                "runtime.dtype must be 'float16' or 'bfloat16'.",
            ),
            CheckResult(
                "target_framework_inference",
                NOT_RUN,
                "Inference was skipped because dtype is invalid.",
            ),
        ]

    trust_remote_code = runtime.get("trust_remote_code") is True
    model_class = runtime.get("model_class", "causal_lm")
    model_factory = {
        "causal_lm": AutoModelForCausalLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
    }.get(model_class)
    if model_factory is None:
        return [
            CheckResult("fp16_or_bf16_model_load", FAIL, "Unsupported model_class."),
            CheckResult(
                "target_framework_inference",
                NOT_RUN,
                "Inference was skipped because model_class is invalid.",
            ),
        ]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model = model_factory.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
        device = runtime.get("device", "cpu")
        model.to(device)
        model.eval()
    except Exception as exc:
        return [
            CheckResult("fp16_or_bf16_model_load", FAIL, repr(exc)),
            CheckResult(
                "target_framework_inference",
                NOT_RUN,
                "Inference was skipped because model loading failed.",
            ),
        ]

    load_check = CheckResult(
        "fp16_or_bf16_model_load",
        PASS,
        f"Loaded {dtype_name} model on {device}.",
    )
    try:
        prompt = runtime.get("smoke_test_prompt", "Hello")
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=1, do_sample=False)
        if output.shape[-1] <= 0:
            raise RuntimeError("The model returned an empty token sequence.")
    except Exception as exc:
        inference_check = CheckResult("target_framework_inference", FAIL, repr(exc))
    else:
        inference_check = CheckResult(
            "target_framework_inference",
            PASS,
            "Generated one token with the configured Transformers backend.",
        )
    return [load_check, inference_check]


def validate_model_input(
    manifest_file: str | Path, runtime_check: bool = False
) -> ValidationReport:
    manifest_path = Path(manifest_file).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    checks, hashes, model_path = _static_checks(manifest, manifest_path)
    static_passed = all(check.status == PASS for check in checks)
    if runtime_check and static_passed:
        checks.extend(_runtime_checks(manifest, model_path))
    else:
        reason = (
            "Run with --runtime-check to load the model."
            if not runtime_check
            else "Runtime check was skipped because static validation failed."
        )
        checks.extend(
            [
                CheckResult(
                    "fp16_or_bf16_model_load",
                    NOT_RUN,
                    reason,
                ),
                CheckResult(
                    "target_framework_inference",
                    NOT_RUN,
                    reason,
                ),
            ]
        )
    return ValidationReport(
        manifest=str(manifest_path),
        model_id=str(manifest.get("model_id", "")),
        checks=checks,
        weight_sha256=hashes,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to model manifest JSON.")
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="Load the model and run one-token inference.",
    )
    parser.add_argument("--report", help="Optional path for the JSON report.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = validate_model_input(args.manifest, args.runtime_check)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
