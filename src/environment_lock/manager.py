"""Create and validate reproducible runtime environment locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, NOT_RUN, PASS, CheckResult


@dataclass
class EnvironmentLockReport:
    nvidia_lock: str
    ascend_lock: str | None
    dependency_lock: str
    container_evidence: str | None
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nvidia_lock": self.nvidia_lock,
            "ascend_lock": self.ascend_lock,
            "dependency_lock": self.dependency_lock,
            "container_evidence": self.container_evidence,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {file_path}: {exc}") from exc
    return file_path, payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_pip_freeze(output_file: str | Path) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.stdout, encoding="utf-8")
    return output_path


def create_nvidia_lock(
    hardware_manifest_file: str | Path,
    benchmark_report_file: str | Path,
    dependency_lock_file: str | Path,
    output_file: str | Path,
) -> dict[str, Any]:
    hardware_path, hardware = _load_json(hardware_manifest_file)
    benchmark_path, benchmark = _load_json(benchmark_report_file)
    dependency_path = Path(dependency_lock_file).expanduser().resolve()
    if not dependency_path.is_file():
        raise ValueError(f"Dependency lock does not exist: {dependency_path}")
    environment = benchmark.get("environment", {})
    frameworks = hardware.get("frameworks", {})
    devices = hardware.get("accelerator", {}).get("devices", [])
    if hardware.get("target") != "nvidia" or not devices:
        raise ValueError("A real NVIDIA hardware manifest is required.")
    version_pairs = (
        ("python", hardware.get("system", {}).get("python")),
        ("torch", frameworks.get("torch")),
        ("transformers", frameworks.get("transformers")),
        ("cuda", hardware.get("software", {}).get("cuda_version")),
        ("gpu", devices[0].get("name")),
    )
    mismatches = [
        name
        for name, expected in version_pairs
        if environment.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "Hardware and benchmark environments differ: "
            + ", ".join(mismatches)
        )
    payload = {
        "schema_version": 1,
        "target": "nvidia",
        "status": "locked",
        "profile_id": f"{hardware['environment_id']}-fp16",
        "runtime": hardware.get("system", {}),
        "hardware": hardware.get("accelerator", {}),
        "software": hardware.get("software", {}),
        "frameworks": frameworks,
        "inference": {
            "framework": "transformers",
            "version": frameworks.get("transformers"),
            "dtype": environment.get("dtype"),
            "device": environment.get("device"),
        },
        "dependencies": {
            "file": str(dependency_path),
            "sha256": _sha256(dependency_path),
            "package_count": len(
                [
                    line
                    for line in dependency_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
            ),
        },
        "source_evidence": {
            "hardware_manifest": str(hardware_path),
            "benchmark_report": str(benchmark_path),
        },
    }
    _write_json(output_file, payload)
    return payload


def validate_environment_lock(
    nvidia_lock_file: str | Path,
    dependency_lock_file: str | Path,
    ascend_lock_file: str | Path | None = None,
    container_evidence_file: str | Path | None = None,
) -> EnvironmentLockReport:
    nvidia_path, nvidia = _load_json(nvidia_lock_file)
    dependency_path = Path(dependency_lock_file).expanduser().resolve()
    dependency_text = (
        dependency_path.read_text(encoding="utf-8")
        if dependency_path.is_file()
        else ""
    )
    ascend_path = None
    ascend = None
    if ascend_lock_file:
        ascend_path, ascend = _load_json(ascend_lock_file)
    container_path = None
    container = None
    if container_evidence_file:
        container_path, container = _load_json(container_evidence_file)

    nvidia_ready = (
        nvidia.get("target") == "nvidia"
        and nvidia.get("status") == "locked"
        and _has_text(nvidia.get("profile_id"))
        and bool(nvidia.get("hardware", {}).get("devices"))
    )
    ascend_ready = (
        ascend is not None
        and ascend.get("target") == "ascend"
        and ascend.get("status") == "locked"
        and _has_text(ascend.get("profile_id"))
    )
    frameworks = nvidia.get("frameworks", {})
    core_versions_ready = all(
        _has_text(value)
        for value in (
            nvidia.get("runtime", {}).get("python"),
            frameworks.get("torch"),
            frameworks.get("transformers"),
        )
    )
    cuda_ready = _has_text(nvidia.get("software", {}).get("cuda_version"))
    cann_ready = ascend_ready and _has_text(
        ascend.get("software", {}).get("cann")
    )
    inference_ready = (
        nvidia.get("inference", {}).get("framework") == "transformers"
        and _has_text(nvidia.get("inference", {}).get("version"))
    )
    container_ready = (
        container is not None
        and container.get("build_status") == "passed"
        and _has_text(container.get("image"))
        and _has_text(container.get("image_digest"))
    )
    dependency_ready = (
        bool(dependency_text.strip())
        and "torch==" in dependency_text
        and "transformers==" in dependency_text
        and nvidia.get("dependencies", {}).get("sha256")
        == _sha256(dependency_path)
    )
    devices = nvidia.get("hardware", {}).get("devices", [])
    hardware_ready = bool(devices) and all(
        _has_text(device.get("name"))
        and isinstance(device.get("memory_mib"), int)
        and _has_text(device.get("driver_version"))
        for device in devices
    )
    checks = [
        CheckResult("nvidia_environment_file_created", PASS if nvidia_ready else FAIL, "Versioned NVIDIA environment lock is complete."),
        CheckResult("ascend_environment_file_created", PASS if ascend_ready else NOT_RUN if ascend is not None and ascend.get("status") == "pending_real_hardware" else FAIL, "Ascend lock requires evidence from a real Ascend host."),
        CheckResult("python_torch_transformers_versions_pinned", PASS if core_versions_ready else FAIL, "Python, PyTorch, and Transformers versions are pinned."),
        CheckResult("cuda_and_cann_versions_pinned", PASS if cuda_ready and cann_ready else NOT_RUN if cuda_ready and not ascend_ready else FAIL, "CUDA is pinned; CANN awaits the real Ascend environment."),
        CheckResult("inference_framework_version_pinned", PASS if inference_ready else FAIL, "The active NVIDIA inference framework is pinned."),
        CheckResult("base_container_image_built", PASS if container_ready else NOT_RUN if container is None else FAIL, "A successful container build with image digest is required."),
        CheckResult("dependency_lock_saved", PASS if dependency_ready else FAIL, "pip freeze is saved, hashed, and includes core packages."),
        CheckResult("hardware_and_driver_recorded", PASS if hardware_ready else FAIL, "NVIDIA model, memory, and driver are recorded."),
    ]
    return EnvironmentLockReport(
        nvidia_lock=str(nvidia_path),
        ascend_lock=str(ascend_path) if ascend_path else None,
        dependency_lock=str(dependency_path),
        container_evidence=str(container_path) if container_path else None,
        checks=checks,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture-pip-freeze")
    capture_parser.add_argument("--output", required=True)
    create_parser = subparsers.add_parser("create-nvidia")
    create_parser.add_argument("--hardware-manifest", required=True)
    create_parser.add_argument("--benchmark-report", required=True)
    create_parser.add_argument("--dependency-lock", required=True)
    create_parser.add_argument("--output", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--nvidia-lock", required=True)
    validate_parser.add_argument("--dependency-lock", required=True)
    validate_parser.add_argument("--ascend-lock")
    validate_parser.add_argument("--container-evidence")
    validate_parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "capture-pip-freeze":
        print(capture_pip_freeze(args.output))
        return 0
    if args.command == "create-nvidia":
        payload = create_nvidia_lock(
            args.hardware_manifest,
            args.benchmark_report,
            args.dependency_lock,
            args.output,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    try:
        report = validate_environment_lock(
            args.nvidia_lock,
            args.dependency_lock,
            args.ascend_lock,
            args.container_evidence,
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    payload = report.to_dict()
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
