"""Collect and validate NVIDIA and Ascend hardware environment manifests."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.input_validation.model_input import FAIL, NOT_RUN, PASS, CheckResult


@dataclass
class HardwareValidationReport:
    nvidia_manifest: str
    ascend_manifest: str | None
    checks: list[CheckResult]

    @property
    def complete(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nvidia_manifest": self.nvidia_manifest,
            "ascend_manifest": self.ascend_manifest,
            "complete": self.complete,
            "checks": [asdict(check) for check in self.checks],
        }


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode, output


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _base_manifest(target: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "target": target,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": f"{socket.gethostname()}-{target}",
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "container_runtime": "google-colab"
            if "COLAB_RELEASE_TAG" in os.environ
            else "host-or-managed-runtime",
            "container_version": os.environ.get(
                "COLAB_RELEASE_TAG", platform.release()
            ),
        },
    }


def collect_nvidia_environment() -> dict[str, Any]:
    manifest = _base_manifest("nvidia")
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    query_code, query_output = _run(query)
    devices = []
    if query_code == 0:
        for line in query_output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 4:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_mib": int(parts[2]),
                        "driver_version": parts[3],
                    }
                )
    topology_code, topology_output = _run(["nvidia-smi", "topo", "-m"])
    try:
        import torch
    except ImportError:
        torch_version = None
        cuda_version = None
        cudnn_version = None
    else:
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    gpu_count = len(devices)
    manifest.update(
        {
            "accelerator": {
                "available": bool(devices),
                "count": gpu_count,
                "devices": devices,
            },
            "software": {
                "cuda_version": cuda_version,
                "cudnn_version": str(cudnn_version) if cudnn_version else None,
                "driver_version": devices[0]["driver_version"] if devices else None,
            },
            "frameworks": {
                "torch": torch_version,
                "transformers": _package_version("transformers"),
                "accelerate": _package_version("accelerate"),
            },
            "communication": {
                "mode": "single_gpu" if gpu_count == 1 else "multi_gpu",
                "topology_recorded": topology_code == 0,
                "topology": topology_output,
                "network_host": socket.gethostname(),
            },
        }
    )
    return manifest


def collect_ascend_environment() -> dict[str, Any]:
    manifest = _base_manifest("ascend")
    info_code, info_output = _run(["npu-smi", "info"])
    manifest.update(
        {
            "accelerator": {
                "available": info_code == 0,
                "count": None,
                "devices": [],
                "raw_npu_smi": info_output,
            },
            "software": {
                "cann_version": os.environ.get("ASCEND_TOOLKIT_VERSION"),
                "driver_version": None,
                "firmware_version": None,
            },
            "frameworks": {
                "torch_npu": _package_version("torch-npu"),
                "mindie": os.environ.get("MINDIE_VERSION"),
                "atb": os.environ.get("ATB_VERSION"),
            },
            "communication": {
                "mode": "unconfirmed",
                "topology_recorded": bool(info_output),
                "topology": info_output,
                "network_host": socket.gethostname(),
            },
        }
    )
    return manifest


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid hardware manifest: {exc}") from exc
    return manifest_path, payload


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _accelerator_ready(manifest: dict[str, Any]) -> bool:
    accelerator = manifest.get("accelerator", {})
    devices = accelerator.get("devices", [])
    return (
        accelerator.get("available") is True
        and isinstance(accelerator.get("count"), int)
        and accelerator["count"] > 0
        and isinstance(devices, list)
        and len(devices) == accelerator["count"]
        and all(
            _has_text(device.get("name"))
            and isinstance(device.get("memory_mib"), int)
            and device["memory_mib"] > 0
            for device in devices
        )
    )


def _communication_ready(manifest: dict[str, Any]) -> bool:
    accelerator = manifest.get("accelerator", {})
    communication = manifest.get("communication", {})
    if accelerator.get("count") == 1:
        return communication.get("mode") == "single_gpu"
    return (
        communication.get("mode") == "multi_gpu"
        and communication.get("topology_recorded") is True
        and _has_text(communication.get("network_host"))
    )


def validate_hardware_inputs(
    nvidia_manifest_file: str | Path,
    ascend_manifest_file: str | Path | None = None,
) -> HardwareValidationReport:
    nvidia_path, nvidia = _load_manifest(nvidia_manifest_file)
    ascend_path = None
    ascend = None
    if ascend_manifest_file:
        ascend_path, ascend = _load_manifest(ascend_manifest_file)

    nvidia_ready = _accelerator_ready(nvidia)
    nvidia_software = nvidia.get("software", {})
    nvidia_environment_ready = (
        _has_text(nvidia_software.get("cuda_version"))
        and _has_text(nvidia_software.get("driver_version"))
        and _has_text(nvidia.get("system", {}).get("container_version"))
    )
    checks = [
        CheckResult(
            "nvidia_gpu_model_count_and_memory_confirmed",
            PASS if nvidia_ready else FAIL,
            "NVIDIA GPU model, count, and memory are recorded.",
        ),
        CheckResult(
            "cuda_driver_and_container_confirmed",
            PASS if nvidia_environment_ready else FAIL,
            "CUDA, driver, and container runtime versions are recorded.",
        ),
    ]

    if ascend is None:
        checks.extend(
            [
                CheckResult(
                    "ascend_model_count_and_memory_confirmed",
                    NOT_RUN,
                    "Run collection on an Ascend host and provide its manifest.",
                ),
                CheckResult(
                    "cann_driver_and_firmware_confirmed",
                    NOT_RUN,
                    "Ascend software validation awaits an Ascend manifest.",
                ),
            ]
        )
    else:
        ascend_ready = _accelerator_ready(ascend)
        ascend_software = ascend.get("software", {})
        ascend_environment_ready = all(
            _has_text(ascend_software.get(field))
            for field in ("cann_version", "driver_version", "firmware_version")
        )
        checks.extend(
            [
                CheckResult(
                    "ascend_model_count_and_memory_confirmed",
                    PASS if ascend_ready else FAIL,
                    "Ascend model, count, and memory are recorded.",
                ),
                CheckResult(
                    "cann_driver_and_firmware_confirmed",
                    PASS if ascend_environment_ready else FAIL,
                    "CANN, driver, and firmware versions are recorded.",
                ),
            ]
        )

    nvidia_frameworks = nvidia.get("frameworks", {})
    frameworks_ready = _has_text(nvidia_frameworks.get("torch")) and _has_text(
        nvidia_frameworks.get("transformers")
    )
    if ascend is not None:
        ascend_frameworks = ascend.get("frameworks", {})
        frameworks_ready = frameworks_ready and any(
            _has_text(ascend_frameworks.get(name))
            for name in ("torch_npu", "mindie", "atb")
        )
    else:
        frameworks_ready = False
    checks.append(
        CheckResult(
            "inference_frameworks_and_versions_confirmed",
            PASS if frameworks_ready else NOT_RUN if ascend is None else FAIL,
            "NVIDIA and Ascend inference framework versions are recorded.",
        )
    )

    communication_ready = _communication_ready(nvidia)
    if ascend is not None:
        communication_ready = communication_ready and _communication_ready(ascend)
    checks.append(
        CheckResult(
            "multi_card_communication_and_network_confirmed",
            PASS if communication_ready and ascend is not None else NOT_RUN,
            "Single-card mode or multi-card topology and network are recorded for both platforms.",
        )
    )

    manifests_versioned = (
        nvidia.get("schema_version") == 1
        and _has_text(nvidia.get("captured_at"))
        and _has_text(nvidia.get("environment_id"))
        and ascend is not None
        and ascend.get("schema_version") == 1
        and _has_text(ascend.get("captured_at"))
        and _has_text(ascend.get("environment_id"))
    )
    checks.append(
        CheckResult(
            "versioned_environment_configs_saved",
            PASS if manifests_versioned else NOT_RUN if ascend is None else FAIL,
            "Versioned NVIDIA and Ascend environment manifests are saved.",
        )
    )
    return HardwareValidationReport(
        nvidia_manifest=str(nvidia_path),
        ascend_manifest=str(ascend_path) if ascend_path else None,
        checks=checks,
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--target", choices=("nvidia", "ascend"), required=True)
    collect_parser.add_argument("--output", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--nvidia-manifest", required=True)
    validate_parser.add_argument("--ascend-manifest")
    validate_parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "collect":
        payload = (
            collect_nvidia_environment()
            if args.target == "nvidia"
            else collect_ascend_environment()
        )
        _write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("accelerator", {}).get("available") else 1
    try:
        report = validate_hardware_inputs(
            args.nvidia_manifest, args.ascend_manifest
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    payload = report.to_dict()
    if args.report:
        _write_json(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
