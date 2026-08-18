#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path


CORE_PACKAGES = ("torch", "datasets", "transformers", "trl", "accelerate", "peft", "math-verify")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def gpu_summary() -> tuple[int, list[str]]:
    if shutil.which("nvidia-smi") is None:
        return 0, ["nvidia-smi not found"]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return 0, [result.stderr.strip() or "nvidia-smi failed"]
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(lines), lines


def local_dataset_ok(path: str | None) -> bool:
    if not path or not path.endswith((".json", ".jsonl")):
        return True
    return Path(path).exists()


def package_to_module(package: str) -> str:
    return {"math-verify": "math_verify"}.get(package, package.replace("-", "_"))


def parse_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^#\s]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for PCT experiment runs.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--require_vllm", action="store_true")
    parser.add_argument("--min_gpus", type=int, default=1)
    parser.add_argument("--requirements", default="requirements-pct.txt")
    parser.add_argument("--strict_versions", action="store_true")
    parser.add_argument("--require_all_requirements", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    gpu_count, gpus = gpu_summary()
    requirements = parse_requirements(Path(args.requirements))
    if not requirements:
        requirements = {package: "" for package in CORE_PACKAGES}
    versions = {package: installed_version(package) for package in requirements}
    required_packages = requirements if args.require_all_requirements else {
        package: requirements.get(package, "") for package in CORE_PACKAGES
    }
    checks = {
        "python_modules": {
            package_to_module(name): has_module(package_to_module(name))
            for name in required_packages
        },
        "python_package_versions": versions,
        "expected_versions": requirements,
        "vllm": has_module("vllm"),
        "gpu_count": gpu_count,
        "gpus": gpus,
        "dataset_path_exists": local_dataset_ok(args.dataset),
    }
    failures = []
    missing = [name for name, ok in checks["python_modules"].items() if not ok]
    if missing:
        failures.append(f"missing python modules: {', '.join(missing)}")
    if args.require_vllm and not checks["vllm"]:
        failures.append("vllm is required but not installed")
    if args.strict_versions:
        mismatched = [
            f"{package}:installed={installed},expected={expected}"
            for package, expected in requirements.items()
            if expected and (installed := versions.get(package)) != expected
        ]
        if mismatched:
            failures.append("version mismatches: " + "; ".join(mismatched))
    if gpu_count < args.min_gpus:
        failures.append(f"need at least {args.min_gpus} GPU(s), found {gpu_count}")
    if not checks["dataset_path_exists"]:
        failures.append(f"dataset path does not exist: {args.dataset}")

    checks["ok"] = not failures
    checks["failures"] = failures
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
