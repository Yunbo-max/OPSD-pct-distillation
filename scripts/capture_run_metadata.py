#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "vllm",
    "math-verify",
    "sentencepiece",
    "tiktoken",
    "wandb",
)

ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NCCL_P2P_DISABLE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "WANDB_MODE",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
)


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    return result.returncode, (result.stdout.strip() or result.stderr.strip())


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def gpu_info(cwd: Path) -> list[str]:
    code, output = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], cwd)
    if code != 0:
        return [output or "nvidia-smi unavailable"]
    return [line.strip() for line in output.splitlines() if line.strip()]


def git_info(cwd: Path) -> dict[str, str | bool | None]:
    inside_code, inside = run(["git", "rev-parse", "--is-inside-work-tree"], cwd)
    if inside_code != 0 or inside != "true":
        return {"inside_work_tree": False, "commit": None, "dirty": None, "status_short": None}
    _, commit = run(["git", "rev-parse", "HEAD"], cwd)
    _, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    _, status = run(["git", "status", "--short"], cwd)
    return {
        "inside_work_tree": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture reproducibility metadata for PCT experiments.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    cwd = Path.cwd()
    record = {
        "cwd": str(cwd),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(cwd),
        "git": git_info(cwd),
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "notes": args.notes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote\t{out}")


if __name__ == "__main__":
    main()
