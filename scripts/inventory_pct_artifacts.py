#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PATTERNS = {
    "metadata": ["run_metadata.json", "preflight.json"],
    "dataset_provenance": ["**/*dataset_provenance*.json", "**/*provenance*.json"],
    "manifests": ["**/manifest*.jsonl", "**/*manifest*.jsonl"],
    "resume_plans": ["**/resume_plan.json", "**/resume_plan.md"],
    "train_states": ["**/trainer_state.json"],
    "eval_json": ["**/*_avg12.json", "**/*_avg*.json"],
    "tables": ["**/*.tsv", "**/*.csv", "**/tables.tex", "**/report.md"],
    "figures": ["**/*.svg", "**/*.pdf", "**/*.png"],
    "artifact_checksums": ["**/*checksums*.jsonl", "**/*checksums*.json"],
}


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def unique_sorted(paths: list[Path]) -> list[Path]:
    return sorted(set(paths), key=lambda p: str(p))


def collect(root: Path) -> dict[str, list[str]]:
    inventory: dict[str, list[str]] = {}
    for name, patterns in PATTERNS.items():
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(root.glob(pattern))
        inventory[name] = [rel(path, root) for path in unique_sorted(paths) if path.is_file()]
    return inventory


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_records(root: Path, inventory: dict[str, list[str]]) -> list[dict[str, str | int]]:
    records = []
    seen: set[str] = set()
    for group, files in inventory.items():
        if group == "artifact_checksums":
            continue
        for name in files:
            if name in seen:
                continue
            seen.add(name)
            path = root / name
            if not path.is_file():
                continue
            records.append(
                {
                    "path": name,
                    "group": group,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sorted(records, key=lambda row: str(row["path"]))


def missing_required(inventory: dict[str, list[str]]) -> list[str]:
    missing = []
    for key in ("metadata", "dataset_provenance", "manifests", "train_states", "eval_json", "tables"):
        if not inventory.get(key):
            missing.append(key)
    return missing


def markdown(inventory: dict[str, list[str]], missing: list[str]) -> str:
    lines = ["# PCT Artifact Inventory", ""]
    lines.append("Missing required groups: " + (", ".join(missing) if missing else "none"))
    for group, files in inventory.items():
        lines += ["", f"## {group}", ""]
        if not files:
            lines.append("_No files found._")
        else:
            lines.extend(f"- `{path}`" for path in files)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory PCT experiment artifacts under an output root.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--json_out", default=None)
    parser.add_argument("--md_out", default=None)
    parser.add_argument("--checksums_out", default=None)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    inventory = collect(root)
    missing = missing_required(inventory)
    payload = {"root": str(root), "missing_required_groups": missing, "inventory": inventory}
    print(json.dumps(payload, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown(inventory, missing), encoding="utf-8")
    if args.checksums_out:
        out = Path(args.checksums_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = checksum_records(root, inventory)
        out.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    if args.require_complete and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
