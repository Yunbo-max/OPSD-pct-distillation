#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_status(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def compact_ranges(indexes: list[int]) -> str:
    if not indexes:
        return "EMPTY"
    ordered = sorted(set(indexes))
    ranges = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = idx
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def pending_by_task(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {"train": [], "eval": []}
    for row in rows:
        task = row.get("task", "")
        if task in grouped and row.get("status") != "complete":
            grouped[task].append(row)
    return grouped


def build_plan(rows: list[dict[str, str]], manifest: str, eval_root: str, python: str) -> dict:
    grouped = pending_by_task(rows)
    train_indexes = [int(row["index"]) for row in grouped["train"]]
    eval_indexes = [int(row["index"]) for row in grouped["eval"]]
    train_array = compact_ranges(train_indexes)
    eval_array = compact_ranges(eval_indexes)
    return {
        "pending_counts": {"train": len(train_indexes), "eval": len(eval_indexes)},
        "pending_indexes": {"train": train_indexes, "eval": eval_indexes},
        "slurm_arrays": {"train": train_array, "eval": eval_array},
        "commands": {
            "train": None
            if not train_indexes
            else (
                f"{python} scripts/run_pct_train_from_manifest.py --manifest {manifest} "
                f"--index <TRAIN_INDEX> --skip_completed --require_train_metrics"
            ),
            "eval": None
            if not eval_indexes
            else (
                f"{python} scripts/run_pct_eval_from_manifest.py --manifest {manifest} "
                f"--eval_root {eval_root} --index <EVAL_INDEX> --skip_completed"
            ),
            "slurm_train": None
            if not train_indexes
            else f"sbatch --array={train_array} scripts/slurm/pct_21_train_manifest_array.sbatch",
            "slurm_eval": None
            if not eval_indexes
            else f"sbatch --array={eval_array} scripts/slurm/pct_61_eval_manifest_array.sbatch",
        },
        "pending_tasks": grouped,
    }


def markdown(plan: dict) -> str:
    lines = ["# PCT Resume Plan", ""]
    for task in ("train", "eval"):
        lines.append(f"## {task}")
        lines.append("")
        lines.append(f"- Pending count: `{plan['pending_counts'][task]}`")
        lines.append(f"- SLURM array: `{plan['slurm_arrays'][task]}`")
        command = plan["commands"][task]
        slurm_command = plan["commands"][f"slurm_{task}"]
        if command:
            lines.append(f"- Single-task command: `{command}`")
        if slurm_command:
            lines.append(f"- SLURM command: `{slurm_command}`")
        lines.append("")
        tasks = plan["pending_tasks"][task]
        if tasks:
            lines.append("| index | run_name | method | dataset | missing |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in tasks:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row.get("index", ""),
                            row.get("run_name", ""),
                            row.get("method", ""),
                            row.get("dataset", ""),
                            row.get("missing", ""),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resume plan from manifest status TSV output.")
    parser.add_argument("--status_tsv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--json_out", default=None)
    parser.add_argument("--md_out", default=None)
    args = parser.parse_args()

    plan = build_plan(read_status(Path(args.status_tsv)), args.manifest, args.eval_root, args.python)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown(plan) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
