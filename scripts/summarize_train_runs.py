#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "step",
    "loss",
    "on_policy_loss",
    "off_policy_loss",
    "pct_loss",
    "pct_transport_mass",
    "grad_norm",
    "learning_rate",
    "epoch",
)


def find_state_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("**/trainer_state.json")))
        else:
            files.append(path)
    return files


def last_metric_record(state: dict[str, Any]) -> dict[str, Any]:
    for record in reversed(state.get("log_history", [])):
        if any(field in record for field in FIELDS):
            return record
    return {}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OPSD/PCT Trainer state files.")
    parser.add_argument("paths", nargs="+", help="Run directories or trainer_state.json files.")
    parser.add_argument("--tsv", default=None, help="Optional path to write a TSV copy.")
    args = parser.parse_args()

    rows = []
    for state_file in find_state_files(args.paths):
        with state_file.open("r", encoding="utf-8") as f:
            state = json.load(f)
        run_dir = state_file.parent
        if run_dir.name.startswith("checkpoint-"):
            run_dir = run_dir.parent
        record = last_metric_record(state)
        rows.append({"run": run_dir.name, "state": str(state_file), **{k: record.get(k) for k in FIELDS}})

    header = ("run", *FIELDS, "state")
    lines = ["\t".join(header)]
    for row in sorted(rows, key=lambda x: x["run"]):
        lines.append("\t".join(fmt(row.get(col)) for col in header))
    output = "\n".join(lines)
    print(output)

    if args.tsv:
        out = Path(args.tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
