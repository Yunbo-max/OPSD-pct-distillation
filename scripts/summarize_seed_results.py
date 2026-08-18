#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_eval_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run_name_from_eval_file(name: str, dataset: str, val_n: str) -> str:
    suffix = f"_{dataset}_avg{val_n}.json"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate eval summaries across manifest seeds.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval_tsv", required=True)
    parser.add_argument("--out_tsv", default=None)
    args = parser.parse_args()

    records = {record["run_name"]: record for record in read_manifest(Path(args.manifest))}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    seed_counts: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in read_eval_tsv(Path(args.eval_tsv)):
        dataset = str(row.get("dataset", ""))
        val_n = str(row.get("N", row.get("val_n", "12")))
        run_name = run_name_from_eval_file(str(row.get("file", "")), dataset, val_n)
        record = records.get(run_name)
        if not record:
            continue
        value_raw = row.get("Average@N", "NA")
        if value_raw == "NA":
            continue
        key = (str(record.get("method", "")), dataset)
        grouped[key].append(float(value_raw))
        seed_counts[key].add(str(record.get("seed", "NA")))

    header = ["method", "dataset", "seeds", "mean_average_at_n", "std_average_at_n"]
    lines = ["\t".join(header)]
    for key in sorted(grouped):
        values = grouped[key]
        std = stdev(values) if len(values) > 1 else 0.0
        row = {
            "method": key[0],
            "dataset": key[1],
            "seeds": ",".join(sorted(seed_counts[key])),
            "mean_average_at_n": fmt(mean(values)),
            "std_average_at_n": fmt(std),
        }
        lines.append("\t".join(row[col] for col in header))

    output = "\n".join(lines)
    print(output)
    if args.out_tsv:
        out = Path(args.out_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
