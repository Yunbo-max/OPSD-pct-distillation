#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("paired_results")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--max_solution_chars", type=int, default=6000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    metrics = {}
    for row in map(json.loads, Path(args.paired_results).open()):
        value = row["layers"][str(args.layer)]["dependency_swap:alpha=1"]
        metrics[str(row["id"])] = value["paired_logp_gap"]
    eligible = [
        row for row in rows
        if str(row["id"]) in metrics and len(row["solution"]) <= args.max_solution_chars
    ]
    eligible.sort(key=lambda row: metrics[str(row["id"])], reverse=True)
    selected = eligible[: args.n]
    with Path(args.out).open("w") as handle:
        for row in selected:
            row = dict(row)
            row["teacher_forced_te"] = metrics[str(row["id"])]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"selected={len(selected)} top_te={[round(metrics[str(row['id'])], 4) for row in selected]}")


if __name__ == "__main__":
    main()
