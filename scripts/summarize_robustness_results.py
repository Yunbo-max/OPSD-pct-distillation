#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_eval_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=/path/to/eval.json")
    method, path = value.split("=", 1)
    return method, Path(path)


def perturbation_name(row: dict) -> str:
    meta = row.get("metadata") or {}
    for key in ("perturbation", "perturbation_type", "transform", "variant"):
        if meta.get(key):
            return str(meta[key])
        if row.get(key):
            return str(row[key])
    return "clean"


def score(row: dict, default_val_n: int) -> float:
    val_n = float(row.get("val_n") or default_val_n or 1)
    return float(row.get("num_correct", 0)) / val_n


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize perturbation-group robustness from eval JSON files.")
    parser.add_argument("--eval", action="append", type=parse_eval_arg, required=True)
    parser.add_argument("--clean_name", default="clean")
    parser.add_argument("--tsv", default=None)
    args = parser.parse_args()

    rows = []
    for method, path in args.eval:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = str(data.get("dataset") or path.stem)
        default_val_n = int(data.get("val_n") or 1)
        by_perturbation: dict[str, list[float]] = {}
        for result in data.get("results", []):
            by_perturbation.setdefault(perturbation_name(result), []).append(score(result, default_val_n))
        clean_avg = mean(by_perturbation.get(args.clean_name, [])) if by_perturbation.get(args.clean_name) else None
        for name, vals in sorted(by_perturbation.items()):
            avg = mean(vals)
            delta = None if clean_avg is None else avg - clean_avg
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "perturbation": name,
                    "problems": len(vals),
                    "average_pct": 100.0 * avg,
                    "delta_vs_clean_pct": None if delta is None else 100.0 * delta,
                }
            )

    header = ("dataset", "method", "perturbation", "problems", "average_pct", "delta_vs_clean_pct")
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    str(row["perturbation"]),
                    str(row["problems"]),
                    f"{row['average_pct']:.2f}",
                    "NA" if row["delta_vs_clean_pct"] is None else f"{row['delta_vs_clean_pct']:.2f}",
                ]
            )
        )
    output = "\n".join(lines)
    print(output)
    if args.tsv:
        out = Path(args.tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
