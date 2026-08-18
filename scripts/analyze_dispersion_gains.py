#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pct.eval_protocol import validate_eval_file


def read_dispersion(path: Path, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if metric not in (reader.fieldnames or []):
            raise ValueError(f"Dispersion metric '{metric}' not found in {path}")
        for row in reader:
            key = str(row.get("id") or row.get("problem_id"))
            if not key:
                continue
            raw = row.get(metric)
            if raw not in (None, ""):
                values[key] = float(raw)
    return values


def per_problem_avg(eval_json: Path) -> dict[str, float]:
    with eval_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scores: dict[str, float] = {}
    for idx, row in enumerate(data.get("results", [])):
        key = str(row.get("problem_id", idx))
        val_n = row.get("val_n") or data.get("val_n") or 1
        scores[key] = float(row.get("num_correct", 0)) / float(val_n)
    return scores


def parse_eval_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=/path/to/eval.json")
    method, path = value.split("=", 1)
    return method, Path(path)


def validate_input(path: Path, expected_num_problems: int, expected_val_n: int | None) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = str(data.get("dataset") or path.stem)
    val_n = int(expected_val_n or data.get("val_n", 12))
    failures = validate_eval_file(
        path,
        expected_dataset=dataset,
        expected_val_n=val_n,
        expected_num_problems=expected_num_problems,
        require_generations=True,
    )
    if failures:
        raise ValueError(f"{path}: {'|'.join(failures)}")


def quantile_threshold(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("No values for quantile")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def group_ids(dispersion: dict[str, float], low_q: float, high_q: float) -> dict[str, set[str]]:
    vals = list(dispersion.values())
    low_thr = quantile_threshold(vals, low_q)
    high_thr = quantile_threshold(vals, high_q)
    return {
        "all": set(dispersion),
        "low": {k for k, v in dispersion.items() if v <= low_thr},
        "high": {k for k, v in dispersion.items() if v >= high_thr},
    }


def avg_for_ids(scores: dict[str, float], ids: set[str]) -> float | None:
    vals = [scores[k] for k in ids if k in scores]
    if not vals:
        return None
    return mean(vals)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{100.0 * value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join diagnostic dispersion with per-problem evals and report low/high gains."
    )
    parser.add_argument("--diagnostic_csv", required=True)
    parser.add_argument("--dispersion_metric", default="sinkhorn_correct_correct")
    parser.add_argument("--eval", action="append", type=parse_eval_arg, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--low_quantile", type=float, default=0.33)
    parser.add_argument("--high_quantile", type=float, default=0.67)
    parser.add_argument("--strict_protocol", action="store_true")
    parser.add_argument("--expected_num_problems", type=int, default=30)
    parser.add_argument("--expected_val_n", type=int, default=None)
    parser.add_argument(
        "--min_matched_per_group",
        type=int,
        default=1,
        help="Minimum eval/diagnostic problem-ID overlap required for every method in each dispersion group.",
    )
    parser.add_argument("--tsv", default=None)
    args = parser.parse_args()

    dispersion = read_dispersion(Path(args.diagnostic_csv), args.dispersion_metric)
    groups = group_ids(dispersion, args.low_quantile, args.high_quantile)
    if args.strict_protocol:
        for _, path in args.eval:
            validate_input(path, args.expected_num_problems, args.expected_val_n)
    method_scores = {method: per_problem_avg(path) for method, path in args.eval}
    if args.baseline not in method_scores:
        raise ValueError(f"Baseline '{args.baseline}' not present in --eval entries")

    header = ("group", "method", "average_pct", f"gain_vs_{args.baseline}_pct", "matched_problems")
    lines = ["\t".join(header)]
    for group_name, ids in groups.items():
        baseline_avg = avg_for_ids(method_scores[args.baseline], ids)
        for method, scores in sorted(method_scores.items()):
            avg = avg_for_ids(scores, ids)
            matched = len([k for k in ids if k in scores])
            if args.strict_protocol and matched < args.min_matched_per_group:
                raise ValueError(
                    f"{group_name}:{method}:matched_problems:{matched}<min:{args.min_matched_per_group}"
                )
            gain = None if avg is None or baseline_avg is None else avg - baseline_avg
            lines.append("\t".join([group_name, method, fmt(avg), fmt(gain), str(matched)]))

    output = "\n".join(lines)
    print(output)
    if args.tsv:
        out = Path(args.tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
