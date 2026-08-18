#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pct.eval_protocol import validate_eval_file


def parse_eval_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=/path/to/eval.json")
    method, path = value.split("=", 1)
    return method, Path(path)


def per_problem_scores(path: Path) -> tuple[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = str(data.get("dataset") or path.stem)
    scores: dict[str, float] = {}
    for idx, row in enumerate(data.get("results", [])):
        key = str(row.get("problem_id", idx))
        val_n = float(row.get("val_n") or data.get("val_n") or 1)
        scores[key] = float(row.get("num_correct", 0)) / val_n
    return dataset, scores


def validate_input(path: Path, expected_num_problems: int, expected_val_n: int | None) -> str:
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
    return dataset


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bootstrap_mean_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = []
    n = len(values)
    for _ in range(n_boot):
        draws.append(mean(values[rng.randrange(n)] for _ in range(n)))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def bootstrap_delta_ci(a: list[float], b: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    draws = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        draws.append(mean(a[i] - b[i] for i in idxs))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Average@N eval matrix with paired method deltas.")
    parser.add_argument("--eval", action="append", type=parse_eval_arg, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict_protocol", action="store_true")
    parser.add_argument("--expected_num_problems", type=int, default=30)
    parser.add_argument("--expected_val_n", type=int, default=None)
    parser.add_argument(
        "--require_paired_problem_ids",
        action="store_true",
        help="Require every method/baseline pair in a dataset to share expected_num_problems problem IDs.",
    )
    parser.add_argument("--tsv", default=None)
    args = parser.parse_args()

    by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for method, path in args.eval:
        if args.strict_protocol:
            validate_input(path, args.expected_num_problems, args.expected_val_n)
        dataset, scores = per_problem_scores(path)
        by_dataset.setdefault(dataset, {})[method] = scores

    header = (
        "dataset",
        "method",
        "problems",
        "average_pct",
        "ci95_low_pct",
        "ci95_high_pct",
        f"delta_vs_{args.baseline}_pct",
        "delta_ci95_low_pct",
        "delta_ci95_high_pct",
    )
    lines = ["\t".join(header)]
    for dataset, method_scores in sorted(by_dataset.items()):
        if args.baseline not in method_scores:
            raise ValueError(f"Missing baseline '{args.baseline}' for dataset '{dataset}'")
        baseline_scores = method_scores[args.baseline]
        for method, scores in sorted(method_scores.items()):
            ids = sorted(set(scores) & set(baseline_scores))
            if args.require_paired_problem_ids and len(ids) != args.expected_num_problems:
                raise ValueError(
                    f"{dataset}:{method}:paired_problem_ids:{len(ids)}!=expected:{args.expected_num_problems}"
                )
            values = [scores[i] for i in ids]
            avg = mean(values) if values else float("nan")
            ci_low, ci_high = bootstrap_mean_ci(values, args.n_boot, args.seed) if values else (float("nan"), float("nan"))
            if method == args.baseline:
                delta = 0.0
                delta_low = 0.0
                delta_high = 0.0
            else:
                baseline_values = [baseline_scores[i] for i in ids]
                deltas = [a - b for a, b in zip(values, baseline_values)]
                delta = mean(deltas) if deltas else float("nan")
                delta_low, delta_high = (
                    bootstrap_delta_ci(values, baseline_values, args.n_boot, args.seed)
                    if deltas
                    else (float("nan"), float("nan"))
                )
            lines.append(
                "\t".join(
                    [
                        dataset,
                        method,
                        str(len(ids)),
                        fmt_pct(avg),
                        fmt_pct(ci_low),
                        fmt_pct(ci_high),
                        fmt_pct(delta),
                        fmt_pct(delta_low),
                        fmt_pct(delta_high),
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
