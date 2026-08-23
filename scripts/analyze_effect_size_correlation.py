#!/usr/bin/env python3
"""Relate per-example behavioral damage to paired rescue and necessity effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    return float((left @ right) / np.sqrt((left @ left) * (right @ right)))


def design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categories = sorted({row["category"] for row in rows})
    # Intercept, TE, and C-1 category indicators.
    x = np.ones((len(rows), 2 + len(categories) - 1), dtype=np.float64)
    x[:, 1] = [row["te"] for row in rows]
    for column, category in enumerate(categories[1:], start=2):
        x[:, column] = [float(row["category"] == category) for row in rows]
    rescue = np.asarray([row["rescue"] for row in rows])
    necessity = np.asarray([row["necessity_magnitude"] for row in rows])
    return x, rescue, necessity


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.lstsq(x, y, rcond=None)[0][1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()
    source = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    layers = sorted({int(layer) for row in source for layer in row["layers"]})
    rng = np.random.default_rng(args.seed)
    report = {"n_problems": len(source), "layers": {}}

    for layer in layers:
        observations = []
        for problem_index, row in enumerate(source):
            for condition, metrics in row["layers"][str(layer)].items():
                category, alpha = condition.split(":alpha=")
                if alpha != "1":
                    continue
                observations.append(
                    {
                        "problem": problem_index,
                        "category": category,
                        "te": metrics["paired_logp_gap"],
                        "rescue": metrics["rescue_gain"],
                        "necessity_magnitude": -metrics["necessity_change"],
                    }
                )
        x, rescue, necessity = design(observations)
        te = x[:, 1]
        problem_ids = np.asarray([row["problem"] for row in observations])
        rescue_slopes, necessity_slopes = [], []
        for _ in range(args.bootstrap):
            sampled = rng.integers(0, len(source), size=len(source))
            indices = np.concatenate([np.flatnonzero(problem_ids == index) for index in sampled])
            rescue_slopes.append(slope(x[indices], rescue[indices]))
            necessity_slopes.append(slope(x[indices], necessity[indices]))
        report["layers"][str(layer)] = {
            "n_pairs": len(observations),
            "pearson_te_rescue": correlation(te, rescue),
            "spearman_te_rescue": correlation(rankdata(te), rankdata(rescue)),
            "pearson_te_necessity": correlation(te, necessity),
            "spearman_te_necessity": correlation(rankdata(te), rankdata(necessity)),
            "category_adjusted_beta_te_rescue": slope(x, rescue),
            "category_adjusted_beta_te_rescue_ci": list(
                map(float, np.quantile(rescue_slopes, [0.025, 0.975]))
            ),
            "category_adjusted_beta_te_necessity": slope(x, necessity),
            "category_adjusted_beta_te_necessity_ci": list(
                map(float, np.quantile(necessity_slopes, [0.025, 0.975]))
            ),
        }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
