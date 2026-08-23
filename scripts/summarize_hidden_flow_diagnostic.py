#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DISTANCES = (
    "euclidean",
    "phf_cosine",
    "sinkhorn",
    "fgw",
    "uot_objective",
    "uot_transport_cost_raw",
    "uot_transport_cost_normalized",
)


def auc(positive: list[float], negative: list[float], *, higher_is_positive: bool = False) -> float:
    if not positive or not negative:
        return float("nan")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos == neg:
                wins += 0.5
            elif (pos > neg) if higher_is_positive else (pos < neg):
                wins += 1.0
    return wins / (len(positive) * len(negative))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize hidden-flow separation diagnostics.")
    parser.add_argument("diagnostic")
    parser.add_argument("--summary_tsv", required=True)
    parser.add_argument("--per_problem_csv", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.diagnostic).read_text(encoding="utf-8").splitlines() if line]
    per_problem = []
    all_groups: dict[str, dict[str, list[float]]] = {}
    metrics = (*DISTANCES, "uot_mass")
    for record in records:
        groups = {
            metric: {group: [] for group in ("correct_correct", "correct_wrong", "correct_shuffled")}
            for metric in metrics
        }
        for pair in record["pairs"]:
            for metric in metrics:
                groups[metric][pair["group"]].append(float(pair[metric]))
                all_groups.setdefault(metric, {}).setdefault(pair["group"], []).append(float(pair[metric]))
        row: dict[str, object] = {"id": record["id"]}
        for metric in metrics:
            for group, values in groups[metric].items():
                row[f"{metric}_{group}_mean"] = sum(values) / len(values) if values else ""
            row[f"{metric}_wrong_auc"] = auc(
                groups[metric]["correct_correct"],
                groups[metric]["correct_wrong"],
                higher_is_positive=metric == "uot_mass",
            )
            row[f"{metric}_shuffled_auc"] = auc(
                groups[metric]["correct_correct"],
                groups[metric]["correct_shuffled"],
                higher_is_positive=metric == "uot_mass",
            )
        per_problem.append(row)

    summary_rows = []
    rng = random.Random(args.seed)
    for metric in metrics:
        positives = all_groups[metric]["correct_correct"]
        for negative_group in ("correct_wrong", "correct_shuffled"):
            negatives = all_groups[metric][negative_group]
            auc_key = f"{metric}_{'wrong' if negative_group == 'correct_wrong' else 'shuffled'}_auc"
            problem_aurocs = [float(row[auc_key]) for row in per_problem]
            bootstrap_means = sorted(
                sum(problem_aurocs[rng.randrange(len(problem_aurocs))] for _ in problem_aurocs)
                / len(problem_aurocs)
                for _ in range(args.bootstrap_samples)
            )
            low_idx = int(0.025 * (len(bootstrap_means) - 1))
            high_idx = int(0.975 * (len(bootstrap_means) - 1))
            summary_rows.append(
                {
                    "metric": metric,
                    "contrast": f"correct_correct_vs_{negative_group}",
                    "positive_mean": sum(positives) / len(positives),
                    "negative_mean": sum(negatives) / len(negatives),
                    "auroc": auc(positives, negatives, higher_is_positive=metric == "uot_mass"),
                    "macro_problem_auroc": sum(problem_aurocs) / len(problem_aurocs),
                    "macro_auroc_ci_low": bootstrap_means[low_idx],
                    "macro_auroc_ci_high": bootstrap_means[high_idx],
                    "n_positive": len(positives),
                    "n_negative": len(negatives),
                }
            )

    Path(args.summary_tsv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary_tsv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    with Path(args.per_problem_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_problem[0]))
        writer.writeheader()
        writer.writerows(per_problem)
    for row in summary_rows:
        print(
            f"{row['metric']}\t{row['contrast']}\t"
            f"pos={row['positive_mean']:.6f}\tneg={row['negative_mean']:.6f}\t"
            f"pooled_auroc={row['auroc']:.3f}\tmacro_auroc={row['macro_problem_auroc']:.3f}"
            f"\t95%CI=[{row['macro_auroc_ci_low']:.3f},{row['macro_auroc_ci_high']:.3f}]"
        )


if __name__ == "__main__":
    main()
