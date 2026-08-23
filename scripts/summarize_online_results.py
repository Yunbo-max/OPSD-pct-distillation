#!/usr/bin/env python3
"""Problem-level bootstrap summary for row-oriented online interventions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .run_online_paired_residual import answers_equivalent
except ImportError:
    from run_online_paired_residual import answers_equivalent


def interval(values: np.ndarray, indices: np.ndarray) -> dict:
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    if not rows:
        Path(args.out).write_text(json.dumps({"n": 0, "results": {}}, indent=2) + "\n")
        return

    baseline_correct, baseline_corrupt = [], []
    arm_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        target = row["answer"]
        correct = float(answers_equivalent(row.get("correct_reference_boxed"), target))
        corrupt = float(answers_equivalent(row.get("corrupt_baseline_boxed"), target))
        baseline_correct.append(correct)
        baseline_corrupt.append(corrupt)
        for key, arm in row.get("arms", {}).items():
            arm_values[key].append(float(answers_equivalent(arm.get("boxed"), target)))

    correct = np.asarray(baseline_correct)
    corrupt = np.asarray(baseline_corrupt)
    n = len(rows)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, n, size=(args.bootstrap, n))
    damage = correct - corrupt
    report = {
        "n": n,
        "baseline": {
            "correct_accuracy": interval(correct, indices),
            "corrupt_accuracy": interval(corrupt, indices),
            "total_effect": interval(damage, indices),
            "n_damaged": int(((correct == 1) & (corrupt == 0)).sum()),
            "n_reverse": int(((correct == 0) & (corrupt == 1)).sum()),
        },
        "results": {},
    }
    for key, raw in sorted(arm_values.items()):
        if len(raw) != n:
            report["results"][key] = {"n": len(raw), "incomplete": True}
            continue
        values = np.asarray(raw)
        gain = values - corrupt
        # Aggregate recovery is a ratio of mean effects. Per-example binary RF
        # is undefined whenever that example's total effect is zero.
        boot_denominator = damage[indices].mean(axis=1)
        boot_numerator = gain[indices].mean(axis=1)
        valid = np.abs(boot_denominator) > 1e-12
        recovery_boot = boot_numerator[valid] / boot_denominator[valid]
        recovery = float(gain.mean() / damage.mean()) if abs(damage.mean()) > 1e-12 else None
        report["results"][key] = {
            "n": n,
            "boxed_math_accuracy": interval(values, indices),
            "gain_over_corrupt": interval(gain, indices),
            "aggregate_recovery_fraction": {
                "mean": recovery,
                "ci": [float(x) for x in np.quantile(recovery_boot, [0.025, 0.975])]
                if len(recovery_boot) else None,
            },
        }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n": n, "arms": len(arm_values)}))


if __name__ == "__main__":
    main()
