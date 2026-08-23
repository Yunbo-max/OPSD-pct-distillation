#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .run_online_paired_residual import answers_equivalent
except ImportError:
    from run_online_paired_residual import answers_equivalent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    baseline = next(record for record in records if record["key"] == "baseline")
    arms = [record for record in records if record["key"] != "baseline"]
    expected_keys = {
        f"layer={layer}:alpha={alpha:g}:schedule={schedule}:mode={mode}"
        for layer in (13, 21, 23, 25) for alpha in (0.25, 0.5, 0.75, 1.0)
        for schedule in ("every1", "every2", "every4", "every8", "first8")
        for mode in ("rescue", "reverse", "random")
    }
    observed_keys = {record["key"] for record in arms}
    grouped = defaultdict(list)
    for record in arms:
        strict = float(record["exact_match"])
        mathematical = float(answers_equivalent(record.get("boxed"), record["answer"]))
        grouped[(record["layer"], record["alpha"], record["schedule"], record["mode"])].append(
            (strict, mathematical)
        )
    corrupt_score = float(answers_equivalent(baseline["corrupt"].get("boxed"), baseline["answer"]))
    correct_score = float(answers_equivalent(baseline["correct"].get("boxed"), baseline["answer"]))
    total_effect = correct_score - corrupt_score
    rows = []
    for key, values in sorted(grouped.items()):
        strict_score = sum(value[0] for value in values) / len(values)
        score = sum(value[1] for value in values) / len(values)
        recovery = (score - corrupt_score) / total_effect if total_effect > 0 else None
        rows.append(
            {
                "layer": key[0], "alpha": key[1], "schedule": key[2], "mode": key[3],
                "boxed_math_accuracy": score, "strict_string_accuracy": strict_score,
                "recovery_fraction": recovery, "n": len(values),
            }
        )
    report = {
        "baseline": baseline,
        "baseline_boxed_math_accuracy": {"correct": correct_score, "corrupt": corrupt_score},
        "completed_arms": len(arms),
        "expected_arms": 240,
        "complete": observed_keys == expected_keys,
        "missing_keys": sorted(expected_keys - observed_keys),
        "unexpected_keys": sorted(observed_keys - expected_keys),
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
