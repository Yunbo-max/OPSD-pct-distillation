#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for source in args.inputs
        for line in Path(source).read_text().splitlines() if line
    ]
    grouped = defaultdict(list)
    for row in rows:
        for values in row["curves"].values():
            for value in values:
                grouped[value["lag"]].append(value)
    curve = []
    for lag, values in sorted(grouped.items()):
        result = {
            "lag": lag, "n": len(values),
            "alignment_mean": float(np.mean([x["alignment"] for x in values])),
            "recovery_mean": float(np.mean([x["recovery"] for x in values])),
            "projected_recovery_mean": float(np.mean([x["projected_recovery"] for x in values])),
        }
        if "random_alignment" in values[0]:
            for key in ("random_alignment", "random_recovery", "random_projected_recovery"):
                result[f"{key}_mean"] = float(np.mean([x[key] for x in values]))
        curve.append(result)
    # Projection onto the correct target distinguishes persistence from a large
    # but irrelevant downstream perturbation. Fit the initial positive segment.
    usable = []
    for row in curve:
        if row["projected_recovery_mean"] <= 1e-8:
            break
        usable.append((row["lag"], row["projected_recovery_mean"]))
    if len(usable) >= 2:
        slope, intercept = np.polyfit(
            [x for x, _ in usable], np.log([y for _, y in usable]), 1
        )
    else:
        slope, intercept = float("nan"), float("nan")
    report = {
        "n_problems": len(rows), "curve": curve,
        "decay_fit_points": len(usable),
        "exponential_decay_slope": float(slope),
        "persistence_tau": float(-1.0 / slope) if np.isfinite(slope) and slope < 0 else None,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
