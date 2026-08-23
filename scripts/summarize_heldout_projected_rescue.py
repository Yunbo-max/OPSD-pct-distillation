#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    values = defaultdict(list)
    for row in rows:
        for arm, metrics in row["arms"].items():
            for metric, value in metrics.items():
                values[(arm, metric)].append(float(value))
    rng = np.random.default_rng(args.seed)
    report = {"n": len(rows), "arms": {}}
    for (arm, metric), samples in values.items():
        array = np.asarray(samples)
        boot = array[rng.integers(0, len(array), size=(args.bootstrap, len(array)))].mean(1)
        report["arms"].setdefault(arm, {})[metric] = {
            "mean": float(array.mean()),
            "ci": list(map(float, np.quantile(boot, [0.025, 0.975]))),
        }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
