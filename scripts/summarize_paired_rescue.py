#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    rows = [
        json.loads(line) for source in args.inputs
        for line in Path(source).read_text().splitlines() if line
    ]
    values = defaultdict(list)
    for row in rows:
        for layer, interventions in row["layers"].items():
            for condition, metrics in interventions.items():
                for metric, value in metrics.items():
                    values[(int(layer), condition, metric)].append(float(value))
    rng = np.random.default_rng(args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["layer", "condition", "metric", "mean", "ci_low", "ci_high", "n"])
        for (layer, condition, metric), samples in sorted(values.items()):
            array = np.asarray(samples)
            indices = rng.integers(0, len(array), size=(args.bootstrap, len(array)))
            boot = array[indices].mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            writer.writerow([layer, condition, metric, array.mean(), low, high, len(array)])
    print(f"wrote {output} from {len(rows)} examples")


if __name__ == "__main__":
    main()
