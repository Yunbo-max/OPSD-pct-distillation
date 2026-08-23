#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    values = defaultdict(list)
    for row in rows:
        for layer, kinds in row["layers"].items():
            for kind, metrics in kinds.items():
                for metric, value in metrics.items():
                    values[(int(layer), kind, metric)].append(float(value))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["layer", "kind", "metric", "mean", "n"])
        for (layer, kind, metric), samples in sorted(values.items()):
            writer.writerow([layer, kind, metric, sum(samples) / len(samples), len(samples)])
    print(f"wrote {output} from {len(rows)} examples")


if __name__ == "__main__":
    main()
