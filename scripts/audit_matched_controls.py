#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    failures = []
    similarities = {kind: [] for kind in ("locally_corrupted", "counterfactual_rationale", "dependency_swap")}
    length_ratios = {kind: [] for kind in similarities}
    for row in rows:
        source = row["references"][0]
        for kind in similarities:
            control = row.get("matched_controls", {}).get(kind, "")
            ratio = len(control.split()) / max(1, len(source.split()))
            similarity = SequenceMatcher(None, source, control).ratio()
            similarities[kind].append(similarity)
            length_ratios[kind].append(ratio)
            if len(control.split()) < 20 or not 0.25 <= ratio <= 2.5:
                failures.append({"id": row.get("id"), "kind": kind, "length_ratio": ratio})
    report = {
        "rows": len(rows),
        "failures": failures,
        "metrics": {
            kind: {
                "mean_similarity": sum(similarities[kind]) / max(1, len(rows)),
                "mean_length_ratio": sum(length_ratios[kind]) / max(1, len(rows)),
            }
            for kind in similarities
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
