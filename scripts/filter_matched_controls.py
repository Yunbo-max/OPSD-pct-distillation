#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--require", default="locally_corrupted,counterfactual_rationale,dependency_swap")
    args = parser.parse_args()
    required = args.require.split(",")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    selected = [
        row for row in rows
        if all(row.get("matched_controls_valid", {}).get(kind, False) for kind in required)
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"selected={len(selected)}/{len(rows)} required={required}")


if __name__ == "__main__":
    main()
