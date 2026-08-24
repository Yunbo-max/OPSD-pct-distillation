#!/usr/bin/env python3
"""Build a deterministic short-solution OPSD-Math subset for the RCIP premise gate."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--min_reference_tokens", type=int, default=1)
    parser.add_argument("--max_reference_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    eligible = [
        (index, row) for index, row in enumerate(dataset)
        if row.get("Answer") and row.get("solution")
        and args.min_reference_tokens <= int(row.get("generated_token_count") or 0) <= args.max_reference_tokens
    ]
    rng = random.Random(args.seed); rng.shuffle(eligible)
    selected = eligible[: args.n]
    if len(selected) < args.n:
        raise RuntimeError(f"only {len(selected)} eligible rows")
    destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for source_index, row in selected:
            record = {
                "id": f"opsd-{source_index}", "source_index": source_index,
                "problem": row["problem"], "solution": row["solution"],
                "answer": row["Answer"], "references": [row["solution"]],
                "generated_token_count": row["generated_token_count"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(destination), "n": len(selected), "eligible": len(eligible)}))


if __name__ == "__main__":
    main()
