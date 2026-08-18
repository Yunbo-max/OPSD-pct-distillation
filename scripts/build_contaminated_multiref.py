#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def shuffled_reference(reference: str, rng: random.Random) -> str:
    chunks = [chunk.strip() for chunk in reference.splitlines() if chunk.strip()]
    if len(chunks) < 2:
        chunks = [part.strip() for part in reference.split(".") if part.strip()]
    rng.shuffle(chunks)
    return "\n".join(chunks)


def replacement_for(row: dict[str, Any], mode: str, rng: random.Random) -> str:
    if mode == "wrong" and row.get("wrong_reference"):
        return str(row["wrong_reference"])
    if mode == "shuffled" and row.get("shuffled_reference"):
        return str(row["shuffled_reference"])
    refs = row.get("references") or [row.get("solution", "")]
    if isinstance(refs, str):
        refs = [refs]
    return shuffled_reference(str(refs[0]), rng)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rho-contaminated multi-reference JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--mode", choices=["wrong", "shuffled", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 <= args.rho <= 1.0:
        raise ValueError("--rho must be in [0, 1]")

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    contaminated = 0
    with Path(args.input).open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            refs = row.get("references") or [row.get("solution")]
            if isinstance(refs, str):
                refs = [refs]
            refs = [str(ref) for ref in refs if ref]
            if refs and rng.random() < args.rho:
                mode = rng.choice(["wrong", "shuffled"]) if args.mode == "mixed" else args.mode
                replace_idx = rng.randrange(len(refs))
                refs[replace_idx] = replacement_for(row, mode, rng)
                row["contamination"] = {"rho": args.rho, "mode": mode, "reference_index": replace_idx}
                contaminated += 1
            row["references"] = refs
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"rows\t{total}")
    print(f"contaminated_rows\t{contaminated}")
    print(f"output\t{out}")


if __name__ == "__main__":
    main()
