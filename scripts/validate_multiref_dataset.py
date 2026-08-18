#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


REFERENCE_KEYS = ("references", "reference_solutions", "solutions")


def get_references(row: dict[str, Any]) -> list[str]:
    refs = None
    for key in REFERENCE_KEYS:
        value = row.get(key)
        if value:
            refs = value
            break
    if refs is None:
        refs = [row.get("solution")] if row.get("solution") else []
    if isinstance(refs, str):
        refs = [refs]
    return [str(ref).strip() for ref in refs if str(ref).strip()]


def normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def load_rows(path_or_name: str, split: str) -> list[dict[str, Any]]:
    if path_or_name.endswith((".json", ".jsonl")):
        return [dict(row) for row in load_dataset("json", data_files=path_or_name)["train"]]
    return [dict(row) for row in load_dataset(path_or_name, split=split)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a multi-reference OPSD/PCT dataset.")
    parser.add_argument("--dataset", required=True, help="HF dataset name or local .json/.jsonl file.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min_refs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write_subset", default=None, help="Optional JSONL path for first valid rows.")
    parser.add_argument("--subset_size", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.split)
    if args.limit > 0:
        rows = rows[: args.limit]

    valid_rows = []
    missing_problem = 0
    missing_solution = 0
    too_few_refs = 0
    duplicate_ref_rows = 0
    ref_counts: dict[int, int] = {}

    for row in rows:
        refs = get_references(row)
        ref_counts[len(refs)] = ref_counts.get(len(refs), 0) + 1
        unique_refs = {normalized(ref) for ref in refs}

        bad = False
        if not row.get("problem"):
            missing_problem += 1
            bad = True
        if not row.get("solution") and not refs:
            missing_solution += 1
            bad = True
        if len(refs) < args.min_refs:
            too_few_refs += 1
            bad = True
        if len(unique_refs) < min(len(refs), args.min_refs):
            duplicate_ref_rows += 1
            bad = True

        if not bad:
            row["references"] = refs
            valid_rows.append(row)

    total = len(rows)
    print(f"rows\t{total}")
    print(f"valid_rows\t{len(valid_rows)}")
    print(f"missing_problem\t{missing_problem}")
    print(f"missing_solution\t{missing_solution}")
    print(f"too_few_refs\t{too_few_refs}")
    print(f"duplicate_ref_rows\t{duplicate_ref_rows}")
    for count in sorted(ref_counts):
        print(f"ref_count_{count}\t{ref_counts[count]}")

    if args.write_subset:
        n = args.subset_size if args.subset_size > 0 else len(valid_rows)
        out = Path(args.write_subset)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in valid_rows[:n]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote_subset\t{out}\t{min(n, len(valid_rows))}")

    if len(valid_rows) == 0 or too_few_refs > 0 or duplicate_ref_rows > 0 or missing_problem > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
