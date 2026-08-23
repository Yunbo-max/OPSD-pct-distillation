#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, compact(left), compact(right)).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated multi-reference diagnostic candidates.")
    parser.add_argument("dataset")
    parser.add_argument("--json", required=True)
    parser.add_argument("--max_pair_similarity", type=float, default=0.8)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for record in records:
        refs = record.get("references", [])
        valid = record.get("reference_valid", [])
        pairwise = [
            similarity(refs[i], refs[j])
            for i in range(len(refs))
            for j in range(i + 1, len(refs))
        ]
        checks = {
            "four_references": len(refs) >= 4,
            "four_answer_checks": len(valid) >= 4 and all(valid[:4]),
            "unique_references": len({compact(ref) for ref in refs[:4]}) == 4,
            "lexically_distinct": bool(pairwise) and max(pairwise) < args.max_pair_similarity,
            "wrong_reference_present": bool(record.get("wrong_reference")),
            "shuffled_reference_present": bool(record.get("shuffled_reference")),
            "shuffle_changed": compact(record.get("shuffled_reference", "")) != compact(record.get("solution", "")),
        }
        rows.append(
            {
                "id": record.get("id"),
                "checks": checks,
                "max_pair_similarity": max(pairwise) if pairwise else None,
                "auto_pass": all(checks.values()),
                "manual_strategy_verification_required": True,
            }
        )

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "rows": len(rows),
        "auto_pass_rows": sum(row["auto_pass"] for row in rows),
        "all_auto_checks_pass": bool(rows) and all(row["auto_pass"] for row in rows),
        "manual_strategy_verification_required": True,
        "row_results": rows,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "row_results"}, indent=2))


if __name__ == "__main__":
    main()
