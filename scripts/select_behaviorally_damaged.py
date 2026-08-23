#!/usr/bin/env python3
"""Select paired examples where the corrupt reference changes final-answer behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .run_online_paired_residual import answers_equivalent, boxed
except ImportError:
    from run_online_paired_residual import answers_equivalent, boxed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("baselines", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include_reverse", action="store_true")
    args = parser.parse_args()

    source = {
        str(row.get("id")): row
        for row in map(json.loads, Path(args.dataset).open())
    }
    selected = []
    audit = []
    for baseline in args.baselines:
        for result in map(json.loads, Path(baseline).open()):
            target = result["answer"]
            correct = answers_equivalent(
                result.get("correct_reference_boxed") or boxed(result.get("correct_reference", "")), target
            )
            corrupt = answers_equivalent(
                result.get("corrupt_baseline_boxed") or boxed(result.get("corrupt_baseline", "")), target
            )
            damaged = correct and not corrupt
            reverse = corrupt and not correct
            audit.append({
                "id": result.get("id"), "correct_exact_match": correct,
                "corrupt_exact_match": corrupt, "damaged": damaged, "reverse": reverse,
            })
            if damaged or (args.include_reverse and reverse):
                selected.append(source[str(result.get("id"))])

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps({
        "n_baseline": len(audit), "n_damaged": sum(x["damaged"] for x in audit),
        "n_reverse": sum(x["reverse"] for x in audit), "selected": len(selected),
        "records": audit,
    }, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(selected), "audit": str(audit_path)}))


if __name__ == "__main__":
    main()
