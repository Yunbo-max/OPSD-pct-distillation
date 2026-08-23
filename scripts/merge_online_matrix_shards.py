#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = {}
    duplicate_keys, conflicting_keys = [], []
    for source in args.inputs:
        for line in Path(source).read_text().splitlines():
            if line:
                record = json.loads(line)
                key = record["key"]
                if key in records:
                    duplicate_keys.append(key)
                    comparable_old = {k: v for k, v in records[key].items() if k not in {"seed", "math_equivalent"}}
                    comparable_new = {k: v for k, v in record.items() if k not in {"seed", "math_equivalent"}}
                    if comparable_old != comparable_new:
                        conflicting_keys.append(key)
                records[key] = record
    ordered = [records.pop("baseline")] if "baseline" in records else []
    ordered.extend(records[key] for key in sorted(records))
    with Path(args.out).open("w") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit = {
        "inputs": args.inputs, "records_written": len(ordered),
        "unique_arms": len(ordered) - int(bool(ordered and ordered[0].get("key") == "baseline")),
        "duplicate_keys": sorted(set(duplicate_keys)),
        "conflicting_duplicate_keys": sorted(set(conflicting_keys)),
        "precedence": "later input wins",
    }
    Path(args.out).with_suffix(".merge_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
