#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge manifest JSONL files and reject duplicate run names.")
    parser.add_argument("--out", required=True)
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()

    seen = set()
    records = []
    for raw in args.manifests:
        path = Path(raw)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            run_name = record["run_name"]
            if run_name in seen:
                raise SystemExit(f"duplicate run_name: {run_name}")
            seen.add(run_name)
            records.append(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote\t{out}\t{len(records)}")


if __name__ == "__main__":
    main()
