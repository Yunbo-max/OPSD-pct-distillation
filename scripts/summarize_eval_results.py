#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pct.eval_protocol import validate_eval_file


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_eval_files(records: list[dict]) -> dict[str, dict]:
    expected = {}
    for record in records:
        for eval_record in record.get("eval", []):
            val_n = int(eval_record.get("val_n", 12))
            dataset = str(eval_record["dataset"])
            name = f"{record['run_name']}_{dataset}_avg{val_n}.json"
            expected[name] = {
                "dataset": dataset,
                "val_n": val_n,
                "seed": eval_record.get("seed", record.get("seed")),
            }
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OPSD/PCT eval JSON files.")
    parser.add_argument("paths", nargs="+", help="Eval JSON files or directories containing JSON files.")
    parser.add_argument("--tsv", default=None, help="Optional path to write a TSV copy.")
    parser.add_argument("--manifest", default=None, help="Optional manifest for expected eval files and seeds.")
    parser.add_argument(
        "--strict_protocol",
        action="store_true",
        help="Validate each eval JSON against the Average@N protocol before summarizing.",
    )
    parser.add_argument("--expected_num_problems", type=int, default=30)
    parser.add_argument("--expected_val_n", type=int, default=None)
    args = parser.parse_args()

    files: list[Path] = []
    saw_directory = False
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            saw_directory = True
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    expected = expected_eval_files(load_manifest(Path(args.manifest))) if args.manifest else {}
    if args.strict_protocol and expected:
        present = {path.name for path in files}
        missing = sorted(set(expected) - present)
        if missing and saw_directory:
            for name in missing:
                print(f"FAIL\tmissing_eval_file:{name}")
            raise SystemExit(1)

    rows = []
    for path in files:
        expected_item = expected.get(path.name, {})
        if args.strict_protocol and expected and path.name not in expected:
            print(f"FAIL\tunexpected_eval_file:{path}")
            raise SystemExit(1)
        if args.strict_protocol:
            with path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
            expected_dataset = expected_item.get("dataset", raw_data.get("dataset"))
            expected_val_n = int(args.expected_val_n or expected_item.get("val_n") or raw_data.get("val_n", 12))
            failures = validate_eval_file(
                path,
                expected_dataset=expected_dataset,
                expected_val_n=expected_val_n,
                expected_num_problems=args.expected_num_problems,
                require_generations=True,
                expected_seed=expected_item.get("seed"),
            )
            if failures:
                print(f"FAIL\t{path}\t{'|'.join(failures)}")
                raise SystemExit(1)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "file": path.name,
                "dataset": data.get("dataset"),
                "avg": data.get("average_at_n_pct"),
                "pass": data.get("pass_at_n_pct"),
                "majority": data.get("majority_vote_at_n_pct"),
                "format": data.get("format_rate"),
                "n": data.get("val_n"),
                "problems": data.get("num_problems"),
                "seed": data.get("seed"),
            }
        )

    header = ["file", "dataset", "Average@N", "Pass@N", "Maj@N", "Format", "N", "Problems", "Seed"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                (
                    str(row["file"]),
                    str(row["dataset"]),
                    f"{row['avg']:.2f}" if isinstance(row["avg"], (int, float)) else "NA",
                    f"{row['pass']:.2f}" if isinstance(row["pass"], (int, float)) else "NA",
                    f"{row['majority']:.2f}" if isinstance(row["majority"], (int, float)) else "NA",
                    f"{row['format']:.2f}" if isinstance(row["format"], (int, float)) else "NA",
                    str(row["n"]),
                    str(row["problems"]),
                    str(row["seed"]) if row["seed"] is not None else "NA",
                )
            )
        )
    output = "\n".join(lines)
    print(output)

    if args.tsv:
        out = Path(args.tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
