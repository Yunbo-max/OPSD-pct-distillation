#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pct.eval_protocol import validate_eval_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate OPSD/PCT eval JSON protocol.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--expected_dataset", default=None)
    parser.add_argument("--expected_val_n", type=int, default=12)
    parser.add_argument("--expected_num_problems", type=int, default=30)
    parser.add_argument("--expected_seed", type=int, default=None)
    parser.add_argument("--allow_missing_generations", action="store_true")
    args = parser.parse_args()

    failed = False
    for raw in args.paths:
        path = Path(raw)
        failures = validate_eval_file(
            path,
            expected_dataset=args.expected_dataset,
            expected_val_n=args.expected_val_n,
            expected_num_problems=args.expected_num_problems,
            require_generations=not args.allow_missing_generations,
            expected_seed=args.expected_seed,
        )
        status = "OK" if not failures else "FAIL"
        print(f"{path}\t{status}\t{','.join(failures) if failures else 'none'}")
        failed = failed or bool(failures)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
