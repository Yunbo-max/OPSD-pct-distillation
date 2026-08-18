#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_baseline(value: str) -> tuple[str, str | None]:
    if "=" not in value:
        return value, None
    name, path = value.split("=", 1)
    return name, path or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write eval manifest rows for base/external baseline checkpoints.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--model_tag", default="qwen3_4b")
    parser.add_argument(
        "--baseline",
        action="append",
        type=parse_baseline,
        required=True,
        help="Baseline name or name=/checkpoint. Use 'base' without a path for direct base-model eval.",
    )
    parser.add_argument("--eval_datasets", nargs="+", default=["aime24", "aime25", "hmmt25"])
    parser.add_argument("--eval_val_n", type=int, default=12)
    parser.add_argument("--eval_seed", type=int, default=None, help="Optional fixed seed for Average@N generation.")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for method, checkpoint in args.baseline:
            run_name = f"{args.model_tag}_{method}"
            record = {
                "run_name": run_name,
                "method": method,
                "model": args.base_model,
                "model_tag": args.model_tag,
                "requires_training": checkpoint is not None,
                "output_dir": checkpoint,
                "eval": [
                    {
                        "dataset": dataset,
                        "val_n": args.eval_val_n,
                        "metric": f"Average@{args.eval_val_n}",
                        **({"seed": args.eval_seed} if args.eval_seed is not None else {}),
                    }
                    for dataset in args.eval_datasets
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
