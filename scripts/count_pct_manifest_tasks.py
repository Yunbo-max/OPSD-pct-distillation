#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_tasks(records: list[dict]) -> list[dict[str, str]]:
    tasks = []
    for record in records:
        if record.get("requires_training") is False:
            continue
        tasks.append(
            {
                "run_name": str(record.get("run_name", "")),
                "method": str(record.get("method", "")),
                "model": str(record.get("model", "")),
                "dataset": str(record.get("dataset", "")),
            }
        )
    return tasks


def eval_tasks(records: list[dict]) -> list[dict[str, str]]:
    tasks = []
    for record in records:
        for eval_record in record.get("eval", []):
            val_n = int(eval_record.get("val_n", 12))
            dataset = str(eval_record.get("dataset", ""))
            tasks.append(
                {
                    "run_name": str(record.get("run_name", "")),
                    "method": str(record.get("method", "")),
                    "model": str(record.get("model", "")),
                    "dataset": dataset,
                    "seed": str(eval_record.get("seed", record.get("seed", ""))),
                    "val_n": str(val_n),
                    "output": f"{record.get('run_name', '')}_{dataset}_avg{val_n}.json",
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Count train/eval tasks in a PCT manifest for job-array sizing.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", choices=["train", "eval"], required=True)
    parser.add_argument("--tsv", default=None)
    parser.add_argument("--print_array", action="store_true")
    args = parser.parse_args()

    records = load_manifest(Path(args.manifest))
    tasks = train_tasks(records) if args.task == "train" else eval_tasks(records)
    count = len(tasks)
    max_index = count - 1
    print(f"task\tcount\tmax_array_index")
    print(f"{args.task}\t{count}\t{max_index}")
    if args.print_array:
        print(f"sbatch_array=0-{max_index}" if count else "sbatch_array=EMPTY")

    if args.tsv:
        out = Path(args.tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        if tasks:
            columns = ["index", *tasks[0].keys()]
        else:
            columns = ["index"]
        lines = ["\t".join(columns)]
        for idx, task in enumerate(tasks):
            lines.append("\t".join([str(idx), *[task.get(col, "") for col in columns[1:]]]))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
