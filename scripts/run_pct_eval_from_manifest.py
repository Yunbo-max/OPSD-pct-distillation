#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pct.eval_protocol import validate_eval_file


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def effective_seed(record: dict, eval_record: dict) -> int | None:
    return eval_record.get("seed", record.get("seed"))


def output_complete(path: Path, dataset: str, val_n: int, expected_num_problems: int, seed: int | None) -> bool:
    return not validate_eval_file(
        path,
        expected_dataset=dataset,
        expected_val_n=val_n,
        expected_num_problems=expected_num_problems,
        require_generations=True,
        expected_seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every run/dataset pair in a PCT manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--max_new_tokens", type=int, default=38912)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--expected_num_problems", type=int, default=30)
    parser.add_argument("--index", type=int, default=None, help="Run only the 0-based selected eval task at this index.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.index is not None and args.index < 0:
        raise SystemExit("--index must be >= 0")

    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    task_index = 0
    selected_count = 0
    for record in load_manifest(Path(args.manifest)):
        for eval_record in record.get("eval", []):
            val_n = int(eval_record.get("val_n", 12))
            dataset = str(eval_record["dataset"])
            seed = effective_seed(record, eval_record)
            output_file = eval_root / f"{record['run_name']}_{dataset}_avg{val_n}.json"
            if args.index is not None and task_index != args.index:
                task_index += 1
                continue
            if args.index is None and task_index % args.num_shards != args.shard_index:
                task_index += 1
                continue
            if args.skip_completed and output_complete(output_file, dataset, val_n, args.expected_num_problems, seed):
                print(f"skip_completed\t{output_file}")
                task_index += 1
                selected_count += 1
                continue
            cmd = [
                args.python,
                "eval/evaluate_math.py",
                "--backend",
                args.backend,
                "--base_model",
                str(record["model"]),
            ]
            if record.get("output_dir"):
                cmd += ["--checkpoint_dir", str(record["output_dir"])]
            cmd += [
                "--dataset",
                dataset,
                "--val_n",
                str(val_n),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--top_k",
                str(args.top_k),
                "--max_model_len",
                str(args.max_model_len),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--tensor_parallel_size",
                str(args.tensor_parallel_size),
                "--output_file",
                str(output_file),
            ]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            if eval_record.get("dataset_jsonl"):
                cmd += ["--dataset_jsonl", str(eval_record["dataset_jsonl"])]
            print(shell_join(cmd))
            selected_count += 1
            if not args.dry_run:
                subprocess.check_call(cmd)
            task_index += 1
    if selected_count == 0:
        raise SystemExit("No manifest eval tasks selected.")


if __name__ == "__main__":
    main()
