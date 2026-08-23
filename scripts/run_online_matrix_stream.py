#!/usr/bin/env python3
"""Arm-level resumable runner for large online paired-residual matrices."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .run_online_paired_residual import (
        arm_seed, answers_equivalent, boxed, normalize_answer, online_generate,
        online_teacher_prompt, ordinary_generate,
    )
    from .run_privileged_effect_diagnostic import parse_layers
except ImportError:  # Direct `python scripts/...` execution.
    from run_online_paired_residual import (
        arm_seed, answers_equivalent, boxed, normalize_answer, online_generate,
        online_teacher_prompt, ordinary_generate,
    )
    from run_privileged_effect_diagnostic import parse_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--layers", default="13,21,23,25")
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1")
    parser.add_argument("--schedules", default="every1,every2,every4,every8,first8")
    parser.add_argument("--modes", default="rescue,reverse,random")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()
    row = next(
        row for row in map(json.loads, Path(args.dataset).open()) if str(row.get("id")) == args.id
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in output.read_text().splitlines() if line] if output.exists() else []
    completed = {record["key"] for record in existing}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    correct_text = online_teacher_prompt(tokenizer, row["problem"], row["references"][0], False)
    corrupt_text = online_teacher_prompt(
        tokenizer, row["problem"], row["matched_controls"]["dependency_swap"], False
    )
    correct_prefix = tokenizer(correct_text, add_special_tokens=False)["input_ids"]
    corrupt_prefix = tokenizer(corrupt_text, add_special_tokens=False)["input_ids"]
    started = time.time()
    with output.open("a") as handle:
        if "baseline" not in completed:
            baseline = {"key": "baseline", "id": row["id"], "answer": row["answer"]}
            for name, prefix in (("correct", correct_prefix), ("corrupt", corrupt_prefix)):
                ids = torch.tensor([prefix], device=model.device)
                tokens = ordinary_generate(model, ids, args.max_new_tokens)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                answer = boxed(text)
                baseline[name] = {
                    "text": text, "boxed": answer,
                    "exact_match": normalize_answer(answer) == normalize_answer(row["answer"]),
                    "math_equivalent": answers_equivalent(answer, row["answer"]),
                    "tokens": len(tokens),
                }
            handle.write(json.dumps(baseline, ensure_ascii=False) + "\n")
            handle.flush()
        layers = parse_layers(args.layers, model.config.num_hidden_layers)
        jobs = [
            (layer, float(alpha), schedule, mode)
            for layer in layers for alpha in args.alphas.split(",")
            for schedule in args.schedules.split(",") for mode in args.modes.split(",")
        ]
        jobs = [job for index, job in enumerate(jobs) if index % args.shard_count == args.shard_index]
        for index, (layer, alpha, schedule, mode) in enumerate(jobs):
            key = f"layer={layer}:alpha={alpha:g}:schedule={schedule}:mode={mode}"
            if key in completed:
                continue
            seed = arm_seed(args.seed, row["id"], layer, alpha, schedule, mode)
            tokens, norm = online_generate(
                model, tokenizer, correct_prefix, corrupt_prefix, layer, alpha, mode,
                schedule, args.max_new_tokens,
                seed,
            )
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            answer = boxed(text)
            record = {
                "key": key, "id": row["id"], "answer": row["answer"],
                "layer": layer, "alpha": alpha, "schedule": schedule, "mode": mode,
                "text": text, "boxed": answer,
                "exact_match": normalize_answer(answer) == normalize_answer(row["answer"]),
                "math_equivalent": answers_equivalent(answer, row["answer"]),
                "tokens": len(tokens), "mean_residual_norm": norm,
                "seed": seed,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"arm={index + 1}/{len(jobs)} key={key} em={record['exact_match']} "
                f"elapsed={time.time() - started:.1f}s", flush=True,
            )


if __name__ == "__main__":
    main()
