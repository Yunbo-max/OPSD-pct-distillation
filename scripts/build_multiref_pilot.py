#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


STRATEGIES = (
    "an algebraic or symbolic derivation distinct from the supplied solution",
    "a concise alternative insight, invariant, or counting argument",
)


def normalize_answer(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^\\\\boxed\{(.*)\}$", r"\1", text)
    return re.sub(r"\s+", "", text).lower()


def final_marker(text: str) -> str | None:
    matches = re.findall(r"<FINAL>(.*?)</FINAL>", text, flags=re.DOTALL | re.IGNORECASE)
    return normalize_answer(matches[-1]) if matches else None


def shuffled_control(solution: str, seed: int) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", solution) if chunk.strip()]
    if len(chunks) < 2:
        chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", solution) if chunk.strip()]
    rng = random.Random(seed)
    rng.shuffle(chunks)
    shuffled = "\n\n".join(chunks)
    if re.sub(r"\s+", " ", shuffled).strip() == re.sub(r"\s+", " ", solution).strip():
        words = solution.split()
        if len(words) > 1:
            pivot = max(1, len(words) // 2)
            shuffled = " ".join(words[pivot:] + words[:pivot])
    return shuffled


def prompt(problem: str, answer: str, strategy: str) -> str:
    return f"""Solve the following problem with {strategy}.

The verified final answer is: {answer}

Requirements:
- Give a self-contained, mathematically valid derivation, not a paraphrase.
- Be concise enough to finish.
- End with the exact machine-readable marker <FINAL>{answer}</FINAL>.

Problem:
{problem}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable multi-reference diagnostic pilot.")
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--retry_max_new_tokens", type=int, default=3072)
    parser.add_argument("--batch_examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if output.exists():
        completed = sum(1 for line in output.read_text(encoding="utf-8").splitlines() if line.strip())
    if completed >= args.n_examples:
        print(f"already complete: {completed} rows")
        return

    rows = list(load_dataset(args.dataset, split=args.split).select(range(args.n_examples + 1)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    passed = 0
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        for batch_start in range(completed, args.n_examples, args.batch_examples):
            indexes = list(range(batch_start, min(batch_start + args.batch_examples, args.n_examples)))
            flat_items = []
            for index in indexes:
                row = rows[index]
                answer = str(row["Answer"]).strip()
                for strategy_idx, strategy in enumerate(STRATEGIES):
                    message = [{"role": "user", "content": prompt(row["problem"], answer, strategy)}]
                    flat_items.append((index, strategy_idx, answer, message))
            texts = [
                tokenizer.apply_chat_template(
                    item[3], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for item in flat_items
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(args.seed + batch_start)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            batch_results: dict[int, dict[str, list]] = {
                index: {"alternatives": ["", ""], "valid": [False, False]} for index in indexes
            }
            for item, sample, prompt_ids in zip(flat_items, generated, inputs.input_ids):
                index, strategy_idx, answer, _ = item
                decoded = tokenizer.decode(sample[prompt_ids.shape[0] :], skip_special_tokens=True)
                batch_results[index]["alternatives"][strategy_idx] = decoded
                batch_results[index]["valid"][strategy_idx] = final_marker(decoded) == normalize_answer(answer)
            del generated, inputs
            torch.cuda.empty_cache()

            retry_items = []
            for index in indexes:
                row = rows[index]
                answer = str(row["Answer"]).strip()
                for strategy_idx, is_valid in enumerate(batch_results[index]["valid"]):
                    if not is_valid:
                        message = [
                            {
                                "role": "user",
                                "content": prompt(row["problem"], answer, STRATEGIES[strategy_idx]),
                            }
                        ]
                        retry_items.append((index, strategy_idx, answer, message))
            if retry_items:
                retry_texts = [
                    tokenizer.apply_chat_template(
                        item[3], tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    for item in retry_items
                ]
                retry_inputs = tokenizer(retry_texts, return_tensors="pt", padding=True).to(model.device)
                torch.manual_seed(args.seed + 100_000 + batch_start)
                with torch.inference_mode():
                    retry_generated = model.generate(
                        **retry_inputs,
                        max_new_tokens=args.retry_max_new_tokens,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                    )
                for item, sample, prompt_ids in zip(
                    retry_items, retry_generated, retry_inputs.input_ids
                ):
                    index, strategy_idx, answer, _ = item
                    retry = tokenizer.decode(
                        sample[prompt_ids.shape[0] :], skip_special_tokens=True
                    )
                    batch_results[index]["alternatives"][strategy_idx] = retry
                    batch_results[index]["valid"][strategy_idx] = (
                        final_marker(retry) == normalize_answer(answer)
                    )
                del retry_generated, retry_inputs
                torch.cuda.empty_cache()

            for index in indexes:
                row = rows[index]
                answer = str(row["Answer"]).strip()
                alternatives = batch_results[index]["alternatives"]
                valid = batch_results[index]["valid"]
                passed += sum(valid)
                record = {
                    "id": index,
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "answer": answer,
                    "references": [row["solution"], row["COT_Reason"], *alternatives],
                    "reference_valid": [True, bool(row["correct"]), *valid],
                    "reference_strategies": ["original", "source_cot", *STRATEGIES],
                    "wrong_reference": rows[index + 1]["solution"],
                    "shuffled_reference": shuffled_control(row["solution"], args.seed + index),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            elapsed = time.time() - started
            done = indexes[-1] - completed + 1
            print(
                f"rows={indexes[0]}-{indexes[-1]} elapsed={elapsed:.1f}s "
                f"rows_per_hour={done * 3600 / elapsed:.1f} "
                f"peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}",
                flush=True,
            )
    print(f"generated answer-marker pass rate: {passed}/{2 * (args.n_examples - completed)}")


if __name__ == "__main__":
    main()
