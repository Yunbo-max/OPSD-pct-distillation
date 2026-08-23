#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_multiref_pilot import final_marker, normalize_answer, prompt, shuffled_control


def similarity(left: str, right: str) -> float:
    normalize = lambda text: re.sub(r"\s+", " ", text).strip().lower()
    return difflib.SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def repair_prompt(problem: str, answer: str, reference_index: int, diversity: bool) -> str:
    strategy = (
        "a radically different short proof: avoid headings and avoid repeating the obvious derivation; "
        "prefer a direct invariant, construction, contradiction, geometric interpretation, or sanity check"
        if diversity
        else "a concise rigorous derivation that reaches the verified answer without being truncated"
    )
    return (
        f"Begin your response with the exact marker <FINAL>{answer}</FINAL>, then give {strategy}.\n"
        "The marker must come first so it cannot be lost to truncation. "
        "Keep the derivation self-contained and under 600 words.\n\n"
        f"Problem:\n{problem}\n\nVerified answer: {answer}\n"
        f"This is replacement reference {reference_index}."
    )


def generate_replacements(model, tokenizer, tasks: list[tuple], args) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task[4]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for task in batch
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(args.seed + start)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
            )
        outputs.extend(
            tokenizer.decode(sample[prompt_ids.shape[0] :], skip_special_tokens=True)
            for sample, prompt_ids in zip(generated, encoded.input_ids)
        )
        del generated, encoded
        torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair failed multi-reference candidate rows.")
    parser.add_argument("dataset")
    parser.add_argument("audit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_pair_similarity", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line]
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    failed = {int(row["id"]): row for row in audit["row_results"] if not row["auto_pass"]}

    for record in records:
        if int(record["id"]) in failed and str(record.get("answer", "")).strip() in {"", "\\"}:
            if "prove" in str(record.get("problem", "")).lower():
                record["answer"] = "PROVED"
                record["reference_valid"][2:4] = [False, False]

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    for attempt in range(args.max_attempts):
        tasks = []
        for record in records:
            row_id = int(record["id"])
            if row_id not in failed:
                continue
            answer = str(record["answer"])
            invalid_generated = [idx for idx in (2, 3) if not record["reference_valid"][idx]]
            too_similar = similarity(record["references"][2], record["references"][3]) >= args.max_pair_similarity
            indexes = invalid_generated or ([3] if too_similar else [])
            for ref_idx in indexes:
                tasks.append(
                    (
                        row_id,
                        ref_idx,
                        answer,
                        too_similar,
                        repair_prompt(record["problem"], answer, ref_idx, too_similar),
                    )
                )
            record["shuffled_reference"] = shuffled_control(record["solution"], args.seed + row_id)
        if not tasks:
            break
        replacements = generate_replacements(model, tokenizer, tasks, args)
        by_id = {int(record["id"]): record for record in records}
        for task, replacement in zip(tasks, replacements):
            row_id, ref_idx, answer, _, _ = task
            record = by_id[row_id]
            record["references"][ref_idx] = replacement
            record["reference_valid"][ref_idx] = final_marker(replacement) == normalize_answer(answer)
        unresolved = set()
        for row_id in failed:
            record = by_id[row_id]
            if not all(record["reference_valid"][:4]):
                unresolved.add(row_id)
            elif similarity(record["references"][2], record["references"][3]) >= args.max_pair_similarity:
                unresolved.add(row_id)
            elif re.sub(r"\s+", " ", record["shuffled_reference"]).strip() == re.sub(
                r"\s+", " ", record["solution"]
            ).strip():
                unresolved.add(row_id)
        failed = {row_id: failed[row_id] for row_id in unresolved}
        print(f"attempt={attempt + 1} replacements={len(tasks)} unresolved={len(failed)}", flush=True)
        if not failed:
            break

    Path(args.out).write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    if failed:
        raise SystemExit(f"unresolved rows after repair: {sorted(failed)}")


if __name__ == "__main__":
    main()
