#!/usr/bin/env python3
"""Generation-only screening for NVIP frontier problems (1..K-1 successes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .extract_predictable_field_dataset import deployment_prompt
    from .run_online_paired_residual import answers_equivalent, boxed
except ImportError:
    from extract_predictable_field_dataset import deployment_prompt
    from run_online_paired_residual import answers_equivalent, boxed


def seed_for(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{identity}".encode()).digest()[:8], "big") % (2**31 - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n_problems", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = list(map(json.loads, Path(args.dataset).open()))[: args.n_problems]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval(); started = time.time()
    for index, row in enumerate(rows):
        destination = out_dir / f"{index:06d}.pt"
        if destination.exists():
            continue
        prompt = deployment_prompt(tokenizer, row["problem"])
        prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        parts = []
        for begin in range(0, args.group_size, args.generation_batch_size):
            count = min(args.generation_batch_size, args.group_size - begin)
            torch.manual_seed(seed_for(args.seed, f'{row["id"]}|group={begin}'))
            with torch.inference_mode():
                parts.append(model.generate(
                    prefix, attention_mask=torch.ones_like(prefix), do_sample=True,
                    temperature=args.temperature, top_p=args.top_p, num_return_sequences=count,
                    max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[:, prefix.shape[1]:].cpu())
        width = max(part.shape[1] for part in parts)
        completions = torch.cat([
            F.pad(part, (0, width - part.shape[1]), value=tokenizer.pad_token_id) for part in parts
        ])
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        predictions = [boxed(text) for text in texts]
        rewards = torch.tensor([
            float(answers_equivalent(prediction, row["answer"])) for prediction in predictions
        ])
        payload = {
            "dataset_index": index, "id": row["id"], "answer": row["answer"],
            "completion_ids": completions, "predictions": predictions,
            "rewards": rewards, "successes": int(rewards.sum()), "config": vars(args),
        }
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary); temporary.replace(destination)
        print(f"row={index} successes={int(rewards.sum())}/{args.group_size} "
              f"frontier={0 < rewards.sum() < args.group_size} elapsed={time.time()-started:.1f}s", flush=True)
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
