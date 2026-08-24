#!/usr/bin/env python3
"""Fresh-rollout causal validation of p, q, and NVIP q* on audited nodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pct.rcip import node_value_i_projection

try:
    from .extract_predictable_field_dataset import deployment_prompt
    from .run_online_paired_residual import answers_equivalent, boxed
except ImportError:
    from extract_predictable_field_dataset import deployment_prompt
    from run_online_paired_residual import answers_equivalent, boxed


def seed_for(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "big") % (2**31 - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("group_cache")
    parser.add_argument("targeted")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    rows = {str(row["id"]): row for row in map(json.loads, Path(args.dataset).open())}
    cache = {}
    for path in Path(args.group_cache).glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cache[str(payload["id"])] = payload
    audited = list(map(json.loads, Path(args.targeted).open()))
    harmful = []
    for item in audited:
        p = torch.tensor(item["student_probs"]); q = torch.tensor(item["teacher_probs"])
        values = torch.tensor(item["branch_success"])
        qstar, _, metrics = node_value_i_projection(p, q, values)
        if float(metrics["teacher_advantage"]) < -1e-7:
            harmful.append((item, p, q, qstar))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval(); results = []
    for item, p, q, qstar in harmful:
        problem_id = str(item["problem_id"]); row = rows[problem_id]; stored = cache[problem_id]
        prefix_tokens = stored["completion_ids"][item["rollout"], : item["position"]]
        prompt = tokenizer(
            deployment_prompt(tokenizer, row["problem"]), return_tensors="pt", add_special_tokens=False
        ).input_ids
        candidate_ids = torch.tensor(item["candidate_ids"])
        result = {"id": item["id"], "arms": {}}
        for name, distribution in (("student", p), ("teacher", q), ("nvip", qstar)):
            generator = torch.Generator().manual_seed(seed_for(args.seed, f'{item["id"]}|{name}'))
            chosen = candidate_ids[torch.multinomial(
                distribution.float(), args.rollouts, replacement=True, generator=generator
            )]
            common = torch.cat((prompt[0], prefix_tokens)).unsqueeze(0).expand(args.rollouts, -1)
            branch_prefix = torch.cat((common, chosen.unsqueeze(1)), 1).to(model.device)
            torch.manual_seed(seed_for(args.seed, f'{item["id"]}|{name}|continuation'))
            with torch.inference_mode():
                generated = model.generate(
                    branch_prefix, attention_mask=torch.ones_like(branch_prefix), do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            suffix = generated[:, branch_prefix.shape[1]:].cpu()
            completions = torch.cat((
                prefix_tokens.unsqueeze(0).expand(args.rollouts, -1), chosen.unsqueeze(1), suffix
            ), 1)
            texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
            rewards = [float(answers_equivalent(boxed(text), row["answer"])) for text in texts]
            result["arms"][name] = {
                "successes": int(sum(rewards)), "n": len(rewards),
                "chosen_tokens": chosen.tolist(), "predictions": [boxed(text) for text in texts],
            }
        results.append(result)
        print(item["id"], {name: arm["successes"] for name, arm in result["arms"].items()}, flush=True)
    Path(args.out).write_text(json.dumps({"n": len(results), "records": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
