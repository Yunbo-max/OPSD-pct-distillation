#!/usr/bin/env python3
"""Compare cheap RCIP chi with targeted-rollout ideal teacher alignment.

One token node is sampled independently of chi from each eligible held-out
problem. At that node we form a small branch set from student/teacher top tokens
plus the observed action, estimate branch success with student continuations,
and compute the candidate-restricted ideal policy gradient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .extract_predictable_field_dataset import deployment_prompt
    from .run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt
except ImportError:
    from extract_predictable_field_dataset import deployment_prompt
    from run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt


def seed_for(base: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{identity}".encode()).digest()[:8], "big") % (2**31 - 1)


def next_logits(model, ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(ids, attention_mask=torch.ones_like(ids), use_cache=True,
                     logits_to_keep=1, return_dict=True).logits[0, -1].float()


def choose_nodes(cache_dir: Path, start_problem: int, n_nodes: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    for path in sorted(cache_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["dataset_index"]) < start_problem:
            continue
        nonzero_rollouts = torch.where(payload["advantages"].abs() > 1e-6)[0].tolist()
        if not nonzero_rollouts:
            continue
        rollout = rng.choice(nonzero_rollouts)
        valid_positions = torch.where(payload["valid_mask"][rollout])[0].tolist()
        if len(valid_positions) < 8:
            continue
        lo, hi = len(valid_positions) // 10, max(len(valid_positions) // 10 + 1, 9 * len(valid_positions) // 10)
        position = rng.choice(valid_positions[lo:hi])
        candidates.append({
            "cache": str(path), "id": str(payload["id"]), "rollout": rollout,
            "position": position, "chi": float(payload["chi"][rollout, position]),
            "sample_alignment": float(payload["sample_alignment"][rollout, position]),
        })
    rng.shuffle(candidates)
    return candidates[:n_nodes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("cache_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--start_problem", type=int, default=100)
    parser.add_argument("--n_nodes", type=int, default=64)
    parser.add_argument("--top_student", type=int, default=3)
    parser.add_argument("--top_teacher", type=int, default=3)
    parser.add_argument("--branch_rollouts", type=int, default=4)
    parser.add_argument("--max_total_tokens", type=int, default=1024)
    parser.add_argument("--max_branch_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = {str(row["id"]): row for row in map(json.loads, Path(args.dataset).open())}
    nodes = choose_nodes(Path(args.cache_dir), args.start_problem, args.n_nodes, args.seed)
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); output.touch(exist_ok=True)
    completed = {json.loads(line)["id"] for line in output.read_text().splitlines() if line}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval(); started = time.time()

    with output.open("a") as handle:
        for node_index, node in enumerate(nodes):
            node_key = f'{node["id"]}|{node["rollout"]}|{node["position"]}'
            if node_key in completed:
                continue
            cached = torch.load(node["cache"], map_location="cpu", weights_only=False)
            row = rows[node["id"]]
            completion = cached["completion_ids"][node["rollout"]]
            prefix_tokens = completion[: node["position"]]
            observed = int(completion[node["position"]])
            student_prompt = deployment_prompt(tokenizer, row["problem"])
            teacher_prompt = online_teacher_prompt(tokenizer, row["problem"], row["references"][0], True)
            student_prefix = tokenizer(student_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            teacher_prefix = tokenizer(teacher_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            prefix_tokens_gpu = prefix_tokens.unsqueeze(0).to(model.device)
            student_node = torch.cat((student_prefix, prefix_tokens_gpu), 1)
            teacher_node = torch.cat((teacher_prefix, prefix_tokens_gpu), 1)
            p_logits = next_logits(model, student_node)
            q_logits = next_logits(model, teacher_node)
            candidate_ids = torch.unique(torch.cat((
                torch.tensor([observed], device=model.device),
                torch.topk(p_logits, args.top_student).indices,
                torch.topk(q_logits, args.top_teacher).indices,
            )), sorted=False)
            p = torch.softmax(p_logits[candidate_ids], 0)
            q = torch.softmax(q_logits[candidate_ids], 0)

            remaining = max(1, min(args.max_branch_tokens, args.max_total_tokens - node["position"] - 1))
            branch_rewards = []
            for candidate in candidate_ids.tolist():
                branch_prefix = torch.cat((
                    student_node, torch.tensor([[candidate]], device=model.device)
                ), 1)
                torch.manual_seed(seed_for(args.seed, f"{node_key}|{candidate}"))
                with torch.inference_mode():
                    generated = model.generate(
                        branch_prefix, attention_mask=torch.ones_like(branch_prefix), do_sample=True,
                        temperature=args.temperature, top_p=args.top_p,
                        num_return_sequences=args.branch_rollouts, max_new_tokens=remaining,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )
                suffix = generated[:, branch_prefix.shape[1]:]
                full_completions = torch.cat((
                    prefix_tokens.unsqueeze(0).expand(args.branch_rollouts, -1),
                    torch.full((args.branch_rollouts, 1), candidate, dtype=torch.long),
                    suffix.cpu(),
                ), 1)
                texts = tokenizer.batch_decode(full_completions, skip_special_tokens=True)
                rewards = [float(answers_equivalent(boxed(text), row["answer"])) for text in texts]
                branch_rewards.append(sum(rewards) / len(rewards))

            values = torch.tensor(branch_rewards, device=model.device)
            expected = (p * values).sum()
            ideal = p * (values - expected)
            teacher = q - p
            ideal_norm = ideal.norm(); teacher_norm = teacher.norm()
            ideal_alignment = float(
                (ideal * teacher).sum() / (ideal_norm * teacher_norm).clamp_min(1e-12)
            )
            teacher_advantage = float((q * values).sum() - expected)
            result = {
                "id": node_key, "problem_id": node["id"], "rollout": node["rollout"],
                "position": node["position"], "chi": node["chi"],
                "sample_alignment": node["sample_alignment"], "observed_token": observed,
                "candidate_ids": candidate_ids.tolist(), "candidate_tokens": [
                    tokenizer.decode([value]) for value in candidate_ids.tolist()
                ],
                "student_probs": p.tolist(), "teacher_probs": q.tolist(),
                "branch_success": branch_rewards, "ideal_alignment": ideal_alignment,
                "teacher_advantage": teacher_advantage, "ideal_norm": float(ideal_norm),
                "teacher_update_norm": float(teacher_norm), "remaining_tokens": remaining,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            print(f"node={node_index} id={node_key} chi={node['chi']:+.3g} ideal={ideal_alignment:+.3g} "
                  f"elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
