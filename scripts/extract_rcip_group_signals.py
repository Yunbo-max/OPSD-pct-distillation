#!/usr/bin/env python3
"""Generate on-policy groups and cache RCIP token compatibility signals."""
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
    from .run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt
except ImportError:
    from extract_predictable_field_dataset import deployment_prompt
    from run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt


def stable_seed(seed: int, record_id: object) -> int:
    digest = hashlib.sha256(f"{seed}|{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def model_logits(model, ids, mask, start: int, length: int):
    with torch.inference_mode():
        logits = model(ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    return logits[:, start - 1 : start + length - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n_problems", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--teacher_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--signal_chunk", type=int, default=16)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--score_batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    rows = list(map(json.loads, Path(args.dataset).open()))[: args.n_problems]
    selected = [(index, row) for index, row in enumerate(rows)
                if index % args.shard_count == args.shard_index]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    started = time.time()

    for local_index, (index, row) in enumerate(selected):
        destination = out_dir / f"{index:06d}.pt"
        if destination.exists():
            continue
        student_prompt = deployment_prompt(tokenizer, row["problem"])
        teacher_prompt = online_teacher_prompt(
            tokenizer, row["problem"], row["references"][0], args.teacher_thinking
        )
        student_prefix = tokenizer(student_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        generated_parts = []
        for group_begin in range(0, args.group_size, args.generation_batch_size):
            count = min(args.generation_batch_size, args.group_size - group_begin)
            torch.manual_seed(stable_seed(args.seed, f'{row["id"]}|group={group_begin}'))
            with torch.inference_mode():
                generated_parts.append(model.generate(
                    student_prefix,
                    attention_mask=torch.ones_like(student_prefix),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=count,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                ))
        width = max(part.shape[1] for part in generated_parts)
        generated = torch.cat([
            F.pad(part, (0, width - part.shape[1]), value=tokenizer.pad_token_id)
            for part in generated_parts
        ], 0)
        student_length = student_prefix.shape[1]
        completions = generated[:, student_length:]
        valid = completions != tokenizer.pad_token_id
        # Preserve EOS as an observed action; tokens after the first EOS are padding.
        if tokenizer.eos_token_id is not None:
            eos = completions == tokenizer.eos_token_id
            after_eos = eos.cumsum(-1) > 1
            valid &= ~after_eos
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        rewards = torch.tensor(
            [float(answers_equivalent(boxed(text), row["answer"])) for text in texts],
            device=model.device,
        )
        std = rewards.std(unbiased=False)
        advantages = (rewards - rewards.mean()) / (std + 1e-4) if std > 0 else torch.zeros_like(rewards)

        teacher_prefix = tokenizer(
            teacher_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        chi_parts, align_parts, py_parts, qy_parts = [], [], [], []
        for batch_begin in range(0, args.group_size, args.score_batch_size):
            batch_end = min(batch_begin + args.score_batch_size, args.group_size)
            student_ids = generated[batch_begin:batch_end]
            student_mask = torch.ones_like(student_ids)
            student_mask[:, student_length:] = valid[batch_begin:batch_end].long()
            student_logits = model_logits(
                model, student_ids, student_mask, student_length, completions.shape[1]
            )
            teacher_prefix_batch = teacher_prefix.expand(batch_end - batch_begin, -1)
            teacher_ids = torch.cat((teacher_prefix_batch, completions[batch_begin:batch_end]), dim=1)
            teacher_mask = torch.cat((
                torch.ones_like(teacher_prefix_batch), valid[batch_begin:batch_end].long()
            ), dim=1)
            teacher_logits = model_logits(
                model, teacher_ids, teacher_mask, teacher_prefix.shape[1], completions.shape[1]
            )
            batch_chi, batch_align, batch_py, batch_qy = [], [], [], []
            for begin in range(0, completions.shape[1], args.signal_chunk):
                end = min(begin + args.signal_chunk, completions.shape[1])
                p = torch.softmax(student_logits[:, begin:end].float(), -1)
                q = torch.softmax(teacher_logits[:, begin:end].float(), -1)
                delta = q - p
                action = completions[batch_begin:batch_end, begin:end].unsqueeze(-1)
                sampled_delta = delta.gather(-1, action).squeeze(-1)
                baseline = (p * delta).sum(-1)
                advantage = advantages[batch_begin:batch_end, None]
                chi = advantage * (sampled_delta - baseline)
                p_y = p.gather(-1, action).squeeze(-1)
                q_y = q.gather(-1, action).squeeze(-1)
                reward_norm = advantage.abs() * torch.sqrt(
                    (1 - 2 * p_y + p.square().sum(-1)).clamp_min(0)
                )
                teacher_norm = delta.norm(dim=-1)
                alignment = chi / (reward_norm * teacher_norm).clamp_min(1e-12)
                batch_chi.append(chi.cpu()); batch_align.append(alignment.cpu())
                batch_py.append(p_y.cpu()); batch_qy.append(q_y.cpu())
                del p, q, delta
            chi_parts.append(torch.cat(batch_chi, 1)); align_parts.append(torch.cat(batch_align, 1))
            py_parts.append(torch.cat(batch_py, 1)); qy_parts.append(torch.cat(batch_qy, 1))
            del student_logits, teacher_logits

        payload = {
            "dataset_index": index, "id": row["id"], "problem": row["problem"],
            "answer": row["answer"], "reference": row["references"][0],
            "completion_ids": completions.cpu(), "valid_mask": valid.cpu(),
            "texts": texts, "boxed": [boxed(text) for text in texts],
            "rewards": rewards.cpu(), "advantages": advantages.cpu(),
            "chi": torch.cat(chi_parts, 1).to(torch.float16),
            "sample_alignment": torch.cat(align_parts, 1).to(torch.float16),
            "student_sampled_prob": torch.cat(py_parts, 1).to(torch.float16),
            "teacher_sampled_prob": torch.cat(qy_parts, 1).to(torch.float16),
            "student_prompt_ids": student_prefix.cpu(),
            "config": vars(args),
        }
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary); temporary.replace(destination)
        del generated, completions, payload
        torch.cuda.empty_cache()
        print(
            f"row={index} local={local_index + 1}/{len(selected)} reward_mean={float(rewards.mean()):.3f} "
            f"elapsed={time.time()-started:.1f}s", flush=True,
        )

    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
