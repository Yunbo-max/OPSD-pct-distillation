#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pct.losses import Flow, fgw_flow_loss, ot_flow_loss, phf_direction_loss, uot_flow_loss


TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def teacher_prompt(tokenizer, problem: str, reference: str) -> str:
    content = (
        f"Problem: {problem}\n\nHere is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{reference}\n=== Reference Solution End ===\n"
        f"{TRANSITION}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def student_prompt(tokenizer, problem: str) -> str:
    content = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def conditioned_flow(model, tokenizer, prompt: str, continuation: torch.Tensor) -> Flow:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    input_ids = torch.cat([encoded, continuation], dim=1)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    response_hidden = output.last_hidden_state[:, encoded.shape[1] :, :].float()
    z = F.normalize(response_hidden[:, 1:, :] - response_hidden[:, :-1, :], dim=-1).unsqueeze(1)
    mask = torch.ones((1, z.shape[2]), dtype=torch.bool, device=z.device)
    return Flow(z=z, mask=mask)


def conditioned_flows(
    model, tokenizer, prompts: list[str], continuation: torch.Tensor
) -> list[Flow]:
    continuation_ids = continuation[0].tolist()
    sequences = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"] + continuation_ids
        for prompt in prompts
    ]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    batch = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(model.device)
    tokenizer.padding_side = original_padding_side
    with torch.inference_mode():
        output = model.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        )
    response_hidden = output.last_hidden_state[:, -continuation.shape[1] :, :].float()
    z = F.normalize(response_hidden[:, 1:, :] - response_hidden[:, :-1, :], dim=-1).unsqueeze(1)
    return [
        Flow(
            z=z[index : index + 1],
            mask=torch.ones((1, z.shape[2]), dtype=torch.bool, device=z.device),
        )
        for index in range(z.shape[0])
    ]


def reference_flow(
    model, tokenizer, problem: str, reference: str, max_reference_tokens: int
) -> Flow:
    prefix = tokenizer(
        f"Problem: {problem}\n\nReference reasoning:\n",
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(model.device)
    reference_ids = tokenizer(
        reference,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_reference_tokens,
    ).input_ids.to(model.device)
    input_ids = torch.cat([prefix, reference_ids], dim=1)
    with torch.inference_mode():
        output = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
    hidden = output.last_hidden_state[:, prefix.shape[1] :, :].float()
    z = F.normalize(hidden[:, 1:, :] - hidden[:, :-1, :], dim=-1).unsqueeze(1)
    mask = torch.ones((1, z.shape[2]), dtype=torch.bool, device=z.device)
    return Flow(z=z, mask=mask)


def pair_metrics(left: Flow, right: Flow, args: argparse.Namespace) -> dict[str, float]:
    length = min(left.z.shape[2], right.z.shape[2])
    left_aligned = Flow(left.z[:, :, :length], left.mask[:, :length])
    right_aligned = Flow(right.z[:, :, :length], right.mask[:, :length])
    euclidean = (left_aligned.z - right_aligned.z).norm(dim=-1).mean()
    phf = phf_direction_loss(left_aligned, right_aligned)
    sinkhorn, sinkhorn_mass = ot_flow_loss(
        left, right, max_atoms=args.max_atoms, epsilon=args.epsilon, sinkhorn_iters=args.sinkhorn_iters
    )
    uot, uot_raw, uot_normalized, uot_mass = uot_flow_loss(
        left,
        right,
        max_atoms=args.max_atoms,
        epsilon=args.epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        rho=args.uot_rho,
    )
    fgw, fgw_mass = fgw_flow_loss(
        left,
        right,
        max_atoms=args.max_atoms,
        epsilon=args.epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        fgw_outer=args.fgw_outer,
    )
    return {
        "euclidean": float(euclidean.cpu()),
        "phf_cosine": float(phf.cpu()),
        "sinkhorn": float(sinkhorn.cpu()),
        "sinkhorn_mass": float(sinkhorn_mass.cpu()),
        "fgw": float(fgw.cpu()),
        "fgw_mass": float(fgw_mass.cpu()),
        "uot_objective": float(uot.cpu()),
        "uot_transport_cost_raw": float(uot_raw.cpu()),
        "uot_transport_cost_normalized": float(uot_normalized.cpu()),
        "uot_mass": float(uot_mass.cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run teacher-conditioned hidden-flow diagnostics.")
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--continuation_tokens", type=int, default=128)
    parser.add_argument("--mode", choices=("reference", "conditioned"), default="reference")
    parser.add_argument("--max_reference_tokens", type=int, default=4096)
    parser.add_argument("--max_atoms", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iters", type=int, default=40)
    parser.add_argument("--uot_rho", type=float, default=0.5)
    parser.add_argument("--fgw_outer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line]
    records = records[: args.n_examples]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if output.exists():
        completed = sum(1 for line in output.read_text(encoding="utf-8").splitlines() if line)
    if completed >= len(records):
        print(f"already complete: {completed} rows")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        for position, record in enumerate(records[completed:], start=completed):
            references = [*record["references"][:4], record["wrong_reference"], record["shuffled_reference"]]
            kinds = ["correct"] * 4 + ["wrong", "shuffled"]
            continuation = None
            generated = None
            if args.mode == "conditioned":
                prompt = student_prompt(tokenizer, record["problem"])
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                torch.manual_seed(args.seed + position)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.continuation_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                continuation = generated[:, encoded.input_ids.shape[1] :]
                flows = [
                    conditioned_flow(
                        model,
                        tokenizer,
                        teacher_prompt(tokenizer, record["problem"], ref),
                        continuation,
                    )
                    for ref in references
                ]
            else:
                flows = [
                    reference_flow(
                        model, tokenizer, record["problem"], ref, args.max_reference_tokens
                    )
                    for ref in references
                ]
            pairs = []
            for left in range(len(flows)):
                for right in range(left + 1, len(flows)):
                    if kinds[left] == kinds[right] == "correct":
                        group = "correct_correct"
                    elif "correct" in (kinds[left], kinds[right]) and "wrong" in (kinds[left], kinds[right]):
                        group = "correct_wrong"
                    elif "correct" in (kinds[left], kinds[right]) and "shuffled" in (kinds[left], kinds[right]):
                        group = "correct_shuffled"
                    else:
                        continue
                    pairs.append(
                        {
                            "left": left,
                            "right": right,
                            "group": group,
                            **pair_metrics(flows[left], flows[right], args),
                        }
                    )
            result = {
                "id": record.get("id", position),
                "mode": args.mode,
                "continuation_tokens": int(continuation.shape[1]) if continuation is not None else 0,
                "continuation": tokenizer.decode(continuation[0], skip_special_tokens=True)
                if continuation is not None
                else None,
                "pairs": pairs,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.time() - started
            print(f"row={position} pairs={len(pairs)} elapsed={elapsed:.1f}s", flush=True)
            del flows, continuation, generated
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
