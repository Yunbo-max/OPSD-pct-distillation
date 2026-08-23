#!/usr/bin/env python3
"""Matched privileged-effect diagnostics and causal residual interventions.

The problem, continuation, model, and token positions are held fixed.  Only the
privileged reference changes, so each delta is an aligned intervention effect:
    delta = state(problem, reference, continuation) - state(problem, continuation)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )


def student_prompt(tokenizer, problem: str) -> str:
    content = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def parse_layers(value: str, n_layers: int) -> list[int]:
    if value == "all":
        return list(range(n_layers))
    layers = sorted({int(item) for item in value.split(",") if item.strip()})
    if not layers or layers[0] < 0 or layers[-1] >= n_layers:
        raise ValueError(f"layers must be in [0, {n_layers - 1}]")
    return layers


def encode(tokenizer, prompt: str, continuation: torch.Tensor, device: torch.device):
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids = torch.cat([prefix, continuation], dim=1)
    return ids, torch.ones_like(ids), prefix.shape[1]


@contextmanager
def projection_capture(model, layers: list[int], continuation_length: int):
    captured: dict[str, dict[int, torch.Tensor]] = {"k": {}, "v": {}}
    handles = []

    def hook(kind: str, layer: int):
        def save(_module, _inputs, output):
            captured[kind][layer] = output[:, -continuation_length:, :].detach().float().cpu()
        return save

    for layer in layers:
        attention = model.model.layers[layer].self_attn
        handles.append(attention.k_proj.register_forward_hook(hook("k", layer)))
        handles.append(attention.v_proj.register_forward_hook(hook("v", layer)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def extract_states(model, ids, mask, prefix_length: int, layers: list[int]):
    continuation_length = ids.shape[1] - prefix_length
    with projection_capture(model, layers, continuation_length) as projections:
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        residual = {
            layer: output.hidden_states[layer + 1][:, -continuation_length:, :]
            .detach().float().cpu()
            for layer in layers
        }
        copied = {
            kind: {layer: value.clone() for layer, value in values.items()}
            for kind, values in projections.items()
        }
    return {"h": residual, **copied}


def cosine_tokens(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=-1).mean())


def effect_metrics(deltas: list[torch.Tensor], wrong: torch.Tensor) -> dict[str, float]:
    cc = [
        cosine_tokens(deltas[left], deltas[right])
        for left in range(len(deltas))
        for right in range(left + 1, len(deltas))
    ]
    cw = [cosine_tokens(delta, wrong) for delta in deltas]
    mean = torch.stack(deltas).mean(dim=0)
    # Tokenwise effective rank of the mean effect, computed from its small Gram matrix.
    singular = torch.linalg.svdvals(mean.squeeze(0))
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    effective_rank = float(entropy.exp())
    rank90 = int((energy.cumsum(0) < 0.9 * energy.sum()).sum()) + 1
    return {
        "correct_correct_cosine": sum(cc) / len(cc),
        "correct_wrong_cosine": sum(cw) / len(cw),
        "cosine_margin": sum(cc) / len(cc) - sum(cw) / len(cw),
        "correct_delta_norm": float(torch.stack(deltas).norm(dim=-1).mean()),
        "wrong_delta_norm": float(wrong.norm(dim=-1).mean()),
        "mean_delta_effective_rank": effective_rank,
        "mean_delta_rank90": rank90,
    }


def token_logp(logits: torch.Tensor, continuation: torch.Tensor) -> float:
    # A hidden state at continuation position t predicts continuation token t+1.
    scores = logits[:, -continuation.shape[1] : -1, :].float()
    targets = continuation[:, 1:]
    values = F.log_softmax(scores, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(values.mean())


def injected_logp(model, ids, mask, continuation, layer: int, delta: torch.Tensor, alpha: float):
    addition = (alpha * delta).to(device=model.device, dtype=next(model.parameters()).dtype)

    def inject(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        modified[:, -addition.shape[1] :, :] += addition
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = model.model.layers[layer].register_forward_hook(inject)
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return token_logp(logits, continuation)
    finally:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--continuation_tokens", type=int, default=128)
    parser.add_argument("--layers", default="3,7,11,15,19,23,27,31,35")
    parser.add_argument("--inject_layers", default="11,23,35")
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    records = records[: args.n_examples]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for line in output.read_text().splitlines() if line) if output.exists() else 0
    if completed >= len(records):
        print(f"already complete: {completed} rows")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    inject_layers = parse_layers(args.inject_layers, model.config.num_hidden_layers)
    alphas = [float(value) for value in args.alphas.split(",")]
    if not set(inject_layers).issubset(layers):
        raise ValueError("inject_layers must be included in layers")

    started = time.time()
    with output.open("a") as handle:
        for position, record in enumerate(records[completed:], start=completed):
            # Use verified solution tokens as a correctness-bearing, exactly matched rollout.
            continuation = tokenizer(
                record["solution"],
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.continuation_tokens,
            ).input_ids.to(model.device)
            base_ids, base_mask, base_prefix = encode(
                tokenizer, student_prompt(tokenizer, record["problem"]), continuation, model.device
            )
            base = extract_states(model, base_ids, base_mask, base_prefix, layers)
            privileged = []
            controls = record.get("matched_controls") or {"legacy_wrong": record["wrong_reference"]}
            control_names = list(controls)
            references = [*record["references"][:4], *[controls[name] for name in control_names]]
            for reference in references:
                ids, mask, prefix = encode(
                    tokenizer,
                    teacher_prompt(tokenizer, record["problem"], reference),
                    continuation,
                    model.device,
                )
                privileged.append(extract_states(model, ids, mask, prefix, layers))

            result = {
                "id": record.get("id", position),
                "continuation_tokens": int(continuation.shape[1]),
                "layers": {},
            }
            with torch.inference_mode():
                base_logits = model(input_ids=base_ids, attention_mask=base_mask, use_cache=False).logits
            base_logp = token_logp(base_logits, continuation)
            generator = torch.Generator(device="cpu").manual_seed(args.seed + position)

            for layer in layers:
                layer_result = {}
                h_correct = [item["h"][layer] - base["h"][layer] for item in privileged[:4]]
                for kind in ("h", "k", "v"):
                    correct = [item[kind][layer] - base[kind][layer] for item in privileged[:4]]
                    for control_index, control_name in enumerate(control_names, start=4):
                        wrong = privileged[control_index][kind][layer] - base[kind][layer]
                        layer_result[f"{kind}:{control_name}"] = effect_metrics(correct, wrong)

                if args.causal and layer in inject_layers:
                    mean_correct = torch.stack(h_correct).mean(dim=0)
                    random = torch.randn(mean_correct.shape, generator=generator)
                    random = F.normalize(random, dim=-1) * mean_correct.norm(dim=-1, keepdim=True)
                    for alpha in alphas:
                        correct_logp = injected_logp(
                            model, base_ids, base_mask, continuation, layer, mean_correct, alpha
                        )
                        random_logp = injected_logp(
                            model, base_ids, base_mask, continuation, layer, random, alpha
                        )
                        causal = {
                            "base_logp": base_logp,
                            "correct_logp": correct_logp,
                            "random_logp": random_logp,
                            "correct_gain": correct_logp - base_logp,
                            "random_gain": random_logp - base_logp,
                        }
                        for control_index, control_name in enumerate(control_names, start=4):
                            control_delta = privileged[control_index]["h"][layer] - base["h"][layer]
                            # Compare directions at identical tokenwise intervention energy.
                            control_delta = F.normalize(control_delta, dim=-1) * mean_correct.norm(
                                dim=-1, keepdim=True
                            )
                            control_logp = injected_logp(
                                model, base_ids, base_mask, continuation, layer, control_delta, alpha
                            )
                            causal[f"{control_name}_logp"] = control_logp
                            causal[f"{control_name}_gain"] = control_logp - base_logp
                        layer_result[f"causal:alpha={alpha:g}"] = causal
                result["layers"][str(layer)] = layer_result

            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(f"row={position} elapsed={time.time() - started:.1f}s", flush=True)
            del base, privileged, base_logits
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
