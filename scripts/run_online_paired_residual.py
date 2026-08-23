#!/usr/bin/env python3
"""Online autoregressive paired-residual interventions.

At every generated step, correct and corrupted references are evaluated on the
same *current generated prefix*. Their layer-state difference is recomputed and
optionally injected into the corrupted arm before choosing the next token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .run_privileged_effect_diagnostic import TRANSITION, parse_layers
except ImportError:  # Direct `python scripts/...` execution.
    from run_privileged_effect_diagnostic import TRANSITION, parse_layers


def online_teacher_prompt(tokenizer, problem: str, reference: str, enable_thinking: bool) -> str:
    content = (
        f"Problem: {problem}\n\nHere is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{reference}\n=== Reference Solution End ===\n"
        f"{TRANSITION}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking,
    )


def boxed(text: str) -> str | None:
    position = text.rfind("\\boxed{")
    if position < 0:
        return None
    start, depth = position + 7, 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return None


def normalize_answer(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def answers_equivalent(predicted: str | None, target: str) -> bool:
    """Math-aware boxed-answer grading with deterministic string fallback."""
    if predicted is None:
        return False
    try:
        left = parse(f"${predicted}$", fallback_mode="no_fallback")
        right = parse(f"${target}$", fallback_mode="no_fallback")
        return bool(verify(right, left, timeout_seconds=5))
    except Exception:
        return normalize_answer(predicted) == normalize_answer(target)


def schedule_active(schedule: str, step: int) -> bool:
    if schedule == "first8":
        return step < 8
    return step % int(schedule.removeprefix("every")) == 0


def arm_seed(base_seed: int, record_id: object, layer: int, alpha: float, schedule: str, mode: str) -> int:
    """Stable per-arm seed, independent of process ordering and Python hash randomization."""
    identity = f"{base_seed}|{record_id}|{layer}|{alpha:.8g}|{schedule}|{mode}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


def padded_pair(tokenizer, correct_prefix, corrupt_prefix, generated, device):
    sequences = [correct_prefix + generated, corrupt_prefix + generated]
    width = max(map(len, sequences))
    pad = tokenizer.pad_token_id
    ids = torch.tensor([[pad] * (width - len(seq)) + seq for seq in sequences], device=device)
    mask = torch.tensor([[0] * (width - len(seq)) + [1] * len(seq) for seq in sequences], device=device)
    return ids, mask


def cached_forward(model, input_ids, layer: int, past_key_values=None, addition=None):
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["state"] = hidden[:, -1, :].detach().float()
        if addition is None:
            return output
        modified = hidden.clone()
        modified[:, -1, :] += addition.to(hidden.device, hidden.dtype)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids, past_key_values=past_key_values,
                use_cache=True, logits_to_keep=1, return_dict=True,
            )
        return captured["state"], output.logits[:, -1, :], output.past_key_values
    finally:
        handle.remove()


def online_generate(
    model, tokenizer, correct_prefix, corrupt_prefix, layer, alpha, mode,
    schedule, max_new_tokens, seed, projection=None,
):
    generated: list[int] = []
    generator = torch.Generator(device=model.device).manual_seed(seed)
    intervention_norms = []
    correct_ids = torch.tensor([correct_prefix], device=model.device)
    corrupt_ids = torch.tensor([corrupt_prefix], device=model.device)
    correct_state, _, correct_cache = cached_forward(model, correct_ids, layer)
    corrupt_state, _, corrupt_cache = cached_forward(model, corrupt_ids, layer)
    residual = correct_state[0] - corrupt_state[0]
    if projection is not None:
        residual = residual @ projection @ projection.T

    def make_addition(step: int, value: torch.Tensor):
        if not schedule_active(schedule, step):
            return None
        if mode == "rescue":
            return alpha * value
        if mode == "reverse":
            return -alpha * value
        if mode == "random":
            noise = torch.randn(value.shape, generator=generator, device=model.device)
            return alpha * F.normalize(noise.float(), dim=0) * value.norm()
        raise ValueError(mode)

    # This third cache carries the causal history of earlier patches.
    _, logits, patched_cache = cached_forward(
        model, corrupt_ids, layer, addition=make_addition(0, residual)
    )
    for step in range(max_new_tokens):
        intervention_norms.append(float(residual.norm()))
        token = int(logits.argmax(dim=-1))
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        token_ids = torch.tensor([[token]], device=model.device)
        correct_state, _, correct_cache = cached_forward(
            model, token_ids, layer, past_key_values=correct_cache
        )
        corrupt_state, _, corrupt_cache = cached_forward(
            model, token_ids, layer, past_key_values=corrupt_cache
        )
        residual = correct_state[0] - corrupt_state[0]
        if projection is not None:
            residual = residual @ projection @ projection.T
        _, logits, patched_cache = cached_forward(
            model, token_ids, layer, past_key_values=patched_cache,
            addition=make_addition(step + 1, residual),
        )
    return generated, sum(intervention_norms) / len(intervention_norms)


def ordinary_generate(model, input_ids, max_new_tokens):
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    return output[0, input_ids.shape[1] :].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--n_examples", type=int, default=4)
    parser.add_argument("--ids", default=None, help="Comma-separated record ids; overrides prefix selection.")
    parser.add_argument("--layers", default="25")
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--schedules", default="every1")
    parser.add_argument("--modes", default="rescue,reverse,random")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--projection_basis", default=None)
    parser.add_argument("--projection_kind", choices=("pca", "contrastive"), default="pca")
    parser.add_argument("--projection_rank", type=int, default=None)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    if args.ids:
        selected_ids = set(args.ids.split(","))
        rows = [row for row in rows if str(row.get("id")) in selected_ids]
    else:
        rows = rows[: args.n_examples]
    rows = [
        row for index, row in enumerate(rows)
        if index % args.shard_count == args.shard_index
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    completed = sum(1 for line in output.read_text().splitlines() if line) if output.exists() else 0
    if completed >= len(rows):
        print(f"already complete: {completed} rows")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    alphas = [float(value) for value in args.alphas.split(",")]
    schedules = args.schedules.split(",")
    modes = [] if args.modes == "none" else args.modes.split(",")
    projection = None
    if args.projection_basis:
        if args.projection_rank is None:
            parser.error("--projection_rank is required with --projection_basis")
        basis_payload = torch.load(args.projection_basis, map_location="cpu", weights_only=False)
        key = f"{args.projection_kind}_basis"
        projection = basis_payload[key][:, : args.projection_rank].float().to(model.device)
    started = time.time()

    with output.open("a") as handle:
        for position, row in enumerate(rows[completed:], start=completed):
            correct_text = online_teacher_prompt(
                tokenizer, row["problem"], row["references"][0], args.enable_thinking
            )
            corrupt_text = online_teacher_prompt(
                tokenizer, row["problem"], row["matched_controls"]["dependency_swap"],
                args.enable_thinking,
            )
            correct_prefix = tokenizer(correct_text, add_special_tokens=False)["input_ids"]
            corrupt_prefix = tokenizer(corrupt_text, add_special_tokens=False)["input_ids"]
            correct_ids = torch.tensor([correct_prefix], device=model.device)
            corrupt_ids = torch.tensor([corrupt_prefix], device=model.device)
            correct_tokens = ordinary_generate(model, correct_ids, args.max_new_tokens)
            corrupt_tokens = ordinary_generate(model, corrupt_ids, args.max_new_tokens)
            result = {
                "id": row.get("id", position),
                "answer": row["answer"],
                "correct_reference": tokenizer.decode(correct_tokens, skip_special_tokens=True),
                "corrupt_baseline": tokenizer.decode(corrupt_tokens, skip_special_tokens=True),
                "arms": {},
            }
            for layer in layers:
                for alpha in alphas:
                    for schedule in schedules:
                        for mode in modes:
                            key = f"layer={layer}:alpha={alpha:g}:schedule={schedule}:mode={mode}"
                            seed = arm_seed(args.seed, row.get("id", position), layer, alpha, schedule, mode)
                            tokens, norm = online_generate(
                                model, tokenizer, correct_prefix, corrupt_prefix, layer,
                                alpha, mode, schedule, args.max_new_tokens,
                                seed,
                                projection=projection,
                            )
                            text = tokenizer.decode(tokens, skip_special_tokens=True)
                            result["arms"][key] = {
                                "text": text,
                                "boxed": boxed(text),
                                "exact_match": normalize_answer(boxed(text)) == normalize_answer(row["answer"]),
                                "math_equivalent": answers_equivalent(boxed(text), row["answer"]),
                                "mean_residual_norm": norm,
                                "tokens": len(tokens),
                                "seed": seed,
                                "projection": (
                                    {"kind": args.projection_kind, "rank": args.projection_rank}
                                    if projection is not None else None
                                ),
                            }
            for name in ("correct_reference", "corrupt_baseline"):
                answer = boxed(result[name])
                result[f"{name}_boxed"] = answer
                result[f"{name}_exact_match"] = normalize_answer(answer) == normalize_answer(row["answer"])
                result[f"{name}_math_equivalent"] = answers_equivalent(answer, row["answer"])
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"row={position} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
