#!/usr/bin/env python3
"""Reference-free online intervention with learned privileged-field predictors."""
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
    from .analyze_predictable_field import predict
    from .extract_predictable_field_dataset import deployment_prompt
    from .run_online_paired_residual import (
        answers_equivalent, boxed, cached_forward, online_teacher_prompt, ordinary_generate,
    )
except ImportError:
    from analyze_predictable_field import predict
    from extract_predictable_field_dataset import deployment_prompt
    from run_online_paired_residual import (
        answers_equivalent, boxed, cached_forward, online_teacher_prompt, ordinary_generate,
    )


def stable_seed(base: int, record_id: object, arm: str) -> int:
    value = f"{base}|{record_id}|{arm}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63 - 1)


def controller_forward(model, input_ids, layer: int, controller: dict, past=None,
                       direction: str = "forward", generator=None, centered: bool = False):
    saved = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        state = hidden[:, -1, :].detach().float()
        field = predict(state, controller)[0]
        if centered:
            field = field - controller["mean_c"]
        if direction == "reverse":
            field = -field
        elif direction == "random":
            noise = torch.randn(field.shape, generator=generator, device=field.device)
            field = F.normalize(noise.float(), dim=0) * field.norm()
        saved["norm"] = float(field.norm())
        modified = hidden.clone()
        modified[:, -1, :] += field.to(hidden.dtype)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, past_key_values=past, use_cache=True,
                           logits_to_keep=1, return_dict=True)
        return output.logits[:, -1, :], output.past_key_values, saved["norm"]
    finally:
        handle.remove()


def generate_controller(model, tokenizer, prefix: list[int], layer: int, controller: dict,
                        direction: str, max_new_tokens: int, seed: int, centered: bool = False):
    generator = torch.Generator(device=model.device).manual_seed(seed)
    ids = torch.tensor([prefix], device=model.device)
    logits, cache, norm = controller_forward(
        model, ids, layer, controller, direction=direction, generator=generator, centered=centered
    )
    generated, norms = [], [norm]
    for _ in range(max_new_tokens):
        token = int(logits.argmax(-1)); generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        ids = torch.tensor([[token]], device=model.device)
        logits, cache, norm = controller_forward(
            model, ids, layer, controller, past=cache, direction=direction,
            generator=generator, centered=centered
        )
        norms.append(norm)
    return generated, sum(norms) / len(norms)


def generate_oracle(model, tokenizer, student_prefix, correct_prefix, corrupt_prefix,
                    layer: int, max_new_tokens: int):
    student_ids = torch.tensor([student_prefix], device=model.device)
    correct_ids = torch.tensor([correct_prefix], device=model.device)
    corrupt_ids = torch.tensor([corrupt_prefix], device=model.device)
    correct_state, _, correct_cache = cached_forward(model, correct_ids, layer)
    corrupt_state, _, corrupt_cache = cached_forward(model, corrupt_ids, layer)
    field = correct_state[0] - corrupt_state[0]
    _, logits, student_cache = cached_forward(model, student_ids, layer, addition=field)
    generated, norms = [], [float(field.norm())]
    for _ in range(max_new_tokens):
        token = int(logits.argmax(-1)); generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        ids = torch.tensor([[token]], device=model.device)
        correct_state, _, correct_cache = cached_forward(
            model, ids, layer, past_key_values=correct_cache
        )
        corrupt_state, _, corrupt_cache = cached_forward(
            model, ids, layer, past_key_values=corrupt_cache
        )
        field = correct_state[0] - corrupt_state[0]
        _, logits, student_cache = cached_forward(
            model, ids, layer, past_key_values=student_cache, addition=field
        )
        norms.append(float(field.norm()))
    return generated, sum(norms) / len(norms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("split_summary")
    parser.add_argument("controllers")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--methods", default="mean,ridge,rrr64,cca64")
    parser.add_argument("--control_method", default="rrr64")
    parser.add_argument("--control_centered", action="store_true")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    rows = {str(row["id"]): row for row in map(json.loads, Path(args.dataset).open())}
    split = json.loads(Path(args.split_summary).read_text())
    ids = split["split_ids"]["test"][: args.n_examples]
    selected = [(position, rows[str(record_id)]) for position, record_id in enumerate(ids)
                if position % args.shard_count == args.shard_index]
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); output.touch(exist_ok=True)
    completed = sum(bool(line) for line in output.read_text().splitlines())
    selected = selected[completed:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    payload = torch.load(args.controllers, map_location="cpu", weights_only=False)
    layer = int(payload["x_layer"])
    learned = {}
    for name, value in payload["models"].items():
        learned[name] = {key: item.float().to(model.device) if isinstance(item, torch.Tensor) else item
                         for key, item in value.items()}
    mean_controller = {"kind": "mean", "mean_x": payload["mean_x"].float().to(model.device),
                       "mean_c": payload["mean_c"].float().to(model.device)}
    methods = args.methods.split(",")
    started = time.time()
    with output.open("a") as handle:
        for local_position, (position, row) in enumerate(selected, start=completed):
            student_text = deployment_prompt(tokenizer, row["problem"])
            correct_text = online_teacher_prompt(tokenizer, row["problem"], row["references"][0], False)
            corrupt_text = online_teacher_prompt(
                tokenizer, row["problem"], row["matched_controls"]["dependency_swap"], False
            )
            prefixes = {
                "student": tokenizer(student_text, add_special_tokens=False)["input_ids"],
                "correct": tokenizer(correct_text, add_special_tokens=False)["input_ids"],
                "corrupt": tokenizer(corrupt_text, add_special_tokens=False)["input_ids"],
            }
            result = {"id": row["id"], "answer": row["answer"], "arms": {}}
            for name in ("student", "correct", "corrupt"):
                tokens = ordinary_generate(
                    model, torch.tensor([prefixes[name]], device=model.device), args.max_new_tokens
                )
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                result["arms"][name] = {"text": text, "boxed": boxed(text), "tokens": len(tokens)}
            oracle_tokens, oracle_norm = generate_oracle(
                model, tokenizer, prefixes["student"], prefixes["correct"], prefixes["corrupt"],
                layer, args.max_new_tokens,
            )
            text = tokenizer.decode(oracle_tokens, skip_special_tokens=True)
            result["arms"]["oracle"] = {
                "text": text, "boxed": boxed(text), "tokens": len(oracle_tokens),
                "mean_field_norm": oracle_norm,
            }
            for name in methods:
                centered = name.startswith("centered_")
                base_name = name.removeprefix("centered_")
                direction = "forward"
                if base_name == "reverse_mean":
                    base_name, direction = "mean", "reverse"
                controller = mean_controller if base_name == "mean" else learned[base_name]
                tokens, norm = generate_controller(
                    model, tokenizer, prefixes["student"], layer, controller, direction,
                    args.max_new_tokens, stable_seed(args.seed, row["id"], name),
                    centered=centered,
                )
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                result["arms"][name] = {
                    "text": text, "boxed": boxed(text), "tokens": len(tokens), "mean_field_norm": norm,
                }
            control = learned[args.control_method]
            for direction in ("reverse", "random"):
                qualifier = "centered_" if args.control_centered else ""
                name = f"{direction}_{qualifier}{args.control_method}"
                tokens, norm = generate_controller(
                    model, tokenizer, prefixes["student"], layer, control, direction,
                    args.max_new_tokens, stable_seed(args.seed, row["id"], name),
                    centered=args.control_centered,
                )
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                result["arms"][name] = {
                    "text": text, "boxed": boxed(text), "tokens": len(tokens), "mean_field_norm": norm,
                }
            for arm in result["arms"].values():
                arm["math_equivalent"] = answers_equivalent(arm["boxed"], row["answer"])
            handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            print(f"row={position} completed={local_position + 1} elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
