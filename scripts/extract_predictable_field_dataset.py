#!/usr/bin/env python3
"""Cache deployment-visible states and oracle paired residuals problem by problem.

For the same problem, continuation, and token positions, this extracts

    X_t = h_t(problem, continuation)                         (deployment visible)
    C_t = h_t(problem, correct reference, continuation)
          - h_t(problem, corrupt reference, continuation)   (training-only oracle)

Each problem is written atomically to make long 4B extraction resumable without
partially valid tensor files. Dataset order is preserved so problem-level split
assignment can be reproduced downstream from a single seed.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .run_online_paired_residual import online_teacher_prompt
except ImportError:
    from run_online_paired_residual import online_teacher_prompt


def deployment_prompt(tokenizer, problem: str) -> str:
    content = (
        f"Problem: {problem}\n\nPlease reason step by step, and put your final answer "
        "within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


@contextmanager
def capture_layers(model, layers: list[int], continuation_length: int):
    values: dict[int, torch.Tensor] = {}
    handles = []

    def hook(layer: int):
        def save(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            values[layer] = hidden[0, -continuation_length:, :].detach().cpu().to(torch.float16)
        return save

    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(hook(layer)))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def extract(model, input_ids: torch.Tensor, layers: list[int], length: int) -> dict[int, torch.Tensor]:
    with capture_layers(model, layers, length) as captured:
        with torch.inference_mode():
            model.model(input_ids=input_ids, use_cache=False)
    return {layer: value.clone() for layer, value in captured.items()}


def encode(tokenizer, prompt: str, continuation: torch.Tensor, device) -> torch.Tensor:
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return torch.cat((prefix, continuation), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--x_layers", default="21,23,25")
    parser.add_argument("--target_layer", type=int, default=23)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard_index must be in [0, shard_count)")
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    rows = rows[: args.n_examples]
    indexed = [(index, row) for index, row in enumerate(rows)
               if index % args.shard_count == args.shard_index]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_layers = sorted({int(value) for value in args.x_layers.split(",")})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    all_layers = sorted(set(x_layers + [args.target_layer]))
    if all_layers[0] < 0 or all_layers[-1] >= model.config.num_hidden_layers:
        parser.error(f"layers must be in [0, {model.config.num_hidden_layers - 1}]")

    started, completed = time.time(), 0
    for index, row in indexed:
        destination = out_dir / f"{index:06d}.pt"
        if destination.exists():
            completed += 1
            continue
        continuation = tokenizer(
            row["solution"], return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=args.tokens,
        ).input_ids.to(model.device)
        length = continuation.shape[1]
        student_ids = encode(tokenizer, deployment_prompt(tokenizer, row["problem"]), continuation, model.device)
        correct_ids = encode(
            tokenizer,
            online_teacher_prompt(tokenizer, row["problem"], row["references"][0], False),
            continuation, model.device,
        )
        corrupt_ids = encode(
            tokenizer,
            online_teacher_prompt(
                tokenizer, row["problem"], row["matched_controls"]["dependency_swap"], False
            ),
            continuation, model.device,
        )
        student = extract(model, student_ids, x_layers, length)
        correct = extract(model, correct_ids, [args.target_layer], length)[args.target_layer]
        corrupt = extract(model, corrupt_ids, [args.target_layer], length)[args.target_layer]
        payload = {
            "dataset_index": index,
            "id": row["id"],
            "tokens": length,
            "x_layers": x_layers,
            "target_layer": args.target_layer,
            "x": student,
            "c": correct - corrupt,
        }
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        completed += 1
        print(
            f"row={index} shard={args.shard_index}/{args.shard_count} tokens={length} "
            f"completed={completed}/{len(indexed)} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    manifest = {
        "dataset": str(Path(args.dataset)), "model": args.model,
        "n_examples": len(rows), "tokens": args.tokens,
        "x_layers": x_layers, "target_layer": args.target_layer,
        "shard_count": args.shard_count,
    }
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"completed": completed, "assigned": len(indexed), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
