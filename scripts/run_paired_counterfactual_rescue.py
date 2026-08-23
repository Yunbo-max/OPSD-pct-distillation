#!/usr/bin/env python3
"""Paired sufficiency/necessity interventions for correctness residuals.

For a correct reference and its locally matched corrupted control, define
    c_l,t = h_l,t(correct) - h_l,t(control).
Rescue injects +alpha*c into the control run; necessity injects -alpha*c into
the correct run. The problem, continuation, token positions, and most reference
text remain matched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_privileged_effect_diagnostic import (
    encode,
    parse_layers,
    teacher_prompt,
    token_logp,
)


def states_and_logp(model, ids, mask, continuation, layers):
    with torch.inference_mode():
        output = model(
            input_ids=ids, attention_mask=mask, use_cache=False,
            output_hidden_states=True, return_dict=True,
        )
    length = continuation.shape[1]
    states = {
        layer: output.hidden_states[layer + 1][:, -length:, :].detach().float().cpu()
        for layer in layers
    }
    return states, token_logp(output.logits, continuation)


def window_injected_logp(
    model, ids, mask, continuation, layer: int, delta: torch.Tensor,
    alpha: float, window: int,
) -> float:
    addition = (alpha * delta[:, :window]).to(
        device=model.device, dtype=next(model.parameters()).dtype
    )
    continuation_length = continuation.shape[1]

    def inject(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        start = hidden.shape[1] - continuation_length
        modified[:, start : start + addition.shape[1], :] += addition
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
    parser.add_argument("--n_examples", type=int, default=108)
    parser.add_argument("--continuation_tokens", type=int, default=128)
    parser.add_argument("--layers", default="9,11,13,15,17,19,21,23,25")
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--windows", default="128")
    parser.add_argument("--controls", default="all")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    rows = rows[: args.n_examples]
    rows = [
        row for index, row in enumerate(rows)
        if index % args.shard_count == args.shard_index
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for line in output.read_text().splitlines() if line) if output.exists() else 0
    if completed >= len(rows):
        print(f"already complete: {completed} rows")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    alphas = [float(value) for value in args.alphas.split(",")]
    requested_windows = [int(value) for value in args.windows.split(",")]
    started = time.time()

    with output.open("a") as handle:
        for position, row in enumerate(rows[completed:], start=completed):
            continuation = tokenizer(
                row["solution"], return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=args.continuation_tokens,
            ).input_ids.to(model.device)
            correct_ids, correct_mask, _ = encode(
                tokenizer,
                teacher_prompt(tokenizer, row["problem"], row["references"][0]),
                continuation,
                model.device,
            )
            correct_states, correct_logp = states_and_logp(
                model, correct_ids, correct_mask, continuation, layers
            )
            controls = row["matched_controls"]
            if args.controls != "all":
                requested_controls = set(args.controls.split(","))
                controls = {key: value for key, value in controls.items() if key in requested_controls}
            result = {"id": row.get("id", position), "layers": {}}

            for control_index, (control_name, control_text) in enumerate(controls.items()):
                control_ids, control_mask, _ = encode(
                    tokenizer,
                    teacher_prompt(tokenizer, row["problem"], control_text),
                    continuation,
                    model.device,
                )
                control_states, control_logp = states_and_logp(
                    model, control_ids, control_mask, continuation, layers
                )
                for layer in layers:
                    residual = correct_states[layer] - control_states[layer]
                    generator = torch.Generator(device="cpu").manual_seed(
                        args.seed + args.shard_index * 1000003 + position * 1009 + layer * 17 + control_index
                    )
                    random = torch.randn(residual.shape, generator=generator)
                    random = F.normalize(random, dim=-1) * residual.norm(dim=-1, keepdim=True)
                    bucket = result["layers"].setdefault(str(layer), {})
                    for alpha in alphas:
                      for requested_window in requested_windows:
                        window = min(requested_window, continuation.shape[1])
                        rescue_logp = window_injected_logp(
                            model, control_ids, control_mask, continuation,
                            layer, residual, alpha, window,
                        )
                        necessity_logp = window_injected_logp(
                            model, correct_ids, correct_mask, continuation,
                            layer, -residual, alpha, window,
                        )
                        random_logp = window_injected_logp(
                            model, control_ids, control_mask, continuation,
                            layer, random, alpha, window,
                        )
                        bucket[f"{control_name}:alpha={alpha:g}:window={window}"] = {
                            "correct_logp": correct_logp,
                            "control_logp": control_logp,
                            "paired_logp_gap": correct_logp - control_logp,
                            "residual_norm": float(residual.norm(dim=-1).mean()),
                            "rescue_logp": rescue_logp,
                            "rescue_gain": rescue_logp - control_logp,
                            "necessity_logp": necessity_logp,
                            "necessity_change": necessity_logp - correct_logp,
                            "random_gain": random_logp - control_logp,
                        }
                del control_states
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(f"row={position} elapsed={time.time() - started:.1f}s", flush=True)
            del correct_states
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
