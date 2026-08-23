#!/usr/bin/env python3
"""Extract paired correct-minus-dependency-swap residual trajectories."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_privileged_effect_diagnostic import encode, teacher_prompt


def capture(model, ids, mask, layer: int, length: int) -> torch.Tensor:
    saved = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["value"] = hidden[:, -length:, :].detach().cpu().to(torch.float16)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        return saved["value"].squeeze(0)
    finally:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--paired_results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--n_examples", type=int, default=108)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line][
        : args.n_examples
    ]
    paired = {
        str(row["id"]): row for row in map(json.loads, Path(args.paired_results).open())
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trajectories, metadata = [], []
    started = time.time()
    for position, row in enumerate(rows):
        continuation = tokenizer(
            row["solution"], return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=args.tokens,
        ).input_ids.to(model.device)
        states = []
        for reference in (row["references"][0], row["matched_controls"]["dependency_swap"]):
            ids, mask, _ = encode(
                tokenizer, teacher_prompt(tokenizer, row["problem"], reference),
                continuation, model.device,
            )
            states.append(capture(model, ids, mask, args.layer, continuation.shape[1]))
        residual = states[0] - states[1]
        condition = paired[str(row["id"])]["layers"][str(args.layer)][
            "dependency_swap:alpha=1"
        ]
        trajectories.append(residual)
        metadata.append(
            {
                "id": row["id"], "tokens": residual.shape[0],
                "te": condition["paired_logp_gap"],
                "rescue": condition["rescue_gain"],
                "necessity_magnitude": -condition["necessity_change"],
            }
        )
        print(f"row={position} elapsed={time.time() - started:.1f}s", flush=True)
    torch.save(
        {"layer": args.layer, "trajectories": trajectories, "metadata": metadata}, args.out
    )
    print(f"saved {len(trajectories)} trajectories to {args.out}")


if __name__ == "__main__":
    main()
