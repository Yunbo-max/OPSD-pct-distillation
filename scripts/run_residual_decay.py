#!/usr/bin/env python3
"""Measure downstream persistence after a one-token residual patch."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_privileged_effect_diagnostic import encode, teacher_prompt


def forward_capture(
    model, ids, mask, continuation_length, inject_layer, readout_layer,
    inject_offset=None, addition=None,
):
    saved = {}

    def inject(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["inject"] = hidden[:, -continuation_length:, :].detach().float().cpu()
        if addition is None:
            return output
        modified = hidden.clone()
        position = hidden.shape[1] - continuation_length + inject_offset
        modified[:, position, :] += addition.to(hidden.device, hidden.dtype)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    def readout(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["readout"] = hidden[:, -continuation_length:, :].detach().float().cpu()

    handles = [
        model.model.layers[inject_layer].register_forward_hook(inject),
        model.model.layers[readout_layer].register_forward_hook(readout),
    ]
    try:
        with torch.inference_mode():
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        return saved
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--n_examples", type=int, default=108)
    parser.add_argument("--inject_layer", type=int, default=25)
    parser.add_argument("--readout_layer", type=int, default=35)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--origins", default="0,8,16,32,64,96")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--random_control", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line][
        : args.n_examples
    ]
    origins = [int(value) for value in args.origins.split(",")]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for line in output.read_text().splitlines() if line) if output.exists() else 0
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    started = time.time()
    with output.open("a") as handle:
        for position, row in enumerate(rows[completed:], start=completed):
            continuation = tokenizer(
                row["solution"], return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=args.tokens,
            ).input_ids.to(model.device)
            length = continuation.shape[1]
            contexts = []
            for reference in (row["references"][0], row["matched_controls"]["dependency_swap"]):
                ids, mask, _ = encode(
                    tokenizer, teacher_prompt(tokenizer, row["problem"], reference),
                    continuation, model.device,
                )
                contexts.append((ids, mask, forward_capture(
                    model, ids, mask, length, args.inject_layer, args.readout_layer
                )))
            target = contexts[0][2]["readout"] - contexts[1][2]["readout"]
            inject_residual = contexts[0][2]["inject"] - contexts[1][2]["inject"]
            curves = {}
            for origin in origins:
                if origin >= length:
                    continue
                patched = forward_capture(
                    model, contexts[1][0], contexts[1][1], length,
                    args.inject_layer, args.readout_layer,
                    inject_offset=origin, addition=inject_residual[:, origin, :],
                )["readout"]
                effect = patched - contexts[1][2]["readout"]
                random_effect = None
                if args.random_control:
                    generator = torch.Generator().manual_seed(
                        args.seed + position * 1009 + origin
                    )
                    residual = inject_residual[:, origin, :]
                    random = torch.randn(residual.shape, generator=generator)
                    random = F.normalize(random, dim=-1) * residual.norm(dim=-1, keepdim=True)
                    random_patched = forward_capture(
                        model, contexts[1][0], contexts[1][1], length,
                        args.inject_layer, args.readout_layer,
                        inject_offset=origin, addition=random,
                    )["readout"]
                    random_effect = random_patched - contexts[1][2]["readout"]
                values = []
                for future in range(origin, length):
                    effect_token, target_token = effect[:, future, :], target[:, future, :]
                    target_energy = target_token.square().sum().clamp_min(1e-12)
                    value = {
                        "lag": future - origin,
                        "alignment": float(F.cosine_similarity(effect_token, target_token, dim=-1)),
                        "recovery": float(effect_token.norm() / target_token.norm().clamp_min(1e-12)),
                        "projected_recovery": float((effect_token * target_token).sum() / target_energy),
                    }
                    if random_effect is not None:
                        random_token = random_effect[:, future, :]
                        value.update({
                            "random_alignment": float(F.cosine_similarity(
                                random_token, target_token, dim=-1
                            )),
                            "random_recovery": float(
                                random_token.norm() / target_token.norm().clamp_min(1e-12)
                            ),
                            "random_projected_recovery": float(
                                (random_token * target_token).sum() / target_energy
                            ),
                        })
                    values.append(value)
                curves[str(origin)] = values
            handle.write(json.dumps({"id": row["id"], "curves": curves}) + "\n")
            handle.flush()
            print(f"row={position} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
