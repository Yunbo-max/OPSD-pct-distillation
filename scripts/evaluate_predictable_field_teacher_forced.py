#!/usr/bin/env python3
"""Teacher-forced causal evaluation of deployment-visible field predictors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .analyze_predictable_field import predict
    from .extract_predictable_field_dataset import deployment_prompt
except ImportError:
    from analyze_predictable_field import predict
    from extract_predictable_field_dataset import deployment_prompt


def sequence_logp(logits: torch.Tensor, continuation: torch.Tensor) -> tuple[float, float]:
    scores = logits[:, -continuation.shape[1]:-1, :].float()
    targets = continuation[:, 1:]
    values = F.log_softmax(scores, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(values.mean()), float(values[:, -min(16, values.shape[1]):].mean())


def forward(model, ids, length: int, layer: int, addition: torch.Tensor | None):
    handle = None
    if addition is not None:
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden.clone()
            modified[:, -length:, :] += addition.to(hidden.device, hidden.dtype)
            return (modified, *output[1:]) if isinstance(output, tuple) else modified
        handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            return model(ids, use_cache=False, return_dict=True).logits
    finally:
        if handle is not None:
            handle.remove()


def bootstrap(values: np.ndarray, indices: np.ndarray) -> dict:
    means = values[indices].mean(1)
    return {"mean": float(values.mean()), "ci": [float(x) for x in np.quantile(means, [.025, .975])]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("cache_dir")
    parser.add_argument("split_summary")
    parser.add_argument("controllers")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--methods", default="mean,ridge,rrr64,cca64")
    parser.add_argument("--control_method", default="rrr64")
    parser.add_argument("--alphas", default="0.25,0.5,1")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = {str(row["id"]): row for row in map(json.loads, Path(args.dataset).open())}
    split = json.loads(Path(args.split_summary).read_text())
    ids = split["split_ids"]["test"]
    cache_index = {}
    for path in Path(args.cache_dir).glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cache_index[str(payload["id"])] = path
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    controller_payload = torch.load(args.controllers, map_location="cpu", weights_only=False)
    layer = int(controller_payload["x_layer"])
    learned = {
        name: {key: value.float().to(model.device) if isinstance(value, torch.Tensor) else value
               for key, value in item.items()}
        for name, item in controller_payload["models"].items()
    }
    mean_model = {
        "kind": "mean", "mean_x": controller_payload["mean_x"].float().to(model.device),
        "mean_c": controller_payload["mean_c"].float().to(model.device),
    }
    methods = args.methods.split(",")
    alphas = [float(value) for value in args.alphas.split(",")]
    records = []
    for position, record_id in enumerate(ids):
        row = rows[str(record_id)]
        cached = torch.load(cache_index[str(record_id)], map_location="cpu", weights_only=False)
        continuation = tokenizer(
            row["solution"], return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=cached["tokens"],
        ).input_ids.to(model.device)
        prefix = tokenizer(
            deployment_prompt(tokenizer, row["problem"]), return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        input_ids = torch.cat((prefix, continuation), 1)
        x = cached["x"][layer].float().to(model.device)
        oracle = cached["c"].float().to(model.device)
        fields = {"oracle": oracle, "mean": predict(x, mean_model)}
        for name in methods:
            if name != "mean": fields[name] = predict(x, learned[name])
        fields.update({
            f"centered_{name}": value - mean_model["mean_c"]
            for name, value in list(fields.items()) if name not in ("oracle", "mean")
        })
        fields["reverse_mean"] = -fields["mean"]
        control = fields[args.control_method]
        generator = torch.Generator(device=model.device).manual_seed(
            int.from_bytes(hashlib.sha256(str(record_id).encode()).digest()[:8], "big")
        )
        noise = torch.randn(control.shape, generator=generator, device=model.device)
        fields[f"reverse_{args.control_method}"] = -control
        fields[f"random_{args.control_method}"] = F.normalize(noise, dim=-1) * control.norm(dim=-1, keepdim=True)
        additions = {
            f"{name}:alpha={alpha:g}": alpha * value
            for name, value in fields.items() for alpha in alphas
        }
        base = sequence_logp(forward(model, input_ids, continuation.shape[1], layer, None), continuation)
        result = {"id": record_id, "tokens": continuation.shape[1],
                  "baseline_logp": base[0], "baseline_tail_logp": base[1], "arms": {}}
        for name, addition in additions.items():
            value = sequence_logp(
                forward(model, input_ids, continuation.shape[1], layer, addition), continuation
            )
            result["arms"][name] = {
                "logp": value[0], "gain": value[0] - base[0],
                "tail_logp": value[1], "tail_gain": value[1] - base[1],
                "mean_field_norm": float(addition.norm(dim=-1).mean()),
            }
        records.append(result)
        print(f"row={position}/{len(ids)}", flush=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(records), size=(args.bootstrap, len(records)))
    names = list(records[0]["arms"])
    summary = {"n": len(records), "results": {}}
    oracle_key = "oracle:alpha=1"
    oracle_gain = np.array([row["arms"][oracle_key]["gain"] for row in records])
    oracle_tail = np.array([row["arms"][oracle_key]["tail_gain"] for row in records])
    for name in names:
        gain = np.array([row["arms"][name]["gain"] for row in records])
        tail = np.array([row["arms"][name]["tail_gain"] for row in records])
        denom = oracle_gain[indices].mean(1); tail_denom = oracle_tail[indices].mean(1)
        valid = np.abs(denom) > 1e-12; tail_valid = np.abs(tail_denom) > 1e-12
        summary["results"][name] = {
            "gain": bootstrap(gain, indices), "tail_gain": bootstrap(tail, indices),
            "orr": {
                "mean": float(gain.mean() / oracle_gain.mean()) if abs(oracle_gain.mean()) > 1e-12 else None,
                "ci": [float(x) for x in np.quantile(gain[indices].mean(1)[valid] / denom[valid], [.025, .975])],
            },
            "tail_orr": {
                "mean": float(tail.mean() / oracle_tail.mean()) if abs(oracle_tail.mean()) > 1e-12 else None,
                "ci": [float(x) for x in np.quantile(tail[indices].mean(1)[tail_valid] / tail_denom[tail_valid], [.025, .975])],
            },
        }
    Path(args.out).write_text(json.dumps({"records": records, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
