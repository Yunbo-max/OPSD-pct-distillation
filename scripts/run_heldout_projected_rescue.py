#!/usr/bin/env python3
"""Held-out causal rescue with train-only PCA/contrastive subspaces."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_paired_counterfactual_rescue import states_and_logp, window_injected_logp
from run_privileged_effect_diagnostic import encode, teacher_prompt


def project(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (value @ basis) @ basis.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("basis")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--ranks", default="1,2,4,8,16,32")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    payload = torch.load(args.basis, map_location="cpu", weights_only=False)
    test_ids = payload["report"]["test_ids"]
    ranks = [int(value) for value in args.ranks.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for line in output.read_text().splitlines() if line) if output.exists() else 0
    started = time.time()
    with output.open("a") as handle:
        for offset, row_index in enumerate(test_ids[completed:], start=completed):
            row = rows[row_index]
            continuation = tokenizer(
                row["solution"], return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=args.tokens,
            ).input_ids.to(model.device)
            contexts = []
            for reference in (row["references"][0], row["matched_controls"]["dependency_swap"]):
                ids, mask, _ = encode(
                    tokenizer, teacher_prompt(tokenizer, row["problem"], reference),
                    continuation, model.device,
                )
                states, logp = states_and_logp(model, ids, mask, continuation, [args.layer])
                contexts.append((ids, mask, states[args.layer], logp))
            residual = contexts[0][2] - contexts[1][2]
            result = {
                "id": row["id"], "row_index": row_index,
                "correct_logp": contexts[0][3], "control_logp": contexts[1][3], "arms": {},
            }
            full_rescue = window_injected_logp(
                model, contexts[1][0], contexts[1][1], continuation,
                args.layer, residual, 1.0, continuation.shape[1],
            )
            full_necessity = window_injected_logp(
                model, contexts[0][0], contexts[0][1], continuation,
                args.layer, -residual, 1.0, continuation.shape[1],
            )
            result["arms"]["full"] = {
                "retained_energy": 1.0,
                "rescue_gain": full_rescue - contexts[1][3],
                "necessity_change": full_necessity - contexts[0][3],
            }
            generator = torch.Generator().manual_seed(args.seed + row_index)
            for rank in ranks:
                bases = {
                    "pca": payload["pca_basis"][:, :rank].float(),
                    "contrastive": payload["contrastive_basis"][:, :rank].float(),
                }
                target_norm = None
                projected = {}
                for name, basis in bases.items():
                    projected[name] = project(residual, basis)
                    if name == "contrastive":
                        target_norm = projected[name].norm(dim=-1, keepdim=True)
                random_basis = torch.randn((residual.shape[-1], rank), generator=generator)
                random_basis = torch.linalg.qr(random_basis).Q
                random_value = project(residual, random_basis)
                random_value = F.normalize(random_value, dim=-1) * target_norm
                projected["random"] = random_value
                for name, value in projected.items():
                    rescue = window_injected_logp(
                        model, contexts[1][0], contexts[1][1], continuation,
                        args.layer, value, 1.0, continuation.shape[1],
                    )
                    necessity = window_injected_logp(
                        model, contexts[0][0], contexts[0][1], continuation,
                        args.layer, -value, 1.0, continuation.shape[1],
                    )
                    result["arms"][f"{name}:rank={rank}"] = {
                        "retained_energy": float(value.square().sum() / residual.square().sum()),
                        "rescue_gain": rescue - contexts[1][3],
                        "necessity_change": necessity - contexts[0][3],
                    }
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(f"row={offset} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
