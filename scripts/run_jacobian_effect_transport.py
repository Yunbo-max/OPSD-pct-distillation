#!/usr/bin/env python3
"""Student-basepoint transport of a privileged causal logit effect.

For one next-token decision per problem, this script:

1. measures the finite source intervention effect produced by c = h+ - h-;
2. compares source- and student-basepoint directional linearizations of c;
3. solves a low-dimensional regularized inverse J_s delta ~= e_source;
4. evaluates the nonlinear student intervention against direct, reverse, and
   norm-matched-random controls.

Only the top-k source-effect logits define the outcome space. Ridge ratios are
emitted independently so they can be selected on validation problems only.
"""
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
    from .extract_predictable_field_dataset import deployment_prompt
    from .run_online_paired_residual import online_teacher_prompt
except ImportError:
    from extract_predictable_field_dataset import deployment_prompt
    from run_online_paired_residual import online_teacher_prompt


def encode(tokenizer, prompt: str, continuation: torch.Tensor, device) -> torch.Tensor:
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return torch.cat((prefix, continuation), dim=1)


def capture_state_logits(model, ids: torch.Tensor, layer: int, addition=None, grad=False):
    saved = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["state"] = hidden[:, -1, :].detach().float()
        if addition is None:
            return output
        modified = hidden.clone()
        modified[:, -1, :] += addition.to(hidden.device, hidden.dtype)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        context = torch.enable_grad() if grad else torch.inference_mode()
        with context:
            out = model(ids, use_cache=False, logits_to_keep=1, return_dict=True)
        return saved["state"][0], out.logits[0, -1].float()
    finally:
        handle.remove()


def logits_with_leaf(model, ids: torch.Tensor, layer: int, state: torch.Tensor):
    leaf = state.to(model.device, torch.bfloat16).detach().requires_grad_(True)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        modified[:, -1, :] = leaf
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        out = model(ids, use_cache=False, logits_to_keep=1, return_dict=True)
        return leaf, out.logits[0, -1].float()
    finally:
        handle.remove()


def explicit_jacobian(model, ids: torch.Tensor, layer: int, state: torch.Tensor,
                      selected: torch.Tensor) -> torch.Tensor:
    leaf, logits = logits_with_leaf(model, ids, layer, state)
    rows = []
    for index in selected:
        rows.append(torch.autograd.grad(logits[index], leaf, retain_graph=True)[0].detach().float())
    return torch.stack(rows)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=0)) if a.norm() and b.norm() else 0.0


def effect_metrics(effect: torch.Tensor, target: torch.Tensor) -> dict:
    denom = float(target.square().sum()) + 1e-12
    return {
        "cosine": cosine(effect, target),
        "relative_squared_error": float((effect - target).square().sum()) / denom,
        "effect_norm": float(effect.norm()),
        "target_norm": float(target.norm()),
    }


def stable_generator(device, seed: int, record_id: str):
    digest = hashlib.sha256(f"{seed}|{record_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(value)


def logp(logits: torch.Tensor, token: int) -> float:
    return float(F.log_softmax(logits.float(), dim=-1)[token])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("split_summary")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--layers", default="23,25")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--ridge_ratios", default="0.001,0.01,0.1,1")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = {str(row["id"]): row for row in map(json.loads, Path(args.dataset).open())}
    summary = json.loads(Path(args.split_summary).read_text())
    split_key = "val" if args.split == "validation" else args.split
    ids = [str(value) for value in summary["split_ids"][split_key]][: args.n_examples]
    layers = [int(value) for value in args.layers.split(",")]
    ridge_ratios = [float(value) for value in args.ridge_ratios.split(",")]
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); output.touch(exist_ok=True)
    completed = {json.loads(line)["id"] for line in output.read_text().splitlines() if line}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    with output.open("a") as handle:
        for position, record_id in enumerate(ids):
            if record_id in completed:
                continue
            row = rows[record_id]
            full = tokenizer(row["solution"], return_tensors="pt", add_special_tokens=False,
                             truncation=True, max_length=args.tokens).input_ids.to(model.device)
            if full.shape[1] < 2:
                continue
            continuation, gold = full[:, :-1], int(full[0, -1])
            prompts = {
                "student": deployment_prompt(tokenizer, row["problem"]),
                "correct": online_teacher_prompt(tokenizer, row["problem"], row["references"][0], False),
                "corrupt": online_teacher_prompt(tokenizer, row["problem"],
                                                   row["matched_controls"]["dependency_swap"], False),
            }
            ids_by_arm = {name: encode(tokenizer, prompt, continuation, model.device)
                          for name, prompt in prompts.items()}
            result = {"id": record_id, "gold_token": gold, "layers": {}}
            for layer in layers:
                hs, zs = capture_state_logits(model, ids_by_arm["student"], layer)
                hp, _ = capture_state_logits(model, ids_by_arm["correct"], layer)
                hm, zm = capture_state_logits(model, ids_by_arm["corrupt"], layer)
                c = hp - hm
                _, z_source = capture_state_logits(model, ids_by_arm["corrupt"], layer, c)
                e_full = z_source - zm
                # The task outcome must remain represented even when the gold token is not
                # among the largest privileged logit changes.
                contrast_k = min(max(args.topk - 1, 0), e_full.numel() - 1)
                contrast = torch.topk(e_full.abs(), contrast_k + 1).indices
                contrast = contrast[contrast != gold][:contrast_k]
                selected = torch.cat((torch.tensor([gold], device=model.device), contrast))
                target = e_full[selected]

                source_jacobian = explicit_jacobian(
                    model, ids_by_arm["corrupt"], layer, hm, selected
                )
                jacobian = explicit_jacobian(
                    model, ids_by_arm["student"], layer, hs, selected
                )
                jmc = source_jacobian @ c
                jsc = jacobian @ c
                gram = jacobian @ jacobian.T
                scale = float(torch.trace(gram) / gram.shape[0]) + 1e-12

                base_logp = logp(zs, gold)
                arms = {
                    "source_linear": effect_metrics(jmc, target),
                    "student_direct_linear": effect_metrics(jsc, target),
                }
                direct_effect = capture_state_logits(
                    model, ids_by_arm["student"], layer, c
                )[1][selected] - zs[selected]
                arms["direct"] = effect_metrics(direct_effect, target) | {
                    "gold_logp_gain": logp(capture_state_logits(
                        model, ids_by_arm["student"], layer, c
                    )[1], gold) - base_logp,
                    "intervention_norm": float(c.norm()),
                }
                for ratio in ridge_ratios:
                    dual = torch.linalg.solve(
                        gram + ratio * scale * torch.eye(gram.shape[0], device=gram.device), target
                    )
                    delta = jacobian.T @ dual
                    transported_logits = capture_state_logits(
                        model, ids_by_arm["student"], layer, delta
                    )[1]
                    effect = transported_logits[selected] - zs[selected]
                    name = f"jt:{ratio:g}"
                    arms[name] = effect_metrics(effect, target) | {
                        "gold_logp_gain": logp(transported_logits, gold) - base_logp,
                        "intervention_norm": float(delta.norm()),
                        "norm_ratio_to_c": float(delta.norm() / (c.norm() + 1e-12)),
                    }
                    reverse_logits = capture_state_logits(
                        model, ids_by_arm["student"], layer, -delta
                    )[1]
                    arms[f"reverse_jt:{ratio:g}"] = effect_metrics(
                        reverse_logits[selected] - zs[selected], target
                    ) | {"gold_logp_gain": logp(reverse_logits, gold) - base_logp}
                    generator = stable_generator(model.device, args.seed, f"{record_id}|{layer}|{ratio:g}")
                    noise = torch.randn(delta.shape, generator=generator, device=model.device)
                    noise = F.normalize(noise.float(), dim=0) * delta.norm()
                    random_logits = capture_state_logits(
                        model, ids_by_arm["student"], layer, noise
                    )[1]
                    arms[f"random_jt:{ratio:g}"] = effect_metrics(
                        random_logits[selected] - zs[selected], target
                    ) | {"gold_logp_gain": logp(random_logits, gold) - base_logp}

                result["layers"][str(layer)] = {
                    "c_norm": float(c.norm()), "base_gold_logp": base_logp,
                    "source_gold_logp_gain": logp(z_source, gold) - logp(zm, gold),
                    "topk_token_ids": selected.tolist(), "arms": arms,
                }
                del source_jacobian, jacobian, gram
                torch.cuda.empty_cache()
            handle.write(json.dumps(result) + "\n"); handle.flush()
            print(f"row={position} id={record_id} elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
