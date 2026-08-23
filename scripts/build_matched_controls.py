#!/usr/bin/env python3
"""Generate length/style-matched causal controls for privileged-effect studies."""
from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


KINDS = ("locally_corrupted", "counterfactual_rationale", "dependency_swap")


def control_prompt(kind: str, row: dict, donor: dict) -> str:
    shared = f"""You are constructing a rigorous mechanistic-control example by making ONE local patch.
Copy one short, exact, contiguous span from the source (normally 1-4 lines) between <OLD> and </OLD>.
Put its replacement between <NEW> and </NEW>. Return only these tags. The OLD text must occur verbatim.
Keep NEW approximately the same length and style as OLD.

Problem:
{row['problem']}

Verified answer: {row['answer']}

Source rationale:
{row['references'][0]}
"""
    if kind == "locally_corrupted":
        rule = """
Change exactly ONE key equation, operator, inequality direction, or arithmetic implication so
the computation is invalid. Do not edit the final answer. The corruption must be locally subtle
but mathematically consequential.
"""
    elif kind == "counterfactual_rationale":
        rule = """
Replace a key derivation span with a locally coherent but false implication that supports a
plausible incorrect conclusion. Include the conclusion in the replaced span when possible.
Do not announce that it is counterfactual or wrong.
"""
    else:
        rule = f"""
Replace exactly one essential premise or intermediate assumption with a compatible-looking
premise adapted from this donor problem:
{donor['problem']}
Make the replacement locally fluent, but ensure it is not a valid dependency for this problem.
"""
    return shared + rule


def apply_edit(source: str, text: str) -> tuple[str, bool]:
    old_start, old_end = text.find("<OLD>"), text.rfind("</OLD>")
    new_start, new_end = text.find("<NEW>"), text.rfind("</NEW>")
    if min(old_start, old_end, new_start, new_end) < 0:
        return source, False
    old = text[old_start + len("<OLD>") : old_end].strip()
    new = text[new_start + len("<NEW>") : new_end].strip()
    if not old or not new or old == new:
        return source, False
    if old in source:
        return source.replace(old, new, 1), True
    # Models sometimes normalize TeX delimiters while copying OLD. Recover the
    # closest contiguous line span, but reject uncertain matches.
    source_lines = source.splitlines()
    old_lines = old.splitlines()
    best = (0.0, 0, 0)
    for width in range(max(1, len(old_lines) - 2), len(old_lines) + 3):
        for start in range(0, len(source_lines) - width + 1):
            candidate = "\n".join(source_lines[start : start + width]).strip()
            score = SequenceMatcher(None, old, candidate).ratio()
            if score > best[0]:
                best = (score, start, start + width)
    if best[0] < 0.72:
        return source, False
    patched = [*source_lines[: best[1]], new, *source_lines[best[2] :]]
    return "\n".join(patched), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/root/models/Qwen3-4B")
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument("--batch_examples", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    rows = rows[: args.n_examples]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for line in output.read_text().splitlines() if line) if output.exists() else 0
    if completed >= len(rows):
        print(f"already complete: {completed} rows")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    started = time.time()
    with output.open("a") as handle:
        for batch_start in range(completed, len(rows), args.batch_examples):
            indexes = list(range(batch_start, min(batch_start + args.batch_examples, len(rows))))
            jobs = []
            for index in indexes:
                donor = rows[(index + 1) % len(rows)]
                for kind in KINDS:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": control_prompt(kind, rows[index], donor)}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    jobs.append((index, kind, text))
            inputs = tokenizer(
                [job[2] for job in jobs], padding=True, truncation=True,
                max_length=args.max_input_tokens, return_tensors="pt",
            ).to(model.device)
            torch.manual_seed(args.seed + batch_start)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.65, top_p=0.9,
                )
            controls = {index: {} for index in indexes}
            valid = {index: {} for index in indexes}
            raw_edits = {index: {} for index in indexes}
            prompt_width = inputs.input_ids.shape[1]
            for (index, kind, _), sample in zip(jobs, generated):
                decoded = tokenizer.decode(sample[prompt_width:], skip_special_tokens=True)
                raw_edits[index][kind] = decoded
                controls[index][kind], valid[index][kind] = apply_edit(
                    rows[index]["references"][0], decoded
                )
            for index in indexes:
                row = dict(rows[index])
                row["matched_controls"] = controls[index]
                row["matched_controls_valid"] = valid[index]
                row["matched_control_raw_edits"] = raw_edits[index]
                row["matched_control_donor_id"] = rows[(index + 1) % len(rows)].get("id")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            elapsed = time.time() - started
            done = indexes[-1] - completed + 1
            print(
                f"rows={indexes[0]}-{indexes[-1]} elapsed={elapsed:.1f}s "
                f"rows_per_hour={done * 3600 / elapsed:.1f}", flush=True
            )
            del inputs, generated
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
