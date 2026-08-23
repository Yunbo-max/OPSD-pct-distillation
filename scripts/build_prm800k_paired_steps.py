#!/usr/bin/env python3
"""Build same-prefix correct/incorrect step pairs from PRM800K phase-1 labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--dataset", default="tasksource/PRM800K")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, records = set(), []
    source = load_dataset(args.dataset, split=args.split, streaming=True)
    for sample_index, sample in enumerate(source):
        problem = sample.get("question", {}).get("problem")
        answer = sample.get("question", {}).get("ground_truth_answer")
        steps = (sample.get("label") or {}).get("steps") or []
        if not problem or answer is None or problem in seen:
            continue
        prefix = []
        for step_index, step in enumerate(steps):
            all_completions = step.get("completions") or []
            completions = [value for value in all_completions if not value.get("flagged")]
            correct = [value["text"] for value in completions if value.get("rating") == 1]
            incorrect = [value["text"] for value in completions if value.get("rating") == -1]
            if correct and incorrect:
                chosen_index = step.get("chosen_completion")
                chosen_value = (
                    all_completions[chosen_index]
                    if chosen_index is not None and chosen_index < len(all_completions)
                    else None
                )
                correct_step = (
                    chosen_value["text"]
                    if chosen_value and chosen_value.get("rating") == 1 and not chosen_value.get("flagged")
                    else correct[0]
                )
                suffix = []
                for later in steps[step_index + 1 :]:
                    later_completions = later.get("completions") or []
                    later_chosen = later.get("chosen_completion")
                    if later_chosen is not None and later_chosen < len(later_completions):
                        suffix.append(later_completions[later_chosen]["text"])
                    elif (later.get("human_completion") or {}).get("text"):
                        suffix.append(later["human_completion"]["text"])
                prefix_text = "\n\n".join(prefix)
                correct_reference = "\n\n".join([*prefix, correct_step])
                corrupt_reference = "\n\n".join([*prefix, incorrect[0]])
                downstream = "\n\n".join(suffix)
                if downstream:
                    downstream += "\n\n"
                downstream += f"The final answer is \\boxed{{{answer}}}."
                records.append(
                    {
                        "id": f"prm-{sample_index}-{step_index}",
                        "problem": problem,
                        "answer": str(answer),
                        "solution": downstream,
                        "references": [correct_reference] * 4,
                        "matched_controls": {"dependency_swap": corrupt_reference},
                        "paired_prefix": prefix_text,
                        "correct_step": correct_step,
                        "incorrect_step": incorrect[0],
                        "downstream_steps": suffix,
                        "source": args.dataset,
                        "source_sample_index": sample_index,
                        "source_step_index": step_index,
                    }
                )
                seen.add(problem)
                break
            chosen = step.get("chosen_completion")
            if chosen is not None and chosen < len(all_completions):
                prefix.append(all_completions[chosen]["text"])
            elif (step.get("human_completion") or {}).get("text"):
                prefix.append(step["human_completion"]["text"])
        if len(records) >= args.n:
            break
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote={len(records)} scanned={sample_index + 1} output={output}")


if __name__ == "__main__":
    main()
