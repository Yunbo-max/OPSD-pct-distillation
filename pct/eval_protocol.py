from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY_METRICS = ("average_at_n_pct", "pass_at_n_pct", "majority_vote_at_n_pct", "format_rate")


def load_eval_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("eval_json_not_object")
    return data


def validate_eval_protocol(
    data: dict[str, Any],
    expected_dataset: str | None = None,
    expected_val_n: int = 12,
    expected_num_problems: int = 30,
    require_generations: bool = True,
    expected_seed: int | None = None,
) -> list[str]:
    failures: list[str] = []
    if expected_dataset is not None and data.get("dataset") != expected_dataset:
        failures.append(f"dataset:{data.get('dataset')}!=expected:{expected_dataset}")
    if int(data.get("val_n", 0)) != expected_val_n:
        failures.append(f"val_n:{data.get('val_n')}!=expected:{expected_val_n}")
    if int(data.get("num_problems", 0)) != expected_num_problems:
        failures.append(f"num_problems:{data.get('num_problems')}!=expected:{expected_num_problems}")
    if expected_seed is not None and data.get("seed") != expected_seed:
        failures.append(f"seed:{data.get('seed')}!=expected:{expected_seed}")
    for metric in SUMMARY_METRICS:
        if not isinstance(data.get(metric), (int, float)):
            failures.append(f"missing_metric:{metric}")

    results = data.get("results")
    if not isinstance(results, list):
        failures.append("results_not_list")
        return failures
    if len(results) != expected_num_problems:
        failures.append(f"results_len:{len(results)}!=expected:{expected_num_problems}")

    seen_ids = set()
    total_correct = 0
    total_formatted = 0
    pass_count = 0
    majority_count = 0
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            failures.append(f"result_{idx}:not_object")
            continue
        problem_id = row.get("problem_id", idx)
        if problem_id in seen_ids:
            failures.append(f"duplicate_problem_id:{problem_id}")
        seen_ids.add(problem_id)
        if int(row.get("val_n", 0)) != expected_val_n:
            failures.append(f"result_{idx}:val_n:{row.get('val_n')}!=expected:{expected_val_n}")
        num_correct = row.get("num_correct")
        if not isinstance(num_correct, int) or not 0 <= num_correct <= expected_val_n:
            failures.append(f"result_{idx}:invalid_num_correct:{num_correct}")
            num_correct = 0
        total_correct += int(num_correct)
        pass_count += int(bool(row.get("pass_at_n")))
        majority_count += int(bool(row.get("majority_vote_correct")))
        generations = row.get("generations")
        if require_generations:
            if not isinstance(generations, list):
                failures.append(f"result_{idx}:generations_not_list")
                generations = []
            if len(generations) != expected_val_n:
                failures.append(f"result_{idx}:generations_len:{len(generations)}!=expected:{expected_val_n}")
            for gen_idx, gen in enumerate(generations):
                if not isinstance(gen, dict):
                    failures.append(f"result_{idx}:generation_{gen_idx}:not_object")
                    continue
                if not isinstance(gen.get("correct"), bool):
                    failures.append(f"result_{idx}:generation_{gen_idx}:correct_not_bool")
                if not isinstance(gen.get("formatted"), bool):
                    failures.append(f"result_{idx}:generation_{gen_idx}:formatted_not_bool")
                total_formatted += int(bool(gen.get("formatted")))

    denom = expected_num_problems * expected_val_n
    if denom > 0:
        expected_avg = 100.0 * total_correct / denom
        actual_avg = data.get("average_at_n_pct")
        if isinstance(actual_avg, (int, float)) and abs(float(actual_avg) - expected_avg) > 1e-6:
            failures.append(f"average_at_n_pct:{actual_avg}!=recomputed:{expected_avg}")
        actual_pass = data.get("pass_at_n_pct")
        expected_pass = 100.0 * pass_count / expected_num_problems
        if isinstance(actual_pass, (int, float)) and abs(float(actual_pass) - expected_pass) > 1e-6:
            failures.append(f"pass_at_n_pct:{actual_pass}!=recomputed:{expected_pass}")
        actual_majority = data.get("majority_vote_at_n_pct")
        expected_majority = 100.0 * majority_count / expected_num_problems
        if isinstance(actual_majority, (int, float)) and abs(float(actual_majority) - expected_majority) > 1e-6:
            failures.append(f"majority_vote_at_n_pct:{actual_majority}!=recomputed:{expected_majority}")
        if require_generations:
            actual_format = data.get("format_rate")
            expected_format = 100.0 * total_formatted / denom
            if isinstance(actual_format, (int, float)) and abs(float(actual_format) - expected_format) > 1e-6:
                failures.append(f"format_rate:{actual_format}!=recomputed:{expected_format}")
    return failures


def validate_eval_file(
    path: Path,
    expected_dataset: str | None = None,
    expected_val_n: int = 12,
    expected_num_problems: int = 30,
    require_generations: bool = True,
    expected_seed: int | None = None,
) -> list[str]:
    try:
        data = load_eval_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"cannot_load:{exc}"]
    return validate_eval_protocol(
        data,
        expected_dataset=expected_dataset,
        expected_val_n=expected_val_n,
        expected_num_problems=expected_num_problems,
        require_generations=require_generations,
        expected_seed=expected_seed,
    )
