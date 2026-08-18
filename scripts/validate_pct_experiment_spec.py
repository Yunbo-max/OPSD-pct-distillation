#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TOP_LEVEL = {
    "name",
    "training_framework",
    "primary_model",
    "training_dataset",
    "diagnostics",
    "short_matrix",
    "full_matrix",
    "scaling_matrix",
    "primary_evaluation",
    "paper_claim_gates",
    "required_artifact_groups",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL\t{message}")


def ok(message: str) -> None:
    print(f"OK\t{message}")


def validate_spec(spec: dict) -> list[str]:
    failures: list[str] = []
    for key in sorted(REQUIRED_TOP_LEVEL - set(spec)):
        fail(f"missing_top_level:{key}", failures)

    train = spec.get("training_dataset", {})
    if int(train.get("examples", 0)) < 29434:
        fail("training_dataset.examples_lt_29434", failures)
    else:
        ok("training_dataset.examples")
    if int(train.get("min_correct_references", 0)) < 4:
        fail("training_dataset.min_correct_references_lt_4", failures)
    else:
        ok("training_dataset.min_correct_references")

    primary_eval = spec.get("primary_evaluation", {})
    if int(primary_eval.get("val_n", 0)) != 12:
        fail("primary_evaluation.val_n_not_12", failures)
    else:
        ok("primary_evaluation.val_n")
    if int(primary_eval.get("problems_per_dataset", 0)) != 30:
        fail("primary_evaluation.problems_per_dataset_not_30", failures)
    else:
        ok("primary_evaluation.problems_per_dataset")
    datasets = set(primary_eval.get("datasets", []))
    missing_eval = {"aime24", "aime25", "hmmt25"} - datasets
    for dataset in sorted(missing_eval):
        fail(f"primary_evaluation.missing_dataset:{dataset}", failures)
    if not missing_eval:
        ok("primary_evaluation.datasets")

    for suite in ("short_matrix", "full_matrix", "scaling_matrix"):
        methods = spec.get(suite, {}).get("required_methods", [])
        if not methods:
            fail(f"{suite}.missing_required_methods", failures)
        else:
            ok(f"{suite}.required_methods:{len(methods)}")

    if not spec.get("paper_claim_gates"):
        fail("missing_paper_claim_gates", failures)
    else:
        ok(f"paper_claim_gates:{len(spec['paper_claim_gates'])}")
    return failures


def validate_manifest(spec: dict, records: list[dict], suite: str) -> list[str]:
    failures: list[str] = []
    suite_spec = spec[suite]
    required_methods = set(suite_spec["required_methods"])
    primary_eval = spec["primary_evaluation"]
    required_eval_datasets = set(primary_eval["datasets"])
    expected_val_n = int(primary_eval["val_n"])

    by_method = {record.get("method"): record for record in records}
    for method in sorted(required_methods - set(by_method)):
        fail(f"{suite}.manifest_missing_method:{method}", failures)

    for method in sorted(required_methods & set(by_method)):
        record = by_method[method]
        evals = {item.get("dataset"): item for item in record.get("eval", [])}
        missing = required_eval_datasets - set(evals)
        for dataset in sorted(missing):
            fail(f"{record.get('run_name')}.missing_eval_dataset:{dataset}", failures)
        for dataset, item in sorted(evals.items()):
            if dataset in required_eval_datasets and int(item.get("val_n", 0)) != expected_val_n:
                fail(f"{record.get('run_name')}.{dataset}.val_n_not_{expected_val_n}", failures)
        if not missing:
            ok(f"{record.get('run_name')}.eval_datasets")

    if suite == "scaling_matrix":
        required_models = set(suite_spec["models"])
        for model in sorted(required_models):
            model_methods = {record.get("method") for record in records if record.get("model") == model}
            for method in sorted(required_methods - model_methods):
                fail(f"scaling_manifest_missing:{model}:{method}", failures)
        if not failures:
            ok("scaling_matrix.models_x_methods")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the PCT experiment spec and optional manifest coverage.")
    parser.add_argument("--spec", default="configs/pct_neurips_spec.json")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--suite", choices=["short_matrix", "full_matrix", "scaling_matrix"], default="short_matrix")
    args = parser.parse_args()

    spec = load_json(Path(args.spec))
    failures = validate_spec(spec)
    if args.manifest:
        failures.extend(validate_manifest(spec, load_manifest(Path(args.manifest)), args.suite))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
