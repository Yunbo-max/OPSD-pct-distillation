#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pct.eval_protocol import validate_eval_file
from scripts.inventory_pct_artifacts import collect as collect_artifacts


DEFAULT_METHODS = (
    "none",
    "phf_single",
    "phf_random",
    "phf_mean",
    "phf_medoid",
    "phf_grassmann",
    "phf_set",
    "set_ot",
    "set_fgw",
    "set_uot",
)

DEFAULT_EVAL_DATASETS = ("aime24", "aime25", "hmmt25")


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_spec(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def spec_defaults(spec: dict, suite: str) -> dict:
    primary_eval = spec.get("primary_evaluation", {})
    suite_spec = spec.get(suite, {})
    return {
        "required_methods": suite_spec.get("required_methods"),
        "required_eval_datasets": primary_eval.get("datasets"),
        "expected_num_problems": primary_eval.get("problems_per_dataset"),
        "required_artifact_groups": spec.get("required_artifact_groups"),
    }


def latest_state_file(output_dir: Path) -> Path | None:
    states = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if states:
        return states[-1]
    state = output_dir / "trainer_state.json"
    return state if state.exists() else None


def checkpoint_complete(output_dir: Path) -> bool:
    if (output_dir / "adapter_model.safetensors").exists() or (output_dir / "adapter_model.bin").exists():
        return True
    return latest_state_file(output_dir) is not None


def final_train_metrics(output_dir: Path) -> dict:
    state_file = latest_state_file(output_dir)
    if state_file is None:
        return {}
    with state_file.open("r", encoding="utf-8") as f:
        state = json.load(f)
    for record in reversed(state.get("log_history", [])):
        if any(key in record for key in ("loss", "train_loss", "pct_loss", "pct_transport_mass")):
            return record
    return {}


def train_metrics_complete(record: dict, metrics: dict) -> tuple[bool, list[str]]:
    missing = []
    if record.get("requires_training") is False:
        return True, missing
    method = record.get("method", "")
    if float(record.get("pct_loss_weight", 0.0)) > 0 and "pct_loss" not in metrics:
        missing.append("pct_loss_metric")
    if method in {"set_ot", "set_fgw", "set_uot"} and "pct_transport_mass" not in metrics:
        missing.append("pct_transport_mass_metric")
    if method == "phf_medoid" and "pct_medoid_index" not in metrics:
        missing.append("pct_medoid_index_metric")
    return not missing, missing


def eval_file(eval_root: Path, run_name: str, dataset: str, val_n: int = 12) -> Path:
    return eval_root / f"{run_name}_{dataset}_avg{val_n}.json"


def eval_failures(
    path: Path,
    dataset: str,
    expected_val_n: int,
    expected_num_problems: int,
    expected_seed: int | None = None,
) -> list[str]:
    return validate_eval_file(
        path,
        expected_dataset=dataset,
        expected_val_n=expected_val_n,
        expected_num_problems=expected_num_problems,
        require_generations=True,
        expected_seed=expected_seed,
    )


def validate_manifest(records: list[dict], required_methods: set[str], required_eval_datasets: set[str]) -> list[str]:
    failures = []
    seen_methods = {record.get("method") for record in records}
    for method in sorted(required_methods - seen_methods):
        failures.append(f"missing_method:{method}")
    for record in records:
        datasets = {item.get("dataset") for item in record.get("eval", [])}
        for dataset in sorted(required_eval_datasets - datasets):
            failures.append(f"{record.get('run_name')}:missing_eval_dataset:{dataset}")
    return failures


def artifact_group_failures(root: Path, required_groups: list[str]) -> list[str]:
    inventory = collect_artifacts(root)
    return [f"missing_artifact_group:{group}" for group in required_groups if not inventory.get(group)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completion of a PCT run manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--spec", default=None, help="Optional experiment spec providing audit defaults.")
    parser.add_argument(
        "--suite",
        default="short_matrix",
        choices=["short_matrix", "full_matrix", "scaling_matrix"],
        help="Experiment suite to read from --spec when deriving required methods.",
    )
    parser.add_argument("--require_eval", action="store_true")
    parser.add_argument("--require_train_metrics", action="store_true")
    parser.add_argument("--expected_num_problems", type=int, default=None)
    parser.add_argument("--required_methods", nargs="+", default=None)
    parser.add_argument("--required_eval_datasets", nargs="+", default=None)
    parser.add_argument("--artifact_root", default=None)
    parser.add_argument("--required_artifact_groups", nargs="+", default=None)
    args = parser.parse_args()

    defaults = spec_defaults(load_spec(args.spec), args.suite)
    required_methods = args.required_methods or defaults.get("required_methods") or list(DEFAULT_METHODS)
    required_eval_datasets = (
        args.required_eval_datasets or defaults.get("required_eval_datasets") or list(DEFAULT_EVAL_DATASETS)
    )
    expected_num_problems = int(args.expected_num_problems or defaults.get("expected_num_problems") or 30)
    required_artifact_groups = args.required_artifact_groups or defaults.get("required_artifact_groups") or []

    records = load_manifest(Path(args.manifest))
    eval_root = Path(args.eval_root)
    failures = []
    manifest_failures = validate_manifest(records, set(required_methods), set(required_eval_datasets))
    failures.extend(manifest_failures)
    for failure in manifest_failures:
        print(f"manifest\tFalse\tFalse\t{failure}")

    if args.artifact_root and required_artifact_groups:
        for failure in artifact_group_failures(Path(args.artifact_root), required_artifact_groups):
            failures.append(failure)
            print(f"artifacts\tFalse\tFalse\t{failure}")

    print("run\ttrain_complete\teval_complete\tmissing")
    for record in records:
        run_name = record["run_name"]
        requires_training = record.get("requires_training", True)
        output_dir = Path(record["output_dir"]) if record.get("output_dir") else None
        train_ok = True if requires_training is False else bool(output_dir and checkpoint_complete(output_dir))
        metrics_ok, metric_missing = train_metrics_complete(
            record, {} if output_dir is None else final_train_metrics(output_dir)
        )
        missing = []
        if not train_ok:
            missing.append("train_checkpoint")
        if args.require_train_metrics and not metrics_ok:
            missing.extend(metric_missing)
        eval_ok = True
        for eval_record in record.get("eval", []):
            val_n = int(eval_record.get("val_n", 12))
            path = eval_file(eval_root, run_name, eval_record["dataset"], val_n)
            failures_for_eval = eval_failures(
                path,
                eval_record["dataset"],
                val_n,
                expected_num_problems,
                eval_record.get("seed", record.get("seed")),
            )
            ok = not failures_for_eval
            eval_ok = eval_ok and ok
            if not ok:
                missing.append(f"{path}:{'|'.join(failures_for_eval)}")
        if missing and (args.require_eval or args.require_train_metrics or not train_ok):
            failures.append(run_name)
        print(f"{run_name}\t{train_ok}\t{eval_ok}\t{','.join(missing) if missing else 'none'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
