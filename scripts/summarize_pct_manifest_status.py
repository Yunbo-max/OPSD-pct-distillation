#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pct.eval_protocol import validate_eval_file


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    try:
        with state_file.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except json.JSONDecodeError:
        return {}
    for record in reversed(state.get("log_history", [])):
        if any(key in record for key in ("loss", "train_loss", "pct_loss", "pct_transport_mass")):
            return record
    return {}


def train_metric_missing(record: dict, metrics: dict) -> list[str]:
    missing = []
    method = record.get("method", "")
    if float(record.get("pct_loss_weight", 0.0)) > 0 and "pct_loss" not in metrics:
        missing.append("pct_loss")
    if method in {"set_ot", "set_fgw", "set_uot"} and "pct_transport_mass" not in metrics:
        missing.append("pct_transport_mass")
    if method == "phf_medoid" and "pct_medoid_index" not in metrics:
        missing.append("pct_medoid_index")
    return missing


def eval_complete(
    path: Path,
    expected_dataset: str,
    expected_val_n: int,
    expected_num_problems: int,
    expected_seed: int | None = None,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_output"
    failures = validate_eval_file(
        path,
        expected_dataset=expected_dataset,
        expected_val_n=expected_val_n,
        expected_num_problems=expected_num_problems,
        require_generations=True,
        expected_seed=expected_seed,
    )
    return not failures, ",".join(failures) if failures else "none"


def train_rows(records: list[dict], require_train_metrics: bool) -> list[dict[str, str]]:
    rows = []
    train_index = 0
    for record in records:
        if record.get("requires_training") is False:
            continue
        output_dir = Path(record["output_dir"])
        checkpoint_ok = checkpoint_complete(output_dir)
        metrics = final_train_metrics(output_dir) if checkpoint_ok else {}
        metric_missing = train_metric_missing(record, metrics) if checkpoint_ok and require_train_metrics else []
        complete = checkpoint_ok and not metric_missing
        missing = metric_missing if checkpoint_ok else ["checkpoint"]
        rows.append(
            {
                "task": "train",
                "index": str(train_index),
                "run_name": str(record.get("run_name", "")),
                "method": str(record.get("method", "")),
                "dataset": str(record.get("dataset", "")),
                "status": "complete" if complete else "pending",
                "missing": "none" if complete else ",".join(missing),
                "artifact": str(output_dir),
            }
        )
        train_index += 1
    return rows


def eval_rows(records: list[dict], eval_root: Path, expected_num_problems: int) -> list[dict[str, str]]:
    rows = []
    eval_index = 0
    for record in records:
        for eval_record in record.get("eval", []):
            val_n = int(eval_record.get("val_n", 12))
            dataset = str(eval_record.get("dataset", ""))
            output = eval_root / f"{record['run_name']}_{dataset}_avg{val_n}.json"
            ok, missing = eval_complete(
                output,
                dataset,
                val_n,
                expected_num_problems,
                eval_record.get("seed", record.get("seed")),
            )
            rows.append(
                {
                    "task": "eval",
                    "index": str(eval_index),
                    "run_name": str(record.get("run_name", "")),
                    "method": str(record.get("method", "")),
                    "dataset": dataset,
                    "status": "complete" if ok else "pending",
                    "missing": missing,
                    "artifact": str(output),
                }
            )
            eval_index += 1
    return rows


def write_table(rows: list[dict[str, str]], out: Path | None) -> None:
    columns = ["task", "index", "run_name", "method", "dataset", "status", "missing", "artifact"]
    lines = ["\t".join(columns)]
    lines.extend("\t".join(row.get(col, "") for col in columns) for row in rows)
    output = "\n".join(lines)
    print(output)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manifest train/eval task completion for resume planning.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--task", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--expected_num_problems", type=int, default=30)
    parser.add_argument("--require_train_metrics", action="store_true")
    parser.add_argument("--only_incomplete", action="store_true")
    parser.add_argument("--tsv", default=None)
    args = parser.parse_args()

    records = load_manifest(Path(args.manifest))
    rows: list[dict[str, str]] = []
    if args.task in {"train", "all"}:
        rows.extend(train_rows(records, require_train_metrics=args.require_train_metrics))
    if args.task in {"eval", "all"}:
        rows.extend(eval_rows(records, Path(args.eval_root), args.expected_num_problems))
    if args.only_incomplete:
        rows = [row for row in rows if row["status"] != "complete"]
    write_table(rows, Path(args.tsv) if args.tsv else None)


if __name__ == "__main__":
    main()
