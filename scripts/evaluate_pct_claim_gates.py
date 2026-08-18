#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_tsv(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def to_float(value: str | None) -> float | None:
    if value in (None, "", "NA", "nan"):
        return None
    return float(value)


def status_line(name: str, status: str, detail: str) -> str:
    return f"{name}\t{status}\t{detail}"


def eval_gate(
    eval_rows: list[dict[str, str]],
    baseline: str,
    methods: list[str],
    required_datasets: list[str],
) -> tuple[str, str]:
    if not eval_rows:
        return "missing", "eval_tsv_missing"
    by_dataset_method: dict[tuple[str, str], float] = {}
    for row in eval_rows:
        dataset = row.get("dataset", "")
        file_name = row.get("file", "")
        avg = to_float(row.get("Average@N"))
        if avg is None:
            continue
        for method in [baseline, *methods]:
            if method in file_name:
                by_dataset_method[(dataset, method)] = avg
    wins = 0
    comparisons = 0
    missing = []
    for dataset in required_datasets:
        base = by_dataset_method.get((dataset, baseline))
        if base is None:
            missing.append(f"{dataset}:{baseline}")
            continue
        candidates = [by_dataset_method[(dataset, method)] for method in methods if (dataset, method) in by_dataset_method]
        best = max(candidates) if candidates else None
        if best is None:
            missing.append(f"{dataset}:pct_method")
            continue
        comparisons += 1
        wins += int(best > base)
    if missing:
        return "missing", ",".join(missing)
    if wins == comparisons and comparisons > 0:
        return "pass", f"pct_best_beats_{baseline}:{wins}/{comparisons}"
    return "fail", f"pct_best_beats_{baseline}:{wins}/{comparisons}"


def seed_summary_gate(
    seed_rows: list[dict[str, str]],
    baseline: str,
    methods: list[str],
    required_datasets: list[str],
) -> tuple[str, str]:
    if not seed_rows:
        return "missing", "seed_summary_tsv_missing"
    by_dataset_method: dict[tuple[str, str], float] = {}
    seed_detail: dict[tuple[str, str], str] = {}
    for row in seed_rows:
        dataset = row.get("dataset", "")
        method = row.get("method", "")
        avg = to_float(row.get("mean_average_at_n"))
        if avg is None:
            continue
        by_dataset_method[(dataset, method)] = avg
        seed_detail[(dataset, method)] = row.get("seeds", "")

    wins = 0
    comparisons = 0
    missing = []
    for dataset in required_datasets:
        base = by_dataset_method.get((dataset, baseline))
        if base is None:
            missing.append(f"{dataset}:{baseline}")
            continue
        candidates = [by_dataset_method[(dataset, method)] for method in methods if (dataset, method) in by_dataset_method]
        best = max(candidates) if candidates else None
        if best is None:
            missing.append(f"{dataset}:pct_method")
            continue
        comparisons += 1
        wins += int(best > base)
    if missing:
        return "missing", ",".join(missing)
    seeds = sorted({value for value in seed_detail.values() if value})
    detail = f"seed_mean_pct_best_beats_{baseline}:{wins}/{comparisons},seeds={';'.join(seeds)}"
    if wins == comparisons and comparisons > 0:
        return "pass", detail
    return "fail", detail


def dispersion_gate(rows: list[dict[str, str]], baseline: str, methods: list[str]) -> tuple[str, str]:
    if not rows:
        return "missing", "dispersion_gain_tsv_missing"
    high = [row for row in rows if row.get("group") == "high" and row.get("method") in methods]
    low = [row for row in rows if row.get("group") == "low" and row.get("method") in methods]
    if not high or not low:
        return "missing", "high_or_low_group_missing"
    best_high = max(to_float(row.get(f"gain_vs_{baseline}_pct")) or 0.0 for row in high)
    best_low = max(to_float(row.get(f"gain_vs_{baseline}_pct")) or 0.0 for row in low)
    status = "pass" if best_high > 0 and best_high > best_low else "fail"
    return status, f"best_high_gain={best_high:.2f},best_low_gain={best_low:.2f}"


def diagnostic_gate(rows: list[dict[str, str]], threshold: float) -> tuple[str, str]:
    if not rows:
        return "missing", "diagnostic_summary_tsv_missing"
    euclidean = None
    best = None
    best_metric = ""
    for row in rows:
        score = to_float(row.get("tauroc_cc_lt_cw"))
        if score is None:
            continue
        if row.get("metric") == "euclidean":
            euclidean = score
        if best is None or score > best:
            best = score
            best_metric = row.get("metric", "")
    if best is None:
        return "missing", "diagnostic_auroc_missing"
    if euclidean is None:
        return "missing", "euclidean_auroc_missing"
    status = "pass" if best >= threshold and best > euclidean else "fail"
    return status, f"best={best_metric}:{best:.3f},euclidean={euclidean:.3f}"


def intervention_gate(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "missing", "intervention_tsv_missing"
    values = {row.get("vector"): to_float(row.get("mean_delta_logp")) for row in rows}
    nearest = values.get("nearest_valid")
    mean_val = values.get("mean")
    wrong = values.get("wrong")
    if nearest is None or mean_val is None or wrong is None:
        return "missing", "nearest_valid_mean_or_wrong_missing"
    status = "pass" if nearest > 0 and nearest > mean_val and wrong < 0 else "fail"
    return status, f"nearest_valid={nearest:.4f},mean={mean_val:.4f},wrong={wrong:.4f}"


def robustness_gate(rows: list[dict[str, str]], mean_method: str, methods: list[str]) -> tuple[str, str]:
    if not rows:
        return "missing", "robustness_tsv_missing"
    drops: dict[str, float] = {}
    for row in rows:
        method = row.get("method", "")
        perturbation = row.get("perturbation", "")
        delta = to_float(row.get("delta_vs_clean_pct"))
        if method and perturbation != "clean" and delta is not None:
            drops[method] = min(drops.get(method, 0.0), delta)
    if mean_method not in drops:
        return "missing", f"{mean_method}_robustness_missing"
    pct_drops = [drops[m] for m in methods if m in drops]
    if not pct_drops:
        return "missing", "pct_robustness_methods_missing"
    best_pct_drop = max(pct_drops)
    status = "pass" if best_pct_drop > drops[mean_method] else "fail"
    return status, f"best_pct_drop={best_pct_drop:.2f},{mean_method}_drop={drops[mean_method]:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCT paper-claim gates from generated TSV artifacts.")
    parser.add_argument("--spec", default="configs/pct_neurips_spec.json")
    parser.add_argument("--eval_tsv", default=None)
    parser.add_argument("--seed_summary_tsv", default=None)
    parser.add_argument("--dispersion_gain_tsv", default=None)
    parser.add_argument("--diagnostic_summary_tsv", default=None)
    parser.add_argument("--intervention_tsv", default=None)
    parser.add_argument("--robustness_tsv", default=None)
    parser.add_argument("--baseline", default="phf_single")
    parser.add_argument("--mean_method", default="phf_mean")
    parser.add_argument("--pct_methods", nargs="+", default=["phf_set", "set_ot", "set_fgw", "set_uot"])
    parser.add_argument("--diagnostic_auroc_threshold", type=float, default=0.6)
    parser.add_argument("--out_tsv", default=None)
    parser.add_argument("--require_all_pass", action="store_true")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    required_datasets = list(spec.get("primary_evaluation", {}).get("datasets", []))
    seed_rows = read_tsv(args.seed_summary_tsv)
    eval_status, eval_detail = (
        seed_summary_gate(seed_rows, args.baseline, args.pct_methods, required_datasets)
        if seed_rows
        else eval_gate(read_tsv(args.eval_tsv), args.baseline, args.pct_methods, required_datasets)
    )
    rows = [
        status_line(
            "average12_vs_single_reference",
            eval_status,
            eval_detail,
        ),
        status_line(
            "high_dispersion_gain",
            *dispersion_gate(read_tsv(args.dispersion_gain_tsv), args.mean_method, args.pct_methods),
        ),
        status_line(
            "transport_diagnostic_separation",
            *diagnostic_gate(read_tsv(args.diagnostic_summary_tsv), args.diagnostic_auroc_threshold),
        ),
        status_line("causal_intervention", *intervention_gate(read_tsv(args.intervention_tsv))),
        status_line("contamination_robustness", *robustness_gate(read_tsv(args.robustness_tsv), args.mean_method, args.pct_methods)),
    ]
    output = "gate\tstatus\tdetail\n" + "\n".join(rows)
    print(output)
    if args.out_tsv:
        out = Path(args.out_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output + "\n", encoding="utf-8")
    if args.require_all_pass and any("\tpass\t" not in row for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
