#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    return "\n".join(lines)


def latex_table(rows: list[dict[str, str]], columns: list[str], caption: str, label: str) -> str:
    if not rows:
        return ""
    colspec = "l" * len(columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(columns).replace("_", "\\_") + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = [row.get(col, "").replace("_", "\\_") for col in columns]
        lines.append(" & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def maybe_section(title: str, path: str | None, columns: list[str], caption: str, label: str) -> tuple[str, str]:
    if not path:
        return f"## {title}\n\n_Not provided._\n", ""
    rows = read_tsv(Path(path))
    md = f"## {title}\n\n" + markdown_table(rows, columns) + "\n"
    tex = latex_table(rows, columns, caption, label)
    return md, tex


def provenance_section(path: str | None) -> str:
    if not path:
        return "## Dataset Provenance\n\n_Not provided._\n"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source = data.get("source", {})
    stats = data.get("reference_stats", {})
    lines = [
        "## Dataset Provenance",
        "",
        f"- Dataset: `{data.get('dataset')}`",
        f"- Source type: `{source.get('type')}`",
        f"- Rows: `{data.get('num_rows')}`",
        f"- Problem rows: `{data.get('problem_rows')}`",
        f"- Solution rows: `{data.get('solution_rows')}`",
        f"- Source SHA256: `{source.get('sha256', source.get('fingerprint'))}`",
        f"- Row-hash SHA256: `{data.get('hashes', {}).get('row_hash_sha256')}`",
        f"- Reference count min/mean/max: `{stats.get('reference_count_min')}` / `{stats.get('reference_count_mean')}` / `{stats.get('reference_count_max')}`",
        f"- Too-few-reference rows: `{stats.get('too_few_reference_rows')}`",
        f"- Duplicate-reference rows: `{stats.get('duplicate_reference_rows')}`",
        f"- Wrong/shuffled reference rows: `{stats.get('rows_with_wrong_reference')}` / `{stats.get('rows_with_shuffled_reference')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready Markdown and LaTeX tables from PCT TSVs.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_tsv", default=None)
    parser.add_argument("--eval_tsv", default=None)
    parser.add_argument("--seed_summary_tsv", default=None)
    parser.add_argument("--bootstrap_tsv", default=None)
    parser.add_argument("--dispersion_gain_tsv", default=None)
    parser.add_argument("--intervention_tsv", default=None)
    parser.add_argument("--robustness_tsv", default=None)
    parser.add_argument("--claim_gates_tsv", default=None)
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--dataset_provenance_json", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = ["# PCT Experiment Report\n"]
    latex_parts = []
    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
        git = metadata.get("git", {})
        sections.append(
            "## Run Metadata\n\n"
            f"- Git commit: `{git.get('commit')}`\n"
            f"- Dirty worktree: `{git.get('dirty')}`\n"
            f"- Platform: `{metadata.get('platform')}`\n"
        )
    sections.append(provenance_section(args.dataset_provenance_json))

    specs = [
        (
            "Training Summary",
            args.train_tsv,
            ["run", "step", "loss", "pct_loss", "pct_transport_mass", "grad_norm"],
            "Training smoke/full-run summary.",
            "tab:pct_train",
        ),
        (
            "Evaluation Summary",
            args.eval_tsv,
            ["file", "dataset", "Average@N", "Pass@N", "Maj@N", "Format", "N", "Problems", "Seed"],
            "Average@N evaluation summary.",
            "tab:pct_eval",
        ),
        (
            "Seed Summary",
            args.seed_summary_tsv,
            ["method", "dataset", "seeds", "mean_average_at_n", "std_average_at_n"],
            "Average@N aggregated across seeds.",
            "tab:pct_seed_summary",
        ),
        (
            "Bootstrap Evaluation",
            args.bootstrap_tsv,
            [
                "dataset",
                "method",
                "problems",
                "average_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "delta_vs_mean_pct",
                "delta_ci95_low_pct",
                "delta_ci95_high_pct",
            ],
            "Bootstrap confidence intervals and paired deltas.",
            "tab:pct_bootstrap",
        ),
        (
            "Dispersion Gain",
            args.dispersion_gain_tsv,
            ["group", "method", "average_pct", "gain_vs_mean_pct", "matched_problems"],
            "Performance grouped by reference dispersion.",
            "tab:pct_dispersion",
        ),
        (
            "Causal Intervention",
            args.intervention_tsv,
            ["vector", "mean_delta_logp", "positive_rate", "n"],
            "Hidden-flow causal intervention summary.",
            "tab:pct_intervention",
        ),
        (
            "Robustness By Perturbation",
            args.robustness_tsv,
            ["dataset", "method", "perturbation", "problems", "average_pct", "delta_vs_clean_pct"],
            "Robustness grouped by deterministic perturbation.",
            "tab:pct_robustness",
        ),
        (
            "Claim Gates",
            args.claim_gates_tsv,
            ["gate", "status", "detail"],
            "Claim-gate status from generated experiment artifacts.",
            "tab:pct_claim_gates",
        ),
    ]

    for spec in specs:
        md, tex = maybe_section(*spec)
        sections.append(md)
        if tex:
            latex_parts.append(tex)

    (out_dir / "report.md").write_text("\n".join(sections), encoding="utf-8")
    (out_dir / "tables.tex").write_text("\n\n".join(latex_parts) + "\n", encoding="utf-8")
    print(f"wrote\t{out_dir / 'report.md'}")
    print(f"wrote\t{out_dir / 'tables.tex'}")


if __name__ == "__main__":
    main()
