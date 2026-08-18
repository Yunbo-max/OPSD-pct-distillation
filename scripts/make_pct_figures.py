#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def read_tsv(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(value: str | None) -> float | None:
    if value in (None, "", "NA", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width = max(640, 90 * max(1, len(labels)))
    height = 420
    margin_left = 70
    margin_bottom = 105
    margin_top = 50
    plot_w = width - margin_left - 30
    plot_h = height - margin_top - margin_bottom
    max_val = max([0.0, *values])
    upper = max(1.0, max_val * 1.15)
    bar_w = plot_w / max(1, len(values)) * 0.65
    gap = plot_w / max(1, len(values))

    def y(v: float) -> float:
        return margin_top + plot_h - (v / upper) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
        f'<text x="20" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 20 {margin_top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(ylabel)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - 30}" y2="{margin_top + plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>',
    ]
    for tick in range(6):
        val = upper * tick / 5
        yy = y(val)
        lines.append(f'<line x1="{margin_left - 4}" y1="{yy:.1f}" x2="{margin_left}" y2="{yy:.1f}" stroke="#222"/>')
        lines.append(
            f'<text x="{margin_left - 8}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{val:.2f}</text>'
        )
        if tick:
            lines.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width - 30}" y2="{yy:.1f}" stroke="#ddd"/>')

    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + i * gap + (gap - bar_w) / 2
        yy = y(value)
        h = margin_top + plot_h - yy
        lines.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2f6f9f"/>')
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{yy - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 18}" text-anchor="end" transform="rotate(-35 {x + bar_w / 2:.1f} {margin_top + plot_h + 18})" font-family="Arial" font-size="11">{html.escape(label)}</text>'
        )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostic_auroc(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = []
    values = []
    for row in rows:
        val = as_float(row.get("auroc_cc_lt_cw"))
        if val is not None:
            labels.append(row["metric"])
            values.append(val)
    if values:
        write_bar_svg(out_dir / "diagnostic_auroc_correct_vs_wrong.svg", "Diagnostic AUROC: correct refs closer than wrong", labels, values, "AUROC")


def eval_average(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = []
    values = []
    for row in rows:
        val = as_float(row.get("Average@N"))
        if val is not None:
            labels.append(f"{row.get('dataset', '')}:{row.get('file', '')}".replace("_avg12.json", ""))
            values.append(val)
    if values:
        write_bar_svg(out_dir / "eval_average_at_n.svg", "Evaluation Average@N", labels, values, "percent")


def dispersion_gain(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = []
    values = []
    gain_key = next((key for key in rows[0].keys() if key.startswith("gain_vs_")), None) if rows else None
    if not gain_key:
        return
    for row in rows:
        if row.get("group") not in {"low", "high"}:
            continue
        val = as_float(row.get(gain_key))
        if val is not None:
            labels.append(f"{row.get('group')}:{row.get('method')}")
            values.append(val)
    if values:
        write_bar_svg(out_dir / "dispersion_group_gain.svg", "Gain by reference dispersion group", labels, values, "percentage points")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dependency-free SVG figures from PCT TSV artifacts.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--diagnostic_summary_tsv", default=None)
    parser.add_argument("--eval_tsv", default=None)
    parser.add_argument("--dispersion_gain_tsv", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    diagnostic_auroc(read_tsv(args.diagnostic_summary_tsv), out_dir)
    eval_average(read_tsv(args.eval_tsv), out_dir)
    dispersion_gain(read_tsv(args.dispersion_gain_tsv), out_dir)
    print(f"wrote_figures\t{out_dir}")


if __name__ == "__main__":
    main()
