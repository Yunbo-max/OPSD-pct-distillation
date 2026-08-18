#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import load_dataset


REFERENCE_KEYS = ("references", "reference_solutions", "solutions")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row(row: dict[str, Any]) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def get_references(row: dict[str, Any]) -> list[str]:
    refs = None
    for key in REFERENCE_KEYS:
        value = row.get(key)
        if value:
            refs = value
            break
    if refs is None:
        refs = [row.get("solution")] if row.get("solution") else []
    if isinstance(refs, str):
        refs = [refs]
    return [str(ref).strip() for ref in refs if str(ref).strip()]


def normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def load_rows(path_or_name: str, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path_or_name.endswith((".json", ".jsonl")):
        path = Path(path_or_name)
        dataset = load_dataset("json", data_files=str(path))["train"]
        source = {
            "type": "local_json",
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "split": "train",
        }
    else:
        dataset = load_dataset(path_or_name, split=split)
        source = {
            "type": "huggingface_dataset",
            "name": path_or_name,
            "split": split,
            "fingerprint": getattr(dataset, "_fingerprint", None),
        }
    return [dict(row) for row in dataset], source


def row_hashes(rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    limit = len(rows) if max_examples <= 0 else min(max_examples, len(rows))
    samples: list[str] = []
    for idx, row in enumerate(rows[:limit]):
        row_digest = hashlib.sha256(canonical_row(row)).hexdigest()
        digest.update(row_digest.encode("ascii"))
        if idx < 3:
            samples.append(row_digest)
    return {
        "num_hashed_rows": limit,
        "row_hash_sha256": digest.hexdigest(),
        "first_row_hashes": samples,
    }


def reference_stats(rows: list[dict[str, Any]], min_refs: int) -> dict[str, Any]:
    ref_counts: list[int] = []
    rows_with_refs = 0
    too_few_refs = 0
    duplicate_reference_rows = 0
    rows_with_wrong_reference = 0
    rows_with_shuffled_reference = 0

    for row in rows:
        refs = get_references(row)
        ref_counts.append(len(refs))
        if refs:
            rows_with_refs += 1
        if len(refs) < min_refs:
            too_few_refs += 1
        unique_refs = {normalized(ref) for ref in refs}
        if len(unique_refs) < min(len(refs), min_refs):
            duplicate_reference_rows += 1
        if str(row.get("wrong_reference", "")).strip():
            rows_with_wrong_reference += 1
        if str(row.get("shuffled_reference", "")).strip():
            rows_with_shuffled_reference += 1

    return {
        "min_refs_required": min_refs,
        "rows_with_refs": rows_with_refs,
        "too_few_reference_rows": too_few_refs,
        "duplicate_reference_rows": duplicate_reference_rows,
        "rows_with_wrong_reference": rows_with_wrong_reference,
        "rows_with_shuffled_reference": rows_with_shuffled_reference,
        "reference_count_min": min(ref_counts) if ref_counts else 0,
        "reference_count_mean": mean(ref_counts) if ref_counts else 0.0,
        "reference_count_max": max(ref_counts) if ref_counts else 0,
        "reference_count_histogram": {str(k): ref_counts.count(k) for k in sorted(set(ref_counts))},
    }


def build_record(dataset: str, split: str, min_refs: int, max_hash_examples: int) -> dict[str, Any]:
    rows, source = load_rows(dataset, split)
    fields = sorted({key for row in rows for key in row})
    return {
        "dataset": dataset,
        "source": source,
        "num_rows": len(rows),
        "fields": fields,
        "problem_rows": sum(1 for row in rows if row.get("problem")),
        "solution_rows": sum(1 for row in rows if row.get("solution")),
        "reference_stats": reference_stats(rows, min_refs),
        "hashes": row_hashes(rows, max_hash_examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture auditable provenance for a PCT training dataset.")
    parser.add_argument("--dataset", required=True, help="HF dataset name or local .json/.jsonl file.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min_refs", type=int, default=4)
    parser.add_argument("--max_hash_examples", type=int, default=0, help="0 means hash every row.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    record = build_record(args.dataset, args.split, args.min_refs, args.max_hash_examples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
