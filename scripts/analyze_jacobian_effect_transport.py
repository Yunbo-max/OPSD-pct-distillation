#!/usr/bin/env python3
"""Summarize a fixed Jacobian-effect-transport gate with paired bootstrap CIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def estimate(values: np.ndarray, rng: np.random.Generator, bootstrap: int) -> dict:
    if not len(values):
        return {"n": 0, "mean": None, "ci": [None, None]}
    indices = rng.integers(0, len(values), size=(bootstrap, len(values)))
    means = values[indices].mean(1)
    return {
        "n": int(len(values)), "mean": float(values.mean()),
        "ci": [float(value) for value in np.quantile(means, [.025, .975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ridge_ratio", default="0.01")
    parser.add_argument("--active_threshold", type=float, default=1e-4)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = list(map(json.loads, Path(args.input).open()))
    rng = np.random.default_rng(args.seed)
    result = {
        "input": args.input, "n": len(rows), "ridge_ratio": args.ridge_ratio,
        "active_threshold": args.active_threshold, "layers": {},
    }
    jt = f"jt:{args.ridge_ratio}"
    reverse = f"reverse_jt:{args.ridge_ratio}"
    random = f"random_jt:{args.ridge_ratio}"
    for layer in sorted(rows[0]["layers"], key=int):
        records = [row["layers"][layer] for row in rows]
        source = np.array([item["source_gold_logp_gain"] for item in records])
        masks = {
            "all": np.ones(len(source), dtype=bool),
            "source_positive": source > args.active_threshold,
            "source_negative": source < -args.active_threshold,
            "source_inert": np.abs(source) <= args.active_threshold,
        }
        layer_result = {
            "counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "source_gold_gain": estimate(source, rng, args.bootstrap),
            "effect_geometry": {}, "strata": {},
        }
        for name in ("source_linear", "student_direct_linear", "direct", jt):
            layer_result["effect_geometry"][name] = {
                metric: estimate(np.array([item["arms"][name][metric] for item in records]), rng,
                                 args.bootstrap)
                for metric in ("cosine", "relative_squared_error")
            }
        for metric in ("cosine", "relative_squared_error"):
            source_values = np.array([
                item["arms"]["source_linear"][metric] for item in records
            ])
            student_values = np.array([
                item["arms"]["student_direct_linear"][metric] for item in records
            ])
            direct_values = np.array([item["arms"]["direct"][metric] for item in records])
            jt_values = np.array([item["arms"][jt][metric] for item in records])
            layer_result["effect_geometry"][f"source_minus_student_linear_{metric}"] = estimate(
                source_values - student_values, rng, args.bootstrap
            )
            layer_result["effect_geometry"][f"jt_minus_direct_{metric}"] = estimate(
                jt_values - direct_values, rng, args.bootstrap
            )
        norm_ratios = np.array([item["arms"][jt]["norm_ratio_to_c"] for item in records])
        layer_result["jt_norm_ratio_to_c"] = estimate(norm_ratios, rng, args.bootstrap)
        for stratum, mask in masks.items():
            entry = {}
            gains = {}
            for name in ("direct", jt, reverse, random):
                values = np.array([item["arms"][name]["gold_logp_gain"] for item in records])[mask]
                gains[name] = values
                entry[name] = estimate(values, rng, args.bootstrap)
            if mask.any():
                source_selected = source[mask]
                entry["aggregate_recovery_fraction"] = (
                    float(gains[jt].mean() / source_selected.mean())
                    if abs(source_selected.mean()) > 1e-12 else None
                )
                entry["jt_minus_direct"] = estimate(gains[jt] - gains["direct"], rng, args.bootstrap)
                entry["jt_minus_reverse"] = estimate(gains[jt] - gains[reverse], rng, args.bootstrap)
                entry["jt_minus_random"] = estimate(gains[jt] - gains[random], rng, args.bootstrap)
            layer_result["strata"][stratum] = entry
        active = masks["source_positive"]
        if active.sum() > 1:
            jt_gain = np.array([item["arms"][jt]["gold_logp_gain"] for item in records])
            layer_result["source_to_jt_gain_correlation_active"] = float(
                np.corrcoef(source[active], jt_gain[active])[0, 1]
            )
        result["layers"][layer] = layer_result

    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
