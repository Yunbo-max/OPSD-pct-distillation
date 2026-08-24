#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def exponential_fit(points: list[tuple[int, float]]) -> tuple[float, float, float | None]:
    if len(points) < 2:
        return float("nan"), float("nan"), None
    slope, intercept = np.polyfit(
        [x for x, _ in points], np.log([y for _, y in points]), 1
    )
    tau = float(-1.0 / slope) if np.isfinite(slope) and slope < 0 else None
    return float(slope), float(intercept), tau


def first_crossing(curve: list[dict], key: str, threshold: float) -> float | None:
    """Linearly interpolate the first lag where `key` falls below threshold."""
    if not curve or curve[0][key] <= threshold:
        return 0.0 if curve else None
    for previous, current in zip(curve, curve[1:]):
        y0, y1 = previous[key], current[key]
        if y1 <= threshold < y0:
            if y0 == y1:
                return float(current["lag"])
            fraction = (y0 - threshold) / (y0 - y1)
            return float(previous["lag"] + fraction * (current["lag"] - previous["lag"]))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for source in args.inputs
        for line in Path(source).read_text().splitlines() if line
    ]
    grouped = defaultdict(list)
    for row in rows:
        for values in row["curves"].values():
            for value in values:
                grouped[value["lag"]].append(value)
    curve = []
    for lag, values in sorted(grouped.items()):
        result = {
            "lag": lag, "n": len(values),
            "alignment_mean": float(np.mean([x["alignment"] for x in values])),
            "recovery_mean": float(np.mean([x["recovery"] for x in values])),
            "projected_recovery_mean": float(np.mean([x["projected_recovery"] for x in values])),
        }
        if "random_alignment" in values[0]:
            for key in ("random_alignment", "random_recovery", "random_projected_recovery"):
                result[f"{key}_mean"] = float(np.mean([x[key] for x in values]))
            result["excess_alignment_mean"] = (
                result["alignment_mean"] - result["random_alignment_mean"]
            )
            result["excess_projected_recovery_mean"] = (
                result["projected_recovery_mean"]
                - result["random_projected_recovery_mean"]
            )
        curve.append(result)

    # The raw projected recovery has a positive random-control floor. Fitting
    # that floor produces an arbitrarily long and misleading time constant.
    # Keep the old raw fit as a diagnostic, but define persistence using the
    # random-subtracted excess and only its initial decay down to 5% of lag 0.
    raw_usable = []
    for row in curve:
        if row["projected_recovery_mean"] <= 1e-8:
            break
        raw_usable.append((row["lag"], row["projected_recovery_mean"]))
    raw_slope, raw_intercept, raw_tau = exponential_fit(raw_usable)

    excess_key = "excess_projected_recovery_mean"
    excess_fit = []
    initial_excess = curve[0].get(excess_key, 0.0) if curve else 0.0
    fit_floor = max(initial_excess * 0.05, 1e-8)
    if initial_excess > 0:
        for row in curve:
            value = row.get(excess_key, 0.0)
            if value <= 0:
                break
            excess_fit.append((row["lag"], value))
            if row["lag"] > 0 and value <= fit_floor:
                break
    slope, intercept, tau = exponential_fit(excess_fit)
    one_step_retention = (
        curve[1].get(excess_key, 0.0) / initial_excess
        if len(curve) > 1 and initial_excess > 0 else None
    )
    report = {
        "n_problems": len(rows), "curve": curve,
        "decay_signal": "projected_recovery_minus_norm_matched_random",
        "decay_fit_threshold_fraction": 0.05,
        "decay_fit_points": len(excess_fit),
        "exponential_decay_slope": float(slope),
        "persistence_tau": tau,
        "one_step_excess_retention": one_step_retention,
        "half_life_tokens": first_crossing(curve, excess_key, initial_excess * 0.5)
        if initial_excess > 0 else None,
        "e_fold_crossing_tokens": first_crossing(curve, excess_key, initial_excess / np.e)
        if initial_excess > 0 else None,
        "legacy_raw_fit": {
            "fit_points": len(raw_usable),
            "exponential_decay_slope": raw_slope,
            "intercept": raw_intercept,
            "persistence_tau": raw_tau,
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
