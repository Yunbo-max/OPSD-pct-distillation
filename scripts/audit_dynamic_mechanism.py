#!/usr/bin/env python3
"""Audit the concrete artifacts required by the dynamic-mechanism checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def jsonl_count(path: Path) -> int:
    return sum(bool(line) for line in path.read_text().splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/privileged_effect")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    checks = {}

    def check(name: str, condition: bool, evidence: object) -> None:
        checks[name] = {"passed": bool(condition), "evidence": evidence}

    matrix_path = root / "online_id288_matrix_summary.json"
    if matrix_path.exists():
        matrix = load_json(matrix_path)
        check("online_matrix_240_complete", matrix.get("complete") is True, {
            "completed": matrix.get("completed_arms"), "missing": len(matrix.get("missing_keys", [])),
        })
        random_rows = [row for row in matrix.get("results", []) if row.get("mode") == "random"]
        check("online_random_80_complete", len(random_rows) == 80, len(random_rows))
        check("online_five_arms", bool(matrix.get("baseline")) and len(matrix.get("results", [])) == 240,
              {"baseline": bool(matrix.get("baseline")), "interventions": len(matrix.get("results", []))})
    else:
        check("online_matrix_240_complete", False, "missing")
        check("online_random_80_complete", False, "missing")
        check("online_five_arms", False, "missing")

    deterministic_paths = [
        root / "online_id288_random_deterministic_shard0.jsonl",
        root / "online_id288_random_deterministic_shard1.jsonl",
    ]
    if all(path.exists() for path in deterministic_paths):
        deterministic = [
            json.loads(line) for path in deterministic_paths
            for line in path.read_text().splitlines() if line
        ]
        random = [x for x in deterministic if x.get("mode") == "random"]
        check("deterministic_random_seeds", len(random) == 80 and all(x.get("seed") is not None for x in random),
              {"random_arms": len(random), "seeded": sum(x.get("seed") is not None for x in random)})
    else:
        check("deterministic_random_seeds", False, "missing")

    paired_paths = [
        root / "prm800k_paired_rescue_100_shard0.jsonl",
        root / "prm800k_paired_rescue_100_shard1.jsonl",
    ]
    paired_count = sum(jsonl_count(path) for path in paired_paths if path.exists())
    check("prm_teacher_forced_100", all(path.exists() for path in paired_paths) and paired_count == 100,
          paired_count if any(path.exists() for path in paired_paths) else "missing")
    baseline_paths = [
        root / "prm800k_online_baseline_100_shard0.jsonl",
        root / "prm800k_online_baseline_100_shard1.jsonl",
    ]
    baseline_count = sum(jsonl_count(path) for path in baseline_paths if path.exists())
    check("prm_online_baseline_100", all(path.exists() for path in baseline_paths) and baseline_count == 100,
          baseline_count if any(path.exists() for path in baseline_paths) else "missing")

    decay_paths = [
        root / "residual_decay_108_l25_to_l35_shard0.jsonl",
        root / "residual_decay_108_l25_to_l35_shard1.jsonl",
    ]
    decay_summary_path = root / "residual_decay_108_l25_to_l35_summary.json"
    decay_count = sum(jsonl_count(path) for path in decay_paths if path.exists())
    check("residual_decay_108", all(path.exists() for path in decay_paths) and decay_count == 108,
          decay_count if any(path.exists() for path in decay_paths) else "missing")
    if decay_summary_path.exists():
        decay = load_json(decay_summary_path)
        check("residual_decay_summary", len(decay.get("curve", [])) > 1 and "persistence_tau" in decay,
              {"lags": len(decay.get("curve", [])), "tau": decay.get("persistence_tau")})
    else:
        check("residual_decay_summary", False, "missing")

    fpca_path = root / "spatiotemporal_fpca_layer25.json"
    fpca = load_json(fpca_path) if fpca_path.exists() else {}
    check("spatiotemporal_fpca", bool(fpca.get("heldout_explained_variance")), {
        "n_test": fpca.get("n_test"), "temporal_rank95": fpca.get("temporal_rank95"),
    })
    heldout_path = root / "heldout_projected_rescue_layer25_with_full_summary.json"
    heldout = load_json(heldout_path) if heldout_path.exists() else {}
    arms = heldout.get("arms", {})
    check("heldout_subspace", "full" in arms and "pca:rank=16" in arms,
          {"n": heldout.get("n"), "arms": len(arms)})

    report = {
        "complete": all(item["passed"] for item in checks.values()),
        "passed": sum(item["passed"] for item in checks.values()),
        "total": len(checks), "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
