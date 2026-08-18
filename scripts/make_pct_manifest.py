#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducible PCT experiment manifest.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--model_tag", default="qwen3_4b")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--train_num_samples", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--pct_num_references", type=int, default=4)
    parser.add_argument("--pct_loss_weight", type=float, default=0.1)
    parser.add_argument("--pct_layers", default="last")
    parser.add_argument("--pct_tau", type=float, default=0.05)
    parser.add_argument("--pct_geometry_weight", type=float, default=0.0)
    parser.add_argument("--pct_max_atoms", type=int, default=64)
    parser.add_argument("--pct_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--pct_sinkhorn_iters", type=int, default=40)
    parser.add_argument("--pct_uot_rho", type=float, default=0.5)
    parser.add_argument("--pct_fgw_outer", type=int, default=4)
    parser.add_argument("--pct_fgw_feature_weight", type=float, default=0.5)
    parser.add_argument("--pct_grassmann_rank", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=20000)
    parser.add_argument("--eval_datasets", nargs="+", default=["aime24", "aime25", "hmmt25"])
    parser.add_argument("--eval_val_n", type=int, default=12)
    parser.add_argument("--eval_seed", type=int, default=None, help="Optional fixed seed for Average@N generation.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed expansion, e.g. --seeds 0 1 2.")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_tag = "full" if args.train_num_samples == 0 else f"n{args.train_num_samples}"
    seeds = args.seeds if args.seeds is not None else [None]

    with out.open("w", encoding="utf-8") as f:
        for method in args.methods:
            for seed in seeds:
                run_name = f"{args.model_tag}_{method}_steps{args.max_steps}_{sample_tag}"
                if seed is not None:
                    run_name += f"_seed{seed}"
                record = {
                    "run_name": run_name,
                    "method": method,
                    "model": args.model,
                    "model_tag": args.model_tag,
                    "dataset": args.dataset,
                    "train_num_samples": args.train_num_samples,
                    "max_steps": args.max_steps,
                    "pct_num_references": args.pct_num_references,
                    "pct_loss_weight": 0.0 if method == "none" else args.pct_loss_weight,
                    "pct_layers": args.pct_layers,
                    "pct_tau": args.pct_tau,
                    "pct_geometry_weight": args.pct_geometry_weight,
                    "pct_max_atoms": args.pct_max_atoms,
                    "pct_sinkhorn_epsilon": args.pct_sinkhorn_epsilon,
                    "pct_sinkhorn_iters": args.pct_sinkhorn_iters,
                    "pct_uot_rho": args.pct_uot_rho,
                    "pct_fgw_outer": args.pct_fgw_outer,
                    "pct_fgw_feature_weight": args.pct_fgw_feature_weight,
                    "pct_grassmann_rank": args.pct_grassmann_rank,
                    "max_completion_length": args.max_completion_length,
                    "max_length": args.max_length,
                    "output_dir": str(Path(args.output_root) / run_name),
                    "eval": [
                        {
                            "dataset": dataset,
                            "val_n": args.eval_val_n,
                            "metric": f"Average@{args.eval_val_n}",
                            **({"seed": args.eval_seed} if args.eval_seed is not None else {}),
                        }
                        for dataset in args.eval_datasets
                    ],
                }
                if seed is not None:
                    record["seed"] = seed
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
