#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODELS = (
    "qwen3_1p7b=Qwen/Qwen3-1.7B",
    "qwen3_4b=Qwen/Qwen3-4B",
    "qwen3_8b=Qwen/Qwen3-8B",
)

DEFAULT_METHODS = ("none", "phf_single", "phf_mean", "phf_grassmann", "phf_set", "set_fgw", "set_uot")


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected tag=model, for example qwen3_4b=Qwen/Qwen3-4B")
    tag, model = value.split("=", 1)
    return tag, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a multi-model PCT scaling manifest.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--models", nargs="+", type=parse_pair, default=[parse_pair(item) for item in DEFAULT_MODELS])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--train_num_samples", type=int, default=29434)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means epoch/budget configured by the training launcher.")
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
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional training seed expansion.")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_tag = "full" if args.train_num_samples == 0 else f"n{args.train_num_samples}"
    step_tag = "full" if args.max_steps == 0 else f"steps{args.max_steps}"
    seeds = args.seeds if args.seeds is not None else [None]

    with out.open("w", encoding="utf-8") as f:
        for model_tag, model in args.models:
            for method in args.methods:
                for seed in seeds:
                    run_name = f"{model_tag}_{method}_{step_tag}_{sample_tag}"
                    if seed is not None:
                        run_name += f"_seed{seed}"
                    record = {
                        "run_name": run_name,
                        "method": method,
                        "model": model,
                        "model_tag": model_tag,
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
