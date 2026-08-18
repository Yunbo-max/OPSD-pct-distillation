#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def pct_arg(record: dict, key: str, default):
    return record[key] if key in record else default


def checkpoint_complete(output_dir: Path) -> bool:
    if (output_dir / "adapter_model.safetensors").exists() or (output_dir / "adapter_model.bin").exists():
        return True
    if (output_dir / "trainer_state.json").exists():
        return True
    return any(output_dir.glob("checkpoint-*/trainer_state.json"))


def latest_state_file(output_dir: Path) -> Path | None:
    states = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if states:
        return states[-1]
    state = output_dir / "trainer_state.json"
    return state if state.exists() else None


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


def train_complete(record: dict, require_train_metrics: bool) -> bool:
    output_dir = Path(record["output_dir"])
    if not checkpoint_complete(output_dir):
        return False
    if not require_train_metrics:
        return True
    return not train_metric_missing(record, final_train_metrics(output_dir))


def build_command(record: dict, args: argparse.Namespace) -> list[str]:
    run_name = str(record["run_name"])
    method = str(record.get("method", "none"))
    output_dir = str(record["output_dir"])
    pct_weight = float(record.get("pct_loss_weight", 0.0 if method == "none" else args.pct_loss_weight))

    cmd = [
        args.accelerate,
        "launch",
        "--config_file",
        args.accelerate_config,
        "--num_processes",
        str(args.num_processes),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "opsd_train.py",
        "--model_name_or_path",
        str(record["model"]),
        "--pct_dataset_name",
        str(record["dataset"]),
        "--pct_train_num_samples",
        str(record.get("train_num_samples", args.train_num_samples)),
        "--learning_rate",
        str(args.learning_rate),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--output_dir",
        output_dir,
        "--run_config",
        run_name,
        "--max_completion_length",
        str(record.get("max_completion_length", args.max_completion_length)),
        "--save_steps",
        str(args.save_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--attn_implementation",
        args.attn_implementation,
        "--torch_dtype",
        args.torch_dtype,
        "--max_length",
        str(record.get("max_length", args.max_length)),
        "--beta",
        str(args.beta),
        "--use_peft",
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_target_modules",
        *args.lora_target_modules,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--lmbda",
        str(args.lmbda),
        "--fixed_teacher",
        "--jsd_token_clip",
        str(args.jsd_token_clip),
        "--report_to",
        args.report_to,
        "--pct_method",
        method,
        "--pct_loss_weight",
        str(pct_weight),
        "--pct_num_references",
        str(record.get("pct_num_references", args.pct_num_references)),
        "--pct_layers",
        str(pct_arg(record, "pct_layers", args.pct_layers)),
        "--pct_tau",
        str(pct_arg(record, "pct_tau", args.pct_tau)),
        "--pct_geometry_weight",
        str(pct_arg(record, "pct_geometry_weight", args.pct_geometry_weight)),
        "--pct_max_atoms",
        str(pct_arg(record, "pct_max_atoms", args.pct_max_atoms)),
        "--pct_sinkhorn_epsilon",
        str(pct_arg(record, "pct_sinkhorn_epsilon", args.pct_sinkhorn_epsilon)),
        "--pct_sinkhorn_iters",
        str(pct_arg(record, "pct_sinkhorn_iters", args.pct_sinkhorn_iters)),
        "--pct_uot_rho",
        str(pct_arg(record, "pct_uot_rho", args.pct_uot_rho)),
        "--pct_fgw_outer",
        str(pct_arg(record, "pct_fgw_outer", args.pct_fgw_outer)),
        "--pct_fgw_feature_weight",
        str(pct_arg(record, "pct_fgw_feature_weight", args.pct_fgw_feature_weight)),
        "--pct_grassmann_rank",
        str(pct_arg(record, "pct_grassmann_rank", args.pct_grassmann_rank)),
    ]
    if "seed" in record:
        cmd += ["--seed", str(record["seed"])]
    max_steps = int(record.get("max_steps", args.max_steps))
    if max_steps > 0:
        cmd += ["--max_steps", str(max_steps)]
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.trust_remote_code:
        cmd += ["--trust_remote_code", "true"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train every PCT run in a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--accelerate", default="accelerate")
    parser.add_argument("--accelerate_config", default="accelerate.yaml")
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method filter.")
    parser.add_argument("--run_names", nargs="*", default=None, help="Optional run-name filter.")
    parser.add_argument("--index", type=int, default=None, help="Run only the 0-based selected manifest record at this index.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--require_train_metrics", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--train_num_samples", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", default="5e-6")
    parser.add_argument("--max_grad_norm", default="0.1")
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--beta", default="0")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--temperature", default="1.1")
    parser.add_argument("--top_p", default="0.95")
    parser.add_argument("--top_k", default="20")
    parser.add_argument("--lmbda", default="1")
    parser.add_argument("--jsd_token_clip", default="0.05")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--pct_loss_weight", type=float, default=0.1)
    parser.add_argument("--pct_num_references", type=int, default=4)
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
    args = parser.parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.index is not None and args.index < 0:
        raise SystemExit("--index must be >= 0")

    methods = set(args.methods or [])
    run_names = set(args.run_names or [])
    selected = []
    eligible_pos = 0
    selected_or_skipped = 0
    for record in load_manifest(Path(args.manifest)):
        if methods and record.get("method") not in methods:
            continue
        if run_names and record.get("run_name") not in run_names:
            continue
        if record.get("requires_training") is False:
            continue
        if args.index is not None and eligible_pos != args.index:
            eligible_pos += 1
            continue
        if args.index is None and eligible_pos % args.num_shards != args.shard_index:
            eligible_pos += 1
            continue
        if args.skip_completed and train_complete(record, args.require_train_metrics):
            print(f"skip_completed\t{record['run_name']}")
            selected_or_skipped += 1
            eligible_pos += 1
            continue
        selected.append(record)
        selected_or_skipped += 1
        eligible_pos += 1

    if not selected and selected_or_skipped == 0:
        raise SystemExit("No manifest records selected for training.")

    for record in selected:
        cmd = build_command(record, args)
        print(shell_join(cmd))
        if not args.dry_run:
            subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
