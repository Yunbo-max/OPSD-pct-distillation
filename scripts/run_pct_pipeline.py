#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Step:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str] | None = None


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def add_env(prefix: dict[str, str] | None, command: str) -> str:
    if not prefix:
        return command
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in prefix.items()) + " " + command


def sample_tag(train_num_samples: int) -> str:
    return "full" if train_num_samples == 0 else f"n{train_num_samples}"


def run_name(model_tag: str, method: str, max_steps: int, train_num_samples: int) -> str:
    return f"{model_tag}_{method}_steps{max_steps}_{sample_tag(train_num_samples)}"


def eval_json(eval_root: Path, model_tag: str, method: str, max_steps: int, train_num_samples: int, dataset: str) -> Path:
    return eval_root / f"{run_name(model_tag, method, max_steps, train_num_samples)}_{dataset}_avg12.json"


def build_steps(args) -> list[Step]:
    opsd = Path(args.opsd_root).resolve()
    standalone = Path(args.set_phf_root).resolve()
    out = Path(args.out_root).resolve()
    source_data = Path(args.dataset).resolve() if args.dataset.endswith((".json", ".jsonl")) else Path(args.dataset)
    candidate_data = out / "data" / f"multiref_candidates_{args.candidate_examples or 'all'}.jsonl"
    data = candidate_data if args.use_generated_multiref else source_data
    subset = out / "data" / f"multiref_{args.train_num_samples}.jsonl"
    diagnostic = out / "diagnostics" / f"ot_diag_{args.diagnostic_examples}.jsonl"
    diagnostic_summary_tsv = out / "diagnostics" / f"ot_diag_{args.diagnostic_examples}_summary.tsv"
    diagnostic_csv = out / "diagnostics" / f"ot_diag_{args.diagnostic_examples}_per_problem.csv"
    intervention = out / "diagnostics" / f"intervention_{args.intervention_examples}.jsonl"
    intervention_tsv = out / "diagnostics" / f"intervention_{args.intervention_examples}_summary.tsv"
    manifest = out / "runs" / "manifest.jsonl"
    train_tasks_tsv = out / "runs" / "train_tasks.tsv"
    eval_tasks_tsv = out / "runs" / "eval_tasks.tsv"
    manifest_status_tsv = out / "runs" / "manifest_status.tsv"
    resume_plan_json = out / "runs" / "resume_plan.json"
    resume_plan_md = out / "runs" / "resume_plan.md"
    train_root = out / "runs" / "pct_short_matrix"
    eval_root = out / "eval" / "pct_average12"
    eval_tsv = eval_root / "eval_summary.tsv"
    seed_eval_tsv = eval_root / "seed_summary.tsv"
    secondary_eval_root = out / "eval" / "pct_secondary"
    secondary_eval_tsv = secondary_eval_root / "eval_summary.tsv"
    report_root = out / "report"
    bootstrap_tsv = eval_root / f"bootstrap_{args.analysis_dataset}.tsv"
    dispersion_gain_tsv = report_root / f"dispersion_gain_{args.analysis_dataset}.tsv"
    claim_gates_tsv = report_root / "claim_gates.tsv"
    checksums_jsonl = report_root / "artifact_checksums.jsonl"
    metadata = out / "run_metadata.json"
    preflight_json = out / "preflight.json"
    source_provenance = out / "data" / "source_dataset_provenance.json"
    train_provenance = out / "data" / "train_dataset_provenance.json"
    spec = opsd / "configs" / "pct_neurips_spec.json"

    steps: list[Step] = []
    explicit_candidate_steps = any(
        stage in args.stages for stage in ("build-multiref-candidates", "audit-multiref-candidates")
    )
    wants_candidate_steps = args.use_generated_multiref or explicit_candidate_steps
    if wants_candidate_steps:
        build_candidate_cmd = [
            args.python,
            "scripts/build_multiref_candidates.py",
            "--out",
            str(candidate_data),
            "--model",
            args.candidate_model,
            "--dataset_split",
            args.dataset_split,
            "--max_new_tokens",
            str(args.candidate_max_new_tokens),
            "--temperature",
            str(args.candidate_temperature),
            "--top_p",
            str(args.candidate_top_p),
            "--min_verified_refs",
            str(args.min_refs),
        ]
        if args.candidate_examples:
            build_candidate_cmd += ["--n_examples", str(args.candidate_examples)]
        if args.dataset.endswith((".json", ".jsonl")):
            build_candidate_cmd += ["--dataset_jsonl", str(source_data)]
        else:
            build_candidate_cmd += ["--dataset_name", args.dataset]
        steps.append(Step("build-multiref-candidates", build_candidate_cmd, standalone))
        steps.append(
            Step(
                "audit-multiref-candidates",
                [
                    args.python,
                    "scripts/audit_multiref_candidates.py",
                    str(candidate_data),
                    "--min_refs",
                    str(args.min_refs),
                    "--require_wrong_reference",
                    "--require_shuffled_reference",
                ],
                standalone,
            )
        )
    steps.append(
        Step(
            "capture-metadata",
            [
                args.python,
                "scripts/capture_run_metadata.py",
                "--out",
                str(metadata),
                "--manifest",
                str(manifest),
                "--notes",
                args.metadata_notes,
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "preflight",
            [
                args.python,
                "scripts/preflight_pct.py",
                "--dataset",
                str(data),
                "--min_gpus",
                str(args.min_gpus),
                "--json",
                str(preflight_json),
            ]
            + (["--require_vllm"] if args.require_vllm else []),
            opsd,
        )
    )
    steps.append(
        Step(
            "validate-data",
            [
                args.python,
                "scripts/validate_multiref_dataset.py",
                "--dataset",
                str(data),
                "--min_refs",
                str(args.min_refs),
                "--write_subset",
                str(subset),
                "--subset_size",
                str(args.train_num_samples),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "source-data-provenance",
            [
                args.python,
                "scripts/capture_dataset_provenance.py",
                "--dataset",
                str(data),
                "--split",
                args.dataset_split,
                "--min_refs",
                str(args.min_refs),
                "--out",
                str(source_provenance),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "train-data-provenance",
            [
                args.python,
                "scripts/capture_dataset_provenance.py",
                "--dataset",
                str(subset),
                "--split",
                "train",
                "--min_refs",
                str(args.min_refs),
                "--out",
                str(train_provenance),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "validate-spec",
            [
                args.python,
                "scripts/validate_pct_experiment_spec.py",
                "--spec",
                str(spec),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "manifest",
            [
                args.python,
                "scripts/make_pct_manifest.py",
                "--out",
                str(manifest),
                "--model",
                args.model,
                "--dataset",
                str(subset),
                "--output_root",
                str(train_root),
                "--model_tag",
                args.model_tag,
                "--train_num_samples",
                str(args.train_num_samples),
                "--max_steps",
                str(args.max_steps),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "validate-manifest",
            [
                args.python,
                "scripts/validate_pct_experiment_spec.py",
                "--spec",
                str(spec),
                "--manifest",
                str(manifest),
                "--suite",
                "short_matrix",
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "count-train-tasks",
            [
                args.python,
                "scripts/count_pct_manifest_tasks.py",
                "--manifest",
                str(manifest),
                "--task",
                "train",
                "--print_array",
                "--tsv",
                str(train_tasks_tsv),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "count-eval-tasks",
            [
                args.python,
                "scripts/count_pct_manifest_tasks.py",
                "--manifest",
                str(manifest),
                "--task",
                "eval",
                "--print_array",
                "--tsv",
                str(eval_tasks_tsv),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "manifest-status",
            [
                args.python,
                "scripts/summarize_pct_manifest_status.py",
                "--manifest",
                str(manifest),
                "--eval_root",
                str(eval_root),
                "--require_train_metrics",
                "--tsv",
                str(manifest_status_tsv),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "resume-plan",
            [
                args.python,
                "scripts/make_pct_resume_plan.py",
                "--status_tsv",
                str(manifest_status_tsv),
                "--manifest",
                str(manifest),
                "--eval_root",
                str(eval_root),
                "--json_out",
                str(resume_plan_json),
                "--md_out",
                str(resume_plan_md),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "ot-diagnostic",
            [
                args.python,
                "scripts/run_ot_diagnostic.py",
                "--model",
                args.model,
                "--dataset_jsonl",
                str(subset),
                "--n_examples",
                str(args.diagnostic_examples),
                "--max_rollout_tokens",
                str(args.diagnostic_rollout_tokens),
                "--max_atoms",
                str(args.max_atoms),
                "--out",
                str(diagnostic),
            ],
            standalone,
        )
    )
    steps.append(
        Step(
            "summarize-diagnostic",
            [
                args.python,
                "scripts/summarize_ot_diagnostic.py",
                str(diagnostic),
                "--summary_tsv",
                str(diagnostic_summary_tsv),
                "--per_problem_csv",
                str(diagnostic_csv),
            ],
            standalone,
        )
    )
    steps.append(
        Step(
            "causal-intervention",
            [
                args.python,
                "scripts/run_causal_intervention.py",
                "--model",
                args.model,
                "--dataset_jsonl",
                str(subset),
                "--n_examples",
                str(args.intervention_examples),
                "--max_rollout_tokens",
                str(args.intervention_rollout_tokens),
                "--max_atoms",
                str(args.max_atoms),
                "--max_positions",
                str(args.max_positions),
                "--alpha",
                str(args.alpha),
                "--out",
                str(intervention),
            ],
            standalone,
        )
    )
    steps.append(
        Step(
            "summarize-intervention",
            [
                args.python,
                "scripts/summarize_causal_intervention.py",
                str(intervention),
                "--tsv",
                str(intervention_tsv),
            ],
            standalone,
        )
    )
    steps.append(
        Step(
            "train-short-matrix",
            [
                args.python,
                "scripts/run_pct_train_from_manifest.py",
                "--manifest",
                str(manifest),
                "--num_processes",
                str(args.num_processes),
                "--max_steps",
                str(args.max_steps),
                "--train_num_samples",
                str(args.train_num_samples),
                "--max_completion_length",
                str(args.train_rollout_tokens),
                "--max_length",
                str(args.max_length),
                "--batch_size",
                str(args.batch_size),
                "--grad_accum",
                str(args.grad_accum),
                "--skip_completed",
                "--require_train_metrics",
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "summarize-train",
            [args.python, "scripts/summarize_train_runs.py", str(train_root), "--tsv", str(train_root / "train_summary.tsv")],
            opsd,
        )
    )
    steps.append(
        Step(
            "eval-average12",
            [
                args.python,
                "scripts/run_pct_eval_from_manifest.py",
                "--manifest",
                str(manifest),
                "--eval_root",
                str(eval_root),
                "--python",
                args.python,
                "--backend",
                args.eval_backend,
                "--tensor_parallel_size",
                str(args.tensor_parallel_size),
                "--max_model_len",
                str(args.eval_max_model_len),
                "--max_new_tokens",
                str(args.eval_max_new_tokens),
                "--expected_num_problems",
                str(args.expected_eval_num_problems),
                "--skip_completed",
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "summarize-eval",
            [
                args.python,
                "scripts/summarize_eval_results.py",
                str(eval_root),
                "--tsv",
                str(eval_tsv),
                "--manifest",
                str(manifest),
                "--strict_protocol",
                "--expected_num_problems",
                str(args.expected_eval_num_problems),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "summarize-seeds",
            [
                args.python,
                "scripts/summarize_seed_results.py",
                "--manifest",
                str(manifest),
                "--eval_tsv",
                str(eval_tsv),
                "--out_tsv",
                str(seed_eval_tsv),
            ],
            opsd,
        )
    )
    analysis_eval_args = []
    for method in [args.dispersion_baseline, *args.analysis_methods]:
        analysis_eval_args.extend(
            [
                "--eval",
                f"{method}={eval_json(eval_root, args.model_tag, method, args.max_steps, args.train_num_samples, args.analysis_dataset)}",
            ]
        )
    steps.append(
        Step(
            "bootstrap-eval",
            [
                args.python,
                "scripts/bootstrap_eval_matrix.py",
                *analysis_eval_args,
                "--baseline",
                args.dispersion_baseline,
                "--n_boot",
                str(args.bootstrap_samples),
                "--seed",
                str(args.bootstrap_seed),
                "--strict_protocol",
                "--expected_num_problems",
                str(args.expected_eval_num_problems),
                "--require_paired_problem_ids",
                "--tsv",
                str(bootstrap_tsv),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "dispersion-gain",
            [
                args.python,
                "scripts/analyze_dispersion_gains.py",
                "--diagnostic_csv",
                str(diagnostic_csv),
                "--dispersion_metric",
                args.dispersion_metric,
                *analysis_eval_args,
                "--baseline",
                args.dispersion_baseline,
                "--strict_protocol",
                "--expected_num_problems",
                str(args.expected_eval_num_problems),
                "--min_matched_per_group",
                str(args.min_matched_per_dispersion_group),
                "--tsv",
                str(dispersion_gain_tsv),
            ],
            opsd,
        )
    )
    if args.secondary_eval_jsonls:
        steps.append(
            Step(
                "eval-secondary",
                ["bash", "scripts/run_pct_eval_secondary.sh"],
                opsd,
                env={
                    "BASE_MODEL": args.model,
                    "CHECKPOINT_ROOT": str(train_root),
                    "OUT_DIR": str(secondary_eval_root),
                    "TP": str(args.tensor_parallel_size),
                    "MAX_MODEL_LEN": str(args.eval_max_model_len),
                    "MAX_NEW_TOKENS": str(args.eval_max_new_tokens),
                    "DATASET_JSONLS": " ".join(args.secondary_eval_jsonls),
                },
            )
        )
        steps.append(
            Step(
                "summarize-secondary-eval",
                [args.python, "scripts/summarize_eval_results.py", str(secondary_eval_root), "--tsv", str(secondary_eval_tsv)],
                opsd,
            )
        )
    steps.append(
        Step(
            "claim-gates",
            [
                args.python,
                "scripts/evaluate_pct_claim_gates.py",
                "--spec",
                str(spec),
                "--eval_tsv",
                str(eval_tsv),
                "--seed_summary_tsv",
                str(seed_eval_tsv),
                "--dispersion_gain_tsv",
                str(dispersion_gain_tsv),
                "--diagnostic_summary_tsv",
                str(diagnostic_summary_tsv),
                "--intervention_tsv",
                str(intervention_tsv),
                "--out_tsv",
                str(claim_gates_tsv),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "audit-completion",
            [
                args.python,
                "scripts/audit_pct_completion.py",
                "--manifest",
                str(manifest),
                "--eval_root",
                str(eval_root),
                "--spec",
                str(spec),
                "--suite",
                "short_matrix",
                "--require_eval",
                "--require_train_metrics",
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "paper-report",
            [
                args.python,
                "scripts/make_paper_report.py",
                "--out_dir",
                str(report_root),
                "--train_tsv",
                str(train_root / "train_summary.tsv"),
                "--eval_tsv",
                str(eval_tsv),
                "--seed_summary_tsv",
                str(seed_eval_tsv),
                "--bootstrap_tsv",
                str(bootstrap_tsv),
                "--dispersion_gain_tsv",
                str(dispersion_gain_tsv),
                "--metadata_json",
                str(metadata),
                "--dataset_provenance_json",
                str(train_provenance),
                "--intervention_tsv",
                str(intervention_tsv),
                "--claim_gates_tsv",
                str(claim_gates_tsv),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "paper-figures",
            [
                args.python,
                "scripts/make_pct_figures.py",
                "--out_dir",
                str(report_root / "figures"),
                "--diagnostic_summary_tsv",
                str(diagnostic_summary_tsv),
                "--eval_tsv",
                str(eval_tsv),
                "--dispersion_gain_tsv",
                str(dispersion_gain_tsv),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "artifact-inventory",
            [
                args.python,
                "scripts/inventory_pct_artifacts.py",
                "--root",
                str(out),
                "--json_out",
                str(report_root / "artifact_inventory.json"),
                "--md_out",
                str(report_root / "artifact_inventory.md"),
                "--checksums_out",
                str(checksums_jsonl),
            ],
            opsd,
        )
    )
    steps.append(
        Step(
            "audit-artifacts",
            [
                args.python,
                "scripts/audit_pct_completion.py",
                "--manifest",
                str(manifest),
                "--eval_root",
                str(eval_root),
                "--spec",
                str(spec),
                "--suite",
                "short_matrix",
                "--artifact_root",
                str(out),
            ],
            opsd,
        )
    )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage runner for the PCT experiment pipeline.")
    parser.add_argument("--opsd_root", default="/root/OPSD")
    parser.add_argument("--set_phf_root", default="/root/set-phf")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--model_tag", default="qwen3_4b")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--stages", nargs="+", default=["all"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--metadata_notes", default="pct experiment run")
    parser.add_argument("--use_generated_multiref", action="store_true")
    parser.add_argument("--candidate_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--candidate_examples", type=int, default=0)
    parser.add_argument("--candidate_max_new_tokens", type=int, default=2048)
    parser.add_argument("--candidate_temperature", type=float, default=0.7)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    parser.add_argument("--min_refs", type=int, default=4)
    parser.add_argument("--min_gpus", type=int, default=1)
    parser.add_argument("--require_vllm", action="store_true")
    parser.add_argument("--train_num_samples", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--diagnostic_examples", type=int, default=500)
    parser.add_argument("--intervention_examples", type=int, default=100)
    parser.add_argument("--diagnostic_rollout_tokens", type=int, default=512)
    parser.add_argument("--intervention_rollout_tokens", type=int, default=256)
    parser.add_argument("--train_rollout_tokens", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=20000)
    parser.add_argument("--max_atoms", type=int, default=64)
    parser.add_argument("--max_positions", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--eval_backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--expected_eval_num_problems", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_max_model_len", type=int, default=40960)
    parser.add_argument("--eval_max_new_tokens", type=int, default=38912)
    parser.add_argument("--analysis_dataset", default="aime24")
    parser.add_argument("--analysis_methods", nargs="+", default=["set_uot"])
    parser.add_argument("--dispersion_baseline", default="phf_mean")
    parser.add_argument("--dispersion_metric", default="sinkhorn_correct_correct")
    parser.add_argument("--min_matched_per_dispersion_group", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument(
        "--secondary_eval_jsonls",
        nargs="*",
        default=[],
        help="Optional local eval sets as label=/path.jsonl, for example aimo=/data/aimo.jsonl rrb-aime=/data/rrb.jsonl.",
    )
    args = parser.parse_args()

    all_steps = build_steps(args)
    selected = set(args.stages)
    if "all" not in selected:
        all_steps = [step for step in all_steps if step.name in selected]
    if not all_steps:
        raise SystemExit("No stages selected.")

    for step in all_steps:
        command = add_env(step.env, shell_join(step.cmd))
        print(f"[{step.name}] cd {step.cwd} && {command}")
        if not args.dry_run:
            env = None
            if step.env:
                import os

                env = os.environ.copy()
                env.update(step.env)
            subprocess.check_call(step.cmd, cwd=step.cwd, env=env)


if __name__ == "__main__":
    main()
