# Privileged Computation Transport in OPSD

This fork keeps OPSD as the training framework and adds PCT as an auxiliary hidden-flow/transport loss.

Default OPSD behavior is unchanged when `--pct_loss_weight 0`.

The machine-readable experiment contract is `configs/pct_neurips_spec.json`. It declares the method matrix, datasets, gates, and artifact groups expected for a full paper-grade run.

## Implemented Methods

- `phf_single`: PHF-style transition direction loss against reference 1.
- `phf_random`: random correct reference baseline.
- `phf_mean`: Euclidean mean hidden-flow baseline.
- `phf_medoid`: medoid correct-reference hidden-flow baseline.
- `phf_grassmann`: Grassmann/subspace ablation; each token/layer student flow is penalized by residual distance to the span of correct-reference flows.
- `phf_set`: soft-min over PHF-style losses.
- `set_ot`: soft-min over balanced Sinkhorn OT distances.
- `set_fgw`: soft-min over approximate fused Gromov-Wasserstein distances, combining atom feature alignment and within-trajectory geometry.
- `set_uot`: soft-min over unbalanced Sinkhorn OT distances.

The OT cost is `1 - cos(d_i^S, d_j^T)`, where `d = normalize(h[t+1] - h[t])`.
The FGW cost adds relational structure from pairwise hidden-flow distances within each trajectory.

## Data

OPSD's original dataset uses `problem` and `solution`.

For multi-reference PCT, each row should provide:

```json
{"problem":"...","solution":"r0","references":["r0","r1","r2","r3"]}
```

If `references` is missing, the collator falls back to the single `solution`.

Generate candidate references from the upstream OPSD data when a real multi-reference file is not available yet:

```bash
python /root/set-phf/scripts/build_multiref_candidates.py \
  --dataset_name siyanzhao/Openthoughts_math_30k_opsd \
  --dataset_split train \
  --model Qwen/Qwen3-4B \
  --n_examples 5000 \
  --out /path/to/multiref_opsd_5k_candidates.jsonl \
  --min_verified_refs 4
```

Audit candidates before using them for training or diagnostics:

```bash
python /root/set-phf/scripts/audit_multiref_candidates.py \
  /path/to/multiref_opsd_5k_candidates.jsonl \
  --min_refs 4 \
  --require_wrong_reference \
  --require_shuffled_reference
```

Validate the real multi-reference file before any training run:

```bash
python scripts/validate_multiref_dataset.py \
  --dataset /path/to/multiref_opsd.jsonl \
  --min_refs 4 \
  --write_subset /path/to/multiref_opsd_5k.jsonl \
  --subset_size 5000
```

Capture an auditable provenance record for both the source data and the exact training subset:

```bash
python scripts/capture_dataset_provenance.py \
  --dataset /path/to/multiref_opsd_5k.jsonl \
  --min_refs 4 \
  --out /path/to/pct_experiment/data/train_dataset_provenance.json
```

The provenance JSON records row counts, schema fields, local file SHA256, canonical row-hash digest, and reference statistics including duplicate, wrong-reference, and shuffled-reference coverage.

## Recommended First Runs

PHF reproduction:

```bash
accelerate launch --config_file accelerate.yaml opsd_train.py \
  --model_name_or_path Qwen/Qwen3-4B \
  --dataset_name siyanzhao/Openthoughts_math_30k_opsd \
  --output_dir runs/qwen3_4b_phf_single \
  --max_steps 100 \
  --use_peft true \
  --fixed_teacher \
  --reason_first false \
  --pct_method phf_single \
  --pct_loss_weight 0.1 \
  --pct_geometry_weight 0.1
```

Set-UOT:

```bash
bash scripts/run_pct_4b_set_uot.sh
```

## Experiment Order

0. Local smoke tests on `Qwen/Qwen3-0.6B`:
   - one-step `phf_single`;
   - one-step `set_uot` with 4 references;
   - one-problem AIME24 eval with `--backend transformers`;
   - eval JSON summarization.
1. Run 500-problem diagnostic with the standalone `/root/set-phf` tooling to choose Euclidean/PHF/OT/UOT/FGW.
   Summarize it with:

```bash
python /root/set-phf/scripts/summarize_ot_diagnostic.py \
  /path/to/ot_diagnostic_500.jsonl \
  --summary_tsv /path/to/ot_diagnostic_500_summary.tsv \
  --per_problem_csv /path/to/ot_diagnostic_500_per_problem.csv
```

2. In OPSD, run the 100-step Qwen3-4B reproduction matrix:
   - OPSD only;
   - `phf_single`;
   - `phf_random`;
   - `phf_mean`;
   - `phf_medoid`;
   - `phf_grassmann`;
   - `phf_set`;
   - `set_ot`;
   - `set_fgw`;
   - `set_uot`.
3. If `set_uot` or `set_fgw` separates high-dispersion cases, run the same matrix on 5k examples.
4. Full 29,434-example train and PHF protocol evaluation:
   - AIME 2024;
   - AIME 2025;
   - HMMT 2025;
   - `Average@12`.
5. Scaling after the Qwen3-4B 5k gate:
   - Qwen3-1.7B;
   - Qwen3-4B;
   - Qwen3-8B;
   - core methods: `none`, `phf_single`, `phf_mean`, `phf_grassmann`, `phf_set`, `set_fgw`, `set_uot`.

## Launchers

Environment setup:

```bash
conda env create -f environment.yml
# or
python -m pip install -r requirements-pct.txt
```

Preflight the target machine:

```bash
python scripts/capture_run_metadata.py \
  --out /path/to/pct_experiment/run_metadata.json \
  --notes "qwen3-4b pct matrix"

python scripts/preflight_pct.py \
  --dataset /path/to/multiref_opsd.jsonl \
  --min_gpus 8 \
  --require_vllm \
  --requirements requirements-pct.txt \
  --strict_versions \
  --json /path/to/pct_experiment/preflight.json
```

Top-level staged pipeline dry-run:

```bash
python scripts/run_pct_pipeline.py \
  --dataset /path/to/multiref_opsd.jsonl \
  --out_root /path/to/pct_experiment \
  --dry_run
```

Top-level pipeline with candidate generation from upstream OPSD:

```bash
python scripts/run_pct_pipeline.py \
  --dataset siyanzhao/Openthoughts_math_30k_opsd \
  --dataset_split train \
  --out_root /path/to/pct_experiment \
  --use_generated_multiref \
  --candidate_examples 5000 \
  --dry_run
```

Run selected stages:

```bash
python scripts/run_pct_pipeline.py \
  --dataset /path/to/multiref_opsd.jsonl \
  --out_root /path/to/pct_experiment \
  --stages validate-data manifest ot-diagnostic summarize-diagnostic bootstrap-eval dispersion-gain
```

Write a manifest for the training/eval matrix:

```bash
python scripts/make_pct_manifest.py \
  --out /path/to/runs/pct_short_matrix/manifest.jsonl \
  --dataset /path/to/multiref_opsd_5k.jsonl \
  --output_root /path/to/runs/pct_short_matrix \
  --train_num_samples 5000 \
  --max_steps 100 \
  --seeds 0 1 2 \
  --eval_seed 123
```

Validate the spec and generated manifest before training:

```bash
python scripts/validate_pct_experiment_spec.py \
  --spec configs/pct_neurips_spec.json \
  --manifest /path/to/runs/pct_short_matrix/manifest.jsonl \
  --suite short_matrix
```

Write a full-data scaling manifest:

```bash
python scripts/make_pct_scaling_manifest.py \
  --out /path/to/pct_experiment/runs/pct_scaling_matrix/manifest.jsonl \
  --dataset /path/to/multiref_opsd.jsonl \
  --output_root /path/to/pct_experiment/runs/pct_scaling_matrix \
  --train_num_samples 0 \
  --max_steps 1000 \
  --seeds 0 1 2 \
  --eval_seed 123
```

Write an external-baseline eval manifest for checkpoints produced outside this PCT launcher:

```bash
python scripts/make_external_baseline_manifest.py \
  --out /path/to/pct_experiment/runs/external_baselines.jsonl \
  --base_model Qwen/Qwen3-4B \
  --baseline base \
  --baseline sft=/path/to/sft_checkpoint \
  --baseline grpo=/path/to/grpo_checkpoint \
  --baseline opsd=/path/to/opsd_checkpoint \
  --baseline oprd=/path/to/oprd_checkpoint \
  --baseline phf=/path/to/phf_checkpoint \
  --eval_seed 123
```

Merge PCT and external-baseline manifests for a single eval/audit table:

```bash
python scripts/merge_manifests.py \
  --out /path/to/pct_experiment/runs/all_methods_manifest.jsonl \
  /path/to/pct_experiment/runs/external_baselines.jsonl \
  /path/to/pct_experiment/runs/pct_short_matrix/manifest.jsonl
```

Short matrix:

```bash
python scripts/run_pct_train_from_manifest.py \
  --manifest /path/to/runs/pct_short_matrix/manifest.jsonl \
  --num_processes 8 \
  --batch_size 4 \
  --grad_accum 1
```

The environment-variable wrapper remains available for direct one-off launches:

```bash
MODEL=Qwen/Qwen3-4B \
DATASET=/path/to/multiref_opsd.jsonl \
OUT_ROOT=/path/to/runs/pct_short_matrix \
NUM_PROCESSES=8 \
MAX_STEPS=100 \
TRAIN_NUM_SAMPLES=5000 \
MAX_COMPLETION_LENGTH=1024 \
MAX_LENGTH=20000 \
bash scripts/run_pct_short_matrix.sh
```

PHF-protocol Average@12 evaluation should use the manifest-driven runner so dataset labels, generation counts, and eval seeds are audit-safe:

```bash
python scripts/run_pct_eval_from_manifest.py \
  --manifest /path/to/pct_experiment/runs/manifest.jsonl \
  --eval_root /path/to/eval_results/pct_average12 \
  --backend vllm \
  --tensor_parallel_size 8 \
  --max_model_len 40960 \
  --max_new_tokens 38912 \
  --expected_num_problems 30 \
  --skip_completed
```

`scripts/run_pct_eval_average12.sh` remains available as a legacy/manual helper for unseeded fixed-directory smoke runs.

Secondary/generalization and robustness evaluation from local JSONL files:

```bash
BASE_MODEL=Qwen/Qwen3-4B \
CHECKPOINT_ROOT=/path/to/runs/pct_short_matrix \
OUT_DIR=/path/to/eval_results/pct_secondary \
TP=8 \
DATASET_JSONLS="aimo=/path/to/aimo.jsonl rrb-aime=/path/to/rrb_aime.jsonl" \
bash scripts/run_pct_eval_secondary.sh
```

The same optional secondary sets can be run through the staged pipeline:

```bash
python scripts/run_pct_pipeline.py \
  --dataset /path/to/multiref_opsd.jsonl \
  --out_root /path/to/pct_experiment \
  --stages eval-secondary summarize-secondary-eval \
  --secondary_eval_jsonls aimo=/path/to/aimo.jsonl rrb-aime=/path/to/rrb_aime.jsonl
```

Summarize evaluation JSON files:

```bash
python scripts/summarize_eval_results.py \
  /path/to/eval_results/pct_average12 \
  --manifest /path/to/runs/pct_short_matrix/manifest.jsonl \
  --strict_protocol \
  --expected_num_problems 30
```

Write the same table for the paper report:

```bash
python scripts/summarize_eval_results.py \
  /path/to/eval_results/pct_average12 \
  --tsv /path/to/eval_results/pct_average12/eval_summary.tsv \
  --manifest /path/to/runs/pct_short_matrix/manifest.jsonl \
  --strict_protocol \
  --expected_num_problems 30

python scripts/summarize_seed_results.py \
  --manifest /path/to/runs/pct_short_matrix/manifest.jsonl \
  --eval_tsv /path/to/eval_results/pct_average12/eval_summary.tsv \
  --out_tsv /path/to/eval_results/pct_average12/seed_summary.tsv
```

Bootstrap Average@12 confidence intervals and paired method deltas:

```bash
python scripts/bootstrap_eval_matrix.py \
  --eval mean=/path/to/qwen3_4b_phf_mean_aime24_avg12.json \
  --eval set_uot=/path/to/qwen3_4b_set_uot_aime24_avg12.json \
  --baseline mean \
  --n_boot 10000 \
  --strict_protocol \
  --expected_num_problems 30 \
  --require_paired_problem_ids \
  --tsv /path/to/eval_results/pct_average12/bootstrap_aime24.tsv
```

Join diagnostic dispersion with per-problem evals:

```bash
python scripts/analyze_dispersion_gains.py \
  --diagnostic_csv /path/to/ot_diagnostic_500_per_problem.csv \
  --dispersion_metric sinkhorn_correct_correct \
  --eval mean=/path/to/qwen3_4b_phf_mean_aime24_avg12.json \
  --eval set_uot=/path/to/qwen3_4b_set_uot_aime24_avg12.json \
  --baseline mean \
  --strict_protocol \
  --expected_num_problems 30 \
  --min_matched_per_group 10 \
  --tsv /path/to/dispersion_gain_aime24.tsv
```

Build contamination robustness datasets:

```bash
python scripts/build_contaminated_multiref.py \
  --input /path/to/multiref_opsd.jsonl \
  --output /path/to/multiref_opsd_rho025_mixed.jsonl \
  --rho 0.25 \
  --mode mixed \
  --seed 0
```

Run the contamination robustness matrix for `rho in {0,0.25,0.5}`:

```bash
SOURCE_DATASET=/path/to/multiref_opsd.jsonl \
OUT_ROOT=/path/to/pct_experiment/runs/pct_contamination_matrix \
RHOS="0 0.25 0.5" \
METHODS="phf_mean phf_set set_fgw set_uot" \
bash scripts/run_pct_contamination_matrix.sh
```

Evaluate the contamination robustness matrix:

```bash
OUT_ROOT=/path/to/pct_experiment/runs/pct_contamination_matrix \
EVAL_ROOT=/path/to/pct_experiment/eval/pct_contamination_average12 \
bash scripts/run_pct_contamination_eval.sh
```

Summarize training logs:

```bash
python scripts/summarize_train_runs.py \
  /path/to/runs/pct_short_matrix \
  --tsv /path/to/runs/pct_short_matrix/train_summary.tsv
```

Run the full-data Qwen3-1.7B/4B/8B scaling matrix:

```bash
DATASET=/path/to/multiref_opsd.jsonl \
OUT_ROOT=/path/to/pct_experiment/runs/pct_scaling_matrix \
TRAIN_NUM_SAMPLES=0 \
MAX_STEPS=1000 \
SEEDS="0 1 2" \
EVAL_SEED=123 \
bash scripts/run_pct_scaling_matrix.sh
```

`run_pct_scaling_matrix.sh` writes or reuses `manifest.jsonl`, validates it against the `scaling_matrix` suite in `configs/pct_neurips_spec.json`, then trains through `scripts/run_pct_train_from_manifest.py` with strict skip/resume checks.

Evaluate any manifest-generated matrix:

```bash
python scripts/run_pct_eval_from_manifest.py \
  --manifest /path/to/pct_experiment/runs/pct_scaling_matrix/manifest.jsonl \
  --eval_root /path/to/pct_experiment/eval/pct_scaling_average12 \
  --backend vllm \
  --tensor_parallel_size 8 \
  --expected_num_problems 30 \
  --skip_completed
```

Validate finished eval JSONs against the exact Average@12 protocol:

```bash
python scripts/validate_eval_results.py \
  /path/to/pct_experiment/eval/pct_average12/qwen3_4b_set_uot_steps100_n5000_aime24_avg12.json \
  --expected_dataset aime24 \
  --expected_val_n 12 \
  --expected_num_problems 30
```

Evaluate paper-claim gates from produced artifacts:

```bash
python scripts/evaluate_pct_claim_gates.py \
  --spec configs/pct_neurips_spec.json \
  --eval_tsv /path/to/eval_results/pct_average12/eval_summary.tsv \
  --seed_summary_tsv /path/to/eval_results/pct_average12/seed_summary.tsv \
  --diagnostic_summary_tsv /path/to/pct_experiment/diagnostics/ot_diag_500_summary.tsv \
  --dispersion_gain_tsv /path/to/dispersion_gain_aime24.tsv \
  --intervention_tsv /path/to/pct_experiment/diagnostics/intervention_100_summary.tsv \
  --robustness_tsv /path/to/robustness_summary.tsv \
  --out_tsv /path/to/pct_experiment/report/claim_gates.tsv
```

Build paper-ready Markdown and LaTeX tables:

```bash
python scripts/make_paper_report.py \
  --out_dir /path/to/pct_experiment/report \
  --train_tsv /path/to/runs/pct_short_matrix/train_summary.tsv \
  --eval_tsv /path/to/eval_results/pct_average12/eval_summary.tsv \
  --seed_summary_tsv /path/to/eval_results/pct_average12/seed_summary.tsv \
  --metadata_json /path/to/pct_experiment/run_metadata.json \
  --dataset_provenance_json /path/to/pct_experiment/data/train_dataset_provenance.json \
  --bootstrap_tsv /path/to/eval_results/pct_average12/bootstrap_aime24.tsv \
  --dispersion_gain_tsv /path/to/dispersion_gain_aime24.tsv \
  --intervention_tsv /path/to/intervention_summary.tsv \
  --claim_gates_tsv /path/to/pct_experiment/report/claim_gates.tsv
```

Build dependency-free SVG figures from the same artifacts:

```bash
python scripts/make_pct_figures.py \
  --out_dir /path/to/pct_experiment/report/figures \
  --diagnostic_summary_tsv /path/to/pct_experiment/diagnostics/ot_diag_500_summary.tsv \
  --eval_tsv /path/to/eval_results/pct_average12/eval_summary.tsv \
  --dispersion_gain_tsv /path/to/dispersion_gain_aime24.tsv
```

Inventory artifacts before packaging results:

```bash
python scripts/inventory_pct_artifacts.py \
  --root /path/to/pct_experiment \
  --json_out /path/to/pct_experiment/report/artifact_inventory.json \
  --md_out /path/to/pct_experiment/report/artifact_inventory.md \
  --checksums_out /path/to/pct_experiment/report/artifact_checksums.jsonl \
  --require_complete
```

Audit expected train/eval artifacts after a run:

```bash
python scripts/audit_pct_completion.py \
  --manifest /path/to/pct_experiment/runs/manifest.jsonl \
  --eval_root /path/to/pct_experiment/eval/pct_average12 \
  --spec configs/pct_neurips_spec.json \
  --suite short_matrix \
  --require_eval \
  --require_train_metrics
```

For a final packaging audit, also require the artifact groups declared by the experiment contract:

```bash
python scripts/audit_pct_completion.py \
  --manifest /path/to/pct_experiment/runs/manifest.jsonl \
  --eval_root /path/to/pct_experiment/eval/pct_average12 \
  --spec configs/pct_neurips_spec.json \
  --suite short_matrix \
  --artifact_root /path/to/pct_experiment
```

SLURM handoff templates:

```bash
export OPSD_ROOT=/path/to/OPSD
export SET_PHF_ROOT=/path/to/set-phf
export DATASET=/path/to/multiref_opsd.jsonl
export OUT_ROOT=/path/to/pct_experiment
export MODEL=Qwen/Qwen3-4B

bash scripts/slurm/submit_pct_short_pipeline.sh --dry-run
bash scripts/slurm/submit_pct_short_pipeline.sh

# Three-seed short matrix when the manifest does not exist before submission:
TRAIN_ARRAY=0-29 EVAL_ARRAY=0-89 bash scripts/slurm/submit_pct_short_pipeline.sh --dry-run
TRAIN_ARRAY=0-29 EVAL_ARRAY=0-89 bash scripts/slurm/submit_pct_short_pipeline.sh

python scripts/count_pct_manifest_tasks.py \
  --manifest /path/to/pct_experiment/runs/manifest.jsonl \
  --task train \
  --print_array

python scripts/count_pct_manifest_tasks.py \
  --manifest /path/to/pct_experiment/runs/manifest.jsonl \
  --task eval \
  --print_array

python scripts/summarize_pct_manifest_status.py \
  --manifest /path/to/pct_experiment/runs/manifest.jsonl \
  --eval_root /path/to/pct_experiment/eval/pct_average12 \
  --require_train_metrics \
  --only_incomplete \
  --tsv /path/to/pct_experiment/runs/manifest_status.tsv

python scripts/make_pct_resume_plan.py \
  --status_tsv /path/to/pct_experiment/runs/manifest_status.tsv \
  --manifest /path/to/pct_experiment/runs/manifest.jsonl \
  --eval_root /path/to/pct_experiment/eval/pct_average12 \
  --json_out /path/to/pct_experiment/runs/resume_plan.json \
  --md_out /path/to/pct_experiment/runs/resume_plan.md

# Eval completion is strict: dataset label, Average@N protocol, generation count,
# problem count, and manifest-declared eval seed must match before skip/resume
# tools treat an eval artifact as complete.

sbatch scripts/slurm/pct_00_preflight.sbatch
sbatch scripts/slurm/pct_10_data_and_diagnostics.sbatch
sbatch scripts/slurm/pct_20_train_short_matrix.sbatch
sbatch --array=0-9 scripts/slurm/pct_21_train_manifest_array.sbatch
sbatch scripts/slurm/pct_30_eval_average12.sbatch
sbatch scripts/slurm/pct_40_audit_and_report.sbatch
sbatch scripts/slurm/pct_50_scaling_full_matrix.sbatch
sbatch scripts/slurm/pct_60_scaling_eval_from_manifest.sbatch
sbatch --array=0-29 scripts/slurm/pct_61_eval_manifest_array.sbatch
sbatch scripts/slurm/pct_70_contamination_matrix.sbatch
sbatch scripts/slurm/pct_80_contamination_eval.sbatch
```

If `DATASET` is the upstream single-reference OPSD dataset instead of a prepared multi-reference JSONL, generate and audit candidates in the diagnostic job:

```bash
export DATASET=siyanzhao/Openthoughts_math_30k_opsd
export USE_GENERATED_MULTIREF=1
export CANDIDATE_EXAMPLES=5000
export CANDIDATE_MODEL=Qwen/Qwen3-4B
sbatch scripts/slurm/pct_10_data_and_diagnostics.sbatch
```

If secondary local eval sets are available:

```bash
export SECONDARY_EVAL_JSONLS="aimo=/path/to/aimo.jsonl rrb-aime=/path/to/rrb_aime.jsonl"
sbatch scripts/slurm/pct_30_eval_average12.sbatch
```

Summarize robustness by deterministic perturbation type:

```bash
python scripts/summarize_robustness_results.py \
  --eval set_fgw=/path/to/qwen3_4b_set_fgw_rrb-aime_avg12.json \
  --eval set_uot=/path/to/qwen3_4b_set_uot_rrb-aime_avg12.json \
  --tsv /path/to/pct_experiment/eval/pct_secondary/robustness_by_perturbation.tsv
```

Local dependency-light eval smoke test:

```bash
python eval/evaluate_math.py \
  --backend transformers \
  --base_model Qwen/Qwen3-0.6B \
  --dataset aime24 \
  --num_samples 1 \
  --val_n 1 \
  --seed 123 \
  --max_new_tokens 8 \
  --output_file eval_results/smoke_aime24_qwen3_06b.json
```

Seeded vLLM evaluation requires a vLLM `SamplingParams` version with seed support; the evaluator fails explicitly if `--seed` is requested but unsupported.

## Compute Notes

The local validation host has one RTX A4000 with 16 GB VRAM. It is suitable for smoke
tests and small diagnostics, not final Qwen3-4B/8B Average@12 training/evaluation.
The full matrix should run on a multi-GPU node with vLLM installed.

Paper naming: call the method Privileged Computation Transport (PCT) or Multi-Reference Privileged Transport (MPT), not PHF-OT.
