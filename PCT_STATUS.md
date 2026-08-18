# PCT Experiment Status

Date: 2026-08-18

## Local Environment

- GPU: 1x NVIDIA RTX A4000, 16 GB VRAM.
- Suitable for: code validation, tiny smoke training, tiny eval smoke.
- Not suitable for: final Qwen3-4B/8B PHF-protocol training or Average@12 evaluation.
- Installed local missing deps: `datasets`, `accelerate`, `trl`, `peft`, `wandb`, `math-verify`, `sentencepiece`, `tiktoken`.
- `vllm` is not installed locally; full LoRA Average@12 eval launcher expects it on the target GPU node.
- Reproducible environment pins are in `environment.yml` and `requirements-pct.txt`.

## Completed Validation

Training smoke, direct Transformers generation, `Qwen/Qwen3-0.6B`, 1 sample, 1 step:

- `none`: passed.
- `phf_single`: passed and logged `pct_loss`.
- `phf_random`: implemented in the full matrix; covered by unit tests.
- `phf_mean`: passed and logged `pct_loss`.
- `phf_medoid`: passed one-step trainer smoke and logged `pct_medoid_index`.
- `phf_grassmann`: passed one-step trainer smoke and logged `pct_loss=0.218501`; included in short/scaling manifests.
- `phf_set`: passed and logged `pct_loss`.
- `set_ot`: passed and logged `pct_transport_mass` near 1.0.
- `set_fgw`: passed one-step trainer smoke; logged `pct_loss=0.0614352` and balanced `pct_transport_mass=1.0`.
- `set_uot`: passed and logged relaxed `pct_transport_mass` below 1.0.

Eval smoke:

- `eval/evaluate_math.py --backend transformers`
- Dataset: AIME 2024
- Samples: 1 problem, `val_n=1`, `max_new_tokens=8`
- Output: `eval_results/smoke_aime24_qwen3_06b.json`

Verification:

- OPSD syntax compile passed.
- `/root/set-phf` tests passed.
- OPSD multi-reference validator tests passed: `3 passed`.
- OPSD PCT loss/unit tests passed.
- OPSD method matrix now includes `phf_random`, `phf_medoid`, `phf_grassmann`, and `set_fgw`.
- Eval summarizer produced a tabular row from the smoke JSON.
- Train-run summarizer produced tables from local smoke runs.
- Manifest generator wrote the short-matrix run manifest.
- Dispersion-gain analyzer smoke joined diagnostic CSV and eval JSON fixtures.
- Contamination builder smoke wrote a deterministic rho=1.0 wrong-reference dataset.
- Bootstrap eval analyzer smoke produced Average@N confidence intervals and paired deltas.
- Top-level pipeline dry-run emitted staged commands for data validation, manifest creation, and diagnostics.
- Paper report builder smoke produced Markdown and LaTeX tables from train/bootstrap/dispersion TSVs.
- Preflight checker reported local deps/GPU and confirmed `vllm` is absent here.
- Preflight checker can compare installed package versions against `requirements-pct.txt` with `--strict_versions`.
- Local preflight smoke passed with core packages, `min_gpus=0`, and version reporting from `requirements-pct.txt`.
- Completion auditor tests passed for complete, missing-eval, and incomplete-metric manifests.
- Completion auditor now enforces the full default method matrix, AIME24/AIME25/HMMT25 eval presence, `num_problems=30`, numeric Average@12 metrics, PCT train metrics, and transport-mass metrics for OT/FGW/UOT.
- Experiment spec validator passed against `configs/pct_neurips_spec.json` and a generated short-matrix manifest.
- SLURM templates passed shell syntax checks and tests verify canonical script references.
- Eval summarizer now writes `eval_summary.tsv`; the top-level pipeline and final SLURM report job generate both audit output and paper-report inputs.
- Multi-reference validator smoke:
  - valid synthetic JSONL passed and wrote a 1-row subset;
  - invalid duplicated/too-short reference JSONL failed as expected.
- OT diagnostic summarizer smoke produced AUROC and per-problem CSV from a fixture.
- Causal intervention helpers and CLI compile/help passed in `/root/set-phf`.
- Causal intervention summarizer smoke passed on a fixture.
- OPSD tests passed: `55 passed, 1 warning`.
- `/root/set-phf` tests passed: `15 passed, 1 warning`.
- Strict completion-audit negative path was run against a fresh manifest and correctly failed with missing train/eval artifacts.
- Multi-reference candidate builder CLI compile/help passed in `/root/set-phf`.
- Multi-reference candidate audit now requires unique verified-correct references and can require incorrect `wrong_reference` plus present `shuffled_reference`.
- Multi-reference candidate audit passed a good fixture and failed bad duplicate/too-short and missing diagnostic-reference fixtures.
- OPSD pipeline now has an optional `--use_generated_multiref` path that generates candidates from upstream OPSD, audits them, then feeds the generated JSONL into OPSD validation/subsetting.
- SLURM diagnostic handoff supports `USE_GENERATED_MULTIREF=1`, `CANDIDATE_EXAMPLES`, and `CANDIDATE_MODEL`.
- Eval script supports local JSON/JSONL datasets via `--dataset_jsonl`, with labels such as `aimo` and `rrb-aime` for secondary/generalization and robustness runs.
- Strict eval-result validator checks dataset label, `val_n`, 30-problem count, per-problem generation counts, per-problem correctness ranges, unique problem IDs, and recomputed summary metrics.
- Pipeline and eval SLURM handoff support optional secondary local eval sets through `--secondary_eval_jsonls` / `SECONDARY_EVAL_JSONLS`.
- Training run naming is model-tagged, so the same scripts can produce Qwen3-1.7B/4B/8B artifacts without hardcoded `qwen3_4b` names.
- Manifest-driven train launcher executes exactly the JSONL run records used later by eval and audit, including PCT hyperparameters.
- Manifest-driven train launcher dry-run is covered by tests, including method hyperparameters and skipping `requires_training=false` baseline rows.
- Manifest train/eval runners support job-array `--index`, coarse sharding, and strict `--skip_completed` restart behavior.
- Manifest task counter reports train/eval task counts and array bounds from the actual JSONL manifest.
- Manifest status summarizer reports complete/pending train and eval task indexes for restart planning.
- Eval skip/resume checks now use the strict eval protocol, including dataset label and manifest-declared seed.
- Resume-plan generator writes pending train/eval indexes, compact SLURM array ranges, and restart commands from `manifest_status.tsv`.
- Machine-readable NeurIPS-level experiment spec records the method matrix, primary/secondary evals, gates, and expected artifact groups.
- Experiment spec validator checks the spec and optional manifest coverage for short/full/scaling suites.
- Short and scaling manifests can expand across explicit seeds, with seed-tagged run names and seed forwarding into training.
- Evaluation generation accepts and records an optional seed; short, scaling, and external-baseline manifest eval records can pin an eval seed separately from training seed.
- Seed-result summarizer aggregates Average@N by method and dataset across manifest seeds.
- Claim-gate evaluator can use seed-aggregated Average@N summaries instead of filename-based per-run rows.
- Standard pipeline/report jobs now build strict bootstrap CI and dispersion-gain TSVs before claim gates and paper tables.
- Completion audit can derive suite-specific required methods, datasets, problem counts, and artifact groups from `configs/pct_neurips_spec.json`.
- Full-data scaling manifest and launcher support Qwen3-1.7B/4B/8B core-method runs, with manifest-driven training, strict skip/resume checks, and manifest-driven Average@12 evaluation.
- Dependency-aware SLURM submitter wires the short-matrix run as preflight -> diagnostics/manifest -> train array -> eval array -> report/audit, with manifest-derived or explicit array ranges for seeded runs.
- OT diagnostic summarizer writes both summary AUROC TSV and per-problem dispersion CSV.
- Dependency-free SVG figure builder generates diagnostic AUROC, eval Average@N, and dispersion-gain figures from TSV artifacts.
- External baseline manifest support covers base-model eval plus existing SFT/GRPO/OPSD/OPRD/PHF checkpoints, and manifests can be merged with duplicate-run protection.
- Run metadata capture writes git commit/dirty state, package versions, GPU inventory, selected environment variables, manifest path, and notes; pipeline and report generation consume it.
- Dataset provenance capture writes source/subset row counts, schema fields, local file SHA256, canonical row-hash digests, and multi-reference coverage statistics.
- Contamination robustness train/eval launchers cover `rho in {0,0.25,0.5}` for Mean/Set/FGW/UOT-style comparisons, with manifest generation and merged eval support.
- Eval results preserve lightweight local-dataset metadata such as perturbation type, and robustness summarizer reports Average@N by perturbation with delta versus clean.
- Claim-gate evaluator reports pass/fail/missing for Average@12 improvement, high-dispersion gain, diagnostic separation, causal intervention, and contamination robustness.
- Causal intervention summarizer now writes TSV, and the pipeline passes it into the paper report.
- Paper report includes dataset provenance for the exact training subset used by the run.
- Completion audit can enforce final artifact groups, including dataset provenance and figures, after report/inventory generation.
- Artifact inventory script records metadata, manifests, train states, eval JSON, tables, reports, and figures; pipeline and report SLURM produce JSON/Markdown inventories plus a SHA256 checksum manifest.

## Ready Launchers

- Local smoke matrix: `scripts/run_pct_local_smoke_matrix.sh`
- Manifest-driven training matrix: `scripts/run_pct_train_from_manifest.py`
- Short 5k/100-step training matrix: `scripts/run_pct_short_matrix.sh`
- Experiment spec: `configs/pct_neurips_spec.json`
- Pip requirements: `requirements-pct.txt`
- Experiment spec validator: `scripts/validate_pct_experiment_spec.py`
- Seed-result summarizer: `scripts/summarize_seed_results.py`
- Claim-gate evaluator: `scripts/evaluate_pct_claim_gates.py`
- Manifest task counter: `scripts/count_pct_manifest_tasks.py`
- Manifest status/resume table: `scripts/summarize_pct_manifest_status.py`
- Resume-plan generator: `scripts/make_pct_resume_plan.py`
- Full-data/scaling training matrix: `scripts/run_pct_scaling_matrix.sh`
- Average@12 AIME24/AIME25/HMMT25 eval: `scripts/run_pct_eval_from_manifest.py`
- Legacy fixed-directory eval helper: `scripts/run_pct_eval_average12.sh`
- Dependency-aware short-matrix SLURM submitter: `scripts/slurm/submit_pct_short_pipeline.sh`
- Manifest-driven Average@12 eval: `scripts/run_pct_eval_from_manifest.py`
- Strict eval-result validator: `scripts/validate_eval_results.py`
- Secondary/robustness local JSONL eval: `scripts/run_pct_eval_secondary.sh`
- Robustness perturbation summarizer: `scripts/summarize_robustness_results.py`
- Eval JSON summarizer: `scripts/summarize_eval_results.py`
- Bootstrap eval matrix analyzer: `scripts/bootstrap_eval_matrix.py`
- Top-level staged runner: `scripts/run_pct_pipeline.py`
- Preflight checker: `scripts/preflight_pct.py`
- Run metadata capture: `scripts/capture_run_metadata.py`
- Dataset provenance capture: `scripts/capture_dataset_provenance.py`
- Artifact inventory: `scripts/inventory_pct_artifacts.py`
- Completion auditor: `scripts/audit_pct_completion.py`
- Paper report builder: `scripts/make_paper_report.py`
- SVG figure builder: `scripts/make_pct_figures.py`
- SLURM handoff templates: `scripts/slurm/pct_*.sbatch`
- SLURM manifest train/eval array templates: `scripts/slurm/pct_21_train_manifest_array.sbatch`, `scripts/slurm/pct_61_eval_manifest_array.sbatch`
- Training log summarizer: `scripts/summarize_train_runs.py`
- Experiment manifest writer: `scripts/make_pct_manifest.py`
- External baseline manifest writer: `scripts/make_external_baseline_manifest.py`
- Manifest merger: `scripts/merge_manifests.py`
- Scaling manifest writer: `scripts/make_pct_scaling_manifest.py`
- Dispersion/eval gain analyzer: `scripts/analyze_dispersion_gains.py`
- Contamination robustness builder: `scripts/build_contaminated_multiref.py`
- Contamination robustness matrix: `scripts/run_pct_contamination_matrix.sh`
- Contamination robustness eval: `scripts/run_pct_contamination_eval.sh`
- Multi-reference dataset validator/subsetter: `scripts/validate_multiref_dataset.py`
- OT diagnostic summarizer: `/root/set-phf/scripts/summarize_ot_diagnostic.py`
- Causal intervention diagnostic: `/root/set-phf/scripts/run_causal_intervention.py`
- Causal intervention summarizer: `/root/set-phf/scripts/summarize_causal_intervention.py`
- Candidate multi-reference generator: `/root/set-phf/scripts/build_multiref_candidates.py`
- Candidate multi-reference audit: `/root/set-phf/scripts/audit_multiref_candidates.py`

## Required Before Full Run

1. Build or provide a real multi-reference OPSD JSONL with:

```json
{"problem":"...","solution":"r0","references":["r0","r1","r2","r3"]}
```

2. If a prepared JSONL is unavailable, run the pipeline with `--use_generated_multiref` or set `USE_GENERATED_MULTIREF=1` in the diagnostic SLURM job.
3. Run the short matrix on a multi-GPU node with Qwen3-4B.
4. Install `vllm` on the evaluation node.
5. Evaluate checkpoints with `val_n=12` on AIME24, AIME25, HMMT25.
6. If available, run secondary `aimo` and robustness `rrb-aime` local JSONL evals with the optional secondary eval launcher.
7. Run contamination robustness matrix for `rho in {0,0.25,0.5}` if wrong/shuffled diagnostic controls show meaningful separation.
8. Only expand to full 29,434 examples and Qwen3-1.7B/8B after the 5k matrix shows Set/PCT gains on high-dispersion examples.
