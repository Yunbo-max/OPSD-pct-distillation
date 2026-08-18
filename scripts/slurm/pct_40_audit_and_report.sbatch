#!/usr/bin/env bash
#SBATCH --job-name=pct-report
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_TAG="${MODEL_TAG:-qwen3_4b}"
MAX_STEPS="${MAX_STEPS:-100}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-5000}"
ANALYSIS_DATASET="${ANALYSIS_DATASET:-aime24}"
DISPERSION_BASELINE="${DISPERSION_BASELINE:-phf_mean}"
ANALYSIS_METHODS="${ANALYSIS_METHODS:-set_uot}"
DISPERSION_METRIC="${DISPERSION_METRIC:-sinkhorn_correct_correct}"
MIN_MATCHED_PER_DISPERSION_GROUP="${MIN_MATCHED_PER_DISPERSION_GROUP:-10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}"

if [ "${TRAIN_NUM_SAMPLES}" = "0" ]; then
  SAMPLE_TAG="full"
else
  SAMPLE_TAG="n${TRAIN_NUM_SAMPLES}"
fi

ANALYSIS_EVAL_ARGS=()
for METHOD in ${DISPERSION_BASELINE} ${ANALYSIS_METHODS}; do
  RUN_NAME="${MODEL_TAG}_${METHOD}_steps${MAX_STEPS}_${SAMPLE_TAG}"
  ANALYSIS_EVAL_ARGS+=(--eval "${METHOD}=${OUT_ROOT}/eval/pct_average12/${RUN_NAME}_${ANALYSIS_DATASET}_avg12.json")
done

cd "${OPSD_ROOT}"
"${PYTHON_BIN}" scripts/audit_pct_completion.py \
  --manifest "${OUT_ROOT}/runs/manifest.jsonl" \
  --eval_root "${OUT_ROOT}/eval/pct_average12" \
  --spec "${OPSD_ROOT}/configs/pct_neurips_spec.json" \
  --suite short_matrix \
  --require_eval \
  --require_train_metrics

"${PYTHON_BIN}" scripts/summarize_eval_results.py \
  "${OUT_ROOT}/eval/pct_average12" \
  --tsv "${OUT_ROOT}/eval/pct_average12/eval_summary.tsv" \
  --manifest "${OUT_ROOT}/runs/manifest.jsonl" \
  --strict_protocol \
  --expected_num_problems 30

"${PYTHON_BIN}" scripts/summarize_seed_results.py \
  --manifest "${OUT_ROOT}/runs/manifest.jsonl" \
  --eval_tsv "${OUT_ROOT}/eval/pct_average12/eval_summary.tsv" \
  --out_tsv "${OUT_ROOT}/eval/pct_average12/seed_summary.tsv"

"${PYTHON_BIN}" scripts/bootstrap_eval_matrix.py \
  "${ANALYSIS_EVAL_ARGS[@]}" \
  --baseline "${DISPERSION_BASELINE}" \
  --n_boot "${BOOTSTRAP_SAMPLES}" \
  --seed "${BOOTSTRAP_SEED}" \
  --strict_protocol \
  --expected_num_problems 30 \
  --require_paired_problem_ids \
  --tsv "${OUT_ROOT}/eval/pct_average12/bootstrap_${ANALYSIS_DATASET}.tsv"

"${PYTHON_BIN}" scripts/analyze_dispersion_gains.py \
  --diagnostic_csv "${OUT_ROOT}/diagnostics/ot_diag_500_per_problem.csv" \
  --dispersion_metric "${DISPERSION_METRIC}" \
  "${ANALYSIS_EVAL_ARGS[@]}" \
  --baseline "${DISPERSION_BASELINE}" \
  --strict_protocol \
  --expected_num_problems 30 \
  --min_matched_per_group "${MIN_MATCHED_PER_DISPERSION_GROUP}" \
  --tsv "${OUT_ROOT}/report/dispersion_gain_${ANALYSIS_DATASET}.tsv"

"${PYTHON_BIN}" scripts/evaluate_pct_claim_gates.py \
  --spec "${OPSD_ROOT}/configs/pct_neurips_spec.json" \
  --eval_tsv "${OUT_ROOT}/eval/pct_average12/eval_summary.tsv" \
  --seed_summary_tsv "${OUT_ROOT}/eval/pct_average12/seed_summary.tsv" \
  --dispersion_gain_tsv "${OUT_ROOT}/report/dispersion_gain_${ANALYSIS_DATASET}.tsv" \
  --diagnostic_summary_tsv "${OUT_ROOT}/diagnostics/ot_diag_500_summary.tsv" \
  --intervention_tsv "${OUT_ROOT}/diagnostics/intervention_100_summary.tsv" \
  --out_tsv "${OUT_ROOT}/report/claim_gates.tsv"

"${PYTHON_BIN}" scripts/make_paper_report.py \
  --out_dir "${OUT_ROOT}/report" \
  --train_tsv "${OUT_ROOT}/runs/pct_short_matrix/train_summary.tsv" \
  --eval_tsv "${OUT_ROOT}/eval/pct_average12/eval_summary.tsv" \
  --seed_summary_tsv "${OUT_ROOT}/eval/pct_average12/seed_summary.tsv" \
  --bootstrap_tsv "${OUT_ROOT}/eval/pct_average12/bootstrap_${ANALYSIS_DATASET}.tsv" \
  --dispersion_gain_tsv "${OUT_ROOT}/report/dispersion_gain_${ANALYSIS_DATASET}.tsv" \
  --metadata_json "${OUT_ROOT}/run_metadata.json" \
  --dataset_provenance_json "${OUT_ROOT}/data/train_dataset_provenance.json" \
  --intervention_tsv "${OUT_ROOT}/diagnostics/intervention_100_summary.tsv" \
  --claim_gates_tsv "${OUT_ROOT}/report/claim_gates.tsv"

"${PYTHON_BIN}" scripts/make_pct_figures.py \
  --out_dir "${OUT_ROOT}/report/figures" \
  --diagnostic_summary_tsv "${OUT_ROOT}/diagnostics/ot_diag_500_summary.tsv" \
  --eval_tsv "${OUT_ROOT}/eval/pct_average12/eval_summary.tsv" \
  --dispersion_gain_tsv "${OUT_ROOT}/report/dispersion_gain_${ANALYSIS_DATASET}.tsv"

"${PYTHON_BIN}" scripts/inventory_pct_artifacts.py \
  --root "${OUT_ROOT}" \
  --json_out "${OUT_ROOT}/report/artifact_inventory.json" \
  --md_out "${OUT_ROOT}/report/artifact_inventory.md" \
  --checksums_out "${OUT_ROOT}/report/artifact_checksums.jsonl"

"${PYTHON_BIN}" scripts/audit_pct_completion.py \
  --manifest "${OUT_ROOT}/runs/manifest.jsonl" \
  --eval_root "${OUT_ROOT}/eval/pct_average12" \
  --spec "${OPSD_ROOT}/configs/pct_neurips_spec.json" \
  --suite short_matrix \
  --artifact_root "${OUT_ROOT}"
