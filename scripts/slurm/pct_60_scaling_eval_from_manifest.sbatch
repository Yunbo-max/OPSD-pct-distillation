#!/usr/bin/env bash
#SBATCH --job-name=pct-scale-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
RUN_ROOT="${RUN_ROOT:-/path/to/pct_experiment/runs/pct_scaling_matrix}"
EVAL_ROOT="${EVAL_ROOT:-/path/to/pct_experiment/eval/pct_scaling_average12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPECTED_EVAL_NUM_PROBLEMS="${EXPECTED_EVAL_NUM_PROBLEMS:-30}"

cd "${OPSD_ROOT}"
"${PYTHON_BIN}" scripts/run_pct_eval_from_manifest.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --eval_root "${EVAL_ROOT}" \
  --python "${PYTHON_BIN}" \
  --backend vllm \
  --tensor_parallel_size "${TP:-8}" \
  --max_model_len "${MAX_MODEL_LEN:-40960}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-38912}" \
  --expected_num_problems "${EXPECTED_EVAL_NUM_PROBLEMS}" \
  --skip_completed

"${PYTHON_BIN}" scripts/summarize_eval_results.py \
  "${EVAL_ROOT}" \
  --tsv "${EVAL_ROOT}/eval_summary.tsv" \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --strict_protocol \
  --expected_num_problems "${EXPECTED_EVAL_NUM_PROBLEMS}"

"${PYTHON_BIN}" scripts/summarize_seed_results.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --eval_tsv "${EVAL_ROOT}/eval_summary.tsv" \
  --out_tsv "${EVAL_ROOT}/seed_summary.tsv"
