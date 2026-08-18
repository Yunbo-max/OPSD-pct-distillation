#!/usr/bin/env bash
#SBATCH --job-name=pct-contam-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment/runs/pct_contamination_matrix}"
EVAL_ROOT="${EVAL_ROOT:-/path/to/pct_experiment/eval/pct_contamination_average12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${OPSD_ROOT}"
PYTHON_BIN="${PYTHON_BIN}" \
OUT_ROOT="${OUT_ROOT}" \
EVAL_ROOT="${EVAL_ROOT}" \
BACKEND="${BACKEND:-vllm}" \
TP="${TP:-8}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}" \
bash scripts/run_pct_contamination_eval.sh
