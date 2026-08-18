#!/usr/bin/env bash
#SBATCH --job-name=pct-contam
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
SOURCE_DATASET="${SOURCE_DATASET:-/path/to/multiref_opsd.jsonl}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment/runs/pct_contamination_matrix}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
MODEL_TAG="${MODEL_TAG:-qwen3_4b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${OPSD_ROOT}"
PYTHON_BIN="${PYTHON_BIN}" \
MODEL="${MODEL}" \
MODEL_TAG="${MODEL_TAG}" \
SOURCE_DATASET="${SOURCE_DATASET}" \
OUT_ROOT="${OUT_ROOT}" \
RHOS="${RHOS:-0 0.25 0.5}" \
METHODS="${METHODS:-phf_mean phf_set set_fgw set_uot}" \
MAX_STEPS="${MAX_STEPS:-100}" \
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-5000}" \
NUM_PROCESSES="${NUM_PROCESSES:-8}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACCUM="${GRAD_ACCUM:-1}" \
bash scripts/run_pct_contamination_matrix.sh
