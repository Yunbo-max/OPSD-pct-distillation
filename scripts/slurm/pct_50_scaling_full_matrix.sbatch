#!/usr/bin/env bash
#SBATCH --job-name=pct-scale
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
DATASET="${DATASET:-/path/to/multiref_opsd.jsonl}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment/runs/pct_scaling_matrix}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS="${MODELS:-qwen3_1p7b=Qwen/Qwen3-1.7B qwen3_4b=Qwen/Qwen3-4B qwen3_8b=Qwen/Qwen3-8B}"
METHODS="${METHODS:-none phf_single phf_mean phf_grassmann phf_set set_fgw set_uot}"
MAX_STEPS="${MAX_STEPS:-1000}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-0}"
SEEDS="${SEEDS:-}"
EVAL_SEED="${EVAL_SEED:-}"

cd "${OPSD_ROOT}"
MANIFEST_CMD=(
  "${PYTHON_BIN}" scripts/make_pct_scaling_manifest.py
  --out "${OUT_ROOT}/manifest.jsonl"
  --dataset "${DATASET}"
  --output_root "${OUT_ROOT}"
  --train_num_samples "${TRAIN_NUM_SAMPLES}"
  --max_steps "${MAX_STEPS}"
  --models ${MODELS}
  --methods ${METHODS}
)
if [ -n "${SEEDS}" ]; then
  MANIFEST_CMD+=(--seeds ${SEEDS})
fi
if [ -n "${EVAL_SEED}" ]; then
  MANIFEST_CMD+=(--eval_seed "${EVAL_SEED}")
fi
"${MANIFEST_CMD[@]}"

DATASET="${DATASET}" \
OUT_ROOT="${OUT_ROOT}" \
MANIFEST="${OUT_ROOT}/manifest.jsonl" \
PYTHON_BIN="${PYTHON_BIN}" \
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}" \
MODELS="${MODELS}" \
METHODS="${METHODS}" \
MAX_STEPS="${MAX_STEPS}" \
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES}" \
SEEDS="${SEEDS}" \
EVAL_SEED="${EVAL_SEED}" \
NUM_PROCESSES="${NUM_PROCESSES:-8}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACCUM="${GRAD_ACCUM:-1}" \
bash scripts/run_pct_scaling_matrix.sh

"${PYTHON_BIN}" scripts/summarize_train_runs.py \
  "${OUT_ROOT}" \
  --tsv "${OUT_ROOT}/train_summary.tsv"
