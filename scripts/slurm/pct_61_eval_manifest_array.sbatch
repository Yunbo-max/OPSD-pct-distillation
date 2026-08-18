#!/usr/bin/env bash
#SBATCH --job-name=pct-eval-array
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --array=0-29
#SBATCH --output=slurm-%x-%A-%a.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
MANIFEST="${MANIFEST:-/path/to/pct_experiment/runs/manifest.jsonl}"
EVAL_ROOT="${EVAL_ROOT:-/path/to/pct_experiment/eval/pct_average12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${OPSD_ROOT}"
"${PYTHON_BIN}" scripts/run_pct_eval_from_manifest.py \
  --manifest "${MANIFEST}" \
  --eval_root "${EVAL_ROOT}" \
  --python "${PYTHON_BIN}" \
  --backend vllm \
  --tensor_parallel_size "${TP:-8}" \
  --max_model_len "${MAX_MODEL_LEN:-40960}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-38912}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --skip_completed
