#!/usr/bin/env bash
#SBATCH --job-name=pct-train-array
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A-%a.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
MANIFEST="${MANIFEST:-/path/to/pct_experiment/runs/manifest.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${OPSD_ROOT}"
"${PYTHON_BIN}" scripts/run_pct_train_from_manifest.py \
  --manifest "${MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --skip_completed \
  --num_processes "${NUM_PROCESSES:-8}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --grad_accum "${GRAD_ACCUM:-1}" \
  --require_train_metrics
