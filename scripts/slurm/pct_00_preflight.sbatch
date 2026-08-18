#!/usr/bin/env bash
#SBATCH --job-name=pct-preflight
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
DATASET="${DATASET:-/path/to/multiref_opsd.jsonl}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${OPSD_ROOT}"
"${PYTHON_BIN}" scripts/capture_run_metadata.py \
  --out "${OUT_ROOT}/run_metadata.json" \
  --notes "pct preflight"

"${PYTHON_BIN}" scripts/preflight_pct.py \
  --dataset "${DATASET}" \
  --min_gpus 8 \
  --require_vllm \
  --json "${OUT_ROOT}/preflight.json"
