#!/usr/bin/env bash
#SBATCH --job-name=pct-train5k
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
SET_PHF_ROOT="${SET_PHF_ROOT:-/path/to/set-phf}"
DATASET="${DATASET:-/path/to/multiref_opsd.jsonl}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${OPSD_ROOT}"
"${PYTHON_BIN}" scripts/run_pct_pipeline.py \
  --opsd_root "${OPSD_ROOT}" \
  --set_phf_root "${SET_PHF_ROOT}" \
  --python "${PYTHON_BIN}" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --out_root "${OUT_ROOT}" \
  --stages train-short-matrix summarize-train \
  --num_processes 8 \
  --tensor_parallel_size 8 \
  --batch_size 4 \
  --grad_accum 1
