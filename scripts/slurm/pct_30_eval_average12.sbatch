#!/usr/bin/env bash
#SBATCH --job-name=pct-eval
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
SECONDARY_EVAL_JSONLS="${SECONDARY_EVAL_JSONLS:-}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EXPECTED_EVAL_NUM_PROBLEMS="${EXPECTED_EVAL_NUM_PROBLEMS:-30}"

EXTRA_ARGS=()
STAGES=(eval-average12 summarize-eval)
if [[ -n "${SECONDARY_EVAL_JSONLS}" ]]; then
  STAGES+=(eval-secondary summarize-secondary-eval)
  EXTRA_ARGS+=(--secondary_eval_jsonls ${SECONDARY_EVAL_JSONLS})
fi

cd "${OPSD_ROOT}"
"${PYTHON_BIN}" scripts/run_pct_pipeline.py \
  --opsd_root "${OPSD_ROOT}" \
  --set_phf_root "${SET_PHF_ROOT}" \
  --python "${PYTHON_BIN}" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --out_root "${OUT_ROOT}" \
  --stages "${STAGES[@]}" \
  --tensor_parallel_size 8 \
  --eval_backend "${EVAL_BACKEND}" \
  --expected_eval_num_problems "${EXPECTED_EVAL_NUM_PROBLEMS}" \
  --eval_max_model_len 40960 \
  --eval_max_new_tokens 38912 \
  "${EXTRA_ARGS[@]}"
