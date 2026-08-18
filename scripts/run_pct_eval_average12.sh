#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/OPSD/runs/pct_short_matrix}"
OUT_DIR="${OUT_DIR:-/root/OPSD/eval_results/pct_average12}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMP="${TEMP:-1.0}"
DATASETS="${DATASETS:-aime24 aime25 hmmt25}"
METHOD_DIRS="${METHOD_DIRS:-qwen3_4b_none_steps100_n5000 qwen3_4b_phf_single_steps100_n5000 qwen3_4b_phf_random_steps100_n5000 qwen3_4b_phf_mean_steps100_n5000 qwen3_4b_phf_medoid_steps100_n5000 qwen3_4b_phf_grassmann_steps100_n5000 qwen3_4b_phf_set_steps100_n5000 qwen3_4b_set_ot_steps100_n5000 qwen3_4b_set_fgw_steps100_n5000 qwen3_4b_set_uot_steps100_n5000}"

mkdir -p "${OUT_DIR}"

for METHOD_DIR in ${METHOD_DIRS}; do
  CHECKPOINT="${CHECKPOINT_ROOT}/${METHOD_DIR}"
  for DATASET in ${DATASETS}; do
    "${PYTHON_BIN}" eval/evaluate_math.py \
      --base_model "${BASE_MODEL}" \
      --checkpoint_dir "${CHECKPOINT}" \
      --dataset "${DATASET}" \
      --val_n 12 \
      --temperature "${TEMP}" \
      --top_p 0.95 \
      --top_k -1 \
      --max_model_len "${MAX_MODEL_LEN}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --tensor_parallel_size "${TP}" \
      --output_file "${OUT_DIR}/${METHOD_DIR}_${DATASET}_avg12.json"
  done
done
