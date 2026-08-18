#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
MODEL_TAG="${MODEL_TAG:-qwen3_4b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DATASET="${SOURCE_DATASET:?Set SOURCE_DATASET to a clean multi-reference JSONL}"
OUT_ROOT="${OUT_ROOT:-/root/OPSD/runs/pct_contamination_matrix}"
METHODS="${METHODS:-phf_mean phf_set set_fgw set_uot}"
RHOS="${RHOS:-0 0.25 0.5}"
MODE="${MODE:-mixed}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-100}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-5000}"
RUN_TRAIN="${RUN_TRAIN:-1}"

mkdir -p "${OUT_ROOT}/data" "${OUT_ROOT}/manifests" "${OUT_ROOT}/runs"

for RHO in ${RHOS}; do
  RHO_TAG="rho${RHO//./p}"
  DATASET="${OUT_ROOT}/data/multiref_${RHO_TAG}_${MODE}.jsonl"
  if [[ "${RHO}" == "0" || "${RHO}" == "0.0" || "${RHO}" == "0.00" ]]; then
    cp "${SOURCE_DATASET}" "${DATASET}"
  else
    "${PYTHON_BIN}" scripts/build_contaminated_multiref.py \
      --input "${SOURCE_DATASET}" \
      --output "${DATASET}" \
      --rho "${RHO}" \
      --mode "${MODE}" \
      --seed "${SEED}"
  fi

  RUN_ROOT="${OUT_ROOT}/runs/${RHO_TAG}_${MODE}"
  "${PYTHON_BIN}" scripts/make_pct_manifest.py \
    --out "${OUT_ROOT}/manifests/${RHO_TAG}_${MODE}.jsonl" \
    --model "${MODEL}" \
    --model_tag "${MODEL_TAG}_${RHO_TAG}_${MODE}" \
    --dataset "${DATASET}" \
    --output_root "${RUN_ROOT}" \
    --methods ${METHODS} \
    --train_num_samples "${TRAIN_NUM_SAMPLES}" \
    --max_steps "${MAX_STEPS}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    MODEL="${MODEL}" \
    MODEL_TAG="${MODEL_TAG}_${RHO_TAG}_${MODE}" \
    DATASET="${DATASET}" \
    OUT_ROOT="${RUN_ROOT}" \
    METHODS="${METHODS}" \
    MAX_STEPS="${MAX_STEPS}" \
    TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES}" \
    bash scripts/run_pct_short_matrix.sh
  fi
done
