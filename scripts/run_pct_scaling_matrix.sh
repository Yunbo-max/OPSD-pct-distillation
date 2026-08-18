#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-/path/to/multiref_opsd.jsonl}"
OUT_ROOT="${OUT_ROOT:-/root/OPSD/runs/pct_scaling_matrix}"
MANIFEST="${MANIFEST:-${OUT_ROOT}/manifest.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
SPEC="${SPEC:-configs/pct_neurips_spec.json}"
VALIDATE_SPEC="${VALIDATE_SPEC:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PCT_WEIGHT="${PCT_WEIGHT:-0.1}"
PCT_REFS="${PCT_REFS:-4}"
METHODS="${METHODS:-none phf_single phf_mean phf_grassmann phf_set set_fgw set_uot}"
MODELS="${MODELS:-qwen3_1p7b=Qwen/Qwen3-1.7B qwen3_4b=Qwen/Qwen3-4B qwen3_8b=Qwen/Qwen3-8B}"
SEEDS="${SEEDS:-}"
EVAL_SEED="${EVAL_SEED:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -s "${MANIFEST}" ]; then
  MANIFEST_CMD=(
    "${PYTHON_BIN}" scripts/make_pct_scaling_manifest.py
    --out "${MANIFEST}"
    --dataset "${DATASET}"
    --output_root "${OUT_ROOT}"
    --train_num_samples "${TRAIN_NUM_SAMPLES}"
    --max_steps "${MAX_STEPS}"
    --pct_loss_weight "${PCT_WEIGHT}"
    --pct_num_references "${PCT_REFS}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --max_length "${MAX_LENGTH}"
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
fi

if [ "${VALIDATE_SPEC}" = "1" ]; then
  "${PYTHON_BIN}" scripts/validate_pct_experiment_spec.py \
    --spec "${SPEC}" \
    --manifest "${MANIFEST}" \
    --suite scaling_matrix
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" scripts/run_pct_train_from_manifest.py
  --manifest "${MANIFEST}"
  --accelerate "${ACCELERATE_BIN}"
  --num_processes "${NUM_PROCESSES}"
  --max_steps "${MAX_STEPS}"
  --train_num_samples "${TRAIN_NUM_SAMPLES}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --max_length "${MAX_LENGTH}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --pct_loss_weight "${PCT_WEIGHT}"
  --pct_num_references "${PCT_REFS}"
  --skip_completed
  --require_train_metrics
)
if [ "${DRY_RUN}" = "1" ]; then
  TRAIN_CMD+=(--dry_run)
fi
"${TRAIN_CMD[@]}"
