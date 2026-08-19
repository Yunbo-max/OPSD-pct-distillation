#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_CFG="${ACCELERATE_CFG:-${REPO_ROOT}/accelerate.yaml}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
MODEL_TAG="${MODEL_TAG:-qwen3_4b}"
DATASET="${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/pct_short_matrix}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-5000}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PCT_WEIGHT="${PCT_WEIGHT:-0.1}"
PCT_REFS="${PCT_REFS:-4}"
SAMPLE_TAG="${SAMPLE_TAG:-$([ "${TRAIN_NUM_SAMPLES}" = "0" ] && printf full || printf n%s "${TRAIN_NUM_SAMPLES}")}"

METHODS="${METHODS:-none phf_single phf_random phf_mean phf_medoid phf_grassmann phf_set set_ot set_fgw set_uot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HAS_ACCELERATE=0
if [[ -f "${ACCELERATE_CFG}" ]]; then
  HAS_ACCELERATE=1
fi
if [[ "${HAS_ACCELERATE}" -eq 0 ]]; then
  echo "[run_pct_short_matrix] accelerate.yaml not found at ${ACCELERATE_CFG}; using direct python launcher."
fi

for METHOD in ${METHODS}; do
  RUN_NAME="${MODEL_TAG}_${METHOD}_steps${MAX_STEPS}_${SAMPLE_TAG}"
  if [[ "${HAS_ACCELERATE}" -eq 1 ]]; then
    CMD=(accelerate launch --config_file "${ACCELERATE_CFG}" --num_processes "${NUM_PROCESSES}" --gradient_accumulation_steps "${GRAD_ACCUM}")
    CMD+=(opsd_train.py)
  else
    CMD=("${PYTHON_BIN}" opsd_train.py)
  fi
  "${CMD[@]}" \
    --model_name_or_path "${MODEL}" \
    --pct_dataset_name "${DATASET}" \
    --pct_train_num_samples "${TRAIN_NUM_SAMPLES}" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --output_dir "${OUT_ROOT}" \
    --run_config "${RUN_NAME}" \
    --max_steps "${MAX_STEPS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length "${MAX_LENGTH}" \
    --beta 0 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --report_to none \
    --pct_method "${METHOD}" \
    --pct_loss_weight "$([ "${METHOD}" = "none" ] && printf 0 || printf %s "${PCT_WEIGHT}")" \
    --pct_num_references "${PCT_REFS}" \
    --pct_layers last \
    --pct_tau 0.05 \
    --pct_geometry_weight 0.0 \
    --pct_max_atoms 64 \
    --pct_sinkhorn_epsilon 0.05 \
    --pct_sinkhorn_iters 40 \
    --pct_uot_rho 0.5 \
    --pct_fgw_outer 4 \
    --pct_fgw_feature_weight 0.5 \
    --pct_grassmann_rank 2
done
