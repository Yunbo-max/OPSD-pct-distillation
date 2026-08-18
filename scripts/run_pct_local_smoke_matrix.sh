#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-/root/OPSD/runs/local_smoke_matrix}"
MAX_STEPS="${MAX_STEPS:-1}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8}"
METHODS="${METHODS:-none phf_single phf_random phf_mean phf_medoid phf_grassmann phf_set set_ot set_fgw set_uot}"

for METHOD in ${METHODS}; do
  "${PYTHON_BIN}" opsd_train.py \
    --model_name_or_path "${MODEL}" \
    --trust_remote_code true \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir "${OUT_ROOT}" \
    --run_config "${MODEL_TAG}_smoke_${METHOD}" \
    --max_steps "${MAX_STEPS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps 1 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --torch_dtype float16 \
    --max_length 2048 \
    --beta 0 \
    --use_peft true \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher true \
    --jsd_token_clip 0.05 \
    --report_to none \
    --pct_method "${METHOD}" \
    --pct_loss_weight "$([ "${METHOD}" = "none" ] && printf 0 || printf 0.1)" \
    --pct_num_references 4 \
    --pct_layers last \
    --pct_train_num_samples "${TRAIN_NUM_SAMPLES}" \
    --pct_fgw_outer 2 \
    --pct_fgw_feature_weight 0.5 \
    --pct_grassmann_rank 2
done
