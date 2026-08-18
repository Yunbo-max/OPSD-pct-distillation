#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B}
DATASET=${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}
OUT=${OUT:-/root/OPSD/runs/qwen3_4b_pct_set_uot}

accelerate launch --config_file accelerate.yaml opsd_train.py \
  --model_name_or_path "$MODEL" \
  --dataset_name "$DATASET" \
  --output_dir "$OUT" \
  --run_config qwen3_4b_pct_set_uot \
  --max_completion_length 1024 \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --max_steps 100 \
  --use_peft true \
  --fixed_teacher \
  --reason_first false \
  --student_thinking false \
  --teacher_thinking true \
  --pct_method set_uot \
  --pct_loss_weight 0.1 \
  --pct_num_references 4 \
  --pct_layers last \
  --pct_tau 0.05 \
  --pct_max_atoms 64 \
  --pct_sinkhorn_epsilon 0.05 \
  --pct_sinkhorn_iters 40 \
  --pct_uot_rho 0.5

