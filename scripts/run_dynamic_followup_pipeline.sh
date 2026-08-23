#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:-$(pwd)}"
shift || true
worker_pids=("$@")
cd "$repo_dir"

for worker_pid in "${worker_pids[@]}"; do
  while kill -0 "$worker_pid" 2>/dev/null; do
    sleep 30
  done
done

# The early matrix used an ordering-dependent random seed. Recompute only the
# random controls with stable per-arm seeds and let these records win at merge.
python3 scripts/run_online_matrix_stream.py \
  artifacts/privileged_effect/online_high_te_24.jsonl \
  --out artifacts/privileged_effect/online_id288_random_deterministic_shard0.jsonl \
  --id 288 --layers 13,21,23,25 --alphas 0.25,0.5,0.75,1 \
  --schedules every1,every2,every4,every8,first8 --modes random \
  --max_new_tokens 1024 --shard_count 2 --shard_index 0 &
random_worker_0=$!
python3 scripts/run_online_matrix_stream.py \
  artifacts/privileged_effect/online_high_te_24.jsonl \
  --out artifacts/privileged_effect/online_id288_random_deterministic_shard1.jsonl \
  --id 288 --layers 13,21,23,25 --alphas 0.25,0.5,0.75,1 \
  --schedules every1,every2,every4,every8,first8 --modes random \
  --max_new_tokens 1024 --shard_count 2 --shard_index 1 &
random_worker_1=$!
wait "$random_worker_0"
wait "$random_worker_1"

python3 scripts/merge_online_matrix_shards.py \
  artifacts/privileged_effect/online_id288_matrix_stream.jsonl \
  artifacts/privileged_effect/online_id288_matrix_shard1.jsonl \
  artifacts/privileged_effect/online_id288_random_deterministic_shard0.jsonl \
  artifacts/privileged_effect/online_id288_random_deterministic_shard1.jsonl \
  --out artifacts/privileged_effect/online_id288_matrix_merged.jsonl
python3 scripts/summarize_online_matrix.py \
  artifacts/privileged_effect/online_id288_matrix_merged.jsonl \
  --out artifacts/privileged_effect/online_id288_matrix_summary.json

python3 scripts/run_paired_counterfactual_rescue.py \
  artifacts/privileged_effect/prm800k_paired_steps_100.jsonl \
  --out artifacts/privileged_effect/prm800k_paired_rescue_100_shard0.jsonl \
  --n_examples 100 --continuation_tokens 128 \
  --layers 13,21,23,25 --alphas 1 --windows 128 \
  --controls dependency_swap --shard_count 2 --shard_index 0 &
prm_paired_worker_0=$!
python3 scripts/run_paired_counterfactual_rescue.py \
  artifacts/privileged_effect/prm800k_paired_steps_100.jsonl \
  --out artifacts/privileged_effect/prm800k_paired_rescue_100_shard1.jsonl \
  --n_examples 100 --continuation_tokens 128 \
  --layers 13,21,23,25 --alphas 1 --windows 128 \
  --controls dependency_swap --shard_count 2 --shard_index 1 &
prm_paired_worker_1=$!
wait "$prm_paired_worker_0"
wait "$prm_paired_worker_1"
python3 scripts/summarize_paired_rescue.py \
  artifacts/privileged_effect/prm800k_paired_rescue_100_shard0.jsonl \
  artifacts/privileged_effect/prm800k_paired_rescue_100_shard1.jsonl \
  --out artifacts/privileged_effect/prm800k_paired_rescue_100_summary.tsv \
  --bootstrap 5000

python3 scripts/run_online_paired_residual.py \
  artifacts/privileged_effect/prm800k_paired_steps_100.jsonl \
  --out artifacts/privileged_effect/prm800k_online_baseline_100_shard0.jsonl \
  --n_examples 100 --layers 25 --alphas 1 \
  --schedules every1 --modes none --max_new_tokens 512 \
  --shard_count 2 --shard_index 0 &
prm_online_worker_0=$!
python3 scripts/run_online_paired_residual.py \
  artifacts/privileged_effect/prm800k_paired_steps_100.jsonl \
  --out artifacts/privileged_effect/prm800k_online_baseline_100_shard1.jsonl \
  --n_examples 100 --layers 25 --alphas 1 \
  --schedules every1 --modes none --max_new_tokens 512 \
  --shard_count 2 --shard_index 1 &
prm_online_worker_1=$!
wait "$prm_online_worker_0"
wait "$prm_online_worker_1"
python3 scripts/select_behaviorally_damaged.py \
  artifacts/privileged_effect/prm800k_paired_steps_100.jsonl \
  artifacts/privileged_effect/prm800k_online_baseline_100_shard0.jsonl \
  artifacts/privileged_effect/prm800k_online_baseline_100_shard1.jsonl \
  --out artifacts/privileged_effect/prm800k_online_damaged.jsonl
python3 scripts/run_online_paired_residual.py \
  artifacts/privileged_effect/prm800k_online_damaged.jsonl \
  --out artifacts/privileged_effect/prm800k_online_rescue_core_shard0.jsonl \
  --n_examples 10 --layers 13,21,23,25 --alphas 0.25,0.5,1 \
  --schedules every1 --modes rescue,reverse,random --max_new_tokens 512 \
  --shard_count 2 --shard_index 0 &
prm_core_worker_0=$!
python3 scripts/run_online_paired_residual.py \
  artifacts/privileged_effect/prm800k_online_damaged.jsonl \
  --out artifacts/privileged_effect/prm800k_online_rescue_core_shard1.jsonl \
  --n_examples 10 --layers 13,21,23,25 --alphas 0.25,0.5,1 \
  --schedules every1 --modes rescue,reverse,random --max_new_tokens 512 \
  --shard_count 2 --shard_index 1 &
prm_core_worker_1=$!
wait "$prm_core_worker_0"
wait "$prm_core_worker_1"
python3 scripts/summarize_online_results.py \
  artifacts/privileged_effect/prm800k_online_rescue_core_shard0.jsonl \
  artifacts/privileged_effect/prm800k_online_rescue_core_shard1.jsonl \
  --out artifacts/privileged_effect/prm800k_online_rescue_core_summary.json \
  --bootstrap 5000
python3 scripts/run_online_paired_residual.py \
  artifacts/privileged_effect/prm800k_online_damaged.jsonl \
  --out artifacts/privileged_effect/prm800k_online_projected_pca16_shard0.jsonl \
  --n_examples 10 --layers 25 --alphas 0.5,1 --schedules every1 \
  --modes rescue,reverse,random --max_new_tokens 512 \
  --projection_basis artifacts/privileged_effect/residual_dynamics_contrastive_layer25.basis.pt \
  --projection_kind pca --projection_rank 16 --shard_count 2 --shard_index 0 &
prm_projected_worker_0=$!
python3 scripts/run_online_paired_residual.py \
  artifacts/privileged_effect/prm800k_online_damaged.jsonl \
  --out artifacts/privileged_effect/prm800k_online_projected_pca16_shard1.jsonl \
  --n_examples 10 --layers 25 --alphas 0.5,1 --schedules every1 \
  --modes rescue,reverse,random --max_new_tokens 512 \
  --projection_basis artifacts/privileged_effect/residual_dynamics_contrastive_layer25.basis.pt \
  --projection_kind pca --projection_rank 16 --shard_count 2 --shard_index 1 &
prm_projected_worker_1=$!
wait "$prm_projected_worker_0"
wait "$prm_projected_worker_1"
python3 scripts/summarize_online_results.py \
  artifacts/privileged_effect/prm800k_online_projected_pca16_shard0.jsonl \
  artifacts/privileged_effect/prm800k_online_projected_pca16_shard1.jsonl \
  --out artifacts/privileged_effect/prm800k_online_projected_pca16_summary.json \
  --bootstrap 5000

python3 scripts/run_residual_decay.py \
  artifacts/privileged_effect/matched_controls_500_valid.jsonl \
  --out artifacts/privileged_effect/residual_decay_108_l25_to_l35_shard0.jsonl \
  --n_examples 108 --inject_layer 25 --readout_layer 35 \
  --tokens 128 --origins 0,8,16,32,64,96 --shard_count 2 --shard_index 0 &
decay_worker_0=$!
python3 scripts/run_residual_decay.py \
  artifacts/privileged_effect/matched_controls_500_valid.jsonl \
  --out artifacts/privileged_effect/residual_decay_108_l25_to_l35_shard1.jsonl \
  --n_examples 108 --inject_layer 25 --readout_layer 35 \
  --tokens 128 --origins 0,8,16,32,64,96 --shard_count 2 --shard_index 1 &
decay_worker_1=$!
wait "$decay_worker_0"
wait "$decay_worker_1"
python3 scripts/summarize_residual_decay.py \
  artifacts/privileged_effect/residual_decay_108_l25_to_l35_shard0.jsonl \
  artifacts/privileged_effect/residual_decay_108_l25_to_l35_shard1.jsonl \
  --out artifacts/privileged_effect/residual_decay_108_l25_to_l35_summary.json

python3 scripts/analyze_spatiotemporal_fpca.py \
  artifacts/privileged_effect/dependency_residual_trajectories_layer25.pt \
  --out artifacts/privileged_effect/spatiotemporal_fpca_layer25.json

python3 scripts/audit_dynamic_mechanism.py \
  --out artifacts/privileged_effect/dynamic_mechanism_audit.json
