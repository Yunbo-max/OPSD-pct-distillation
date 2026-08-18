#!/usr/bin/env bash
#SBATCH --job-name=pct-diag
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
SET_PHF_ROOT="${SET_PHF_ROOT:-/path/to/set-phf}"
DATASET="${DATASET:-/path/to/multiref_opsd.jsonl}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
USE_GENERATED_MULTIREF="${USE_GENERATED_MULTIREF:-0}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-${MODEL}}"
CANDIDATE_EXAMPLES="${CANDIDATE_EXAMPLES:-5000}"

EXTRA_ARGS=()
STAGES=(capture-metadata preflight validate-data source-data-provenance train-data-provenance manifest count-train-tasks count-eval-tasks manifest-status resume-plan ot-diagnostic summarize-diagnostic causal-intervention summarize-intervention)
if [[ "${USE_GENERATED_MULTIREF}" == "1" ]]; then
  STAGES=(build-multiref-candidates audit-multiref-candidates "${STAGES[@]}")
  EXTRA_ARGS+=(
    --use_generated_multiref
    --dataset_split "${DATASET_SPLIT}"
    --candidate_model "${CANDIDATE_MODEL}"
    --candidate_examples "${CANDIDATE_EXAMPLES}"
  )
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
  --min_gpus 8 \
  --require_vllm \
  "${EXTRA_ARGS[@]}"
