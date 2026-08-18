#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/root/OPSD/runs/pct_contamination_matrix}"
EVAL_ROOT="${EVAL_ROOT:-/root/OPSD/eval_results/pct_contamination_average12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND="${BACKEND:-vllm}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
DRY_RUN="${DRY_RUN:-0}"

MANIFESTS=("${OUT_ROOT}"/manifests/*.jsonl)
MERGED="${OUT_ROOT}/manifests/all_contamination.jsonl"

"${PYTHON_BIN}" scripts/merge_manifests.py \
  --out "${MERGED}" \
  "${MANIFESTS[@]}"

"${PYTHON_BIN}" scripts/run_pct_eval_from_manifest.py \
  --manifest "${MERGED}" \
  --eval_root "${EVAL_ROOT}" \
  --python "${PYTHON_BIN}" \
  --backend "${BACKEND}" \
  --tensor_parallel_size "${TP}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  $([ "${DRY_RUN}" = "1" ] && printf -- --dry_run)

"${PYTHON_BIN}" scripts/summarize_eval_results.py \
  "${EVAL_ROOT}" \
  --tsv "${EVAL_ROOT}/eval_summary.tsv"
