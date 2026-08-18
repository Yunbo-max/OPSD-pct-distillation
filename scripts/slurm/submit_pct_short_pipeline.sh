#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_pct_short_pipeline.sh [--dry-run]

Submits the short PCT experiment pipeline with SLURM dependencies:
  preflight -> data/diagnostics/manifest -> train array -> eval array -> report/audit

Required environment:
  OPSD_ROOT=/path/to/OPSD
  SET_PHF_ROOT=/path/to/set-phf
  DATASET=/path/to/multiref_opsd.jsonl
  OUT_ROOT=/path/to/pct_experiment

Optional environment:
  MODEL=Qwen/Qwen3-4B
  PYTHON_BIN=python3
  NUM_PROCESSES=8
  TP=8
  USE_GENERATED_MULTIREF=0
  TRAIN_ARRAY=0-9
  EVAL_ARRAY=0-29
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  usage >&2
  exit 2
fi

OPSD_ROOT="${OPSD_ROOT:-/path/to/OPSD}"
SET_PHF_ROOT="${SET_PHF_ROOT:-/path/to/set-phf}"
DATASET="${DATASET:-/path/to/multiref_opsd.jsonl}"
OUT_ROOT="${OUT_ROOT:-/path/to/pct_experiment}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${DRY_RUN}" == "0" ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found; use --dry-run outside a SLURM login node." >&2
  exit 127
fi

submit() {
  local name="$1"
  shift
  local cmd=(sbatch --parsable "$@")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %s\t' "${name}" >&2
    printf '%q ' "${cmd[@]}" >&2
    printf '\n' >&2
    echo "dry_${name}"
  else
    local job_id
    job_id="$("${cmd[@]}")"
    echo "${name}: ${job_id}" >&2
    echo "${job_id}"
  fi
}

export OPSD_ROOT SET_PHF_ROOT DATASET OUT_ROOT MODEL PYTHON_BIN
export MANIFEST="${OUT_ROOT}/runs/manifest.jsonl"
export EVAL_ROOT="${OUT_ROOT}/eval/pct_average12"

array_range() {
  local task="$1"
  local fallback="$2"
  local env_value="$3"
  if [[ -n "${env_value}" ]]; then
    echo "${env_value}"
    return
  fi
  if [[ -f "${MANIFEST}" ]]; then
    local count
    count="$("${PYTHON_BIN}" "${OPSD_ROOT}/scripts/count_pct_manifest_tasks.py" --manifest "${MANIFEST}" --task "${task}" | awk -F'\t' -v task="${task}" '$1 == task {print $3}')"
    if [[ -n "${count}" && "${count}" != "-1" ]]; then
      echo "0-${count}"
      return
    fi
  fi
  echo "${fallback}"
}

TRAIN_ARRAY_RESOLVED="$(array_range train "0-9" "${TRAIN_ARRAY:-}")"
EVAL_ARRAY_RESOLVED="$(array_range eval "0-29" "${EVAL_ARRAY:-}")"

preflight_job="$(submit preflight scripts/slurm/pct_00_preflight.sbatch)"
diag_job="$(submit diagnostics --dependency="afterok:${preflight_job}" scripts/slurm/pct_10_data_and_diagnostics.sbatch)"
train_job="$(submit train --dependency="afterok:${diag_job}" --array="${TRAIN_ARRAY_RESOLVED}" scripts/slurm/pct_21_train_manifest_array.sbatch)"
eval_job="$(submit eval --dependency="afterok:${train_job}" --array="${EVAL_ARRAY_RESOLVED}" scripts/slurm/pct_61_eval_manifest_array.sbatch)"
report_job="$(submit report --dependency="afterok:${eval_job}" scripts/slurm/pct_40_audit_and_report.sbatch)"

cat <<EOF
Submitted PCT short pipeline:
preflight=${preflight_job}
diagnostics=${diag_job}
train=${train_job}
eval=${eval_job}
report=${report_job}
EOF
