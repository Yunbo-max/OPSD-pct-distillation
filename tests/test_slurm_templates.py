from pathlib import Path
import json
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_slurm_templates_reference_canonical_scripts():
    files = sorted((ROOT / "scripts" / "slurm").glob("pct_*.sbatch"))
    assert len(files) >= 7
    contents = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "scripts/preflight_pct.py" in contents
    assert "scripts/capture_run_metadata.py" in contents
    assert "scripts/run_pct_pipeline.py" in contents
    assert "scripts/audit_pct_completion.py" in contents
    assert "scripts/make_paper_report.py" in contents
    assert "scripts/make_pct_figures.py" in contents
    assert "scripts/inventory_pct_artifacts.py" in contents
    assert "USE_GENERATED_MULTIREF" in contents
    assert "build-multiref-candidates" in contents
    assert "SECONDARY_EVAL_JSONLS" in contents
    assert "eval-secondary" in contents
    assert "scripts/make_pct_scaling_manifest.py" in contents
    assert "scripts/run_pct_eval_from_manifest.py" in contents
    assert "scripts/run_pct_train_from_manifest.py" in contents
    assert "SLURM_ARRAY_TASK_ID" in contents
    assert "scripts/run_pct_contamination_matrix.sh" in contents
    assert "scripts/run_pct_contamination_eval.sh" in contents
    assert "submit_pct_short_pipeline.sh" in "\n".join(path.name for path in files + [ROOT / "scripts" / "slurm" / "submit_pct_short_pipeline.sh"])


def test_slurm_templates_have_strict_shell_mode():
    for path in list((ROOT / "scripts" / "slurm").glob("pct_*.sbatch")) + [
        ROOT / "scripts" / "slurm" / "submit_pct_short_pipeline.sh"
    ]:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "set -euo pipefail" in text


def test_scaling_launcher_uses_manifest_driven_training():
    text = (ROOT / "scripts" / "run_pct_scaling_matrix.sh").read_text(encoding="utf-8")
    assert "scripts/make_pct_scaling_manifest.py" in text
    assert "scripts/validate_pct_experiment_spec.py" in text
    assert "--suite scaling_matrix" in text
    assert "scripts/run_pct_train_from_manifest.py" in text
    assert "--require_train_metrics" in text


def test_scaling_launcher_dry_run_writes_manifest_hyperparams(tmp_path):
    out_root = tmp_path / "runs"
    env = os.environ.copy()
    env.update(
        {
            "DATASET": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "OUT_ROOT": str(out_root),
            "PYTHON_BIN": sys.executable,
            "ACCELERATE_BIN": "echo",
            "MODELS": "small=Qwen/Qwen3-1.7B",
            "METHODS": "set_uot",
            "TRAIN_NUM_SAMPLES": "2",
            "MAX_STEPS": "3",
            "PCT_WEIGHT": "0.2",
            "PCT_REFS": "2",
            "DRY_RUN": "1",
            "VALIDATE_SPEC": "0",
        }
    )
    output = subprocess.check_output(
        ["bash", str(ROOT / "scripts" / "run_pct_scaling_matrix.sh")],
        cwd=ROOT,
        env=env,
        text=True,
    )
    manifest = out_root / "manifest.jsonl"
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert record["pct_loss_weight"] == 0.2
    assert record["pct_num_references"] == 2
    assert record["max_completion_length"] == 1024
    assert "scripts/run_pct_train_from_manifest.py" not in output
    assert "--pct_loss_weight 0.2" in output
    assert "--pct_num_references 2" in output


def test_scaling_eval_template_is_strict_and_seed_summarized():
    text = (ROOT / "scripts" / "slurm" / "pct_60_scaling_eval_from_manifest.sbatch").read_text(encoding="utf-8")
    assert "--expected_num_problems" in text
    assert "--skip_completed" in text
    assert "scripts/summarize_seed_results.py" in text
    assert "seed_summary.tsv" in text


def test_report_template_builds_posthoc_evidence_tables():
    text = (ROOT / "scripts" / "slurm" / "pct_40_audit_and_report.sbatch").read_text(encoding="utf-8")
    assert "scripts/bootstrap_eval_matrix.py" in text
    assert "--require_paired_problem_ids" in text
    assert "scripts/analyze_dispersion_gains.py" in text
    assert "--min_matched_per_group" in text
    assert "--bootstrap_tsv" in text
    assert "--dispersion_gain_tsv" in text


def test_slurm_submitter_dry_run_wires_dependencies(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "OPSD_ROOT": str(ROOT),
            "SET_PHF_ROOT": "/root/set-phf",
            "DATASET": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "OUT_ROOT": str(tmp_path / "out"),
        }
    )
    result = subprocess.run(
        [str(ROOT / "scripts" / "slurm" / "submit_pct_short_pipeline.sh"), "--dry-run"],
        text=True,
        env=env,
        capture_output=True,
        check=True,
    )
    trace = result.stderr
    assert "pct_00_preflight.sbatch" in trace
    assert "pct_10_data_and_diagnostics.sbatch" in trace
    assert "pct_21_train_manifest_array.sbatch" in trace
    assert "pct_61_eval_manifest_array.sbatch" in trace
    assert "pct_40_audit_and_report.sbatch" in trace
    assert "--dependency=afterok:dry_preflight" in trace
    assert "--dependency=afterok:dry_diagnostics" in trace
    assert "--array=0-9" in trace
    assert "--array=0-29" in trace
    assert "report=dry_report" in result.stdout


def test_slurm_submitter_accepts_array_overrides(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "OPSD_ROOT": str(ROOT),
            "SET_PHF_ROOT": "/root/set-phf",
            "DATASET": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "OUT_ROOT": str(tmp_path / "out"),
            "TRAIN_ARRAY": "0-29",
            "EVAL_ARRAY": "0-89",
        }
    )
    result = subprocess.run(
        [str(ROOT / "scripts" / "slurm" / "submit_pct_short_pipeline.sh"), "--dry-run"],
        text=True,
        env=env,
        capture_output=True,
        check=True,
    )
    assert "--array=0-29" in result.stderr
    assert "--array=0-89" in result.stderr
