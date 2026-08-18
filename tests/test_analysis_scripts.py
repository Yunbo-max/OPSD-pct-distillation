import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def valid_eval_payload(dataset: str = "aime24", val_n: int = 12, problems: int = 30) -> dict:
    results = []
    for idx in range(problems):
        generations = [
            {
                "predicted_answer": "1",
                "full_generation": "\\boxed{1}",
                "correct": gen_idx == 0,
                "formatted": True,
            }
            for gen_idx in range(val_n)
        ]
        results.append(
            {
                "problem_id": f"p{idx}",
                "problem": f"problem {idx}",
                "ground_truth": "1",
                "val_n": val_n,
                "generations": generations,
                "num_correct": 1,
                "pass_at_n": True,
                "majority_vote_correct": False,
            }
        )
    return {
        "dataset": dataset,
        "val_n": val_n,
        "num_problems": problems,
        "total_solutions": val_n * problems,
        "pass_at_n_pct": 100.0,
        "average_at_n_pct": 100.0 / val_n,
        "majority_vote_at_n_pct": 0.0,
        "format_rate": 100.0,
        "results": results,
    }


def test_analyze_dispersion_gains_reports_high_group_gain():
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_dispersion_gains.py"),
        "--diagnostic_csv",
        str(ROOT / "tests" / "fixtures" / "dispersion.csv"),
        "--eval",
        f"mean={ROOT / 'tests' / 'fixtures' / 'eval_mean.json'}",
        "--eval",
        f"set={ROOT / 'tests' / 'fixtures' / 'eval_set.json'}",
        "--baseline",
        "mean",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "high\tset\t75.00\t50.00\t2" in output


def test_analyze_dispersion_gains_strict_protocol_rejects_bad_eval(tmp_path):
    bad_path = tmp_path / "bad.json"
    payload = valid_eval_payload("aime24", problems=3)
    payload["pass_at_n_pct"] = 0.0
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_dispersion_gains.py"),
            "--diagnostic_csv",
            str(ROOT / "tests" / "fixtures" / "dispersion.csv"),
            "--eval",
            f"mean={bad_path}",
            "--baseline",
            "mean",
            "--strict_protocol",
            "--expected_num_problems",
            "3",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "pass_at_n_pct" in result.stderr


def test_analyze_dispersion_gains_strict_protocol_requires_overlap(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(valid_eval_payload("aime24", problems=3)), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_dispersion_gains.py"),
            "--diagnostic_csv",
            str(ROOT / "tests" / "fixtures" / "dispersion.csv"),
            "--eval",
            f"mean={eval_path}",
            "--baseline",
            "mean",
            "--strict_protocol",
            "--expected_num_problems",
            "3",
            "--min_matched_per_group",
            "2",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "matched_problems" in result.stderr


def test_build_contaminated_multiref_is_deterministic(tmp_path):
    out = tmp_path / "contam.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_contaminated_multiref.py"),
        "--input",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--output",
        str(out),
        "--rho",
        "1.0",
        "--mode",
        "wrong",
        "--seed",
        "7",
    ]
    subprocess.check_call(cmd)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    assert all("contamination" in row for row in rows)
    assert any("wrong" in row["references"] for row in rows)


def test_contamination_matrix_can_build_manifests_without_training(tmp_path):
    subprocess.check_call(
        [
            "bash",
            str(ROOT / "scripts" / "run_pct_contamination_matrix.sh"),
        ],
        cwd=ROOT,
        env={
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHON_BIN": sys.executable,
            "SOURCE_DATASET": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "OUT_ROOT": str(tmp_path / "contam"),
            "RHOS": "0 0.5",
            "METHODS": "phf_mean set_uot",
            "RUN_TRAIN": "0",
        },
    )
    assert (tmp_path / "contam" / "data" / "multiref_rho0_mixed.jsonl").exists()
    assert (tmp_path / "contam" / "data" / "multiref_rho0p5_mixed.jsonl").exists()
    manifest = (tmp_path / "contam" / "manifests" / "rho0p5_mixed.jsonl").read_text(encoding="utf-8")
    assert "qwen3_4b_rho0p5_mixed_set_uot_steps100_n5000" in manifest


def test_contamination_eval_dry_run_merges_manifests(tmp_path):
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHON_BIN": sys.executable,
        "SOURCE_DATASET": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "OUT_ROOT": str(tmp_path / "contam"),
        "RHOS": "0",
        "METHODS": "phf_mean",
        "RUN_TRAIN": "0",
    }
    subprocess.check_call(["bash", str(ROOT / "scripts" / "run_pct_contamination_matrix.sh")], cwd=ROOT, env=env)
    output = subprocess.check_output(
        ["bash", str(ROOT / "scripts" / "run_pct_contamination_eval.sh")],
        cwd=ROOT,
        env={
            "PATH": env["PATH"],
            "PYTHON_BIN": sys.executable,
            "OUT_ROOT": str(tmp_path / "contam"),
            "EVAL_ROOT": str(tmp_path / "eval"),
            "DRY_RUN": "1",
        },
        text=True,
    )
    assert "all_contamination.jsonl" in output
    assert "eval/evaluate_math.py" in output
    assert "qwen3_4b_rho0_mixed_phf_mean_steps100_n5000_aime24_avg12.json" in output


def test_bootstrap_eval_matrix_reports_paired_delta():
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bootstrap_eval_matrix.py"),
        "--eval",
        f"mean={ROOT / 'tests' / 'fixtures' / 'eval_mean.json'}",
        "--eval",
        f"set={ROOT / 'tests' / 'fixtures' / 'eval_set.json'}",
        "--baseline",
        "mean",
        "--n_boot",
        "100",
        "--seed",
        "0",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "aime24\tset\t3\t83.33" in output
    assert "\t33.33\t" in output


def test_bootstrap_eval_matrix_strict_protocol_accepts_valid_inputs(tmp_path):
    mean_path = tmp_path / "mean.json"
    set_path = tmp_path / "set.json"
    mean_path.write_text(json.dumps(valid_eval_payload("aime24", problems=3)), encoding="utf-8")
    set_path.write_text(json.dumps(valid_eval_payload("aime24", problems=3)), encoding="utf-8")
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "bootstrap_eval_matrix.py"),
            "--eval",
            f"mean={mean_path}",
            "--eval",
            f"set={set_path}",
            "--baseline",
            "mean",
            "--n_boot",
            "10",
            "--strict_protocol",
            "--expected_num_problems",
            "3",
        ],
        text=True,
    )
    assert "aime24\tset\t3\t8.33" in output


def test_bootstrap_eval_matrix_strict_protocol_rejects_bad_inputs(tmp_path):
    mean_path = tmp_path / "mean.json"
    bad_path = tmp_path / "bad.json"
    mean_path.write_text(json.dumps(valid_eval_payload("aime24", problems=3)), encoding="utf-8")
    payload = valid_eval_payload("aime24", problems=3)
    payload["average_at_n_pct"] = 100.0
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bootstrap_eval_matrix.py"),
            "--eval",
            f"mean={mean_path}",
            "--eval",
            f"set={bad_path}",
            "--baseline",
            "mean",
            "--strict_protocol",
            "--expected_num_problems",
            "3",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "average_at_n_pct" in result.stderr


def test_bootstrap_eval_matrix_strict_protocol_requires_paired_ids(tmp_path):
    mean_path = tmp_path / "mean.json"
    set_path = tmp_path / "set.json"
    mean_path.write_text(json.dumps(valid_eval_payload("aime24", problems=3)), encoding="utf-8")
    payload = valid_eval_payload("aime24", problems=3)
    payload["results"][0]["problem_id"] = "different"
    set_path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bootstrap_eval_matrix.py"),
            "--eval",
            f"mean={mean_path}",
            "--eval",
            f"set={set_path}",
            "--baseline",
            "mean",
            "--strict_protocol",
            "--expected_num_problems",
            "3",
            "--require_paired_problem_ids",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "paired_problem_ids" in result.stderr


def test_pipeline_dry_run_prints_selected_stages(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "validate-data",
        "manifest",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[validate-data]" in output
    assert "[manifest]" in output
    assert "[build-multiref-candidates]" not in output
    assert "[train-short-matrix]" not in output


def test_pipeline_dry_run_includes_spec_validation_stages(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "validate-spec",
        "validate-manifest",
        "count-train-tasks",
        "count-eval-tasks",
        "manifest-status",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[validate-spec]" in output
    assert "validate_pct_experiment_spec.py" in output
    assert "[validate-manifest]" in output
    assert "--suite short_matrix" in output
    assert "[count-train-tasks]" in output
    assert "train_tasks.tsv" in output
    assert "[count-eval-tasks]" in output
    assert "eval_tasks.tsv" in output
    assert "[manifest-status]" in output
    assert "manifest_status.tsv" in output


def test_pipeline_dry_run_can_generate_multiref_candidates(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "build-multiref-candidates",
        "audit-multiref-candidates",
        "validate-data",
        "--use_generated_multiref",
        "--candidate_examples",
        "2",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[build-multiref-candidates]" in output
    assert "--dataset_jsonl" in output
    assert "[audit-multiref-candidates]" in output
    assert "--require_wrong_reference" in output
    assert "multiref_candidates_2.jsonl" in output
    assert "[validate-data]" in output


def test_summarize_eval_results_writes_tsv(tmp_path):
    out = tmp_path / "eval.tsv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_eval_results.py"),
        str(ROOT / "tests" / "fixtures" / "eval_mean.json"),
        "--tsv",
        str(out),
    ]
    output = subprocess.check_output(cmd, text=True)
    text = out.read_text(encoding="utf-8")
    assert "Average@N" in output
    assert "Average@N" in text
    assert "aime24" in text


def test_summarize_eval_results_strict_manifest_seed_passes(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    payload = valid_eval_payload("aime24", problems=3)
    payload["seed"] = 123
    (eval_root / "run_aime24_avg12.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"run_name": "run", "eval": [{"dataset": "aime24", "val_n": 12, "seed": 123}]}) + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_eval_results.py"),
            str(eval_root),
            "--manifest",
            str(manifest),
            "--strict_protocol",
            "--expected_num_problems",
            "3",
        ],
        text=True,
    )
    assert "run_aime24_avg12.json\taime24" in output
    assert "\t123" in output


def test_summarize_eval_results_strict_manifest_seed_fails(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    payload = valid_eval_payload("aime24", problems=3)
    payload["seed"] = 999
    (eval_root / "run_aime24_avg12.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"run_name": "run", "eval": [{"dataset": "aime24", "val_n": 12, "seed": 123}]}) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_eval_results.py"),
            str(eval_root),
            "--manifest",
            str(manifest),
            "--strict_protocol",
            "--expected_num_problems",
            "3",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "seed:999!=expected:123" in result.stdout


def test_capture_run_metadata_writes_json(tmp_path):
    out = tmp_path / "metadata.json"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "capture_run_metadata.py"),
            "--out",
            str(out),
            "--notes",
            "test run",
        ],
        cwd=ROOT,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["notes"] == "test run"
    assert "packages" in data
    assert "git" in data


def test_manifest_includes_medoid_and_random(tmp_path):
    out = tmp_path / "manifest.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_manifest.py"),
            "--out",
            str(out),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--output_root",
            str(tmp_path / "runs"),
        ]
    )
    text = out.read_text(encoding="utf-8")
    assert '"method": "phf_random"' in text
    assert '"method": "phf_medoid"' in text
    assert '"method": "phf_grassmann"' in text
    assert '"method": "set_fgw"' in text


def test_manifest_expands_seeded_runs(tmp_path):
    out = tmp_path / "manifest.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_manifest.py"),
            "--out",
            str(out),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--output_root",
            str(tmp_path / "runs"),
            "--methods",
            "phf_single",
            "set_uot",
            "--seeds",
            "0",
            "1",
        ]
    )
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [record["seed"] for record in records] == [0, 1, 0, 1]
    assert records[0]["run_name"].endswith("_seed0")
    assert records[1]["output_dir"].endswith("_seed1")


def test_manifest_can_pin_eval_seed(tmp_path):
    out = tmp_path / "manifest.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_manifest.py"),
            "--out",
            str(out),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--output_root",
            str(tmp_path / "runs"),
            "--methods",
            "set_uot",
            "--seeds",
            "0",
            "1",
            "--eval_seed",
            "123",
        ]
    )
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {record["seed"] for record in records} == {0, 1}
    assert all(item["seed"] == 123 for record in records for item in record["eval"])


def test_train_from_manifest_forwards_seed(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "seeded",
                "method": "phf_single",
                "model": "Qwen/Qwen3-0.6B",
                "dataset": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
                "output_dir": str(tmp_path / "run"),
                "seed": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_train_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--dry_run",
        ],
        text=True,
    )
    assert "--seed 7" in output


def test_summarize_seed_results_aggregates_manifest_seeds(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"run_name": "m_phf_seed0", "method": "phf_single", "seed": 0},
                {"run_name": "m_phf_seed1", "method": "phf_single", "seed": 1},
                {"run_name": "m_uot_seed0", "method": "set_uot", "seed": 0},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    eval_tsv = tmp_path / "eval.tsv"
    eval_tsv.write_text(
        "file\tdataset\tAverage@N\tPass@N\tMaj@N\tFormat\tN\tProblems\n"
        "m_phf_seed0_aime24_avg12.json\taime24\t40.00\t50.00\t30.00\t1.00\t12\t30\n"
        "m_phf_seed1_aime24_avg12.json\taime24\t60.00\t70.00\t50.00\t1.00\t12\t30\n"
        "m_uot_seed0_aime24_avg12.json\taime24\t70.00\t80.00\t60.00\t1.00\t12\t30\n",
        encoding="utf-8",
    )
    out = tmp_path / "seed.tsv"
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_seed_results.py"),
            "--manifest",
            str(manifest),
            "--eval_tsv",
            str(eval_tsv),
            "--out_tsv",
            str(out),
        ],
        text=True,
    )
    assert "phf_single\taime24\t0,1\t50.00\t14.14" in output
    assert "set_uot\taime24\t0\t70.00\t0.00" in out.read_text(encoding="utf-8")


def test_pct_experiment_spec_validates_default_spec():
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_pct_experiment_spec.py"),
            "--spec",
            str(ROOT / "configs" / "pct_neurips_spec.json"),
        ],
        text=True,
    )
    assert "OK\tprimary_evaluation.val_n" in output
    assert "OK\tshort_matrix.required_methods:10" in output


def test_pct_experiment_spec_validates_short_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_manifest.py"),
            "--out",
            str(manifest),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--output_root",
            str(tmp_path / "runs"),
        ]
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_pct_experiment_spec.py"),
            "--spec",
            str(ROOT / "configs" / "pct_neurips_spec.json"),
            "--manifest",
            str(manifest),
            "--suite",
            "short_matrix",
        ],
        text=True,
    )
    assert "qwen3_4b_set_uot_steps100_n5000.eval_datasets" in output


def test_pct_experiment_spec_fails_manifest_missing_method(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "qwen3_4b_none_steps100_n5000",
                "method": "none",
                "model": "Qwen/Qwen3-4B",
                "dataset": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
                "output_dir": str(tmp_path / "runs" / "none"),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_pct_experiment_spec.py"),
            "--spec",
            str(ROOT / "configs" / "pct_neurips_spec.json"),
            "--manifest",
            str(manifest),
            "--suite",
            "short_matrix",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "short_matrix.manifest_missing_method:phf_single" in result.stdout


def test_scaling_manifest_names_models_and_full_sample_tag(tmp_path):
    out = tmp_path / "scaling_manifest.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_scaling_manifest.py"),
            "--out",
            str(out),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--output_root",
            str(tmp_path / "runs"),
            "--models",
            "small=Qwen/Qwen3-1.7B",
            "large=Qwen/Qwen3-8B",
            "--methods",
            "phf_mean",
            "set_fgw",
            "--train_num_samples",
            "0",
            "--max_steps",
            "1000",
            "--eval_seed",
            "321",
            "--seeds",
            "0",
            "1",
        ]
    )
    text = out.read_text(encoding="utf-8")
    assert '"run_name": "small_phf_mean_steps1000_full_seed0"' in text
    assert '"run_name": "large_set_fgw_steps1000_full_seed1"' in text
    records = [json.loads(line) for line in text.splitlines()]
    assert len(records) == 8
    assert {record["seed"] for record in records} == {0, 1}
    assert all(item["seed"] == 321 for record in records for item in record["eval"])


def test_scaling_manifest_keeps_default_unseeded_names(tmp_path):
    out = tmp_path / "scaling_manifest.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_scaling_manifest.py"),
            "--out",
            str(out),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--output_root",
            str(tmp_path / "runs"),
            "--models",
            "small=Qwen/Qwen3-1.7B",
            "--methods",
            "set_fgw",
            "--train_num_samples",
            "0",
            "--max_steps",
            "1000",
        ]
    )
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["run_name"] == "small_set_fgw_steps1000_full"
    assert "seed" not in record


def test_eval_from_manifest_dry_run_uses_record_model_and_output(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    run_dir = tmp_path / "runs" / "method"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "model": "Qwen/Qwen3-4B",
                "output_dir": str(run_dir),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_eval_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(tmp_path / "eval"),
            "--dry_run",
        ],
        text=True,
    )
    assert "--base_model Qwen/Qwen3-4B" in output
    assert "--checkpoint_dir" in output
    assert "method_aime24_avg12.json" in output


def test_pipeline_eval_average12_uses_manifest_runner(tmp_path):
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_pipeline.py"),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--out_root",
            str(tmp_path / "out"),
            "--stages",
            "eval-average12",
            "--dry_run",
        ],
        text=True,
    )
    assert "[eval-average12]" in output
    assert "scripts/run_pct_eval_from_manifest.py" in output
    assert "scripts/run_pct_eval_average12.sh" not in output
    assert "--skip_completed" in output
    assert "--expected_num_problems 30" in output


def test_eval_from_manifest_forwards_eval_seed(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "model": "Qwen/Qwen3-4B",
                "output_dir": str(tmp_path / "runs" / "method"),
                "seed": 3,
                "eval": [{"dataset": "aime24", "val_n": 12, "seed": 99}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_eval_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(tmp_path / "eval"),
            "--dry_run",
        ],
        text=True,
    )
    assert "--seed 99" in output


def test_eval_from_manifest_supports_index_and_skip_completed(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "model": "Qwen/Qwen3-4B",
                "output_dir": str(tmp_path / "runs" / "method"),
                "eval": [{"dataset": "aime24", "val_n": 12}, {"dataset": "aime25", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "method_aime24_avg12.json").write_text(
        json.dumps(valid_eval_payload("aime24")),
        encoding="utf-8",
    )

    indexed = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_eval_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--index",
            "1",
            "--dry_run",
        ],
        text=True,
    )
    assert "method_aime25_avg12.json" in indexed
    assert "method_aime24_avg12.json" not in indexed

    skipped = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_eval_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--index",
            "0",
            "--skip_completed",
            "--dry_run",
        ],
        text=True,
    )
    assert "skip_completed" in skipped
    assert "method_aime24_avg12.json" in skipped


def test_eval_from_manifest_does_not_skip_stale_seeded_output(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "model": "Qwen/Qwen3-4B",
                "output_dir": str(tmp_path / "runs" / "method"),
                "seed": 7,
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    payload = valid_eval_payload("aime24")
    payload["seed"] = 8
    (eval_root / "method_aime24_avg12.json").write_text(json.dumps(payload), encoding="utf-8")
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_eval_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--index",
            "0",
            "--skip_completed",
            "--dry_run",
        ],
        text=True,
    )
    assert "skip_completed" not in output
    assert "--seed 7" in output


def test_validate_eval_results_accepts_full_average12_payload(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(valid_eval_payload("aime24")), encoding="utf-8")
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_eval_results.py"),
            str(path),
            "--expected_dataset",
            "aime24",
            "--expected_val_n",
            "12",
            "--expected_num_problems",
            "30",
        ],
        text=True,
    )
    assert f"{path}\tOK\tnone" in output


def test_validate_eval_results_rejects_bad_recomputed_metric(tmp_path):
    payload = valid_eval_payload("aime24")
    payload["average_at_n_pct"] = 99.0
    path = tmp_path / "eval_bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_eval_results.py"),
            str(path),
            "--expected_dataset",
            "aime24",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "average_at_n_pct" in result.stdout


def test_validate_eval_results_checks_expected_seed(tmp_path):
    payload = valid_eval_payload("aime24")
    payload["seed"] = 8
    path = tmp_path / "eval_seed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_eval_results.py"),
            str(path),
            "--expected_dataset",
            "aime24",
            "--expected_seed",
            "7",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "seed:8!=expected:7" in result.stdout


def test_train_from_manifest_dry_run_uses_record_hyperparameters(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    run_dir = tmp_path / "runs" / "method"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "qwen3_4b_phf_grassmann_steps100_n5000",
                "method": "phf_grassmann",
                "model": "Qwen/Qwen3-4B",
                "dataset": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
                "output_dir": str(run_dir),
                "train_num_samples": 5000,
                "max_steps": 100,
                "pct_loss_weight": 0.1,
                "pct_num_references": 4,
                "pct_grassmann_rank": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_train_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--accelerate",
            "accelerate",
            "--dry_run",
        ],
        text=True,
    )
    assert "opsd_train.py" in output
    assert "--pct_method phf_grassmann" in output
    assert "--pct_grassmann_rank 3" in output
    assert f"--output_dir {run_dir}" in output


def test_train_from_manifest_skips_records_not_requiring_training(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "qwen3_4b_base",
                "method": "base",
                "model": "Qwen/Qwen3-4B",
                "dataset": str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
                "output_dir": "",
                "requires_training": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_train_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--dry_run",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "No manifest records selected" in result.stderr


def test_train_from_manifest_supports_index_and_skip_completed(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    done_dir = tmp_path / "runs" / "done"
    done_dir.mkdir(parents=True)
    (done_dir / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    todo_dir = tmp_path / "runs" / "todo"
    records = [
        {
            "run_name": "done",
            "method": "phf_single",
            "model": "Qwen/Qwen3-4B",
            "dataset": "data.jsonl",
            "output_dir": str(done_dir),
        },
        {
            "run_name": "todo",
            "method": "set_uot",
            "model": "Qwen/Qwen3-4B",
            "dataset": "data.jsonl",
            "output_dir": str(todo_dir),
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    indexed = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_train_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--index",
            "1",
            "--dry_run",
        ],
        text=True,
    )
    assert "--run_config todo" in indexed
    assert "--run_config done" not in indexed

    skipped = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_train_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--index",
            "0",
            "--skip_completed",
            "--dry_run",
        ],
        text=True,
    )
    assert "skip_completed\tdone" in skipped


def test_train_from_manifest_strict_skip_requires_pct_metrics(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    run_dir = tmp_path / "runs" / "set_uot"
    run_dir.mkdir(parents=True)
    (run_dir / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (run_dir / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"loss": 0.1}]}),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "run_name": "set_uot",
                "method": "set_uot",
                "model": "Qwen/Qwen3-4B",
                "dataset": "data.jsonl",
                "output_dir": str(run_dir),
                "pct_loss_weight": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loose = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_train_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--skip_completed",
            "--dry_run",
        ],
        text=True,
    )
    strict = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_train_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--skip_completed",
            "--require_train_metrics",
            "--dry_run",
        ],
        text=True,
    )
    assert "skip_completed\tset_uot" in loose
    assert "skip_completed" not in strict
    assert "--run_config set_uot" in strict


def test_count_pct_manifest_tasks_reports_array_bounds(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_name": "base",
                        "method": "base",
                        "model": "Qwen/Qwen3-4B",
                        "requires_training": False,
                        "eval": [{"dataset": "aime24", "val_n": 12}],
                    }
                ),
                json.dumps(
                    {
                        "run_name": "set_uot",
                        "method": "set_uot",
                        "model": "Qwen/Qwen3-4B",
                        "dataset": "data.jsonl",
                        "output_dir": str(tmp_path / "runs" / "set_uot"),
                        "eval": [{"dataset": "aime24", "val_n": 12}, {"dataset": "aime25", "val_n": 12}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_tsv = tmp_path / "train_tasks.tsv"
    train = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "count_pct_manifest_tasks.py"),
            "--manifest",
            str(manifest),
            "--task",
            "train",
            "--print_array",
            "--tsv",
            str(train_tsv),
        ],
        text=True,
    )
    assert "train\t1\t0" in train
    assert "sbatch_array=0-0" in train
    assert "set_uot" in train_tsv.read_text(encoding="utf-8")

    eval_count = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "count_pct_manifest_tasks.py"),
            "--manifest",
            str(manifest),
            "--task",
            "eval",
            "--print_array",
        ],
        text=True,
    )
    assert "eval\t3\t2" in eval_count
    assert "sbatch_array=0-2" in eval_count


def test_summarize_pct_manifest_status_reports_resume_indexes(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    complete_run = tmp_path / "runs" / "complete"
    complete_run.mkdir(parents=True)
    (complete_run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (complete_run / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"loss": 0.1, "pct_loss": 0.2, "pct_transport_mass": 1.0}]}),
        encoding="utf-8",
    )
    (eval_root / "complete_aime24_avg12.json").write_text(
        json.dumps(valid_eval_payload("aime24")), encoding="utf-8"
    )
    pending_run = tmp_path / "runs" / "pending"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_name": "complete",
                        "method": "set_uot",
                        "pct_loss_weight": 0.1,
                        "output_dir": str(complete_run),
                        "dataset": "data.jsonl",
                        "eval": [{"dataset": "aime24", "val_n": 12}],
                    }
                ),
                json.dumps(
                    {
                        "run_name": "pending",
                        "method": "phf_mean",
                        "pct_loss_weight": 0.1,
                        "output_dir": str(pending_run),
                        "dataset": "data.jsonl",
                        "eval": [{"dataset": "aime24", "val_n": 12}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "status.tsv"
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_pct_manifest_status.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--require_train_metrics",
            "--tsv",
            str(out),
        ],
        text=True,
    )
    assert "train\t0\tcomplete\tset_uot\tdata.jsonl\tcomplete\tnone" in output
    assert "train\t1\tpending\tphf_mean\tdata.jsonl\tpending\tcheckpoint" in output
    assert "eval\t0\tcomplete\tset_uot\taime24\tcomplete\tnone" in output
    assert "eval\t1\tpending\tphf_mean\taime24\tpending\tmissing_output" in out.read_text(encoding="utf-8")


def test_summarize_pct_manifest_status_flags_eval_seed_mismatch(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    run_dir = tmp_path / "runs" / "method"
    run_dir.mkdir(parents=True)
    (run_dir / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    payload = valid_eval_payload("aime24")
    payload["seed"] = 8
    (eval_root / "method_aime24_avg12.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "method": "set_uot",
                "output_dir": str(run_dir),
                "seed": 7,
                "dataset": "data.jsonl",
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_pct_manifest_status.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
        ],
        text=True,
    )
    assert "eval\t0\tmethod\tset_uot\taime24\tpending\tseed:8!=expected:7" in output


def test_external_baseline_manifest_and_base_eval_do_not_require_checkpoint(tmp_path):
    manifest = tmp_path / "external.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_external_baseline_manifest.py"),
            "--out",
            str(manifest),
            "--base_model",
            "Qwen/Qwen3-4B",
            "--baseline",
            "base",
            "--baseline",
            f"sft={tmp_path / 'sft_checkpoint'}",
            "--eval_datasets",
            "aime24",
            "--eval_seed",
            "321",
        ]
    )
    text = manifest.read_text(encoding="utf-8")
    assert '"run_name": "qwen3_4b_base"' in text
    assert '"requires_training": false' in text
    assert '"method": "sft"' in text
    records = [json.loads(line) for line in text.splitlines()]
    assert all(item["seed"] == 321 for record in records for item in record["eval"])

    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pct_eval_from_manifest.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(tmp_path / "eval"),
            "--dry_run",
        ],
        text=True,
    )
    base_line = next(line for line in output.splitlines() if "qwen3_4b_base_aime24_avg12.json" in line)
    sft_line = next(line for line in output.splitlines() if "qwen3_4b_sft_aime24_avg12.json" in line)
    assert "--checkpoint_dir" not in base_line
    assert "--checkpoint_dir" in sft_line


def test_merge_manifests_rejects_duplicate_runs(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text('{"run_name":"same"}\n', encoding="utf-8")
    second.write_text('{"run_name":"same"}\n', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "merge_manifests.py"),
            "--out",
            str(tmp_path / "merged.jsonl"),
            str(first),
            str(second),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "duplicate run_name" in result.stderr


def test_pipeline_dry_run_includes_preflight(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "preflight",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[preflight]" in output
    assert "scripts/preflight_pct.py" in output


def test_preflight_reports_versions_from_requirements(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("pip==0.0.0\nmath-verify==0.0.0\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preflight_pct.py"),
            "--requirements",
            str(req),
            "--strict_versions",
            "--min_gpus",
            "0",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["expected_versions"]["pip"] == "0.0.0"
    assert "python_package_versions" in data
    assert any("version mismatches" in item for item in data["failures"])


def test_pipeline_dry_run_includes_metadata_capture(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "capture-metadata",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[capture-metadata]" in output
    assert "capture_run_metadata.py" in output


def test_capture_dataset_provenance_writes_hashes_and_reference_stats(tmp_path):
    out = tmp_path / "provenance.json"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "capture_dataset_provenance.py"),
            "--dataset",
            str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
            "--min_refs",
            "4",
            "--out",
            str(out),
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["num_rows"] == 2
    assert data["source"]["type"] == "local_json"
    assert len(data["source"]["sha256"]) == 64
    assert data["hashes"]["num_hashed_rows"] == 2
    assert len(data["hashes"]["row_hash_sha256"]) == 64
    stats = data["reference_stats"]
    assert stats["reference_count_min"] == 4
    assert stats["reference_count_max"] == 4
    assert stats["rows_with_wrong_reference"] == 2
    assert stats["rows_with_shuffled_reference"] == 2
    assert stats["too_few_reference_rows"] == 0


def test_pipeline_dry_run_includes_dataset_provenance(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "source-data-provenance",
        "train-data-provenance",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[source-data-provenance]" in output
    assert "[train-data-provenance]" in output
    assert "capture_dataset_provenance.py" in output


def test_pipeline_dry_run_persists_preflight_json(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "preflight",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[preflight]" in output
    assert "--json" in output
    assert "preflight.json" in output


def test_make_pct_resume_plan_writes_pending_commands(tmp_path):
    status = tmp_path / "status.tsv"
    status.write_text(
        "task\tindex\trun_name\tmethod\tdataset\tstatus\tmissing\tartifact\n"
        "train\t0\tr0\tphf_single\tdata.jsonl\tcomplete\tnone\tout0\n"
        "train\t1\tr1\tset_uot\tdata.jsonl\tpending\tcheckpoint\tout1\n"
        "train\t2\tr2\tset_fgw\tdata.jsonl\tpending\tpct_loss\tout2\n"
        "eval\t0\tr0\tphf_single\taime24\tcomplete\tnone\te0\n"
        "eval\t1\tr0\tphf_single\taime25\tpending\tmissing_output\te1\n"
        "eval\t3\tr1\tset_uot\thmmt25\tpending\tinvalid\te3\n",
        encoding="utf-8",
    )
    json_out = tmp_path / "resume.json"
    md_out = tmp_path / "resume.md"
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_resume_plan.py"),
            "--status_tsv",
            str(status),
            "--manifest",
            "manifest.jsonl",
            "--eval_root",
            "eval",
            "--json_out",
            str(json_out),
            "--md_out",
            str(md_out),
        ],
        text=True,
    )
    data = json.loads(output)
    assert data["pending_counts"] == {"eval": 2, "train": 2}
    assert data["slurm_arrays"]["train"] == "1-2"
    assert data["slurm_arrays"]["eval"] == "1,3"
    assert "--index <TRAIN_INDEX>" in data["commands"]["train"]
    assert "--require_train_metrics" in data["commands"]["train"]
    assert "sbatch --array=1,3" in data["commands"]["slurm_eval"]
    assert "set_uot" in md_out.read_text(encoding="utf-8")
    assert json.loads(json_out.read_text(encoding="utf-8"))["pending_indexes"]["train"] == [1, 2]


def test_pipeline_dry_run_includes_resume_plan(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "resume-plan",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[resume-plan]" in output
    assert "make_pct_resume_plan.py" in output
    assert "resume_plan.json" in output


def test_pipeline_train_stage_uses_strict_skip_completed(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "train-short-matrix",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[train-short-matrix]" in output
    assert "--skip_completed" in output
    assert "--require_train_metrics" in output


def test_pipeline_dry_run_includes_final_audit_and_report(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "summarize-eval",
        "claim-gates",
        "bootstrap-eval",
        "dispersion-gain",
        "audit-completion",
        "paper-report",
        "paper-figures",
        "artifact-inventory",
        "audit-artifacts",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[claim-gates]" in output
    assert "evaluate_pct_claim_gates.py" in output
    assert "--seed_summary_tsv" in output
    assert "--dispersion_gain_tsv" in output
    assert "[bootstrap-eval]" in output
    assert "bootstrap_eval_matrix.py" in output
    assert "--require_paired_problem_ids" in output
    assert "[dispersion-gain]" in output
    assert "analyze_dispersion_gains.py" in output
    assert "--min_matched_per_group" in output
    assert "[audit-completion]" in output
    assert "--spec" in output
    assert "--suite short_matrix" in output
    assert "--require_train_metrics" in output
    assert "[paper-report]" in output
    assert "--eval_tsv" in output
    assert "--seed_summary_tsv" in output
    assert "--bootstrap_tsv" in output
    assert "--claim_gates_tsv" in output
    assert "--dataset_provenance_json" in output
    assert "[paper-figures]" in output
    assert "make_pct_figures.py" in output
    assert "[artifact-inventory]" in output
    assert "inventory_pct_artifacts.py" in output
    assert "--checksums_out" in output
    assert "[audit-artifacts]" in output
    assert "--artifact_root" in output
    assert "--strict_protocol" in output


def test_pipeline_dry_run_includes_optional_secondary_eval(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pct_pipeline.py"),
        "--dataset",
        str(ROOT / "tests" / "fixtures" / "multiref_clean.jsonl"),
        "--out_root",
        str(tmp_path / "out"),
        "--stages",
        "eval-secondary",
        "summarize-secondary-eval",
        "--secondary_eval_jsonls",
        f"aimo={ROOT / 'tests' / 'fixtures' / 'custom_eval.jsonl'}",
        f"rrb-aime={ROOT / 'tests' / 'fixtures' / 'custom_eval.jsonl'}",
        "--dry_run",
    ]
    output = subprocess.check_output(cmd, text=True)
    assert "[eval-secondary]" in output
    assert "DATASET_JSONLS=" in output
    assert "[summarize-secondary-eval]" in output


def test_make_paper_report_writes_markdown_and_latex(tmp_path):
    bootstrap = tmp_path / "bootstrap.tsv"
    bootstrap.write_text(
        "dataset\tmethod\tproblems\taverage_pct\tci95_low_pct\tci95_high_pct\t"
        "delta_vs_mean_pct\tdelta_ci95_low_pct\tdelta_ci95_high_pct\n"
        "aime24\tmean\t3\t50.00\t16.67\t100.00\t0.00\t0.00\t0.00\n",
        encoding="utf-8",
    )
    dispersion = tmp_path / "dispersion.tsv"
    dispersion.write_text(
        "group\tmethod\taverage_pct\tgain_vs_mean_pct\tmatched_problems\n"
        "high\tset\t75.00\t50.00\t2\n",
        encoding="utf-8",
    )
    seed_summary = tmp_path / "seed.tsv"
    seed_summary.write_text(
        "method\tdataset\tseeds\tmean_average_at_n\tstd_average_at_n\n"
        "set_uot\taime24\t0,1\t55.00\t7.07\n",
        encoding="utf-8",
    )
    out = tmp_path / "report"
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"git": {"commit": "abc", "dirty": True}, "platform": "test-platform"}),
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "dataset": "train.jsonl",
                "num_rows": 5000,
                "problem_rows": 5000,
                "solution_rows": 5000,
                "source": {"type": "local_json", "sha256": "a" * 64},
                "hashes": {"row_hash_sha256": "b" * 64},
                "reference_stats": {
                    "reference_count_min": 4,
                    "reference_count_mean": 4.0,
                    "reference_count_max": 4,
                    "too_few_reference_rows": 0,
                    "duplicate_reference_rows": 0,
                    "rows_with_wrong_reference": 5000,
                    "rows_with_shuffled_reference": 5000,
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_paper_report.py"),
            "--out_dir",
            str(out),
            "--bootstrap_tsv",
            str(bootstrap),
            "--dispersion_gain_tsv",
            str(dispersion),
            "--metadata_json",
            str(metadata),
            "--dataset_provenance_json",
            str(provenance),
            "--seed_summary_tsv",
            str(seed_summary),
        ]
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    tables = (out / "tables.tex").read_text(encoding="utf-8")
    assert "Run Metadata" in report
    assert "Dataset Provenance" in report
    assert "Row-hash SHA256" in report
    assert "Seed Summary" in report
    assert "Bootstrap Evaluation" in report
    assert "Dispersion Gain" in report
    assert "\\label{tab:pct_bootstrap}" in tables
    assert "\\label{tab:pct_seed_summary}" in tables


def test_evaluate_pct_claim_gates_reports_pass_and_missing(tmp_path):
    eval_tsv = tmp_path / "eval.tsv"
    eval_tsv.write_text(
        "file\tdataset\tAverage@N\tPass@N\tMaj@N\tFormat\tN\tProblems\n"
        "qwen3_4b_phf_single_steps100_n5000_aime24_avg12.json\taime24\t40.00\t50.00\t30.00\t1.00\t12\t30\n"
        "qwen3_4b_set_uot_steps100_n5000_aime24_avg12.json\taime24\t60.00\t70.00\t50.00\t1.00\t12\t30\n"
        "qwen3_4b_phf_single_steps100_n5000_aime25_avg12.json\taime25\t35.00\t45.00\t25.00\t1.00\t12\t30\n"
        "qwen3_4b_set_uot_steps100_n5000_aime25_avg12.json\taime25\t45.00\t55.00\t35.00\t1.00\t12\t30\n"
        "qwen3_4b_phf_single_steps100_n5000_hmmt25_avg12.json\thmmt25\t20.00\t30.00\t15.00\t1.00\t12\t30\n"
        "qwen3_4b_set_uot_steps100_n5000_hmmt25_avg12.json\thmmt25\t25.00\t35.00\t20.00\t1.00\t12\t30\n",
        encoding="utf-8",
    )
    diagnostic_tsv = tmp_path / "diagnostic.tsv"
    diagnostic_tsv.write_text(
        "metric\tcc_mean\tcw_mean\tcs_mean\ttauroc_cc_lt_cw\ttauroc_cc_lt_cs\n"
        "euclidean\t0.2\t0.3\t0.3\t0.55\t0.50\n"
        "fused_gw\t0.1\t0.5\t0.4\t0.85\t0.80\n",
        encoding="utf-8",
    )
    intervention_tsv = tmp_path / "intervention.tsv"
    intervention_tsv.write_text(
        "vector\tmean_delta_logp\tpositive_rate\tn\n"
        "mean\t0.100000\t0.500000\t2\n"
        "nearest_valid\t0.300000\t1.000000\t2\n"
        "wrong\t-0.200000\t0.000000\t2\n",
        encoding="utf-8",
    )
    out = tmp_path / "claim_gates.tsv"
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_pct_claim_gates.py"),
            "--spec",
            str(ROOT / "configs" / "pct_neurips_spec.json"),
            "--eval_tsv",
            str(eval_tsv),
            "--diagnostic_summary_tsv",
            str(diagnostic_tsv),
            "--intervention_tsv",
            str(intervention_tsv),
            "--out_tsv",
            str(out),
        ],
        text=True,
    )
    assert "average12_vs_single_reference\tpass" in output
    assert "transport_diagnostic_separation\tpass" in output
    assert "causal_intervention\tpass" in output
    assert "high_dispersion_gain\tmissing" in output
    assert "contamination_robustness\tmissing" in out.read_text(encoding="utf-8")


def test_evaluate_pct_claim_gates_uses_seed_summary_when_available(tmp_path):
    seed_tsv = tmp_path / "seed.tsv"
    seed_tsv.write_text(
        "method\tdataset\tseeds\tmean_average_at_n\tstd_average_at_n\n"
        "phf_single\taime24\t0,1\t40.00\t1.00\n"
        "set_uot\taime24\t0,1\t50.00\t1.00\n"
        "phf_single\taime25\t0,1\t35.00\t1.00\n"
        "set_uot\taime25\t0,1\t45.00\t1.00\n"
        "phf_single\thmmt25\t0,1\t20.00\t1.00\n"
        "set_uot\thmmt25\t0,1\t25.00\t1.00\n",
        encoding="utf-8",
    )
    eval_tsv = tmp_path / "eval.tsv"
    eval_tsv.write_text(
        "file\tdataset\tAverage@N\tPass@N\tMaj@N\tFormat\tN\tProblems\n"
        "unparseable_aime24_avg12.json\taime24\t0.00\t0.00\t0.00\t1.00\t12\t30\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_pct_claim_gates.py"),
            "--spec",
            str(ROOT / "configs" / "pct_neurips_spec.json"),
            "--eval_tsv",
            str(eval_tsv),
            "--seed_summary_tsv",
            str(seed_tsv),
        ],
        text=True,
    )
    assert "average12_vs_single_reference\tpass\tseed_mean_pct_best_beats_phf_single:3/3" in output


def test_make_paper_report_includes_claim_gates(tmp_path):
    gates = tmp_path / "gates.tsv"
    gates.write_text(
        "gate\tstatus\tdetail\n"
        "transport_diagnostic_separation\tpass\tbest=fused_gw:0.850,euclidean=0.550\n",
        encoding="utf-8",
    )
    out = tmp_path / "report"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_paper_report.py"),
            "--out_dir",
            str(out),
            "--claim_gates_tsv",
            str(gates),
        ]
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    tables = (out / "tables.tex").read_text(encoding="utf-8")
    assert "Claim Gates" in report
    assert "\\label{tab:pct_claim_gates}" in tables


def test_make_pct_figures_writes_svg_files(tmp_path):
    diagnostic = tmp_path / "diagnostic.tsv"
    diagnostic.write_text(
        "metric\tcc_mean\tcw_mean\tcs_mean\tauroc_cc_lt_cw\tauroc_cc_lt_cs\n"
        "sinkhorn\t0.1\t0.4\t0.3\t1.0\t0.9\n",
        encoding="utf-8",
    )
    eval_tsv = tmp_path / "eval.tsv"
    eval_tsv.write_text(
        "file\tdataset\tAverage@N\tPass@N\tMaj@N\tFormat\tN\tProblems\n"
        "m_aime24_avg12.json\taime24\t50.00\t60.00\t40.00\t1.00\t12\t30\n",
        encoding="utf-8",
    )
    dispersion = tmp_path / "dispersion.tsv"
    dispersion.write_text(
        "group\tmethod\taverage_pct\tgain_vs_mean_pct\tmatched_problems\n"
        "high\tset_fgw\t75.00\t25.00\t30\n",
        encoding="utf-8",
    )
    out = tmp_path / "figures"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_pct_figures.py"),
            "--out_dir",
            str(out),
            "--diagnostic_summary_tsv",
            str(diagnostic),
            "--eval_tsv",
            str(eval_tsv),
            "--dispersion_gain_tsv",
            str(dispersion),
        ]
    )
    assert (out / "diagnostic_auroc_correct_vs_wrong.svg").exists()
    assert (out / "eval_average_at_n.svg").exists()
    assert (out / "dispersion_group_gain.svg").exists()


def test_inventory_pct_artifacts_writes_json_and_markdown(tmp_path):
    root = tmp_path / "run"
    (root / "runs" / "method" / "checkpoint-1").mkdir(parents=True)
    (root / "eval" / "pct_average12").mkdir(parents=True)
    (root / "report" / "figures").mkdir(parents=True)
    (root / "run_metadata.json").write_text("{}", encoding="utf-8")
    (root / "data" / "train_dataset_provenance.json").parent.mkdir(parents=True)
    (root / "data" / "train_dataset_provenance.json").write_text("{}", encoding="utf-8")
    (root / "runs" / "manifest.jsonl").write_text('{"run_name":"method"}\n', encoding="utf-8")
    (root / "runs" / "method" / "checkpoint-1" / "trainer_state.json").write_text("{}", encoding="utf-8")
    (root / "eval" / "pct_average12" / "method_aime24_avg12.json").write_text("{}", encoding="utf-8")
    (root / "eval" / "pct_average12" / "eval_summary.tsv").write_text("x\n", encoding="utf-8")
    (root / "report" / "figures" / "fig.svg").write_text("<svg/>", encoding="utf-8")
    json_out = tmp_path / "inventory.json"
    md_out = tmp_path / "inventory.md"
    checksums_out = tmp_path / "checksums.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "inventory_pct_artifacts.py"),
            "--root",
            str(root),
            "--json_out",
            str(json_out),
            "--md_out",
            str(md_out),
            "--checksums_out",
            str(checksums_out),
            "--require_complete",
        ]
    )
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["missing_required_groups"] == []
    assert "method_aime24_avg12.json" in md_out.read_text(encoding="utf-8")
    assert "train_dataset_provenance.json" in md_out.read_text(encoding="utf-8")
    checksum_rows = [json.loads(line) for line in checksums_out.read_text(encoding="utf-8").splitlines()]
    eval_row = next(row for row in checksum_rows if row["path"] == "eval/pct_average12/method_aime24_avg12.json")
    assert eval_row["group"] == "eval_json"
    assert len(eval_row["sha256"]) == 64
    assert eval_row["size_bytes"] > 0


def test_summarize_robustness_results_groups_perturbations(tmp_path):
    out = tmp_path / "robustness.tsv"
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_robustness_results.py"),
            "--eval",
            f"set_fgw={ROOT / 'tests' / 'fixtures' / 'eval_robustness.json'}",
            "--tsv",
            str(out),
        ],
        text=True,
    )
    assert "rrb-aime\tset_fgw\tclean\t2\t50.00\t0.00" in output
    assert "rrb-aime\tset_fgw\tswap_numbers\t2\t25.00\t-25.00" in out.read_text(encoding="utf-8")


def test_audit_pct_completion_detects_complete_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    eval_json = eval_root / "method_aime24_avg12.json"
    eval_json.write_text(json.dumps(valid_eval_payload("aime24")), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "method": "method",
                "output_dir": str(run),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--require_eval",
            "--required_methods",
            "method",
            "--required_eval_datasets",
            "aime24",
        ],
        text=True,
    )
    assert "method\tTrue\tTrue\tnone" in output


def test_audit_pct_completion_uses_spec_defaults(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "method_aime24_avg12.json").write_text(
        json.dumps(valid_eval_payload("aime24", problems=3)),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "method": "method",
                "output_dir": str(run),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "short_matrix": {"required_methods": ["method"]},
                "primary_evaluation": {"datasets": ["aime24"], "problems_per_dataset": 3},
                "required_artifact_groups": ["metadata"],
            }
        ),
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--spec",
            str(spec),
            "--require_eval",
        ],
        text=True,
    )
    assert "method\tTrue\tTrue\tnone" in output


def test_audit_pct_completion_uses_spec_suite_methods(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "scale_aime24_avg12.json").write_text(
        json.dumps(valid_eval_payload("aime24", problems=3)),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "scale",
                "method": "scale_method",
                "output_dir": str(run),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "short_matrix": {"required_methods": ["short_method"]},
                "scaling_matrix": {"required_methods": ["scale_method"]},
                "primary_evaluation": {"datasets": ["aime24"], "problems_per_dataset": 3},
            }
        ),
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--spec",
            str(spec),
            "--suite",
            "scaling_matrix",
            "--require_eval",
        ],
        text=True,
    )
    assert "scale\tTrue\tTrue\tnone" in output


def test_audit_pct_completion_fails_missing_eval(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "method": "method",
                "output_dir": str(run),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(tmp_path / "eval"),
            "--require_eval",
            "--required_methods",
            "method",
            "--required_eval_datasets",
            "aime24",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "False" in result.stdout


def test_audit_pct_completion_fails_eval_seed_mismatch(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    payload = valid_eval_payload("aime24")
    payload["seed"] = 8
    (eval_root / "method_aime24_avg12.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "method": "method",
                "output_dir": str(run),
                "seed": 7,
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--require_eval",
            "--required_methods",
            "method",
            "--required_eval_datasets",
            "aime24",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "seed:8!=expected:7" in result.stdout


def test_audit_pct_completion_fails_missing_artifact_group(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "method_aime24_avg12.json").write_text(json.dumps(valid_eval_payload("aime24")), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "run_metadata.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "method": "method",
                "output_dir": str(run),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--required_methods",
            "method",
            "--required_eval_datasets",
            "aime24",
            "--artifact_root",
            str(artifact_root),
            "--required_artifact_groups",
            "metadata",
            "dataset_provenance",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "missing_artifact_group:dataset_provenance" in result.stdout


def test_audit_pct_completion_fails_incomplete_eval_metrics(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "method_aime24_avg12.json").write_text('{"val_n":12,"num_problems":30}', encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "method",
                "method": "method",
                "output_dir": str(run),
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--require_eval",
            "--required_methods",
            "method",
            "--required_eval_datasets",
            "aime24",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "method\tTrue\tFalse" in result.stdout


def test_audit_pct_completion_allows_base_model_without_checkpoint(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "qwen3_4b_base_aime24_avg12.json").write_text(
        json.dumps(valid_eval_payload("aime24")), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_name": "qwen3_4b_base",
                "method": "base",
                "model": "Qwen/Qwen3-4B",
                "requires_training": False,
                "output_dir": None,
                "eval": [{"dataset": "aime24", "val_n": 12}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_pct_completion.py"),
            "--manifest",
            str(manifest),
            "--eval_root",
            str(eval_root),
            "--require_eval",
            "--require_train_metrics",
            "--required_methods",
            "base",
            "--required_eval_datasets",
            "aime24",
        ],
        text=True,
    )
    assert "qwen3_4b_base\tTrue\tTrue\tnone" in output
