#!/usr/bin/env python3
"""Problem-split predictability gate for a privileged causal field.

The deployable predictors consume only an unprivileged hidden state X. PCA
baselines consume the oracle target C and are explicitly reported as compression
ceilings, not deployable predictors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def split_indices(n: int, seed: int) -> dict[str, list[int]]:
    order = np.random.default_rng(seed).permutation(n).tolist()
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    return {"train": order[:n_train], "val": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:]}


def predict(x: torch.Tensor, model: dict) -> torch.Tensor:
    kind = model["kind"]
    if kind == "zero":
        return torch.zeros((x.shape[0], model["mean_c"].numel()), device=x.device)
    if kind == "mean":
        return model["mean_c"].expand(x.shape[0], -1)
    centered = x - model["mean_x"]
    if kind == "dense":
        value = centered @ model["weight"]
    elif kind == "factorized":
        value = (centered @ model["left"]) @ model["right"]
    else:
        raise ValueError(kind)
    return model["mean_c"] + model.get("scale", 1.0) * value


def aggregate_metrics(records: list[tuple[torch.Tensor, torch.Tensor]], model: dict) -> dict:
    problem_rows, total_sse, total_sst, cosine_sum, norm_pred, norm_target, tokens = [], 0, 0, 0, 0, 0, 0
    mean_c = model["mean_c"]
    for x, target in records:
        estimate = predict(x, model)
        residual = (target - estimate).square().sum()
        baseline = (target - mean_c).square().sum()
        cosine = F.cosine_similarity(estimate, target, dim=-1, eps=1e-8)
        target_norm = target.norm(dim=-1)
        estimate_norm = estimate.norm(dim=-1)
        n = target.shape[0]
        row = {
            "r2": float(1 - residual / baseline.clamp_min(1e-12)),
            "cosine": float(cosine.mean()),
            "norm_ratio": float(estimate_norm.mean() / target_norm.mean().clamp_min(1e-12)),
            "tokens": n,
        }
        problem_rows.append(row)
        total_sse += float(residual); total_sst += float(baseline)
        cosine_sum += float(cosine.sum()); norm_pred += float(estimate_norm.sum())
        norm_target += float(target_norm.sum()); tokens += n
    return {
        "n_problems": len(records), "n_tokens": tokens,
        "token_weighted": {
            "r2": 1 - total_sse / max(total_sst, 1e-12),
            "cosine": cosine_sum / tokens,
            "norm_ratio": norm_pred / max(norm_target, 1e-12),
        },
        "problem_mean": {
            key: float(np.mean([row[key] for row in problem_rows]))
            for key in ("r2", "cosine", "norm_ratio")
        },
    }


def calibration_scale(records: list[tuple[torch.Tensor, torch.Tensor]], model: dict) -> float:
    numerator = denominator = 0.0
    raw = dict(model); raw["scale"] = 1.0
    for x, target in records:
        component = predict(x, raw) - model["mean_c"]
        centered_target = target - model["mean_c"]
        numerator += float((component * centered_target).sum())
        denominator += float(component.square().sum())
    return numerator / max(denominator, 1e-12)


def move_model(model: dict, device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in model.items()}


def load_records(cache: Path, rows: list[dict], indices: list[int], x_layer: int, device) -> list:
    result = []
    for index in indices:
        payload = torch.load(cache / f"{index:06d}.pt", map_location="cpu", weights_only=False)
        assert payload["id"] == rows[index]["id"]
        result.append((payload["x"][x_layer].float().to(device), payload["c"].float().to(device)))
    return result


def concatenate(records: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([row[0] for row in records]), torch.cat([row[1] for row in records])


def cpu_payload(model: dict) -> dict:
    return {key: value.detach().cpu().to(torch.float16) if isinstance(value, torch.Tensor) else value
            for key, value in model.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("cache_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--controller_out", required=True)
    parser.add_argument("--x_layer", type=int, default=23)
    parser.add_argument("--subset", choices=("all", "downstream"), default="all")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--ridge", default="0.0001,0.001,0.01,0.1,1")
    parser.add_argument("--ranks", default="16,64")
    parser.add_argument("--cca_reg", type=float, default=0.01)
    args = parser.parse_args()

    all_rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    cache = Path(args.cache_dir)
    available = [index for index in range(len(all_rows)) if (cache / f"{index:06d}.pt").exists()]
    if args.subset == "downstream":
        available = [index for index in available if all_rows[index].get("downstream_steps")]
    rows = [all_rows[index] for index in available]
    # Reindex only the split array; cache indices remain tied to the original dataset.
    splits_local = split_indices(len(rows), args.seed)
    splits = {name: [available[index] for index in values] for name, values in splits_local.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = {name: load_records(cache, all_rows, values, args.x_layer, device)
               for name, values in splits.items()}
    train_x, train_c = concatenate(records["train"])
    mean_x, mean_c = train_x.mean(0), train_c.mean(0)
    xc, cc = train_x - mean_x, train_c - mean_c
    n = xc.shape[0]
    sxx = xc.T @ xc / n
    sxy = xc.T @ cc / n
    syy = cc.T @ cc / n
    del train_x, train_c, xc, cc
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dimension = sxx.shape[0]
    eye = torch.eye(dimension, device=device)
    x_scale, c_scale = float(sxx.trace() / dimension), float(syy.trace() / dimension)
    eigen_x, basis_x = torch.linalg.eigh(sxx)
    ranks = [int(value) for value in args.ranks.split(",")]
    models = {
        "zero": {"kind": "zero", "mean_c": mean_c, "mean_x": mean_x},
        "mean": {"kind": "mean", "mean_c": mean_c, "mean_x": mean_x},
    }

    ridge_candidates = {}
    for value in [float(item) for item in args.ridge.split(",")]:
        inverse_action = basis_x @ ((basis_x.T @ sxy) / (eigen_x[:, None] + value * x_scale))
        candidate = {"kind": "dense", "mean_x": mean_x, "mean_c": mean_c,
                     "weight": inverse_action, "ridge": value, "scale": 1.0}
        candidate["scale"] = calibration_scale(records["val"], candidate)
        ridge_candidates[value] = candidate
    ridge_scores = {value: aggregate_metrics(records["val"], model)["token_weighted"]["r2"]
                    for value, model in ridge_candidates.items()}
    selected_ridge = max(ridge_scores, key=ridge_scores.get)
    ridge_model = ridge_candidates[selected_ridge]
    models["ridge"] = ridge_model

    # Classical reduced-rank regression: project the ridge fitted values onto
    # their dominant output subspace. This is prediction-aware, unlike PCA(C).
    train_records = records["train"]
    fitted = torch.cat([predict(x, ridge_model) - mean_c for x, _ in train_records])
    _, _, rrr_v = torch.pca_lowrank(fitted, q=max(ranks), center=False, niter=4)
    del fitted
    for rank in ranks:
        output_basis = rrr_v[:, :rank]
        model = {"kind": "factorized", "mean_x": mean_x, "mean_c": mean_c,
                 "left": ridge_model["weight"] @ output_basis,
                 "right": output_basis.T, "rank": rank, "ridge": selected_ridge, "scale": 1.0}
        model["scale"] = calibration_scale(records["val"], model)
        models[f"rrr{rank}"] = model

    # Regularized CCA followed by canonical regression from whitened X to C.
    eigen_c, basis_c = torch.linalg.eigh(syy)
    reg_x, reg_c = args.cca_reg * x_scale, args.cca_reg * c_scale
    invroot_x = basis_x @ torch.diag((eigen_x + reg_x).clamp_min(1e-12).rsqrt()) @ basis_x.T
    invroot_c = basis_c @ torch.diag((eigen_c + reg_c).clamp_min(1e-12).rsqrt()) @ basis_c.T
    root_c = basis_c @ torch.diag((eigen_c + reg_c).clamp_min(1e-12).sqrt()) @ basis_c.T
    canonical = invroot_x @ sxy @ invroot_c
    cca_u, cca_s, cca_v = torch.pca_lowrank(canonical, q=max(ranks), center=False, niter=4)
    for rank in ranks:
        left = invroot_x @ cca_u[:, :rank] @ torch.diag(cca_s[:rank])
        right = cca_v[:, :rank].T @ root_c
        model = {"kind": "factorized", "mean_x": mean_x, "mean_c": mean_c,
                 "left": left, "right": right, "rank": rank,
                 "cca_reg": args.cca_reg, "scale": 1.0}
        model["scale"] = calibration_scale(records["val"], model)
        models[f"cca{rank}"] = model

    # Oracle PCA is a target-dependent reconstruction ceiling and cannot be
    # used by a reference-free deployment controller.
    _, _, pca_v = torch.pca_lowrank(
        torch.cat([target - mean_c for _, target in records["train"]]),
        q=max(ranks), center=False, niter=4,
    )
    results = {name: aggregate_metrics(records["test"], model) for name, model in models.items()}
    oracle_pca = {}
    for rank in ranks:
        basis = pca_v[:, :rank]
        oracle_model = {"kind": "oracle_pca", "rank": rank}
        transformed = []
        for x, target in records["test"]:
            estimate = mean_c + ((target - mean_c) @ basis) @ basis.T
            # Reuse the metric path with a synthetic identity predictor.
            transformed.append((estimate, target))
        identity = {"kind": "dense", "mean_x": torch.zeros_like(mean_c),
                    "mean_c": torch.zeros_like(mean_c), "weight": eye, "scale": 1.0}
        oracle_pca[f"pca{rank}"] = aggregate_metrics(transformed, identity)
        oracle_pca[f"pca{rank}"]["deployment_visible"] = False
        oracle_pca[f"pca{rank}"]["kind"] = oracle_model["kind"]

    report = {
        "definition": "predict E[C_t | h_student_t] with strict problem split",
        "subset": args.subset, "x_layer": args.x_layer,
        "n_problems": len(rows), "split_seed": args.seed,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "split_ids": {key: [all_rows[index]["id"] for index in value] for key, value in splits.items()},
        "ridge_validation_r2": {str(key): value for key, value in ridge_scores.items()},
        "selected_ridge": selected_ridge,
        "results": results, "oracle_compression_ceilings": oracle_pca,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    controller = {
        "definition": report["definition"], "x_layer": args.x_layer,
        "target_layer": 23, "subset": args.subset, "split_seed": args.seed,
        "models": {name: cpu_payload(model) for name, model in models.items()
                   if name not in ("zero", "mean")},
        "mean_x": mean_x.cpu().to(torch.float16), "mean_c": mean_c.cpu().to(torch.float16),
    }
    torch.save(controller, args.controller_out)
    print(json.dumps({
        "n_problems": len(rows), "selected_ridge": selected_ridge,
        "test": {name: value["token_weighted"] for name, value in results.items()},
        "oracle": {name: value["token_weighted"] for name, value in oracle_pca.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
