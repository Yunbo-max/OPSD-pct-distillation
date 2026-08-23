#!/usr/bin/env python3
"""Held-out separable functional PCA for paired-residual trajectories.

The tensor is decomposed into temporal functions and hidden-space directions;
reported variance is always measured on problem-level held-out trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def rank_for(values: torch.Tensor, fraction: float) -> int:
    energy = values.clamp_min(0)
    return int((energy.cumsum(0) < fraction * energy.sum()).sum()) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--max_spatial_rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=89)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    trajectories = torch.stack([value.float() for value in payload["trajectories"]])
    n_examples, n_tokens, hidden = trajectories.shape
    order = torch.randperm(n_examples, generator=torch.Generator().manual_seed(args.seed))
    split = int(args.train_fraction * n_examples)
    train_ids, test_ids = order[:split], order[split:]
    train, test = trajectories[train_ids], trajectories[test_ids]

    # A time-varying mean prevents the dominant generic position profile from
    # being counted as a correctness mode.
    mean_trajectory = train.mean(0, keepdim=True)
    train_centered, test_centered = train - mean_trajectory, test - mean_trajectory
    temporal_cov = torch.einsum("ntd,nsd->ts", train_centered, train_centered)
    temporal_values, temporal_basis = torch.linalg.eigh(temporal_cov)
    temporal_values, temporal_basis = temporal_values.flip(0), temporal_basis.flip(1)

    flattened = train_centered.reshape(-1, hidden)
    spatial_rank = min(args.max_spatial_rank, *flattened.shape)
    _, spatial_singular, spatial_basis = torch.pca_lowrank(
        flattened, q=spatial_rank, center=False, niter=4
    )

    total = test_centered.square().sum().clamp_min(1e-12)
    reconstruction = {}
    temporal_ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    spatial_ranks = [1, 2, 4, 8, 16, 32, 64]
    for tr in temporal_ranks:
        if tr > n_tokens:
            continue
        phi = temporal_basis[:, :tr]
        temporally_projected = torch.einsum("ts,nsd->ntd", phi @ phi.T, test_centered)
        for sr in spatial_ranks:
            if sr > spatial_basis.shape[1]:
                continue
            u = spatial_basis[:, :sr]
            reconstructed = temporally_projected @ u @ u.T
            explained = 1.0 - (test_centered - reconstructed).square().sum() / total
            reconstruction[f"time={tr}:space={sr}"] = float(explained)

    report = {
        "layer": payload.get("layer"), "n_examples": n_examples,
        "n_train": len(train_ids), "n_test": len(test_ids),
        "tokens": n_tokens, "hidden": hidden,
        "centering": "time_varying_train_mean",
        "temporal_rank90": rank_for(temporal_values, 0.90),
        "temporal_rank95": rank_for(temporal_values, 0.95),
        "spatial_rank90": rank_for(spatial_singular.square(), 0.90),
        "spatial_rank95": rank_for(spatial_singular.square(), 0.95),
        "heldout_explained_variance": reconstruction,
        "train_ids": train_ids.tolist(), "test_ids": test_ids.tolist(),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    torch.save({
        "mean_trajectory": mean_trajectory.half(),
        "temporal_basis": temporal_basis.half(),
        "spatial_basis": spatial_basis.half(), "report": report,
    }, Path(args.out).with_suffix(".basis.pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
