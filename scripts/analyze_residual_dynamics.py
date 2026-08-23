#!/usr/bin/env python3
"""Spatial/temporal rank and held-out reconstruction for residual trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def energy_rank(values: torch.Tensor, threshold: float) -> int:
    energy = values.square()
    return int((energy.cumsum(0) < threshold * energy.sum()).sum()) + 1


def corr(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left - left.mean(), right - right.mean()
    return float((left @ right) / ((left.norm() * right.norm()).clamp_min(1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--max_rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=89)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    trajectories = [value.float() for value in payload["trajectories"]]
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(trajectories), generator=generator)
    split = int(args.train_fraction * len(trajectories))
    train_ids, test_ids = permutation[:split], permutation[split:]
    train = torch.cat([trajectories[int(index)] for index in train_ids], dim=0)
    mean = train.mean(dim=0, keepdim=True)
    centered = train - mean
    q = min(args.max_rank, centered.shape[0], centered.shape[1])
    _, singular, spatial = torch.pca_lowrank(centered, q=q, center=False, niter=4)

    heldout = torch.cat([trajectories[int(index)] for index in test_ids], dim=0)
    heldout_centered = heldout - mean
    total = heldout_centered.square().sum()
    reconstruction = {}
    for rank in (1, 2, 4, 8, 16, 32, 64):
        if rank > spatial.shape[1]:
            continue
        basis = spatial[:, :rank]
        projected = heldout_centered @ basis @ basis.T
        reconstruction[str(rank)] = float(1.0 - (heldout_centered - projected).square().sum() / total)

    # Temporal functional modes from per-trajectory hidden-space Gram matrices.
    temporal_gram = torch.zeros((128, 128))
    used = 0
    for index in train_ids:
        value = trajectories[int(index)]
        if value.shape[0] == 128:
            temporal_gram += value @ value.T
            used += 1
    temporal_eigen = torch.linalg.eigvalsh(temporal_gram).flip(0).clamp_min(0).sqrt()

    # Hankel-style one-step dynamics in the learned spatial coordinates.
    coordinates = [(value - mean) @ spatial for value in trajectories]
    past = torch.cat([coordinates[int(index)][:-1] for index in train_ids], dim=0)
    future = torch.cat([coordinates[int(index)][1:] for index in train_ids], dim=0)
    transition = torch.linalg.lstsq(past, future).solution
    predicted = past @ transition
    one_step_r2 = float(1.0 - (future - predicted).square().sum() / future.square().sum())
    hankel = torch.cat([past, future], dim=1)
    hankel_singular = torch.linalg.svdvals(hankel)

    # Active-vs-inert generalized eigenspace in the learned spatial basis.
    train_te = torch.tensor([payload["metadata"][int(index)]["te"] for index in train_ids])
    low, high = torch.quantile(train_te, torch.tensor([0.25, 0.75]))
    active_ids = [int(index) for index in train_ids if payload["metadata"][int(index)]["te"] >= high]
    inert_ids = [int(index) for index in train_ids if payload["metadata"][int(index)]["te"] <= low]

    def covariance(indices, weighted):
        chunks, weights = [], []
        for index in indices:
            value = coordinates[index]
            chunks.append(value)
            weight = max(0.0, payload["metadata"][index]["te"]) if weighted else 1.0
            weights.append(torch.full((value.shape[0],), max(weight, 1e-6)))
        matrix, weight = torch.cat(chunks), torch.cat(weights)
        return (matrix.T * weight) @ matrix / weight.sum()

    active_cov = covariance(active_ids, True)
    inert_cov = covariance(inert_ids, False)
    regularization = 1e-3 * inert_cov.trace() / inert_cov.shape[0]
    denominator = inert_cov + regularization * torch.eye(inert_cov.shape[0])
    eigen_d, eigen_q = torch.linalg.eigh(denominator)
    whitening = eigen_q @ torch.diag(eigen_d.clamp_min(1e-8).rsqrt()) @ eigen_q.T
    contrast = whitening @ active_cov @ whitening
    contrast_values, contrast_vectors = torch.linalg.eigh(contrast)
    order = contrast_values.argsort(descending=True)
    contrast_values = contrast_values[order]
    contrast_basis_small = whitening @ contrast_vectors[:, order]
    contrast_basis_small = torch.linalg.qr(contrast_basis_small).Q
    contrast_basis = spatial @ contrast_basis_small

    heldout_te = torch.tensor([payload["metadata"][int(index)]["te"] for index in test_ids])
    heldout_rescue = torch.tensor([payload["metadata"][int(index)]["rescue"] for index in test_ids])
    contrastive_heldout = {}
    for rank in (1, 2, 4, 8, 16, 32):
        basis = contrast_basis[:, :rank]
        energies = torch.tensor([
            float((trajectories[int(index)] @ basis).square().mean()) for index in test_ids
        ])
        contrastive_heldout[str(rank)] = {
            "energy_te_correlation": corr(energies, heldout_te),
            "energy_rescue_correlation": corr(energies, heldout_rescue),
        }

    report = {
        "layer": payload["layer"],
        "n_train": len(train_ids), "n_test": len(test_ids),
        "spatial_rank90": energy_rank(singular, 0.90),
        "spatial_rank95": energy_rank(singular, 0.95),
        "temporal_rank90": energy_rank(temporal_eigen, 0.90),
        "temporal_rank95": energy_rank(temporal_eigen, 0.95),
        "heldout_spatial_explained_variance": reconstruction,
        "linear_dynamics_one_step_r2": one_step_r2,
        "hankel_rank90": energy_rank(hankel_singular, 0.90),
        "hankel_rank95": energy_rank(hankel_singular, 0.95),
        "active_te_threshold": float(high),
        "inert_te_threshold": float(low),
        "contrastive_eigenvalues_top10": [float(value) for value in contrast_values[:10]],
        "contrastive_heldout": contrastive_heldout,
        "train_ids": [int(value) for value in train_ids],
        "test_ids": [int(value) for value in test_ids],
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    basis_out = str(Path(args.out).with_suffix(".basis.pt"))
    torch.save(
        {
            "mean": mean.half(), "pca_basis": spatial.half(),
            "contrastive_basis": contrast_basis.half(), "report": report,
        },
        basis_out,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
