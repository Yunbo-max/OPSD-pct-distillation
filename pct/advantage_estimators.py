"""Low-variance estimators for node-level privileged-teacher advantage."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdvantageInterval:
    estimate: float
    lower: float
    upper: float
    radius: float
    samples: int


def jordan_signed_measures(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    tolerance: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TV mass and normalized positive/negative parts of q-p."""
    if student_probs.ndim != 1 or student_probs.shape != teacher_probs.shape:
        raise ValueError("student_probs and teacher_probs must be matching vectors")
    p = student_probs.float().clamp_min(0); p = p / p.sum()
    q = teacher_probs.float().clamp_min(0); q = q / q.sum()
    difference = q - p
    positive = difference.clamp_min(0)
    negative = (-difference).clamp_min(0)
    alpha = positive.sum()
    if not torch.allclose(alpha, negative.sum(), atol=tolerance, rtol=0):
        raise ValueError("positive and negative Jordan masses do not match")
    if float(alpha) <= tolerance:
        zeros = torch.zeros_like(p)
        return alpha, zeros, zeros
    return alpha, positive / alpha, negative / alpha


def empirical_bernstein_interval(
    samples: torch.Tensor,
    *,
    low: float,
    high: float,
    confidence: float = 0.95,
) -> AdvantageInterval:
    """A finite-sample empirical-Bernstein interval for bounded samples."""
    values = samples.float().flatten()
    if values.numel() < 2:
        raise ValueError("at least two samples are required")
    if not 0 < confidence < 1 or high <= low:
        raise ValueError("invalid confidence or sample bounds")
    if bool(((values < low) | (values > high)).any()):
        raise ValueError("sample outside declared bounds")
    n = values.numel()
    log_term = math.log(3.0 / (1.0 - confidence))
    variance = values.var(unbiased=False)
    width = high - low
    radius = float(torch.sqrt(2 * variance * log_term / n) + 3 * width * log_term / n)
    mean = float(values.mean())
    lower = max(low, mean - radius)
    upper = min(high, mean + radius)
    return AdvantageInterval(mean, lower, upper, max(mean - lower, upper - mean), n)


def signed_terminal_advantage(
    plus_rewards: torch.Tensor,
    minus_rewards: torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    confidence: float = 0.95,
) -> AdvantageInterval:
    """Estimate alpha * E[R+ - R-] from paired terminal outcomes."""
    if plus_rewards.shape != minus_rewards.shape:
        raise ValueError("plus_rewards and minus_rewards must have matching shapes")
    signed = plus_rewards.float() - minus_rewards.float()
    raw = empirical_bernstein_interval(signed, low=-1, high=1, confidence=confidence)
    scale = float(alpha)
    return AdvantageInterval(
        scale * raw.estimate, scale * raw.lower, scale * raw.upper,
        scale * raw.radius, raw.samples,
    )


def doubly_robust_advantage(
    proxy_differences: torch.Tensor,
    audited_residuals: torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    confidence: float = 0.95,
) -> AdvantageInterval:
    """Estimate advantage with a cheap value proxy and unbiased correction.

    ``proxy_differences`` are independent samples of V(s+)-V(s-) in [-1,1].
    ``audited_residuals`` are independent samples of
    (R+-R-) - (V(s+)-V(s-)) in [-2,2].  Using disjoint sampling pools makes
    the sum unbiased without trusting the value model and permits a simple
    union-bound confidence interval.
    """
    component_confidence = 1.0 - (1.0 - confidence) / 2.0
    proxy = empirical_bernstein_interval(
        proxy_differences, low=-1, high=1, confidence=component_confidence
    )
    residual = empirical_bernstein_interval(
        audited_residuals, low=-2, high=2, confidence=component_confidence
    )
    estimate = proxy.estimate + residual.estimate
    lower = max(-1.0, proxy.lower + residual.lower)
    upper = min(1.0, proxy.upper + residual.upper)
    scale = float(alpha)
    return AdvantageInterval(
        scale * estimate, scale * lower, scale * upper,
        scale * max(estimate - lower, upper - estimate),
        proxy.samples + residual.samples,
    )


def fit_ridge_value_head(
    features: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Fit a small linear prefix-value head; final entry is the intercept."""
    x = features.float()
    y = outcomes.float().flatten()
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be [N,D] and outcomes [N]")
    augmented = torch.cat((x, torch.ones(x.shape[0], 1, device=x.device)), dim=1)
    penalty = torch.eye(augmented.shape[1], device=x.device) * ridge
    penalty[-1, -1] = 0
    return torch.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)


def predict_ridge_value(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Predict a bounded prefix success probability from a fitted ridge head."""
    x = features.float()
    return (x @ weights[:-1] + weights[-1]).clamp(0, 1)
