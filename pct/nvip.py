"""Confidence-calibrated Node-Value Information Projection (SC-NVIP)."""
from __future__ import annotations

import math

import torch


def wilson_interval(successes: torch.Tensor, trials: torch.Tensor | int, z: float = 1.959963984540054):
    """Wilson score interval for independent Bernoulli branch outcomes."""
    count = successes.float()
    total = torch.as_tensor(trials, device=count.device, dtype=torch.float32)
    phat = count / total.clamp_min(1)
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    radius = z / denom * torch.sqrt(
        (phat * (1 - phat) / total + z * z / (4 * total.square())).clamp_min(0)
    )
    return (center - radius).clamp(0, 1), (center + radius).clamp(0, 1)


def teacher_advantage_confidence(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    value_mean: torch.Tensor,
    value_radius: torch.Tensor,
    *,
    delta: float = 0.01,
) -> dict[str, torch.Tensor | str]:
    """Box-uncertainty interval and tri-state decision for teacher advantage."""
    difference = teacher_probs.float() - student_probs.float()
    estimate = (difference * value_mean.float()).sum()
    radius = (difference.abs() * value_radius.float()).sum()
    lower, upper = estimate - radius, estimate + radius
    if float(lower) > delta:
        decision = "helpful"
    elif float(upper) < -delta:
        decision = "harmful"
    else:
        decision = "ambiguous"
    return {
        "estimate": estimate, "radius": radius, "lower": lower, "upper": upper,
        "decision": decision,
        "allocation_priority": difference.abs() * value_radius.float(),
    }


def sequential_audit_decision(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    successes: torch.Tensor,
    trials: torch.Tensor,
    *,
    delta: float = 0.01,
    z: float = 1.959963984540054,
    max_trials: int = 32,
) -> dict[str, torch.Tensor | str | int | None]:
    """Decide or allocate the next rollout in a sequential SC-NVIP audit.

    Branch values use Wilson intervals.  An ambiguous audit spends its next
    sample on the branch contributing most to the teacher-advantage radius,
    ``|q-p| * epsilon``.  Once every branch reaches ``max_trials``, the audit
    abstains instead of forcing a noisy helpful/harmful label.
    """
    if not (
        student_probs.shape == teacher_probs.shape == successes.shape == trials.shape
        and student_probs.ndim == 1
    ):
        raise ValueError("SC-NVIP audit expects matching one-dimensional action vectors")
    if bool((trials <= 0).any()):
        raise ValueError("every branch needs at least one initial rollout")

    lower, upper = wilson_interval(successes, trials, z=z)
    empirical_mean = successes.float() / trials.float()
    # Wilson intervals are asymmetric near 0/1.  Represent the confidence box
    # by its actual midpoint and half-width, not phat +/- half-width.
    mean = (lower + upper) / 2
    radius = (upper - lower) / 2
    confidence = teacher_advantage_confidence(
        student_probs, teacher_probs, mean, radius, delta=delta
    )
    decision = str(confidence["decision"])
    next_action: int | None = None
    if decision == "ambiguous":
        eligible = trials < max_trials
        if bool(eligible.any()):
            priority = confidence["allocation_priority"].clone()
            priority[~eligible] = -1
            next_action = int(priority.argmax().item())
        else:
            decision = "abstain"
    return {
        **confidence,
        "decision": decision,
        "value_mean": mean,
        "empirical_value_mean": empirical_mean,
        "value_lower": lower,
        "value_upper": upper,
        "value_radius": radius,
        "next_action": next_action,
        "total_rollouts": int(trials.sum().item()),
    }


def confidence_nvip_projection(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    value_mean: torch.Tensor,
    value_radius: torch.Tensor,
    *,
    outer_iterations: int = 64,
    inner_iterations: int = 80,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact low-dimensional KL projection under box-robust node values.

    Solves min KL(r||q) subject to
        (r-p)^T Qhat >= sum_a epsilon_a |r_a-p_a|.
    The fixed-dual subproblem is a KL proximal operator with a weighted L1
    center at p; only scalar inner/outer bisections are required.
    """
    if student_probs.ndim != 1 or not (
        teacher_probs.shape == value_mean.shape == value_radius.shape == student_probs.shape
    ):
        raise ValueError("SC-NVIP currently expects matching action vectors")
    p = student_probs.float().clamp_min(1e-30); p = p / p.sum()
    q = teacher_probs.float().clamp_min(1e-30); q = q / q.sum()
    values = value_mean.float()
    epsilon = value_radius.float().clamp_min(0)

    def margin(r: torch.Tensor) -> torch.Tensor:
        change = r - p
        return (change * values).sum() - (change.abs() * epsilon).sum()

    teacher_margin = margin(q)
    if float(teacher_margin) >= -tolerance:
        return teacher_probs.clone(), {
            "teacher_robust_margin": teacher_margin,
            "projected_robust_margin": teacher_margin,
            "repair_kl": torch.tensor(0.0, device=p.device),
            "repaired": torch.tensor(False, device=p.device),
            "dual": torch.tensor(0.0, device=p.device),
        }

    log_p_over_q = p.log() - q.log()

    def fixed_dual(lmbda: torch.Tensor) -> torch.Tensor:
        base = lmbda * values - log_p_over_q
        width = lmbda * epsilon
        low_c = (base - width).amin() - 100
        high_c = (base + width).amax() + 100
        result = p
        for _ in range(inner_iterations):
            c = (low_c + high_c) / 2
            offset = base - c
            above = offset > width
            below = offset < -width
            candidate = torch.where(
                above,
                q * torch.exp((lmbda * (values - epsilon) - c).clamp(-80, 80)),
                torch.where(
                    below,
                    q * torch.exp((lmbda * (values + epsilon) - c).clamp(-80, 80)),
                    p,
                ),
            )
            total = candidate.sum()
            if float(total) > 1:
                low_c = c
            else:
                high_c = c
            result = candidate
        return result / result.sum()

    low = torch.tensor(0.0, device=p.device); high = torch.tensor(1.0, device=p.device)
    projected = fixed_dual(high)
    for _ in range(40):
        if float(margin(projected)) >= -tolerance:
            break
        high *= 2
        projected = fixed_dual(high)
    else:
        projected = p.clone()
    for _ in range(outer_iterations):
        middle = (low + high) / 2
        candidate = fixed_dual(middle)
        if float(margin(candidate)) < 0:
            low = middle
        else:
            high = middle; projected = candidate
    repair_kl = (projected * (projected.clamp_min(1e-30).log() - q.log())).sum()
    return projected.to(teacher_probs.dtype), {
        "teacher_robust_margin": teacher_margin,
        "projected_robust_margin": margin(projected),
        "repair_kl": repair_kl,
        "repaired": torch.tensor(True, device=p.device),
        "dual": high,
    }
