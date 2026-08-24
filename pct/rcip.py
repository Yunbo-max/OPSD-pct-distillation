"""Reward-Compatible I-Projection (RCIP) primitives.

The projection keeps a privileged teacher distribution as close as possible in
forward KL while constraining its student-logit update to have non-negative
inner product with an estimated reward-ascent direction.
"""
from __future__ import annotations

import torch


def sampled_reward_direction(
    student_probs: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Return A * (one_hot(y) - p), the score-function reward direction."""
    direction = -student_probs * advantages.unsqueeze(-1)
    return direction.scatter_add(
        -1, sampled_token_ids.unsqueeze(-1), advantages.unsqueeze(-1)
    )


def compatibility(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    reward_direction: torch.Tensor,
) -> torch.Tensor:
    """Inner product between the proposed teacher update and reward direction."""
    return ((teacher_probs - student_probs) * reward_direction).sum(-1)


def sampled_compatibility(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Memory-light chi for d_R=A(e_y-p), without materializing one-hot tensors."""
    teacher_minus_student = teacher_probs - student_probs
    sampled = teacher_minus_student.gather(-1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)
    baseline = (student_probs * teacher_minus_student).sum(-1)
    return advantages * (sampled - baseline)


def reward_compatible_i_projection(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    reward_direction: torch.Tensor,
    *,
    iterations: int = 64,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL I-project q onto <r-p,d_R> >= 0.

    Returns `(q_star, lambda)` with lambda expressed for the original (not
    internally normalized) reward direction. Safe rows are unchanged and have
    lambda zero. Bisection is vectorized over every leading dimension.
    """
    if student_probs.shape != teacher_probs.shape or student_probs.shape != reward_direction.shape:
        raise ValueError("student, teacher, and reward direction must have identical shapes")
    if student_probs.ndim < 1:
        raise ValueError("probability tensors must have a vocabulary dimension")

    original_shape = student_probs.shape
    vocab = original_shape[-1]
    p = student_probs.reshape(-1, vocab).float()
    q = teacher_probs.reshape(-1, vocab).float()
    d = reward_direction.reshape(-1, vocab).float()
    scale = d.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    dn = d / scale
    target = (p * dn).sum(-1)
    chi = ((q - p) * dn).sum(-1)
    active = chi < -tolerance
    log_q = q.clamp_min(torch.finfo(torch.float32).tiny).log()

    def expectation(lmbda: torch.Tensor) -> torch.Tensor:
        projected = torch.softmax(log_q + lmbda.unsqueeze(-1) * dn, dim=-1)
        return (projected * dn).sum(-1)

    low = torch.zeros_like(target)
    high = torch.ones_like(target)
    for _ in range(32):
        insufficient = active & (expectation(high) < target)
        if not bool(insufficient.any()):
            break
        high = torch.where(insufficient, high * 2, high)
    for _ in range(iterations):
        middle = (low + high) / 2
        below = expectation(middle) < target
        low = torch.where(active & below, middle, low)
        high = torch.where(active & below, high, middle)
    lambda_normalized = torch.where(active, high, torch.zeros_like(high))
    q_star = torch.softmax(log_q + lambda_normalized.unsqueeze(-1) * dn, dim=-1)
    q_star = torch.where(active.unsqueeze(-1), q_star, q)
    lambda_original = lambda_normalized / scale.squeeze(-1)
    return q_star.reshape(original_shape).to(teacher_probs.dtype), lambda_original.reshape(original_shape[:-1])


def fisher_teacher_tangent(student_probs: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """Centered log-density ratio representing q from the Fisher tangent at p."""
    p = student_probs.float().clamp_min(torch.finfo(torch.float32).tiny)
    q = teacher_probs.float().clamp_min(torch.finfo(torch.float32).tiny)
    value = q.log() - p.log()
    return value - (p * value).sum(-1, keepdim=True)


def fisher_inner(student_probs: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (student_probs.float() * left.float() * right.float()).sum(-1)


def fisher_exponential_map(student_probs: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    logits = student_probs.float().clamp_min(torch.finfo(torch.float32).tiny).log() + tangent.float()
    return torch.softmax(logits, -1).to(student_probs.dtype)


def robust_fisher_cone_projection(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    reward_tangents: torch.Tensor,
    *,
    kappa: float = 1.0,
    iterations: int = 256,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Project the teacher score into an empirical robust Fisher compatibility cone.

    This unbatched primitive expects `[vocab]` probabilities and `[samples,vocab]`
    same-state reward score tangents. It solves the low-rank SOCP by projected
    gradient on its Lorentz-cone dual; no vocab-by-vocab covariance is formed.
    """
    if student_probs.ndim != 1 or teacher_probs.shape != student_probs.shape:
        raise ValueError("student_probs and teacher_probs must be matching vectors")
    if reward_tangents.ndim != 2 or reward_tangents.shape[1] != student_probs.shape[0]:
        raise ValueError("reward_tangents must have shape [samples, vocab]")
    if reward_tangents.shape[0] < 1:
        raise ValueError("at least one reward tangent is required")
    p = student_probs.float().clamp_min(torch.finfo(torch.float32).tiny)
    p = p / p.sum()
    sqrt_p = p.sqrt()
    teacher = fisher_teacher_tangent(p, teacher_probs.float())
    rewards = reward_tangents.float()
    rewards = rewards - (p * rewards).sum(-1, keepdim=True)
    mean = rewards.mean(0)
    centered = rewards - mean
    x0 = sqrt_p * teacher
    a = sqrt_p * mean
    if rewards.shape[0] > 1:
        b = sqrt_p.unsqueeze(0) * centered / (rewards.shape[0] - 1) ** 0.5
    else:
        b = centered
    operator = torch.cat((a.unsqueeze(0), float(kappa) * b), 0)
    source = operator @ x0

    def project_lorentz(value: torch.Tensor) -> torch.Tensor:
        time, space = value[0], value[1:]
        norm = space.norm()
        if bool(norm <= time):
            return value
        if bool(norm <= -time):
            return torch.zeros_like(value)
        projected_time = (time + norm) / 2
        return torch.cat((projected_time.unsqueeze(0), projected_time * space / norm.clamp_min(1e-12)))

    if bool(source[0] >= source[1:].norm() - tolerance):
        projected_x = x0
        dual = torch.zeros(operator.shape[0], device=p.device)
    else:
        gram = operator @ operator.T
        lipschitz = torch.linalg.eigvalsh(gram).amax().clamp_min(1e-12)
        step = 1 / lipschitz
        dual = torch.zeros(operator.shape[0], device=p.device)
        # The dual variable is in the self-dual Lorentz cone. The primal
        # projection is x0 + A^T w for min_w 1/2||A^T w||^2 + w^T A x0.
        for _ in range(iterations):
            updated = project_lorentz(dual - step * (gram @ dual + source))
            if float((updated - dual).norm()) <= tolerance * (1 + float(dual.norm())):
                dual = updated
                break
            dual = updated
        projected_x = x0 + operator.T @ dual
    tangent = projected_x / sqrt_p
    tangent = tangent - (p * tangent).sum()
    q_star = fisher_exponential_map(p, tangent)
    margin = a @ projected_x - float(kappa) * (b @ projected_x).norm()
    diagnostics = {
        "robust_margin": margin,
        "teacher_robust_margin": a @ x0 - float(kappa) * (b @ x0).norm(),
        "fisher_distortion": (projected_x - x0).norm(),
        "dual_norm": dual.norm(),
    }
    return q_star.to(teacher_probs.dtype), tangent.to(teacher_probs.dtype), diagnostics
