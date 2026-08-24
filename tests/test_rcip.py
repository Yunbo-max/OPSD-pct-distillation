import torch

from pct.rcip import (
    compatibility,
    reward_compatible_i_projection,
    sampled_compatibility,
    sampled_reward_direction,
    robust_fisher_cone_projection,
    node_value_i_projection,
)


def test_sampled_compatibility_matches_explicit_direction():
    p = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    q = torch.tensor([[0.3, 0.6, 0.1], [0.4, 0.4, 0.2]])
    tokens = torch.tensor([0, 2])
    advantages = torch.tensor([1.5, -0.7])
    direction = sampled_reward_direction(p, tokens, advantages)
    assert torch.allclose(
        sampled_compatibility(p, q, tokens, advantages),
        compatibility(p, q, direction),
        atol=1e-7,
    )


def test_safe_teacher_is_unchanged():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.7, 0.2, 0.1]])
    direction = torch.tensor([[1.0, -0.5, -0.5]])
    projected, lmbda = reward_compatible_i_projection(p, q, direction)
    assert torch.equal(projected, q)
    assert lmbda.item() == 0


def test_harmful_teacher_projects_to_constraint_boundary():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.2, 0.5, 0.3]])
    direction = torch.tensor([[1.0, -0.5, -0.5]])
    projected, lmbda = reward_compatible_i_projection(p, q, direction)
    assert lmbda.item() > 0
    assert abs(compatibility(p, projected, direction).item()) < 1e-6
    assert torch.allclose(projected.sum(-1), torch.ones(1), atol=1e-7)


def test_zero_reward_direction_leaves_teacher_unchanged():
    p = torch.tensor([[0.2, 0.3, 0.5]])
    q = torch.tensor([[0.4, 0.4, 0.2]])
    projected, lmbda = reward_compatible_i_projection(p, q, torch.zeros_like(p))
    assert torch.equal(projected, q)
    assert lmbda.item() == 0


def test_robust_fisher_keeps_robustly_aligned_teacher():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.7, 0.2, 0.1])
    rewards = torch.tensor([[1.0, -0.5, -1.75], [0.8, -0.2, -1.7]])
    projected, _, metrics = robust_fisher_cone_projection(p, q, rewards, kappa=0.5)
    assert torch.allclose(projected, q, atol=1e-6)
    assert metrics["teacher_robust_margin"] >= 0


def test_robust_fisher_projects_conflict_into_cone():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.1, 0.4, 0.5])
    rewards = torch.tensor([[1.0, -0.5, -1.75], [0.8, -0.2, -1.7], [1.1, -0.6, -1.85]])
    projected, tangent, metrics = robust_fisher_cone_projection(
        p, q, rewards, kappa=1.0, iterations=2000
    )
    assert torch.allclose(projected.sum(), torch.tensor(1.0), atol=1e-6)
    assert metrics["teacher_robust_margin"] < 0
    assert metrics["robust_margin"] >= -2e-5
    assert metrics["fisher_distortion"] > 0
    assert torch.isfinite(tangent).all()


def test_nvip_leaves_helpful_teacher_unchanged():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.7, 0.2, 0.1]])
    values = torch.tensor([[1.0, 0.4, 0.0]])
    projected, lmbda, metrics = node_value_i_projection(p, q, values)
    assert torch.equal(projected, q)
    assert lmbda.item() == 0
    assert not metrics["repaired"].item()


def test_nvip_minimally_repairs_harmful_teacher_to_student_value():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.1, 0.3, 0.6]])
    values = torch.tensor([[1.0, 0.4, 0.0]])
    projected, lmbda, metrics = node_value_i_projection(p, q, values)
    assert lmbda.item() > 0
    assert metrics["repair_kl"].item() > 0
    assert torch.allclose(metrics["projected_value"], metrics["student_value"], atol=1e-6)
    assert torch.allclose(projected.sum(-1), torch.ones(1), atol=1e-7)
