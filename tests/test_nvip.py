import torch

from pct.nvip import (
    confidence_nvip_projection,
    sequential_audit_decision,
    teacher_advantage_confidence,
    wilson_interval,
)


def test_wilson_interval_contains_empirical_rate():
    lower, upper = wilson_interval(torch.tensor([0, 4, 8]), 8)
    empirical = torch.tensor([0.0, 0.5, 1.0])
    assert torch.all(lower <= empirical)
    assert torch.all(empirical <= upper)
    assert torch.all((lower >= 0) & (upper <= 1))


def test_teacher_confidence_triage_and_priority():
    p = torch.tensor([0.7, 0.3]); q = torch.tensor([0.2, 0.8])
    decision = teacher_advantage_confidence(
        p, q, torch.tensor([1.0, 0.0]), torch.tensor([0.01, 0.01]), delta=0.01
    )
    assert decision["decision"] == "harmful"
    assert decision["upper"] < -0.01
    assert torch.all(decision["allocation_priority"] > 0)


def test_confidence_projection_keeps_robustly_safe_teacher():
    p = torch.tensor([0.5, 0.3, 0.2]); q = torch.tensor([0.7, 0.2, 0.1])
    values = torch.tensor([1.0, 0.4, 0.0]); radius = torch.full_like(values, 0.01)
    projected, metrics = confidence_nvip_projection(p, q, values, radius)
    assert torch.equal(projected, q)
    assert not metrics["repaired"].item()


def test_confidence_projection_repairs_to_robust_boundary():
    p = torch.tensor([0.5, 0.3, 0.2]); q = torch.tensor([0.1, 0.3, 0.6])
    values = torch.tensor([1.0, 0.4, 0.0]); radius = torch.full_like(values, 0.03)
    projected, metrics = confidence_nvip_projection(p, q, values, radius)
    assert metrics["teacher_robust_margin"] < 0
    assert metrics["projected_robust_margin"] >= -2e-6
    assert metrics["repair_kl"] > 0
    assert torch.allclose(projected.sum(), torch.tensor(1.0), atol=1e-6)


def test_more_uncertainty_requires_at_least_as_much_repair():
    p = torch.tensor([0.5, 0.3, 0.2]); q = torch.tensor([0.1, 0.3, 0.6])
    values = torch.tensor([1.0, 0.4, 0.0])
    _, low = confidence_nvip_projection(p, q, values, torch.full_like(values, 0.01))
    _, high = confidence_nvip_projection(p, q, values, torch.full_like(values, 0.08))
    assert high["repair_kl"] >= low["repair_kl"] - 1e-6


def test_sequential_audit_allocates_to_largest_relevant_uncertainty():
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.2, 0.7, 0.1])
    result = sequential_audit_decision(
        p, q, torch.tensor([2, 2, 2]), torch.tensor([4, 4, 4]), delta=0.01
    )
    assert result["decision"] == "ambiguous"
    assert result["next_action"] in (0, 1)
    assert result["next_action"] != 2


def test_sequential_audit_abstains_at_budget_limit():
    result = sequential_audit_decision(
        torch.tensor([0.7, 0.3]),
        torch.tensor([0.3, 0.7]),
        torch.tensor([2, 2]),
        torch.tensor([4, 4]),
        max_trials=4,
    )
    assert result["decision"] == "abstain"
    assert result["next_action"] is None
