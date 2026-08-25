import torch

from pct.advantage_estimators import (
    doubly_robust_advantage,
    fit_ridge_value_head,
    jordan_signed_measures,
    predict_ridge_value,
    signed_terminal_advantage,
)


def test_jordan_decomposition_reconstructs_teacher_difference():
    p = torch.tensor([0.6, 0.3, 0.1]); q = torch.tensor([0.2, 0.5, 0.3])
    alpha, positive, negative = jordan_signed_measures(p, q)
    assert torch.allclose(alpha * (positive - negative), q - p, atol=1e-6)
    assert torch.allclose(alpha, 0.4 * torch.ones(()), atol=1e-6)


def test_signed_terminal_advantage_respects_tv_bound():
    result = signed_terminal_advantage(
        torch.ones(16), torch.zeros(16), alpha=0.3, confidence=0.95
    )
    assert result.estimate == 0.3
    assert -0.3 <= result.lower <= result.upper <= 0.3


def test_doubly_robust_is_exact_with_perfect_proxy():
    proxy = torch.full((32,), 0.25)
    residual = torch.zeros(8)
    result = doubly_robust_advantage(proxy, residual, alpha=0.4)
    assert abs(result.estimate - 0.1) < 1e-6


def test_ridge_value_head_fits_simple_signal():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([0.0, 0.3, 0.6, 0.9])
    w = fit_ridge_value_head(x, y, ridge=1e-5)
    prediction = predict_ridge_value(x, w)
    assert torch.mean((prediction - y).square()) < 1e-5
