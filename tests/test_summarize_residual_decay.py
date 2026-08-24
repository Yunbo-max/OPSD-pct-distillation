from scripts.summarize_residual_decay import exponential_fit, first_crossing


def test_exponential_fit_recovers_time_constant():
    slope, _, tau = exponential_fit([(0, 1.0), (1, 0.5), (2, 0.25)])
    assert slope < 0
    assert abs(tau - 1.0 / 0.6931471805599453) < 1e-6


def test_first_crossing_interpolates_between_tokens():
    curve = [
        {"lag": 0, "signal": 1.0},
        {"lag": 1, "signal": 0.25},
        {"lag": 2, "signal": 0.1},
    ]
    assert abs(first_crossing(curve, "signal", 0.5) - 2.0 / 3.0) < 1e-12


def test_first_crossing_returns_none_without_crossing():
    curve = [{"lag": 0, "signal": 1.0}, {"lag": 1, "signal": 0.75}]
    assert first_crossing(curve, "signal", 0.5) is None
