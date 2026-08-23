from scripts.run_online_paired_residual import (
    answers_equivalent,
    arm_seed,
    boxed,
    schedule_active,
)


def test_boxed_handles_nested_latex():
    assert boxed(r"first \boxed{1}, final \boxed{\frac{\pi}{2}}") == r"\frac{\pi}{2}"


def test_math_equivalence_is_not_only_string_match():
    assert answers_equivalent(r"\frac{1}{2}", "0.5")
    assert not answers_equivalent("2", "3")


def test_refresh_schedules():
    assert [i for i in range(10) if schedule_active("every4", i)] == [0, 4, 8]
    assert [i for i in range(10) if schedule_active("first8", i)] == list(range(8))


def test_arm_seed_is_stable_and_arm_specific():
    first = arm_seed(83, 288, 25, 0.5, "every1", "random")
    assert first == arm_seed(83, 288, 25, 0.5, "every1", "random")
    assert first != arm_seed(83, 288, 25, 0.5, "every2", "random")
