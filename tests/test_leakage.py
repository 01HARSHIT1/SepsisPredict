from src.evaluation.leakage import run_leakage_checks


def test_leakage_checks_pass():
    passed = run_leakage_checks()
    assert "observation_state_ignores_future_values" in passed
    assert "backward_mean_does_not_treat_nan_as_zero" in passed
    assert len(passed) >= 6
