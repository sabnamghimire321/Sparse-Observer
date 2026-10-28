import numpy as np
import pytest

from sparse_observer.evaluation.metrics import ALC_WEIGHTS, BUDGETS, brier, brier_alc, ece


def test_alc_matches_the_worked_example():
    scores = [0.22, 0.18, 0.145, 0.12, 0.10, 0.09]
    assert brier_alc(scores) == pytest.approx(0.14)


def test_alc_weights_are_trapezoid_areas_over_equal_steps():
    scores = [0.22, 0.18, 0.145, 0.12, 0.10, 0.09]
    assert brier_alc(scores) == pytest.approx(np.trapezoid(scores, dx=1.0) / 5)


def test_alc_weights_sum_to_one_and_match_budgets():
    assert ALC_WEIGHTS.sum() == pytest.approx(1.0)
    assert len(ALC_WEIGHTS) == len(BUDGETS)


def test_brier_of_constant_half_is_a_quarter():
    y = np.array([0, 1, 1, 0, 1])
    assert brier(np.full(5, 0.5), y) == pytest.approx(0.25)


def test_brier_of_perfect_predictions_is_zero():
    y = np.array([0.0, 1.0, 1.0])
    assert brier(y, y) == 0.0


def test_ece_of_constant_ninety_on_all_ones_is_a_tenth():
    assert ece(np.full(100, 0.9), np.ones(100)) == pytest.approx(0.1)


def test_ece_puts_probability_one_in_the_last_bin():
    assert ece(np.ones(10), np.ones(10)) == pytest.approx(0.0)


def test_alc_rejects_a_wrong_number_of_budgets():
    with pytest.raises(ValueError):
        brier_alc([0.2, 0.2])
