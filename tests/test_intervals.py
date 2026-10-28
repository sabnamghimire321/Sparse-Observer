import numpy as np
import pytest

from sparse_observer.evaluation.intervals import clopper_pearson, pool_aware_wilson, wilson


def test_wilson_matches_textbook_value():
    low, high = wilson(5, 10, level=0.95)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)


def test_clopper_pearson_matches_textbook_value():
    low, high = clopper_pearson(5, 10, level=0.95)
    assert low == pytest.approx(0.1871, abs=1e-3)
    assert high == pytest.approx(0.8129, abs=1e-3)


def test_clopper_pearson_zero_successes_has_closed_form():
    low, high = clopper_pearson(0, 10, level=0.95)
    assert low == 0.0
    assert high == pytest.approx(1 - 0.025 ** (1 / 10))


def test_wilson_is_symmetric_in_successes_and_failures():
    low, high = wilson(3, 10)
    low_m, high_m = wilson(7, 10)
    assert low == pytest.approx(1 - high_m)
    assert high == pytest.approx(1 - low_m)


def test_clopper_pearson_is_wider_than_wilson():
    w, c = wilson(3, 10), clopper_pearson(3, 10)
    assert c[1] - c[0] > w[1] - w[0]


def test_pool_aware_interval_is_wider_and_converges_for_a_huge_pool():
    w = wilson(3, 10)
    p = pool_aware_wilson(3, 10, 40)
    assert p[1] - p[0] > w[1] - w[0]
    assert np.allclose(pool_aware_wilson(3, 10, 1e9), w, atol=1e-6)


def test_intervals_narrow_as_labels_increase():
    widths = [wilson(n * 3 // 10, n)[1] - wilson(n * 3 // 10, n)[0] for n in (10, 20, 40)]
    assert widths[0] > widths[1] > widths[2]


def test_no_labels_gives_the_whole_unit_interval():
    assert wilson(0, 0) == (0.0, 1.0)
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    assert pool_aware_wilson(0, 0, 40) == (0.0, 1.0)
