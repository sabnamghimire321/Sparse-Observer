import numpy as np
from scipy.stats import beta, binom

from sparse_observer.experiments.coverage import eb_interval
from sparse_observer.models.empirical_bayes import GRID

POOL = 50
MATRIX = binom.pmf(np.arange(POOL + 1)[None, :], POOL, GRID[:, None])
PRIOR = beta.pdf(GRID, 2, 4) / beta.pdf(GRID, 2, 4).sum()


def test_interval_is_ordered_and_inside_the_unit_interval():
    for k, n in [(0, 1), (1, 1), (3, 10), (5, 10), (31, 31), (0, 31)]:
        low, high = eb_interval(PRIOR, k, n, MATRIX)
        assert 0.0 <= low <= high <= 1.0


def test_more_successes_move_the_interval_up():
    intervals = [eb_interval(PRIOR, k, 10, MATRIX) for k in (0, 3, 6, 10)]
    for (low_a, high_a), (low_b, high_b) in zip(intervals, intervals[1:]):
        assert low_a <= low_b and high_a <= high_b


def test_interval_narrows_as_labels_increase_at_the_same_rate():
    cases = ((3, 10), (6, 20), (12, 40))
    widths = [h - l for l, h in (eb_interval(PRIOR, k, n, MATRIX) for k, n in cases)]
    assert widths[0] > widths[1] > widths[2]


def test_interval_is_calibrated_when_the_data_follow_the_model():
    rng = np.random.default_rng(0)
    n, trials, hits = 7, 3000, 0
    for _ in range(trials):
        rate = rng.choice(GRID, p=PRIOR)
        k = rng.binomial(n, rate)
        theta = rng.binomial(POOL, rate) / POOL
        low, high = eb_interval(PRIOR, k, n, MATRIX)
        hits += low - 1e-12 <= theta <= high + 1e-12
    assert 0.88 <= hits / trials <= 0.97
