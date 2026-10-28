"""Confidence intervals for a proportion, used to ask how reliable a few labels are."""

import numpy as np
from scipy.stats import beta, norm

LEVEL = 0.90


def wilson(k, n, level=LEVEL):
    if n == 0:
        return 0.0, 1.0
    z = norm.ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(center - half, 0.0)), float(min(center + half, 1.0))


def clopper_pearson(k, n, level=LEVEL):
    if n == 0:
        return 0.0, 1.0
    alpha = 1 - level
    low = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(low), float(high)


def pool_aware_wilson(k, n, pool_size, level=LEVEL):
    """Wilson interval for the mean of a finite evaluation pool of `pool_size` items.

    The target has its own sampling variance (about 1/pool_size) on top of the label
    variance (1/n), so the effective size is n * N / (n + N). Derived for this study,
    not a published method; the coverage experiment checks it.
    """
    if n == 0:
        return 0.0, 1.0
    n_eff = n * pool_size / (n + pool_size)
    return wilson(k / n * n_eff, n_eff, level)
