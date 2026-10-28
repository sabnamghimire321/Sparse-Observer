"""Brier score, expected calibration error, and Brier ALC (a weighted mean of Brier over label budgets)."""

import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
# trapezoid areas over equally spaced log2(n + 1) positions: endpoints 0.1, interior 0.2
ALC_WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
N_BINS = 10


def brier(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece(p, y, n_bins=N_BINS):
    """Expected calibration error with equal-width bins, weighted by the number of responses per bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    # a probability of exactly 1.0 falls in the last bin
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def brier_alc(brier_by_budget):
    """Weighted sum of the Brier scores at budgets 0, 1, 3, 7, 15 and 31 (lower is better)."""
    scores = np.asarray(brier_by_budget, dtype=float)
    if scores.shape[-1] != len(BUDGETS):
        raise ValueError(f"expected {len(BUDGETS)} budget scores, got {scores.shape[-1]}")
    return scores @ ALC_WEIGHTS


def main():
    worked_example = [0.22, 0.18, 0.145, 0.12, 0.10, 0.09]
    print(f"example ALC: {brier_alc(worked_example):.4f} (expected 0.140)")
    print(f"trapezoid check: {np.trapezoid(worked_example, dx=1.0) / 5:.4f}")

    y = np.random.default_rng(0).integers(0, 2, size=10000)
    print(f"constant 0.5 Brier: {brier(np.full(10000, 0.5), y):.4f} (expected 0.25)")
    print(f"perfect Brier: {brier(y, y):.4f} (expected 0)")
    print(f"ECE, 0.5 on a fair coin: {ece(np.full(10000, 0.5), y):.4f} (expected near 0)")
    print(f"ECE, constant 0.9 on all ones: {ece(np.full(100, 0.9), np.ones(100)):.4f} (expected 0.1)")


if __name__ == "__main__":
    main()
