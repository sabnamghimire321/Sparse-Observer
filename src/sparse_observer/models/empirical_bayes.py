"""Empirical-Bayes predictor: a prior over pair success rates estimated from training benchmarks."""

import numpy as np

from sparse_observer.data.labels import load_labeled_responses
from sparse_observer.models.base import Predictor
from sparse_observer.simulator.protocol import MIN_ITEMS

GRID = np.linspace(0.01, 0.99, 99)
MIN_PAIRS = 5
BANDWIDTH = 0.05
FLOOR = 0.1
CLIP = 0.01


def smooth(rates):
    """Gaussian-kernel density of the pair success rates on GRID, normalized to sum to one."""
    density = np.exp(-0.5 * ((GRID[:, None] - rates[None, :]) / BANDWIDTH) ** 2).sum(axis=1)
    return density / density.sum()


class EmpiricalBayes(Predictor):
    """Posterior mean of a pair's success rate on a grid. `temper` below 1 weakens the
    likelihood of the revealed labels.
    """

    def __init__(self, temper=1.0):
        self.temper = temper
        self.prior = np.full(len(GRID), 1.0 / len(GRID))
        self.n_pairs = {}
        self.name = f"empirical_bayes_t{temper:g}"

    def fit(self, train_benchmarks, rules):
        """Average the smoothed rate distributions of training benchmarks that have at least 5 eligible pairs."""
        histograms = []
        self.n_pairs = {}
        for benchmark in train_benchmarks:
            resp = load_labeled_responses(benchmark, rules).dropna(subset=["label"])
            rates = [
                group["label"].mean()
                for _, group in resp.groupby("subject_id")
                if group["item_id"].nunique() >= MIN_ITEMS
            ]
            if len(rates) >= MIN_PAIRS:
                histograms.append(smooth(np.array(rates)))
                self.n_pairs[benchmark] = len(rates)
        if histograms:
            # benchmarks count equally, so a large benchmark does not dominate
            mean = np.mean(histograms, axis=0)
            self.prior = (1 - FLOOR) * mean + FLOOR / len(GRID)
        return self

    def posterior_mean(self, k, n):
        log_post = np.log(self.prior) + self.temper * (k * np.log(GRID) + (n - k) * np.log1p(-GRID))
        weights = np.exp(log_post - log_post.max())
        weights /= weights.sum()
        return float(np.clip(weights @ GRID, CLIP, 1 - CLIP))

    def predict_pair(self, benchmark, subject_id, item_ids, labeled):
        n = len(labeled)
        k = sum(label for _, label in labeled)
        return np.full(len(item_ids), self.posterior_mean(k, n))


def main():
    from sparse_observer.data.labels import load_rules
    from sparse_observer.data.loader import BENCHMARKS

    rules = load_rules()
    cases = [(0, 0), (0, 1), (1, 1), (0, 3), (1, 3), (0, 7), (3, 7), (0, 31), (15, 31), (31, 31)]
    for temper in (1.0, 0.5):
        print(f"temper {temper:g}: predictions for (successes, labels)")
        for held_out in BENCHMARKS:
            model = EmpiricalBayes(temper).fit([b for b in BENCHMARKS if b != held_out], rules)
            values = "  ".join(f"{k}/{n}:{model.posterior_mean(k, n):.2f}" for k, n in cases)
            print(f"  held out {held_out:<18} prior mean {float(model.prior @ GRID):.2f}  {values}")
    print(f"\ntraining pairs used when matharena is held out: {model.n_pairs}")


if __name__ == "__main__":
    main()
