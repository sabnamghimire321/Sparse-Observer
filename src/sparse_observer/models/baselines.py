"""The constant predictor and the primary comparator, a shrunk per-pair mean."""

import numpy as np

from sparse_observer.data.labels import load_labeled_responses
from sparse_observer.models.base import Predictor


class Constant(Predictor):
    def __init__(self, p=0.5):
        self.p = p
        self.name = f"constant_{p:g}"

    def predict_pair(self, benchmark, subject_id, item_ids, labeled):
        return np.full(len(item_ids), self.p)


class PairMean(Predictor):
    """Posterior mean of the pair's success rate with a prior of strength `prior_strength`
    centered on the mean success rate of the training benchmarks.
    """

    def __init__(self, prior_strength=2.0):
        self.prior_strength = prior_strength
        self.prior_mean = 0.5
        self.name = f"pair_mean_k{prior_strength:g}"

    def fit(self, train_benchmarks, rules):
        rates = []
        for benchmark in train_benchmarks:
            labels = load_labeled_responses(benchmark, rules)["label"].dropna()
            if len(labels):
                rates.append(labels.mean())
        # mean of benchmark rates, so a large benchmark does not dominate the prior
        self.prior_mean = float(np.mean(rates)) if rates else 0.5
        return self

    def predict_pair(self, benchmark, subject_id, item_ids, labeled):
        total = sum(label for _, label in labeled)
        p = (total + self.prior_strength * self.prior_mean) / (len(labeled) + self.prior_strength)
        return np.full(len(item_ids), p)
