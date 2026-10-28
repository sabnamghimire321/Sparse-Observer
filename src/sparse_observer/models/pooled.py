"""Benchmark-level pooled evidence: other pairs' labels enter the pair's rate with a fixed weight."""

import numpy as np

from sparse_observer.data.labels import load_labeled_responses
from sparse_observer.models.base import Predictor
from sparse_observer.models.online_state import logit, sigmoid
from sparse_observer.simulator.protocol import MIN_ITEMS

MIN_PAIRS = 5
DEFAULT_BENCH_SPREAD = 0.5
DEFAULT_PAIR_SPREAD = 1.0


class PooledOffset(Predictor):

    """Per-pair offset with a prior whose spread comes from the training data. Labels from other
    pairs on the same benchmark enter the likelihood with weight `weight`.
    """

    def __init__(self, weight=0.5):
        self.weight = weight
        self.prior_logit = 0.0
        self.bench_spread = DEFAULT_BENCH_SPREAD
        self.pair_spread = DEFAULT_PAIR_SPREAD
        self.name = f"pooled_w{weight:g}"

    @property
    def tau(self):
        return float(np.hypot(self.bench_spread, self.pair_spread))

    def fit(self, train_benchmarks, rules):
        """Prior mean and spreads (in logit units) from the training benchmarks."""
        rates, bench_logits, pair_spreads = [], [], []
        for benchmark in train_benchmarks:
            resp = load_labeled_responses(benchmark, rules).dropna(subset=["label"])
            if resp.empty:
                continue
            rate = float(resp["label"].mean())
            rates.append(rate)
            bench_logits.append(float(logit(rate)))
            pair_rates = [
                g["label"].mean()
                for _, g in resp.groupby("subject_id")
                if g["item_id"].nunique() >= MIN_ITEMS
            ]
            if len(pair_rates) >= MIN_PAIRS:
                pair_spreads.append(float(np.std(logit(np.array(pair_rates)))))
        # prior: the mean of the training benchmarks' success rates
        self.prior_logit = float(logit(np.mean(rates))) if rates else 0.0
        self.bench_spread = float(np.std(bench_logits)) if len(bench_logits) >= 2 else DEFAULT_BENCH_SPREAD
        self.pair_spread = float(np.mean(pair_spreads)) if pair_spreads else DEFAULT_PAIR_SPREAD
        return self

    def predict_pair(self, benchmark, subject_id, item_ids, labeled, pooled=()):
        """Posterior offset from own labels plus down-weighted labels from other pairs."""
        own = np.array([label for _, label in labeled], dtype=float)
        other = np.array(list(pooled), dtype=float)
        n_eff = len(own) + self.weight * len(other)
        total = own.sum() + self.weight * other.sum()
        b = self.prior_logit
        for _ in range(25):
            p = sigmoid(b)
            grad = n_eff * p - total + (b - self.prior_logit) / self.tau**2
            hess = n_eff * p * (1 - p) + 1 / self.tau**2
            b -= grad / hess
        return np.full(len(item_ids), sigmoid(b))
