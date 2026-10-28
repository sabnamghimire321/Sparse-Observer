"""Item-level model: a per-pair offset plus an item effect learned from other pairs' labels."""

from collections import defaultdict

import numpy as np

from sparse_observer.models.online_state import logit, sigmoid
from sparse_observer.models.pooled import PooledOffset


class ItemPooled(PooledOffset):
    """PooledOffset plus an item effect. The effect is the shrunk log-odds of the item among the
    labels other pairs acquired, relative to their average; items without pooled labels get none.
    """

    def __init__(self, slope=1.0, shrink=2.0):
        super().__init__(weight=0.0)
        self.slope = slope
        self.shrink = shrink
        self.name = f"item_pooled_s{slope:g}"

    def item_ease(self, pooled):
        """Shrunk log-odds of each pooled item relative to the pooled average."""
        if not pooled:
            return {}
        sums, counts = defaultdict(float), defaultdict(int)
        for item, label in pooled:
            sums[item] += label
            counts[item] += 1
        q = float(np.mean([label for _, label in pooled]))
        base = float(logit(q))
        return {
            item: float(logit((sums[item] + self.shrink * q) / (counts[item] + self.shrink))) - base
            for item in sums
        }

    def predict_items(self, eval_items, own, pooled):
        """Probabilities for the evaluation items from the pair's own labels and the pooled labels."""
        ease = self.item_ease(pooled)
        y = np.array([label for _, label in own], dtype=float)
        x = self.slope * np.array([ease.get(item, 0.0) for item, _ in own], dtype=float)
        # Newton steps on the penalized log-likelihood
        b = self.prior_logit
        for _ in range(25):
            p = sigmoid(b + x)
            grad = np.sum(p - y) + (b - self.prior_logit) / self.tau**2
            hess = np.sum(p * (1 - p)) + 1 / self.tau**2
            b -= grad / hess
        z = b + self.slope * np.array([ease.get(item, 0.0) for item in eval_items], dtype=float)
        return sigmoid(z)
