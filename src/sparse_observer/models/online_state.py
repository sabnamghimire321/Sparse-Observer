"""MAP logit offset per pair, optionally shifted by a subject ability from other benchmarks."""

import numpy as np

from sparse_observer.data.labels import load_labeled_responses, load_rules
from sparse_observer.data.loader import BENCHMARKS, load_table
from sparse_observer.models.base import Predictor

CLIP = 0.02


def logit(p):
    # clipping keeps the logit finite for rates of exactly 0 or 1
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class OffsetLogit(Predictor):
    """Per-pair logit offset fitted by MAP. With scale > 0 the pair is shifted by the ability
    that the same model name showed on the training benchmarks (hypothesis H1).
    """

    def __init__(self, scale=1.0, tau=1.0, shrink=5.0):
        self.scale = scale
        self.tau = tau
        self.shrink = shrink
        self.prior_logit = 0.0
        self.ability = {}
        self._names = {}
        self.name = f"offset_logit_s{scale:g}"

    def subject_names(self, benchmark):
        if benchmark not in self._names:
            table = load_table(benchmark, "subjects", columns=["subject_id", "normalized_name"])
            self._names[benchmark] = {
                s: n
                for s, n in zip(table["subject_id"], table["normalized_name"])
                if isinstance(n, str) and n.strip()
            }
        return self._names[benchmark]

    def fit(self, train_benchmarks, rules):
        """Prior from the training benchmarks' mean rate; ability per model name from their subjects."""
        rates, gaps = [], {}
        for benchmark in train_benchmarks:
            resp = load_labeled_responses(benchmark, rules).dropna(subset=["label"])
            if resp.empty:
                continue
            bench_rate = resp["label"].mean()
            rates.append(bench_rate)
            names = self.subject_names(benchmark)
            per_subject = resp.groupby("subject_id")["label"].agg(["mean", "size"])
            for subject_id, row in per_subject.iterrows():
                name = names.get(subject_id)
                if name is None:
                    continue
                # shrink small samples toward the benchmark rate before taking the gap
                rate = (row["mean"] * row["size"] + bench_rate * self.shrink) / (row["size"] + self.shrink)
                gaps.setdefault(name, []).append(logit(rate) - logit(bench_rate))
        self.prior_logit = float(logit(np.mean(rates))) if rates else 0.0
        self.ability = {name: float(np.mean(v)) for name, v in gaps.items()}
        return self

    def predict_pair(self, benchmark, subject_id, item_ids, labeled):
        """MAP offset from the pair's own labels; no item information is used."""
        name = self.subject_names(benchmark).get(subject_id)
        shift = self.scale * self.ability.get(name, 0.0)
        y = np.array([label for _, label in labeled], dtype=float)
        b = self.prior_logit
        for _ in range(25):
            p = sigmoid(b + shift)
            grad = np.sum(p - y) + (b - self.prior_logit) / self.tau**2
            # every label adds curvature, so the likelihood term scales with the number of labels
            hess = len(y) * p * (1 - p) + 1 / self.tau**2
            b -= grad / hess
        return np.full(len(item_ids), sigmoid(b + shift))


def main():
    rules = load_rules()
    model = OffsetLogit()
    for benchmark in BENCHMARKS:
        model.fit([b for b in BENCHMARKS if b != benchmark], rules)
        subject_ids = load_table(benchmark, "subjects", columns=["subject_id"])["subject_id"]
        names = model.subject_names(benchmark)
        known = sum(1 for s in subject_ids if names.get(s) in model.ability)
        spread = np.std(list(model.ability.values())) if model.ability else float("nan")
        print(
            f"{benchmark}: ability known for {known}/{len(subject_ids)} subjects "
            f"({known / len(subject_ids):.2f}), prior logit {model.prior_logit:.2f}, "
            f"ability spread {spread:.2f}"
        )


if __name__ == "__main__":
    main()
