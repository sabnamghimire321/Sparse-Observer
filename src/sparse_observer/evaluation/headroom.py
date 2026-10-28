"""Oracle bounds on what item difficulty could add.

The item oracle uses other subjects' responses and the pair's own evaluation outcomes,
so it is an upper bound and not a method.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sparse_observer.data.labels import load_labeled_responses, load_rules
from sparse_observer.data.loader import BENCHMARKS
from sparse_observer.evaluation.metrics import brier
from sparse_observer.models.online_state import logit, sigmoid
from sparse_observer.simulator.protocol import MIN_ITEMS, split_pair

SEEDS = (0, 1, 2, 3, 4)
SHRINK = 2.0
SCORES = Path("results/main_v1/pair_scores.csv")
OUT = Path("results/headroom_v1/pair_headroom.csv")


def fit_intercept(d, y, ridge=1e-3):
    # the ridge keeps the fit finite when a pair's targets are all 0 or all 1
    a = 0.0
    for _ in range(30):
        p = sigmoid(a + d)
        grad = np.sum(p - y) + ridge * a
        hess = np.sum(p * (1 - p)) + ridge
        a -= grad / hess
    return a


def benchmark_headroom(benchmark, rules):
    """Constant-oracle and item-oracle Brier per pair and seed on one benchmark's evaluation items."""
    resp = load_labeled_responses(benchmark, rules).dropna(subset=["label"])
    if resp.empty or resp["subject_id"].nunique() < 2:
        return []
    bench_rate = resp["label"].mean()
    item_sum = resp.groupby("item_id")["label"].sum()
    item_cnt = resp.groupby("item_id")["label"].size()

    rows = []
    for subject_id, group in resp.groupby("subject_id"):
        if group["item_id"].nunique() < MIN_ITEMS:
            continue
        for seed in SEEDS:
            split = split_pair(benchmark, subject_id, group["item_id"].unique(), seed)
            targets = group[group["item_id"].isin(list(split.evaluation))]
            items = targets["item_id"].to_numpy()
            own = targets.groupby("item_id")["label"].agg(["sum", "size"])

            # leave the subject out so its own responses never inform its item difficulty
            other_sum = item_sum.reindex(items).to_numpy() - own["sum"].reindex(items).to_numpy()
            other_cnt = item_cnt.reindex(items).to_numpy() - own["size"].reindex(items).to_numpy()
            rate = (other_sum + SHRINK * bench_rate) / (other_cnt + SHRINK)
            d = logit(rate) - logit(bench_rate)

            y = targets["label"].to_numpy()
            a = fit_intercept(d, y)
            rows.append({
                "benchmark": benchmark,
                "subject_id": subject_id,
                "seed": seed,
                "n_targets": len(y),
                "constant_oracle": brier(np.full(len(y), y.mean()), y),
                "item_oracle": brier(sigmoid(a + d), y),
                "items_with_other_subjects": float((other_cnt > 0).mean()),
            })
    return rows


def main():
    rules = load_rules()
    rows = []
    for benchmark in BENCHMARKS:
        rows += benchmark_headroom(benchmark, rules)
    oracle = pd.DataFrame(rows).drop(columns="seed")
    # average over seeds so each pair counts once
    oracle = oracle.groupby(["benchmark", "subject_id"], as_index=False).mean()

    scores = pd.read_csv(SCORES)
    s0 = scores[scores["predictor"] == "offset_logit_s0"]
    s0 = s0.groupby(["benchmark", "subject_id"], as_index=False)[["brier_0", "brier_31"]].mean()
    merged = oracle.merge(s0, on=["benchmark", "subject_id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)

    columns = ["constant_oracle", "item_oracle", "brier_31", "brier_0", "items_with_other_subjects"]
    by_benchmark = merged.groupby("benchmark")[columns].mean()
    by_benchmark.insert(0, "pairs", merged.groupby("benchmark").size())
    print(f"seeds {SEEDS}, evaluation items only (brier_0 and brier_31 are offset_logit_s0):")
    print(by_benchmark.round(3).to_string())
    print("\nall pairs:")
    print(merged[columns].mean().round(3).to_string())
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
