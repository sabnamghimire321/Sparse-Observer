"""Main simulation: every predictor on every eligible pair, five seeds, leave-one-benchmark-out."""

from pathlib import Path

import pandas as pd

from sparse_observer.data.labels import load_rules
from sparse_observer.data.loader import BENCHMARKS
from sparse_observer.evaluation.metrics import BUDGETS, brier, brier_alc, ece
from sparse_observer.models.baselines import Constant, PairMean
from sparse_observer.models.empirical_bayes import EmpiricalBayes
from sparse_observer.models.online_state import OffsetLogit
from sparse_observer.simulator.stream import acquired_stream, load_pairs

SEEDS = (0, 1, 2, 3, 4)
SMALL_PAIR_ITEMS = 150
OUT_DIR = Path("results/main_v1")


def evaluate_pair(predictor, benchmark, subject_id, group, seed):
    """Brier and ECE at each budget, and Brier ALC, for one predictor on one pair and seed."""
    split, acquired = acquired_stream(benchmark, subject_id, group, seed)

    targets = group[group["item_id"].isin(list(split.evaluation))]
    eval_items = targets["item_id"].unique()
    position = {item: i for i, item in enumerate(eval_items)}
    index = targets["item_id"].map(position).to_numpy()
    y = targets["label"].to_numpy()

    row = {"predictor": predictor.name, "benchmark": benchmark, "seed": seed,
           "subject_id": subject_id, "n_targets": len(y),
           "n_items": group["item_id"].nunique()}
    for n in BUDGETS:
        p = predictor.predict_pair(benchmark, subject_id, eval_items, acquired[:n])[index]
        row[f"brier_{n}"] = brier(p, y)
        row[f"ece_{n}"] = ece(p, y)
    row["brier_alc"] = brier_alc([row[f"brier_{n}"] for n in BUDGETS])
    return row


def summarize(scores, title):
    columns = [f"brier_{n}" for n in BUDGETS] + ["brier_alc"]
    print(title)
    summary = scores.groupby("predictor")[columns].mean()
    # a subject_id can repeat across benchmarks, so a pair is (benchmark, subject_id)
    pairs = scores.drop_duplicates(["predictor", "benchmark", "subject_id"])
    summary["pairs"] = pairs.groupby("predictor").size()
    print(summary.round(3).to_string())


def main():
    rules = load_rules()
    data = {b: load_pairs(b, rules) for b in BENCHMARKS}
    predictors = [
        Constant(0.5),
        PairMean(2.0),
        OffsetLogit(scale=0),
        OffsetLogit(scale=0.5),
        OffsetLogit(scale=1.0),
        EmpiricalBayes(1.0),
        EmpiricalBayes(0.5),
    ]

    rows = []
    for predictor in predictors:
        for benchmark in BENCHMARKS:
            if not data[benchmark]:
                continue
            predictor.fit([b for b in BENCHMARKS if b != benchmark], rules)
            for seed in SEEDS:
                for subject_id, group in data[benchmark]:
                    rows.append(evaluate_pair(predictor, benchmark, subject_id, group, seed))

    scores = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "pair_scores.csv", index=False)

    columns = [f"brier_{n}" for n in BUDGETS] + ["brier_alc"]
    grouped = scores.groupby(["predictor", "benchmark"])
    by_benchmark = grouped[columns].mean()
    by_benchmark["pairs"] = grouped["subject_id"].nunique()
    print(f"mean over seeds {SEEDS}:")
    print(by_benchmark.round(3).to_string())

    summarize(scores, "\nall pairs, mean over seeds:")
    summarize(scores[scores["n_items"] <= SMALL_PAIR_ITEMS],
              f"\nsmall pairs only (at most {SMALL_PAIR_ITEMS} distinct items):")
    print(f"\nsaved {OUT_DIR / 'pair_scores.csv'}")


if __name__ == "__main__":
    main()
