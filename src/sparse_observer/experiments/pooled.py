"""Benchmark-level pooled evidence (dropped): other pairs' labels weighted into the pair's rate."""

from pathlib import Path

import numpy as np
import pandas as pd

from sparse_observer.data.labels import load_rules
from sparse_observer.data.loader import BENCHMARKS
from sparse_observer.evaluation.metrics import BUDGETS, brier, brier_alc
from sparse_observer.models.pooled import PooledOffset
from sparse_observer.simulator.stream import SEEDS, acquired_stream, load_pairs, sample_sources

OUT_DIR = Path("results/pooled_v1")
MULTI = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
# (weight of pooled labels, number of source pairs); the first entry is the control
VARIANTS = [(0.0, 0), (0.25, 1), (0.5, 1), (1.0, 1), (0.25, 3), (0.5, 3), (1.0, 3)]
PRIMARY = "w0.5_m1"
CONTROL = "w0_m0"
THRESHOLD = -0.003


def label(w, m):
    return f"w{w:g}_m{m}"


def run_benchmark(benchmark, rules, model):
    """Score every variant on the pairs of one held-out benchmark."""
    pairs = load_pairs(benchmark, rules)
    if len(pairs) < 2:
        return []
    model.fit([b for b in BENCHMARKS if b != benchmark], rules)
    print(f"{benchmark}: prior logit {model.prior_logit:.2f}, benchmark spread "
          f"{model.bench_spread:.2f}, pair spread {model.pair_spread:.2f}, tau {model.tau:.2f}")
    subjects = [s for s, _ in pairs]
    rows = []
    for seed in SEEDS:
        streams = {s: acquired_stream(benchmark, s, g, seed) for s, g in pairs}
        for subject_id, group in pairs:
            split, own = streams[subject_id]
            y = group[group["item_id"].isin(list(split.evaluation))]["label"].to_numpy()
            for w, m in VARIANTS:
                model.weight = w
                sources = sample_sources(benchmark, subjects, subject_id, m, seed)
                scores = []
                for n in BUDGETS:
                    pooled = [lab for s in sources for _, lab in streams[s][1][:n]]
                    p = model.predict_pair(benchmark, subject_id, [0], own[:n], pooled)[0]
                    scores.append(brier(np.full(len(y), p), y))
                row = {"benchmark": benchmark, "subject_id": subject_id, "seed": seed,
                       "variant": label(w, m), "brier_alc": brier_alc(scores)}
                row.update({f"brier_{n}": s for n, s in zip(BUDGETS, scores)})
                rows.append(row)
    return rows


def main():
    rules = load_rules()
    model = PooledOffset()
    rows = []
    for benchmark in BENCHMARKS:
        rows += run_benchmark(benchmark, rules, model)
    scores = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "pair_scores.csv", index=False)

    alc = scores.groupby(["benchmark", "variant"])["brier_alc"].mean().unstack("variant")
    order = [label(w, m) for w, m in VARIANTS]
    print("\nmean Brier ALC by benchmark:")
    print(alc[order].round(4).to_string())

    diff = alc[order].sub(alc[CONTROL], axis=0)
    print(f"\ndifference from control {CONTROL} (negative is better):")
    print(diff.drop(columns=CONTROL).round(4).to_string())

    pair_mean = scores.groupby("variant")["brier_alc"].mean()
    print("\nmean over all pairs:")
    print(pair_mean[order].round(4).to_string())

    wins = int((diff.loc[list(MULTI), PRIMARY] <= THRESHOLD).sum())
    print(f"\nprimary variant {PRIMARY}: ALC at least {abs(THRESHOLD)} lower than control in "
          f"{wins} of {len(MULTI)} benchmarks (criterion: at least 3)")

    budgets = [f"brier_{n}" for n in BUDGETS]
    print("\nBrier by budget, control vs primary (all pairs):")
    print(scores[scores["variant"].isin([CONTROL, PRIMARY])].groupby("variant")[budgets].mean().round(4).to_string())
    print(f"\nsaved {OUT_DIR / 'pair_scores.csv'}")


if __name__ == "__main__":
    main()
