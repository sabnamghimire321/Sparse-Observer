"""Coverage of 90% intervals for a pair's evaluation-pool accuracy at small label budgets (H5)."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from sparse_observer.data.labels import load_rules
from sparse_observer.data.loader import BENCHMARKS
from sparse_observer.evaluation.intervals import LEVEL, clopper_pearson, pool_aware_wilson, wilson
from sparse_observer.evaluation.stats import cluster_bootstrap_ci, cluster_labels
from sparse_observer.models.empirical_bayes import GRID, MIN_PAIRS, EmpiricalBayes
from sparse_observer.simulator.stream import acquired_stream, load_pairs

BUDGETS = (1, 3, 7, 15, 31)
N_DRAWS = 200
# fresh seeds, separate from the seeds 0-4 used in the earlier experiments
SEED0 = 1000
METHODS = ("wilson", "clopper_pearson", "pool_aware_wilson", "eb", "eb_wrong_prior")
MULTI = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
H5_BUDGETS = (7, 15, 31)
NOMINAL_LOW, NOMINAL_HIGH = 0.85, 0.95
TOL = 1e-12
OUT_DIR = Path("results/coverage_v1")


def eb_interval(prior, k, n, pool_matrix, level=LEVEL):
    """Empirical-Bayes predictive interval for the evaluation-pool mean.

    Posterior over the pair's rate, then the distribution of the pool mean given that rate;
    equal-tailed quantiles.
    """
    # posterior over the pair's success rate, then the distribution of the evaluation-pool
    # mean given that rate (binomial over the pool's items); equal-tailed quantiles
    log_post = np.log(prior) + k * np.log(GRID) + (n - k) * np.log1p(-GRID)
    weights = np.exp(log_post - log_post.max())
    weights /= weights.sum()
    cdf = np.cumsum(weights @ pool_matrix)
    tail = (1 - level) / 2
    pool = pool_matrix.shape[1] - 1
    low = int(np.searchsorted(cdf, tail))
    high = int(np.searchsorted(cdf, 1 - tail))
    return low / pool, min(high, pool) / pool


def all_intervals(k, n, n_pool, prior, wrong_prior, pool_matrix):
    return [
        wilson(k, n),
        clopper_pearson(k, n),
        pool_aware_wilson(k, n, n_pool),
        eb_interval(prior, k, n, pool_matrix),
        eb_interval(wrong_prior, k, n, pool_matrix),
    ]


def run_pair(benchmark, subject_id, group, prior, wrong_prior, n_draws):
    """Coverage and width of every method and budget over repeated label draws for one pair."""
    covered = np.zeros((len(METHODS), len(BUDGETS)))
    width = np.zeros_like(covered)
    pool_matrix = None
    for d in range(n_draws):
        split, acquired = acquired_stream(benchmark, subject_id, group, SEED0 + d)
        targets = group[group["item_id"].isin(list(split.evaluation))]
        theta = float(targets["label"].mean())
        n_pool = len(split.evaluation)
        if pool_matrix is None:
            pool_matrix = binom.pmf(np.arange(n_pool + 1)[None, :], n_pool, GRID[:, None])
        labels = np.array([label for _, label in acquired])
        for j, n in enumerate(BUDGETS):
            k = float(labels[:n].sum())
            for i, (low, high) in enumerate(all_intervals(k, n, n_pool, prior, wrong_prior, pool_matrix)):
                covered[i, j] += (low - TOL) <= theta <= (high + TOL)
                width[i, j] += high - low
    return covered / n_draws, width / n_draws


def mean_rate(pairs):
    return float(np.mean([group["label"].mean() for _, group in pairs]))


def farthest_benchmark(benchmark, data):
    """Stress test: the training benchmark whose mean rate is farthest from this one."""
    # stress test: the training benchmark whose mean pair rate is farthest from this one
    target = mean_rate(data[benchmark])
    candidates = [b for b in BENCHMARKS if b != benchmark and len(data[b]) >= MIN_PAIRS]
    return max(candidates, key=lambda b: abs(mean_rate(data[b]) - target))


def summarize(frame, clusters, rng):
    """Coverage per benchmark, method and budget, with a bootstrap interval over model names."""
    rows = []
    for (benchmark, method, budget), part in frame.groupby(["benchmark", "method", "budget"]):
        series = part.set_index(["benchmark", "subject_id"])["coverage"]
        low, high = cluster_bootstrap_ci(series, clusters.reindex(series.index), rng)
        rows.append({"benchmark": benchmark, "method": method, "budget": budget,
                     "pairs": len(series), "coverage": series.mean(),
                     "ci_low": low, "ci_high": high, "width": part["width"].mean()})
    return pd.DataFrame(rows)


def h5_verdict(summary):
    """Apply the H5 rule of docs/analysis_plan.md to the empirical-Bayes cells and print the verdict."""
    cells = summary[(summary["method"] == "eb") & summary["benchmark"].isin(MULTI)
                    & summary["budget"].isin(H5_BUDGETS)]
    outside = (cells["ci_high"] < NOMINAL_LOW) | (cells["ci_low"] > NOMINAL_HIGH)
    inside = cells["coverage"].between(NOMINAL_LOW, NOMINAL_HIGH)
    print(f"\nH5 cells (empirical-Bayes interval, budgets {H5_BUDGETS}, {len(cells)} cells):")
    print(cells[["benchmark", "budget", "coverage", "ci_low", "ci_high"]].round(3).to_string(index=False))
    if outside.any():
        print(f"\nH5 FAILS: the whole interval lies outside {NOMINAL_LOW}-{NOMINAL_HIGH} in "
              f"{int(outside.sum())} of {len(cells)} cells")
    elif inside.all():
        print(f"\nH5 SUPPORTED: all {len(cells)} coverage estimates lie in {NOMINAL_LOW}-{NOMINAL_HIGH}")
    else:
        print(f"\nH5 INCONCLUSIVE: {int((~inside).sum())} estimates are outside "
              f"{NOMINAL_LOW}-{NOMINAL_HIGH} but their intervals overlap it")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    args = parser.parse_args()

    rules = load_rules()
    data = {b: load_pairs(b, rules) for b in BENCHMARKS}
    rows = []
    for benchmark in BENCHMARKS:
        if not data[benchmark]:
            continue
        start = time.time()
        others = [b for b in BENCHMARKS if b != benchmark]
        prior = EmpiricalBayes(1.0).fit(others, rules).prior
        wrong = farthest_benchmark(benchmark, data)
        wrong_prior = EmpiricalBayes(1.0).fit([wrong], rules).prior
        for subject_id, group in data[benchmark]:
            coverage, width = run_pair(benchmark, subject_id, group, prior, wrong_prior, args.draws)
            for i, method in enumerate(METHODS):
                for j, budget in enumerate(BUDGETS):
                    rows.append({"benchmark": benchmark, "subject_id": subject_id, "method": method,
                                 "budget": budget, "coverage": coverage[i, j], "width": width[i, j]})
        print(f"done {benchmark}: {len(data[benchmark])} pairs, wrong prior from {wrong}, "
              f"{time.time() - start:.0f} s")

    frame = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "pair_coverage.csv", index=False)
    summary = summarize(frame, cluster_labels(), np.random.default_rng(0))
    summary.to_csv(OUT_DIR / "coverage_summary.csv", index=False)

    pd.set_option("display.width", 200)
    for method in METHODS:
        part = summary[summary["method"] == method]
        print(f"\n{method}: coverage (nominal {LEVEL:.2f}) by benchmark and budget")
        print(part.pivot(index="benchmark", columns="budget", values="coverage").round(3).to_string())
        print(f"{method}: mean interval width")
        print(part.pivot(index="benchmark", columns="budget", values="width").round(3).to_string())
    h5_verdict(summary)
    print(f"\nsaved {OUT_DIR}/pair_coverage.csv and coverage_summary.csv")


if __name__ == "__main__":
    main()
