"""Paired comparisons of predictors, with bootstrap intervals that resample whole model-name clusters."""

from pathlib import Path

import numpy as np
import pandas as pd

from sparse_observer.features.subject_feats import all_subjects

SCORES = Path("results/main_v1/pair_scores.csv")
OUT = Path("tables/comparisons_main_v1.csv")
N_BOOT = 2000
MIN_PAIRS = 5
SMALL_PAIR_ITEMS = 150
# benchmarks where ability is known for at least half of the subjects (the H1 rule)
H1_BENCHMARKS = ("multi_swebench", "real_webagents")
COMPARISONS = [
    ("pair_mean_k2", "constant_0.5", "adapting a base rate to labels"),
    ("offset_logit_s1", "offset_logit_s0", "H1 (pre-registered): subject ability, full scale"),
    ("offset_logit_s0.5", "offset_logit_s0", "exploratory: subject ability, half scale"),
    ("empirical_bayes_t1", "pair_mean_k2", "exploratory: empirical-Bayes prior vs shrunk mean"),
    ("offset_logit_s0", "pair_mean_k2", "exploratory: MAP offset vs shrunk mean"),
    ("empirical_bayes_t0.5", "empirical_bayes_t1", "exploratory: tempered vs full likelihood"),
]


def cluster_labels():
    """Cluster key per (benchmark, subject): the model name; unnamed subjects form their own cluster."""
    # subjects sharing a model name are not independent, so they share a cluster
    subjects = all_subjects()
    name = subjects["normalized_name"].fillna("").str.strip()
    fallback = subjects["benchmark"] + "|" + subjects["subject_id"]
    key = name.where(name != "", fallback)
    return pd.Series(key.to_numpy(), index=pd.MultiIndex.from_arrays(
        [subjects["benchmark"], subjects["subject_id"]]))


def pair_tables(scores):
    # average over seeds first so each pair counts once
    alc = scores.groupby(["benchmark", "subject_id", "predictor"])["brier_alc"].mean().unstack("predictor")
    size = scores.groupby(["benchmark", "subject_id"])["n_items"].first()
    return alc, size


def cluster_bootstrap_ci(diff, clusters, rng, n_boot=N_BOOT):
    """95% bootstrap interval for the mean of `diff`, resampling whole clusters."""
    grouped = diff.groupby(clusters.to_numpy())
    sums = grouped.sum().to_numpy()
    counts = grouped.size().to_numpy()
    idx = rng.integers(len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summarize(diff, clusters, scope, rng):
    own = clusters.reindex(diff.index)
    low, high = cluster_bootstrap_ci(diff, own, rng)
    return {
        "scope": scope,
        "pairs": len(diff),
        "clusters": own.nunique(),
        "mean_diff": diff.mean(),
        "ci_low": low,
        "ci_high": high,
        "share_better": (diff < 0).mean(),
    }


def compare(alc, size, clusters, a, b, rng):
    diff = (alc[a] - alc[b]).dropna()
    benchmarks = diff.index.get_level_values("benchmark")
    rows = []
    for name in sorted(set(benchmarks)):
        part = diff[benchmarks == name]
        if len(part) >= MIN_PAIRS:
            rows.append(summarize(part, clusters, name, rng))
    rows.append(summarize(diff[benchmarks.isin(H1_BENCHMARKS)], clusters,
                          "H1 set: " + " + ".join(H1_BENCHMARKS), rng))
    rest = diff[benchmarks != "swe_rebench"]
    rows.append(summarize(rest, clusters, "all pairs except swe_rebench", rng))
    small = rest[size.reindex(rest.index).to_numpy() <= SMALL_PAIR_ITEMS]
    if len(small) >= MIN_PAIRS:
        rows.append(summarize(small, clusters, f"small pairs (<= {SMALL_PAIR_ITEMS} items)", rng))
    return pd.DataFrame(rows)


def seed_spread(scores, a, b):
    wide = scores.pivot_table(index=["seed", "benchmark", "subject_id"],
                              columns="predictor", values="brier_alc")
    wide = wide[wide.index.get_level_values("benchmark") != "swe_rebench"]
    return (wide[a] - wide[b]).groupby("seed").mean()


def h1_verdict(result):
    """Apply the H1 rule of docs/analysis_plan.md: negative in both benchmarks and a pooled interval below zero."""
    per_benchmark = result.set_index("scope").loc[list(H1_BENCHMARKS)]
    pooled = result[result["scope"].str.startswith("H1 set")].iloc[0]
    negative_in_both = bool((per_benchmark["mean_diff"] < 0).all())
    interval_below_zero = bool(pooled["ci_high"] < 0)
    return negative_in_both, interval_below_zero


def main():
    scores = pd.read_csv(SCORES)
    alc, size = pair_tables(scores)
    clusters = cluster_labels()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 200)

    frames = []
    for a, b, label in COMPARISONS:
        print(f"\n{label}: {a} minus {b} (negative means {a} is better)")
        result = compare(alc, size, clusters, a, b, rng)
        print(result.round(4).to_string(index=False))
        print("per-seed mean difference (all pairs except swe_rebench):",
              seed_spread(scores, a, b).round(4).to_dict())
        if label.startswith("H1"):
            negative, below = h1_verdict(result)
            print(f"pre-registered H1 rule: negative in both benchmarks: {negative}; "
                  f"pooled 95% interval below zero: {below}; supported: {negative and below}")
        frames.append(result.assign(comparison=f"{a} - {b}", question=label))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(OUT, index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
