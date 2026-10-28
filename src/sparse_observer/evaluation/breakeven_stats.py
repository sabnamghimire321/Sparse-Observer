"""Paired intervals for the item-pooled model against its no-pooling control."""

from pathlib import Path

import numpy as np
import pandas as pd

from sparse_observer.evaluation.stats import MIN_PAIRS, cluster_labels, summarize

SCORES = Path("results/breakeven_v1/pair_scores.csv")
OUT = Path("tables/breakeven_v1.csv")
CONTROL = "m0_s1"
TREATED = ["m1_s1", "m3_s1", "m10_s1", "m30_s1"]


def main():
    scores = pd.read_csv(SCORES)
    # average over seeds first so each pair counts once
    alc = scores.groupby(["benchmark", "subject_id", "variant"])["brier_alc"].mean().unstack("variant")
    clusters = cluster_labels()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 200)

    frames = []
    for variant in TREATED:
        diff = (alc[variant] - alc[CONTROL]).dropna()
        benchmarks = diff.index.get_level_values("benchmark")
        rows = [
            summarize(diff[benchmarks == name], clusters, name, rng)
            for name in sorted(set(benchmarks))
            if (benchmarks == name).sum() >= MIN_PAIRS
        ]
        rows.append(summarize(diff, clusters, "all pairs (4 benchmarks)", rng))
        result = pd.DataFrame(rows)
        print(f"\n{variant} minus {CONTROL} (negative means the pooled item model is better)")
        print(result.round(4).to_string(index=False))
        frames.append(result.assign(variant=variant))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(OUT, index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
