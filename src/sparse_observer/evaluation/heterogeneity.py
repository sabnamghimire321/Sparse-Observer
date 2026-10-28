"""Does the item-pooled model help pairs with low, middle and high success rates equally?"""

from pathlib import Path

import pandas as pd

from sparse_observer.data.labels import load_labeled_responses, load_rules
from sparse_observer.data.loader import BENCHMARKS
from sparse_observer.simulator.protocol import MIN_ITEMS

SCORES = Path("results/breakeven_v1/pair_scores.csv")
CONTROL = "m0_s1"
TREATED = ["m1_s1", "m3_s1", "m10_s1"]
EDGES = [0.0, 0.1, 0.3, 0.7, 1.0001]
NAMES = ["rate < 0.1", "0.1 - 0.3", "0.3 - 0.7", "rate >= 0.7"]


def pair_rates(rules):
    rows = []
    for benchmark in BENCHMARKS:
        resp = load_labeled_responses(benchmark, rules).dropna(subset=["label"])
        for subject_id, group in resp.groupby("subject_id"):
            if group["item_id"].nunique() >= MIN_ITEMS:
                rows.append({"benchmark": benchmark, "subject_id": subject_id,
                             "rate": group["label"].mean()})
    return pd.DataFrame(rows).set_index(["benchmark", "subject_id"])


def main():
    scores = pd.read_csv(SCORES)
    # mean over seeds, so each pair counts once
    wide = scores.pivot_table(index=["benchmark", "subject_id"], columns="variant",
                              values=["brier_alc", "brier_31"])
    frame = pair_rates(load_rules())
    frame["bin"] = pd.cut(frame["rate"], EDGES, labels=NAMES, right=False)
    for t in TREATED:
        frame[f"alc_{t}"] = wide[("brier_alc", t)] - wide[("brier_alc", CONTROL)]
        frame[f"b31_{t}"] = wide[("brier_31", t)] - wide[("brier_31", CONTROL)]
    frame = frame.dropna()

    pd.set_option("display.width", 200)
    print(f"pairs in the simulation: {len(frame)}")
    print("pairs per bin:", frame["bin"].value_counts().reindex(NAMES).to_dict())
    for t in TREATED:
        table = frame.groupby("bin", observed=True).agg(
            pairs=(f"alc_{t}", "size"),
            alc_diff=(f"alc_{t}", "mean"),
            share_worse=(f"alc_{t}", lambda s: (s > 1e-12).mean()),
            share_changed=(f"alc_{t}", lambda s: (s.abs() > 1e-12).mean()),
            brier31_diff=(f"b31_{t}", "mean"),
        )
        print(f"\n{t} minus {CONTROL} (negative is better; share_worse among all pairs in the bin)")
        print(table.reindex(NAMES).round(4).to_string())


if __name__ == "__main__":
    main()
