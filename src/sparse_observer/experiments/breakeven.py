"""How many other pairs on a benchmark does an item-level model need to beat its no-pooling control?"""

from pathlib import Path

import pandas as pd

from sparse_observer.data.labels import load_rules
from sparse_observer.data.loader import BENCHMARKS
from sparse_observer.evaluation.metrics import BUDGETS, brier, brier_alc
from sparse_observer.models.item_pooled import ItemPooled
from sparse_observer.simulator.stream import SEEDS, acquired_stream, load_pairs, sample_sources

OUT_DIR = Path("results/breakeven_v1")
MULTI = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
VARIANTS = [(0, 1.0), (1, 1.0), (3, 1.0), (10, 1.0), (30, 1.0), (3, 0.5), (10, 0.5), (30, 0.5)]
CONTROL = "m0_s1"
PRIMARY_SLOPE = 1.0
THRESHOLD = -0.003


def label(m, slope):
    return f"m{m}_s{slope:g}"


def run_benchmark(benchmark, rules, model):
    """Score every variant on the pairs of one held-out benchmark."""
    pairs = load_pairs(benchmark, rules)
    if len(pairs) < 2:
        return []
    model.fit([b for b in BENCHMARKS if b != benchmark], rules)
    subjects = [s for s, _ in pairs]
    rows = []
    for seed in SEEDS:
        streams = {s: acquired_stream(benchmark, s, g, seed) for s, g in pairs}
        for subject_id, group in pairs:
            split, own = streams[subject_id]
            targets = group[group["item_id"].isin(list(split.evaluation))]
            eval_items = targets["item_id"].unique()
            position = {item: i for i, item in enumerate(eval_items)}
            index = targets["item_id"].map(position).to_numpy()
            y = targets["label"].to_numpy()
            for m, slope in VARIANTS:
                model.slope = slope
                sources = sample_sources(benchmark, subjects, subject_id, m, seed)
                full = {item for s in sources for item, _ in streams[s][1][:BUDGETS[-1]]}
                scores = []
                for n in BUDGETS:
                    pooled = [(i, lab) for s in sources for i, lab in streams[s][1][:n]]
                    p = model.predict_items(eval_items, own[:n], pooled)[index]
                    scores.append(brier(p, y))
                row = {"benchmark": benchmark, "subject_id": subject_id, "seed": seed,
                       "variant": label(m, slope), "m": m, "slope": slope,
                       "sources": len(sources),
                       "covered": float(targets["item_id"].isin(full).mean()),
                       "brier_alc": brier_alc(scores)}
                row.update({f"brier_{n}": s for n, s in zip(BUDGETS, scores)})
                rows.append(row)
    return rows


def main():
    rules = load_rules()
    model = ItemPooled()
    rows = []
    for benchmark in BENCHMARKS:
        rows += run_benchmark(benchmark, rules, model)
        print(f"done {benchmark}")
    scores = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "pair_scores.csv", index=False)

    order = [label(m, s) for m, s in VARIANTS]
    alc = scores.groupby(["benchmark", "variant"])["brier_alc"].mean().unstack("variant")[order]
    print("\nmean Brier ALC by benchmark (m = source pairs, s = slope):")
    print(alc.round(4).to_string())
    print("\nmean over all pairs:")
    print(scores.groupby("variant")["brier_alc"].mean()[order].round(4).to_string())

    diff = alc.sub(alc[CONTROL], axis=0).drop(columns=CONTROL)
    print(f"\ndifference from control {CONTROL} (negative is better):")
    print(diff.round(4).to_string())

    cover = scores[scores["slope"] == PRIMARY_SLOPE].groupby(["benchmark", "m"])[["covered", "sources"]].mean()
    print("\nshare of evaluation targets with a pooled label at 31 labels, and sources actually used:")
    print(cover.unstack("m").round(3).to_string())

    print("\ncriterion (slope 1.0): benchmarks where ALC is at least 0.003 lower than control")
    for m, slope in VARIANTS:
        if slope == PRIMARY_SLOPE and m > 0:
            wins = int((diff.loc[list(MULTI), label(m, slope)] <= THRESHOLD).sum())
            print(f"  m = {m:>2}: {wins} of {len(MULTI)} (reachable if at least 3)")
    print(f"\nsaved {OUT_DIR / 'pair_scores.csv'}")


if __name__ == "__main__":
    main()
