"""Acquisition streams and the evidence that other pairs on a benchmark would contribute."""

from collections import Counter

import pandas as pd

from sparse_observer.data.labels import load_labeled_responses, load_rules
from sparse_observer.data.loader import BENCHMARKS
from sparse_observer.evaluation.metrics import BUDGETS
from sparse_observer.simulator.protocol import MIN_ITEMS, pair_rng, split_pair

MAX_LABELS = BUDGETS[-1]
SEEDS = (0, 1, 2, 3, 4)
SOURCE_COUNTS = (1, 3)
CHECK_BUDGETS = (7, 31)


def load_pairs(benchmark, rules):
    """Eligible pairs (at least 80 items with labels) as (subject_id, labeled responses)."""
    resp = load_labeled_responses(benchmark, rules).dropna(subset=["label"])
    return [
        (subject_id, group)
        for subject_id, group in resp.groupby("subject_id")
        if group["item_id"].nunique() >= MIN_ITEMS
    ]


def reveal_labels(group, rng):
    """Reveal one recorded trial per item, chosen at random.

    The task description does not say which trial is revealed; this is an assumption
    of the simulator (docs/protocol_notes.md).
    """
    revealed = {}
    for item_id, labels in group.groupby("item_id")["label"]:
        values = labels.to_numpy()
        revealed[item_id] = float(values[rng.integers(len(values))])
    return revealed


def acquired_stream(benchmark, subject_id, group, seed):
    """Split of one pair and its first 31 acquired labels, in stream order."""
    split = split_pair(benchmark, subject_id, group["item_id"].unique(), seed)
    revealed = reveal_labels(group, pair_rng(f"{benchmark}|reveal", subject_id, seed))
    acquired = [(item, revealed[item]) for item in split.acquisition[:MAX_LABELS]]
    return split, acquired


def sample_sources(benchmark, subjects, target, m, seed):
    """Other pairs on the same benchmark whose labels the target pair may see (sampling is assumed)."""
    # sorted, so the draw does not depend on the order of the input
    others = sorted(s for s in subjects if s != target)
    take = min(m, len(others))
    if take == 0:
        return []
    rng = pair_rng(f"{benchmark}|sources", target, seed)
    return list(rng.choice(others, size=take, replace=False))


def item_coverage(benchmark, pairs, streams, m, budget, seed):
    """Share of the target pair's evaluation responses whose item was labeled by a source pair.

    labels_per_covered is zero-filled for uncovered pairs, so it is biased low and unused.
    """
    subjects = [s for s, _ in pairs]
    rows = []
    for subject_id, group in pairs:
        split = streams[subject_id][0]
        targets = group[group["item_id"].isin(list(split.evaluation))]
        sources = sample_sources(benchmark, subjects, subject_id, m, seed)
        counts = Counter(item for s in sources for item, _ in streams[s][1][:budget])
        per_row = targets["item_id"].map(dict(counts)).fillna(0)
        covered = per_row > 0
        rows.append({
            "benchmark": benchmark,
            "m": m,
            "budget": budget,
            "covered": covered.mean(),
            "labels_per_covered": per_row[covered].mean() if covered.any() else 0.0,
        })
    return rows


def main():
    rules = load_rules()
    rows = []
    for benchmark in BENCHMARKS:
        pairs = load_pairs(benchmark, rules)
        if len(pairs) < 2:
            continue
        print(f"{benchmark}: {len(pairs)} eligible pairs usable as sources")
        for seed in SEEDS:
            streams = {s: acquired_stream(benchmark, s, g, seed) for s, g in pairs}
            for m in SOURCE_COUNTS:
                for budget in CHECK_BUDGETS:
                    rows += item_coverage(benchmark, pairs, streams, m, budget, seed)

    result = pd.DataFrame(rows)
    table = result.groupby(["benchmark", "m", "budget"])[["covered", "labels_per_covered"]].mean()
    print("\nshare of evaluation targets whose item was labeled by a source pair "
          "(and labels per covered item), mean over seeds:")
    print(table.unstack("budget").round(3).to_string())


if __name__ == "__main__":
    main()
