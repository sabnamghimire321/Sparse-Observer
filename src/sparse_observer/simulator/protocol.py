"""Pair splits and nested label budgets for the sparse-label task (see docs/protocol_notes.md)."""

import zlib
from dataclasses import dataclass

import numpy as np

from sparse_observer.data.loader import BENCHMARKS, load_table

MIN_ITEMS = 80


@dataclass(frozen=True)
class PairSplit:
    """Acquisition items in stream order and the held-out evaluation items of one pair."""
    benchmark: str
    subject_id: str
    # candidate items in stream order
    acquisition: tuple
    evaluation: tuple


def pair_rng(benchmark, subject_id, seed):
    """Random generator tied to a pair and a seed, so results do not depend on loop order."""
    # crc32 is stable across runs, unlike the built-in hash
    key = zlib.crc32(f"{benchmark}|{subject_id}".encode())
    return np.random.default_rng([seed, key])


def split_pair(benchmark, subject_id, item_ids, seed=0):
    """Shuffle a pair's items and split them 50/50.

    The extra item of an odd count goes to acquisition, following the task description.
    """
    items = np.array(sorted(item_ids))
    pair_rng(benchmark, subject_id, seed).shuffle(items)
    n_acquisition = (len(items) + 1) // 2
    return PairSplit(
        benchmark,
        subject_id,
        tuple(items[:n_acquisition].tolist()),
        tuple(items[n_acquisition:].tolist()),
    )


def build_splits(benchmark, seed=0, min_items=MIN_ITEMS):
    resp = load_table(benchmark, "response", columns=["subject_id", "item_id"])
    splits = {}
    for subject_id, group in resp.groupby("subject_id"):
        items = group["item_id"].unique()
        if len(items) >= min_items:
            splits[(benchmark, subject_id)] = split_pair(benchmark, subject_id, items, seed)
    return splits


def labels_at_budget(acquired, budget):
    """Budgets are nested: the labels at a smaller budget are a prefix of those at a larger one."""
    return acquired[:budget]


def main():
    total = 0
    all_ok = True
    for benchmark in BENCHMARKS:
        splits = build_splits(benchmark)
        repeat = build_splits(benchmark)
        other_seed = build_splits(benchmark, seed=1)

        disjoint = all(not set(s.acquisition) & set(s.evaluation) for s in splits.values())
        halved = all(len(s.acquisition) == (len(s.acquisition) + len(s.evaluation) + 1) // 2
                     for s in splits.values())
        deterministic = splits == repeat
        changed = np.mean([splits[k] != other_seed[k] for k in splits]) if splits else 0.0
        all_ok &= disjoint and halved and deterministic

        stream = np.median([len(s.acquisition) for s in splits.values()])
        evaluation = np.median([len(s.evaluation) for s in splits.values()])
        print(
            f"{benchmark}: {len(splits)} pairs, median stream {stream:.0f}, "
            f"median eval items {evaluation:.0f}, disjoint {disjoint}, "
            f"half split {halved}, deterministic {deterministic}, seed 1 differs {changed:.2f}"
        )
        total += len(splits)

    prefix_ok = labels_at_budget(list(range(31)), 7) == list(range(7))
    print(f"\ntotal pairs: {total}, nested prefix ok: {prefix_ok}, all checks ok: {all_ok and prefix_ok}")


if __name__ == "__main__":
    main()
