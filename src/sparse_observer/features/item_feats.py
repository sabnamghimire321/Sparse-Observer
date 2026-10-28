"""Checks whether item_features or text length carry a cross-benchmark signal about item ease."""

import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from sparse_observer.data.labels import load_rules
from sparse_observer.data.loader import BENCHMARKS, load_table
from sparse_observer.features.text_embed import item_table

TOP_KEYS = 6


def parse(value):
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def describe_features(benchmark):
    items = load_table(benchmark, "items", columns=["item_id", "item_features"])
    parsed = items["item_features"].map(parse)
    filled = (parsed.map(len) > 0).mean()
    counts = Counter(key for record in parsed for key in record)
    print(f"{benchmark}: item_features present for {filled:.2f} of {len(items)} items, "
          f"{len(counts)} distinct keys")
    for key, count in counts.most_common(TOP_KEYS):
        example = next((str(r[key]) for r in parsed if key in r and r[key] not in ("", None)), "")
        print(f"    {key:<28} {count / len(items):.2f} filled   e.g. {example[:40]!r}")


def length_signal(benchmark, rules):
    table = item_table(benchmark, rules)
    if table is None:
        return None
    content = load_table(benchmark, "items", columns=["item_id", "content"]).set_index("item_id")["content"]
    length = np.log1p(content.fillna("").str.len().reindex(table["item_id"]).to_numpy())
    return spearmanr(length, table["ease"]).statistic


def main():
    rules = load_rules()
    for benchmark in BENCHMARKS:
        describe_features(benchmark)

    print("\nSpearman between log text length and item ease (negative: longer is harder):")
    for benchmark in BENCHMARKS:
        rho = length_signal(benchmark, rules)
        if rho is not None:
            print(f"    {benchmark:<20}{rho:>7.3f}")


if __name__ == "__main__":
    main()
