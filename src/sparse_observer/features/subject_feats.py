"""Subject attributes: how complete they are and how many model names recur across benchmarks."""

import pandas as pd

from sparse_observer.data.loader import BENCHMARKS, load_table

SUBJECT_COLUMNS = [
    "subject_id",
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "harness_version",
    "reasoning_effort",
    "subject_features_extra",
]


def load_subjects(benchmark):
    subjects = load_table(benchmark, "subjects", columns=SUBJECT_COLUMNS)
    subjects["benchmark"] = benchmark
    return subjects


def all_subjects():
    return pd.concat([load_subjects(b) for b in BENCHMARKS], ignore_index=True)


def missing_share(subjects):
    cols = SUBJECT_COLUMNS[1:]
    empty = subjects[cols].isna() | (subjects[cols] == "")
    return empty.groupby(subjects["benchmark"]).mean()


def overlap_matrix(subjects):
    names = {b: set(g["normalized_name"]) for b, g in subjects.groupby("benchmark")}
    return pd.DataFrame(
        [[len(names[a] & names[b]) for b in BENCHMARKS] for a in BENCHMARKS],
        index=BENCHMARKS,
        columns=BENCHMARKS,
    )


def seen_elsewhere(subjects):
    shares = {}
    for benchmark, group in subjects.groupby("benchmark"):
        others = set(subjects.loc[subjects["benchmark"] != benchmark, "normalized_name"])
        names = set(group["normalized_name"])
        shares[benchmark] = len(names & others) / len(names)
    return pd.Series(shares)


def main():
    subjects = all_subjects()
    print("subjects per benchmark / distinct model names:")
    print(subjects.groupby("benchmark")["normalized_name"].agg(subjects="size", names="nunique").to_string())
    print("\nshare of empty values per attribute:")
    print(missing_share(subjects).round(2).to_string())
    print("\nshared model names between benchmarks:")
    print(overlap_matrix(subjects).to_string())
    print("\nshare of a benchmark's model names seen in some other benchmark:")
    print(seen_elsewhere(subjects).round(2).to_string())
    print("\nmost common harness values:")
    print(subjects["harness"].value_counts(dropna=False).head(8).to_string())


if __name__ == "__main__":
    main()
