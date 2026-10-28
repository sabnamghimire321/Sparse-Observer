"""List or download the Measurement Data Bank tables needed here (about 64 MB, not the raw archives)."""

import argparse
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, RepoFile, snapshot_download

REPO_ID = "aims-foundations/measurement-db"
SKIP_FOLDERS = {"raw", "reproduction_checks"}
SKIP_NAMES = {"traces.parquet", "assets.parquet"}


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def select_files():
    """Files to download: everything except raw archives, reproduction checks, traces and assets."""
    api = HfApi()
    tree = api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
    chosen = []
    for entry in tree:
        if not isinstance(entry, RepoFile):
            continue
        parts = entry.path.split("/")
        if len(parts) > 1 and parts[1] in SKIP_FOLDERS:
            continue
        if parts[-1] in SKIP_NAMES:
            continue
        chosen.append((entry.path, entry.size))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    chosen = select_files()
    by_folder = defaultdict(list)
    for path, size in chosen:
        folder = path.split("/")[0] if "/" in path else "(root)"
        by_folder[folder].append((path, size))

    for folder, items in sorted(by_folder.items()):
        print(f"{folder}  ({len(items)} files, {human(sum(s for _, s in items))})")
        for path, size in sorted(items):
            print(f"    {human(size):>10}  {path}")

    total = sum(size for _, size in chosen)
    print(f"\nselected: {len(chosen)} files, {human(total)}")

    if not args.execute:
        print("dry run only; add --execute to download")
        return

    Path(args.out).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=[path for path, _ in chosen],
    )
    print(f"downloaded to {args.out}")


if __name__ == "__main__":
    main()
