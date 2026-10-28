"""Run every stage of the project in order, writing one log per stage to logs/run_all."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path("logs/run_all")
PY = sys.executable

STAGES = [
    ("tests", [PY, "-m", "pytest", "tests", "-q"]),
    ("validate", [PY, "src/sparse_observer/data/validate.py"]),
    ("labels", [PY, "src/sparse_observer/data/labels.py"]),
    ("pair_stats", [PY, "src/sparse_observer/data/pair_stats.py"]),
    ("protocol", [PY, "src/sparse_observer/simulator/protocol.py"]),
    ("subject_overlap", [PY, "src/sparse_observer/features/subject_feats.py"]),
    ("ability_coverage", [PY, "src/sparse_observer/models/online_state.py"]),
    ("main_runner", [PY, "src/sparse_observer/experiments/runner.py"]),
    ("stats", [PY, "src/sparse_observer/evaluation/stats.py"]),
    ("headroom", [PY, "src/sparse_observer/evaluation/headroom.py"]),
    ("coverage", [PY, "src/sparse_observer/simulator/stream.py"]),
    ("pooled", [PY, "src/sparse_observer/experiments/pooled.py"]),
    ("breakeven", [PY, "src/sparse_observer/experiments/breakeven.py"]),
    ("breakeven_stats", [PY, "src/sparse_observer/evaluation/breakeven_stats.py"]),
    ("downside", [PY, "src/sparse_observer/evaluation/downside.py"]),
    ("heterogeneity", [PY, "src/sparse_observer/evaluation/heterogeneity.py"]),
    ("interval_coverage", [PY, "src/sparse_observer/experiments/coverage.py"]),
    ("item_features", [PY, "src/sparse_observer/features/item_feats.py"]),
    ("report_assets", [PY, "scripts/make_report_assets.py"]),
]
# needs sentence-transformers and a model download
OPTIONAL = [("text_transfer", [PY, "src/sparse_observer/features/text_embed.py"])]


def run(name, command):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    log = LOG_DIR / f"{name}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (exit {result.returncode})"
    print(f"{name:<18} {status:<18} {time.time() - start:7.1f} s")
    if not ok:
        print("last lines of the log:")
        print("\n".join((result.stdout + result.stderr).splitlines()[-15:]))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-embeddings", action="store_true")
    parser.add_argument("--only", nargs="+", help="run only stages whose name contains one of these")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    stages = STAGES + (OPTIONAL if args.with_embeddings else [])
    if args.only:
        stages = [s for s in stages if any(key in s[0] for key in args.only)]
    if args.list:
        for name, command in stages:
            print(f"{name:<18} {' '.join(command[1:])}")
        return

    start = time.time()
    for name, command in stages:
        if not run(name, command):
            raise SystemExit(f"stopped at {name}; see {LOG_DIR / (name + '.log')}")
    print(f"\nall {len(stages)} stages finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
