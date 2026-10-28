"""How often, and by how much, the item-pooled model loses to its no-pooling control."""

from pathlib import Path

import numpy as np
import pandas as pd

SCORES = Path("results/breakeven_v1/pair_scores.csv")
CONTROL = "m0_s1"
TREATED = ["m1_s1", "m3_s1", "m10_s1"]
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
DRAW_SIZE = 9
N_DRAWS = 5000


def main():
    scores = pd.read_csv(SCORES)
    wide = scores.pivot_table(index=["benchmark", "subject_id", "seed"],
                              columns="variant", values="brier_alc")
    rng = np.random.default_rng(0)
    for variant in TREATED:
        diff = (wide[variant] - wide[CONTROL]).dropna()
        draws = rng.choice(diff.to_numpy(), size=(N_DRAWS, DRAW_SIZE)).mean(axis=1)
        print(f"\n{variant} minus {CONTROL}, per pair and seed ({len(diff)} rows; negative is better)")
        print("  quantiles:", {q: round(float(diff.quantile(q)), 4) for q in QUANTILES})
        print(f"  mean {diff.mean():.4f}, share worse than +0.001: {(diff > 0.001).mean():.3f}, "
              f"worst {diff.max():.4f}")
        print(f"  random {DRAW_SIZE}-pair draws: 5th percentile {np.percentile(draws, 5):.4f}, "
              f"95th {np.percentile(draws, 95):.4f}, share of draws where the item model is worse: "
              f"{(draws > 0).mean():.3f}")


if __name__ == "__main__":
    main()
