"""Build the report's figures and tables from saved result files only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MAIN = Path("results/main_v1/pair_scores.csv")
COVERAGE = Path("results/coverage_v1/coverage_summary.csv")
BREAKEVEN = Path("tables/breakeven_v1.csv")
FIG_DIR = Path("figures")
TABLE_DIR = Path("tables")

BUDGETS = [0, 1, 3, 7, 15, 31]
LABEL_BUDGETS = [1, 3, 7, 15, 31]
SMALL_PAIR_ITEMS = 150
MULTI = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
ORDER = ["constant_0.5", "pair_mean_k2", "offset_logit_s0", "offset_logit_s0.5",
         "offset_logit_s1", "empirical_bayes_t1", "empirical_bayes_t0.5"]
PLOTTED = ["constant_0.5", "pair_mean_k2", "offset_logit_s0", "empirical_bayes_t1"]
METHODS = ["wilson", "pool_aware_wilson", "eb", "eb_wrong_prior"]
H5_BUDGETS = [7, 15, 31]


def main_table(scores):
    columns = [f"brier_{n}" for n in BUDGETS]
    small = scores[scores["n_items"] <= SMALL_PAIR_ITEMS]
    pairs = scores.drop_duplicates(["predictor", "benchmark", "subject_id"]).groupby("predictor").size()
    table = pd.DataFrame({
        "pairs": pairs,
        "alc_all_pairs": scores.groupby("predictor")["brier_alc"].mean(),
        "alc_small_pairs": small.groupby("predictor")["brier_alc"].mean(),
    }).join(scores.groupby("predictor")[columns].mean())
    return table.loc[ORDER].round(4)


def coverage_table(summary):
    cells = summary[summary["benchmark"].isin(MULTI) & summary["budget"].isin(H5_BUDGETS)]
    wide = cells.pivot_table(index=["benchmark", "budget"], columns="method", values="coverage")
    eb = cells[cells["method"] == "eb"].set_index(["benchmark", "budget"])
    wide["eb_ci_low"] = eb["ci_low"]
    wide["eb_ci_high"] = eb["ci_high"]
    return wide[["eb", "eb_ci_low", "eb_ci_high", "pool_aware_wilson", "wilson", "clopper_pearson",
                 "eb_wrong_prior"]].round(3)


def figure_brier(scores):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    columns = [f"brier_{n}" for n in BUDGETS]
    mean = scores.groupby("predictor")[columns].mean()
    for model in PLOTTED:
        axes[0].plot(range(len(BUDGETS)), mean.loc[model], marker="o", label=model)
    axes[0].set_title("All pairs (mean over seeds)")
    for benchmark in MULTI:
        part = scores[(scores["predictor"] == "pair_mean_k2") & (scores["benchmark"] == benchmark)]
        axes[1].plot(range(len(BUDGETS)), part[columns].mean(), marker="o", label=benchmark)
    axes[1].set_title("pair_mean_k2 by benchmark")
    for ax in axes:
        ax.set_xticks(range(len(BUDGETS)), BUDGETS)
        ax.set_xlabel("label budget")
        ax.set_ylabel("Brier score (lower is better)")
        ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "fig1_brier_by_budget.png", dpi=200)
    plt.close(fig)


def figure_coverage(summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharey=True)
    positions = range(len(LABEL_BUDGETS))
    for ax, benchmark in zip(axes.ravel(), MULTI):
        ax.axhspan(0.85, 0.95, color="0.9", zorder=0)
        ax.axhline(0.90, color="0.4", linewidth=1, linestyle="--")
        for method in METHODS:
            part = summary[(summary["benchmark"] == benchmark) & (summary["method"] == method)]
            part = part.sort_values("budget")
            ax.plot(positions, part["coverage"], marker="o", label=method)
            if method == "eb":
                ax.fill_between(positions, part["ci_low"], part["ci_high"], alpha=0.2)
        ax.set_title(benchmark)
        ax.set_xticks(positions, LABEL_BUDGETS)
        ax.set_xlabel("label budget")
        ax.set_ylabel("empirical coverage of a 90% interval")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.savefig(FIG_DIR / "fig2_coverage_by_budget.png", dpi=200)
    plt.close(fig)


def figure_breakeven(table):
    table = table.assign(m=table["variant"].str.extract(r"m(\d+)_")[0].astype(int))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ms = sorted(table["m"].unique())
    for benchmark in MULTI:
        part = table[table["scope"] == benchmark].sort_values("m")
        yerr = [part["mean_diff"] - part["ci_low"], part["ci_high"] - part["mean_diff"]]
        ax.errorbar(range(len(ms)), part["mean_diff"], yerr=yerr, marker="o", capsize=3, label=benchmark)
    ax.axhline(0, color="0.4", linewidth=1)
    ax.axhline(-0.003, color="0.4", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(ms)), ms)
    ax.set_xlabel("other pairs on the same benchmark (source pairs)")
    ax.set_ylabel("change in Brier ALC vs no pooling")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "fig3_breakeven.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)
    scores = pd.read_csv(MAIN)
    summary = pd.read_csv(COVERAGE)

    main_table(scores).to_csv(TABLE_DIR / "table_main_alc.csv")
    coverage_table(summary).to_csv(TABLE_DIR / "table_coverage_h5_cells.csv")
    figure_brier(scores)
    figure_coverage(summary)
    figure_breakeven(pd.read_csv(BREAKEVEN))
    print(main_table(scores).to_string())
    print()
    print(coverage_table(summary).to_string())
    print("\nwrote figures/fig1_brier_by_budget.png, fig2_coverage_by_budget.png, fig3_breakeven.png")
    print("wrote tables/table_main_alc.csv, table_coverage_h5_cells.csv")


if __name__ == "__main__":
    main()
