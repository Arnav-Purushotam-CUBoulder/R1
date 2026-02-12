from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_mean_std(series: pd.Series) -> str:
    return f"{series.mean():.3f} ± {series.std(ddof=1):.3f}"


def summarize_results(results_dir: Path) -> None:
    raw_dir = results_dir / "raw"
    summary_dir = ensure_dir(results_dir / "summary")
    fig_dir = ensure_dir(results_dir / "figures")

    runs = pd.read_csv(raw_dir / "run_results.csv")
    datasets = pd.read_csv(raw_dir / "dataset_summary.csv")

    aggregated = (
        runs.groupby(["model"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            search_time_mean=("search_time_sec", "mean"),
            fit_time_mean=("fit_time_sec", "mean"),
            inference_time_mean=("inference_time_sec", "mean"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    aggregated.to_csv(summary_dir / "model_summary.csv", index=False)

    per_dataset = (
        runs.groupby(["dataset", "model"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            search_time_mean=("search_time_sec", "mean"),
            fit_time_mean=("fit_time_sec", "mean"),
        )
        .reset_index()
    )
    per_dataset.to_csv(summary_dir / "dataset_model_summary.csv", index=False)

    mlp = per_dataset[per_dataset["model"] == "MLP"].set_index("dataset")
    comparisons = []
    for model in per_dataset["model"].unique():
        if model == "MLP":
            continue
        model_df = per_dataset[per_dataset["model"] == model].set_index("dataset")
        joined = model_df.join(mlp, lsuffix="_model", rsuffix="_mlp")
        try:
            wilcoxon_p = float(stats.wilcoxon(joined["macro_f1_mean_model"], joined["macro_f1_mean_mlp"]).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        comparisons.append(
            {
                "model": model,
                "macro_f1_mean_diff_vs_mlp": (joined["macro_f1_mean_model"] - joined["macro_f1_mean_mlp"]).mean(),
                "macro_f1_wins_vs_mlp": int((joined["macro_f1_mean_model"] > joined["macro_f1_mean_mlp"]).sum()),
                "ece_better_count_vs_mlp": int((joined["ece_mean_model"] < joined["ece_mean_mlp"]).sum()),
                "fit_time_faster_count_vs_mlp": int((joined["fit_time_mean_model"] < joined["fit_time_mean_mlp"]).sum()),
                "wilcoxon_p_macro_f1": wilcoxon_p,
            }
        )
    pd.DataFrame(comparisons).to_csv(summary_dir / "vs_mlp_summary.csv", index=False)

    macro_table = (
        per_dataset.pivot(index="dataset", columns="model", values="macro_f1_mean")
        .reset_index()
        .sort_values("dataset")
    )
    macro_table.to_csv(summary_dir / "macro_f1_by_dataset.csv", index=False)

    win_rows = []
    for dataset, frame in per_dataset.groupby("dataset"):
        best_row = frame.sort_values(["macro_f1_mean", "ece_mean"], ascending=[False, True]).iloc[0]
        win_rows.append(
            {
                "dataset": dataset,
                "best_model": best_row["model"],
                "best_macro_f1": best_row["macro_f1_mean"],
            }
        )
    pd.DataFrame(win_rows).sort_values("dataset").to_csv(summary_dir / "dataset_winners.csv", index=False)

    rank_df = per_dataset.pivot(index="dataset", columns="model", values="macro_f1_mean")
    ranks = rank_df.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean().sort_values()
    mean_ranks.rename("mean_rank").reset_index().to_csv(summary_dir / "macro_f1_mean_ranks.csv", index=False)
    friedman_stat, friedman_p = stats.friedmanchisquare(*[rank_df[col] for col in rank_df.columns])
    pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "friedman_statistic": float(friedman_stat),
                "friedman_pvalue": float(friedman_p),
            }
        ]
    ).to_csv(summary_dir / "significance_tests.csv", index=False)

    stability = (
        per_dataset.groupby("model")
        .agg(
            mean_macro_f1_std=("macro_f1_std", "mean"),
            mean_accuracy_std=("accuracy_std", "mean"),
            mean_brier_std=("brier_std", "mean"),
            mean_ece_std=("ece_std", "mean"),
        )
        .reset_index()
        .sort_values("mean_macro_f1_std")
    )
    stability.to_csv(summary_dir / "stability_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=aggregated.sort_values("macro_f1_mean", ascending=False),
        x="model",
        y="macro_f1_mean",
        hue="model",
        legend=False,
        palette="deep",
    )
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean macro-F1")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(fig_dir / "macro_f1_overall.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=aggregated.sort_values("fit_time_mean"),
        x="model",
        y="fit_time_mean",
        hue="model",
        legend=False,
        palette="muted",
    )
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean final fit time (s)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(fig_dir / "fit_time_overall.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=aggregated,
        x="fit_time_mean",
        y="macro_f1_mean",
        hue="model",
        s=180,
        palette="tab10",
    )
    for _, row in aggregated.iterrows():
        ax.text(row["fit_time_mean"] * 1.03, row["macro_f1_mean"] + 0.0005, row["model"], fontsize=10)
    plt.xlabel("Mean final fit time (s)")
    plt.ylabel("Mean macro-F1")
    plt.tight_layout()
    plt.savefig(fig_dir / "macro_f1_vs_fit_time.png", dpi=200)
    plt.close()

    rank_plot = mean_ranks.rename_axis("model").reset_index(name="mean_rank")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=rank_plot,
        x="model",
        y="mean_rank",
        hue="model",
        legend=False,
        palette="crest",
    )
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean rank (lower is better)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(fig_dir / "macro_f1_mean_rank.png", dpi=200)
    plt.close()

    latex_summary = aggregated.copy()
    for col in ["accuracy_mean", "macro_f1_mean", "brier_mean", "ece_mean"]:
        latex_summary[col] = latex_summary[col].map(lambda v: f"{v:.3f}")
    for col in ["search_time_mean", "fit_time_mean", "inference_time_mean"]:
        latex_summary[col] = latex_summary[col].map(lambda v: f"{v:.3f}")
    latex_summary.to_latex(summary_dir / "model_summary.tex", index=False, escape=False)
    datasets.to_latex(summary_dir / "dataset_summary.tex", index=False, escape=False)
