from __future__ import annotations

from pathlib import Path

import pandas as pd


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_pvalue(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def render_tabular(columns: list[str], rows: list[list[str]], align: str) -> str:
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\hline"]
    lines.append(" & ".join(columns) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def build_assets(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_summary = pd.read_csv(results_dir / "summary" / "model_summary_with_ranks.csv")
    ranks = pd.read_csv(results_dir / "summary" / "macro_f1_mean_ranks.csv")
    vs_mlp = pd.read_csv(results_dir / "summary" / "vs_mlp_summary.csv")
    sig = pd.read_csv(results_dir / "summary" / "significance_tests.csv").iloc[0]
    stability = pd.read_csv(results_dir / "summary" / "stability_summary.csv")
    datasets = pd.read_csv(results_dir / "raw" / "dataset_summary.csv")
    runs = pd.read_csv(results_dir / "raw" / "run_results.csv")

    best_macro_idx = model_summary["macro_f1_mean"].idxmax()
    best_ece_idx = model_summary["ece_mean"].idxmin()
    best_fit_idx = model_summary["fit_time_mean"].idxmin()
    best_search_idx = model_summary["search_time_mean"].idxmin()
    best_rank_idx = model_summary["mean_rank"].idxmin()

    overall_rows = []
    for idx, row in model_summary.iterrows():
        model_name = latex_escape(row["model"])
        overall_rows.append(
            [
                rf"\textbf{{{model_name}}}" if idx == best_rank_idx else model_name,
                rf"\textbf{{{format_float(row['mean_rank'], 2)}}}" if idx == best_rank_idx else format_float(row["mean_rank"], 2),
                rf"\textbf{{{format_float(row['macro_f1_mean'])}}}" if idx == best_macro_idx else format_float(row["macro_f1_mean"]),
                format_float(row["accuracy_mean"]),
                format_float(row["brier_mean"]),
                rf"\textbf{{{format_float(row['ece_mean'])}}}" if idx == best_ece_idx else format_float(row["ece_mean"]),
                rf"\textbf{{{format_float(row['search_time_mean'], 2)}}}" if idx == best_search_idx else format_float(row["search_time_mean"], 2),
                rf"\textbf{{{format_float(row['fit_time_mean'], 3)}}}" if idx == best_fit_idx else format_float(row["fit_time_mean"], 3),
            ]
        )
    (output_dir / "table_overall.tex").write_text(
        render_tabular(
            columns=["Model", "Rank", "Macro-F1", "Acc.", "Brier", "ECE", "Search (s)", "Fit (s)"],
            rows=overall_rows,
            align="lrrrrrrr",
        )
    )

    dataset_rows = []
    for _, row in datasets.sort_values(["instances", "dataset"]).iterrows():
        dataset_rows.append(
            [
                latex_escape(row["dataset"]),
                str(int(row["task_id"])),
                str(int(row["dataset_id"])),
                str(int(row["instances"])),
                str(int(row["features"])),
                str(int(row["classes"])),
                format_float(row["minority_share"], 3),
            ]
        )
    (output_dir / "table_datasets.tex").write_text(
        render_tabular(
            columns=["Dataset", "Task ID", "Data ID", "Samples", "Features", "Classes", "Minority share"],
            rows=dataset_rows,
            align="lrrrrrr",
        )
    )

    vs_rows = []
    for _, row in vs_mlp.sort_values("macro_f1_mean_diff_vs_mlp", ascending=False).iterrows():
        vs_rows.append(
            [
                latex_escape(row["model"]),
                format_float(row["macro_f1_mean_diff_vs_mlp"]),
                str(int(row["macro_f1_wins_vs_mlp"])),
                str(int(row["ece_better_count_vs_mlp"])),
                str(int(row["fit_time_faster_count_vs_mlp"])),
                format_pvalue(row["holm_adjusted_p_macro_f1"]),
            ]
        )
    (output_dir / "table_vs_mlp.tex").write_text(
        render_tabular(
            columns=["Model", r"$\Delta$ Macro-F1", "Wins vs. MLP", "Better ECE", "Faster fit", "Holm p"],
            rows=vs_rows,
            align="lrrrrr",
        )
    )

    stability_rows = []
    for _, row in stability.iterrows():
        stability_rows.append(
            [
                latex_escape(row["model"]),
                format_float(row["mean_macro_f1_std"]),
                format_float(row["mean_accuracy_std"]),
                format_float(row["mean_brier_std"]),
                format_float(row["mean_ece_std"]),
            ]
        )
    (output_dir / "table_stability.tex").write_text(
        render_tabular(
            columns=["Model", "Macro-F1 std", "Acc. std", "Brier std", "ECE std"],
            rows=stability_rows,
            align="lrrrr",
        )
    )

    total_search_fit_minutes = (runs["search_time_sec"].sum() + runs["fit_time_sec"].sum()) / 60.0
    dataset_count = len(datasets)
    seeds = int(runs["seed"].nunique())
    model_count = int(runs["model"].nunique())
    configs_per_model = int(pd.read_csv(results_dir / "raw" / "search_results.csv").groupby(["dataset", "seed", "model"]).size().iloc[0])
    total_runs = len(runs)
    total_model_fits = len(pd.read_csv(results_dir / "raw" / "search_results.csv")) + len(runs)

    mlp_row = model_summary[model_summary["model"] == "MLP"].iloc[0]
    logistic_row = model_summary[model_summary["model"] == "Logistic Regression"].iloc[0]
    logistic_vs_mlp = vs_mlp[vs_mlp["model"] == "Logistic Regression"].iloc[0]
    random_forest_vs_mlp = vs_mlp[vs_mlp["model"] == "Random Forest"].iloc[0]
    extra_trees_vs_mlp = vs_mlp[vs_mlp["model"] == "Extra Trees"].iloc[0]
    hgb_vs_mlp = vs_mlp[vs_mlp["model"] == "HistGradientBoosting"].iloc[0]
    best_macro_row = model_summary.iloc[best_macro_idx]
    best_rank_row = model_summary.iloc[best_rank_idx]
    best_ece_row = model_summary.iloc[best_ece_idx]
    best_macro_name = str(best_macro_row["model"])
    best_rank_name = str(best_rank_row["model"])
    best_ece_name = str(best_ece_row["model"])
    dataset_winners = pd.read_csv(results_dir / "summary" / "dataset_winners.csv")["best_model"].value_counts()
    instances_min = int(datasets["instances"].min())
    instances_max = int(datasets["instances"].max())
    features_min = int(datasets["features"].min())
    features_max = int(datasets["features"].max())
    classes_min = int(datasets["classes"].min())
    classes_max = int(datasets["classes"].max())

    macros = f"""
\\newcommand{{\\DatasetCount}}{{{dataset_count}}}
\\newcommand{{\\SeedCount}}{{{seeds}}}
\\newcommand{{\\ModelCount}}{{{model_count}}}
\\newcommand{{\\SearchConfigsPerModel}}{{{configs_per_model}}}
\\newcommand{{\\TotalRuns}}{{{total_runs}}}
\\newcommand{{\\TotalModelFits}}{{{total_model_fits}}}
\\newcommand{{\\BenchmarkSuiteName}}{{OpenML-CC18}}
\\newcommand{{\\HardwareCPU}}{{Apple M3 Pro}}
\\newcommand{{\\HardwareRAM}}{{18 GB}}
\\newcommand{{\\MinInstances}}{{{instances_min}}}
\\newcommand{{\\MaxInstances}}{{{instances_max}}}
\\newcommand{{\\MinFeatures}}{{{features_min}}}
\\newcommand{{\\MaxFeatures}}{{{features_max}}}
\\newcommand{{\\MinClasses}}{{{classes_min}}}
\\newcommand{{\\MaxClasses}}{{{classes_max}}}
\\newcommand{{\\BestMacroFOneModel}}{{{latex_escape(best_macro_name)}}}
\\newcommand{{\\BestMacroFOneValue}}{{{format_float(best_macro_row['macro_f1_mean'])}}}
\\newcommand{{\\BestRankModel}}{{{latex_escape(best_rank_name)}}}
\\newcommand{{\\BestMeanRank}}{{{format_float(best_rank_row['mean_rank'], 2)}}}
\\newcommand{{\\BestECEModel}}{{{latex_escape(best_ece_name)}}}
\\newcommand{{\\BestECEValue}}{{{format_float(best_ece_row['ece_mean'])}}}
\\newcommand{{\\MLPMacroFOneValue}}{{{format_float(mlp_row['macro_f1_mean'])}}}
\\newcommand{{\\MLPMeanRank}}{{{format_float(mlp_row['mean_rank'], 2)}}}
\\newcommand{{\\MLPMeanECE}}{{{format_float(mlp_row['ece_mean'])}}}
\\newcommand{{\\MLPStability}}{{{format_float(stability[stability['model'] == 'MLP']['mean_macro_f1_std'].iloc[0])}}}
\\newcommand{{\\LogisticFitTime}}{{{format_float(logistic_row['fit_time_mean'], 3)}}}
\\newcommand{{\\MLPFitTime}}{{{format_float(mlp_row['fit_time_mean'], 3)}}}
\\newcommand{{\\BenchmarkWallMinutes}}{{{format_float(total_search_fit_minutes, 1)}}}
\\newcommand{{\\FriedmanPValue}}{{{format_pvalue(sig['friedman_pvalue'])}}}
\\newcommand{{\\MLPTaskWins}}{{{int(dataset_winners.get('MLP', 0))}}}
\\newcommand{{\\ExtraTreesTaskWins}}{{{int(dataset_winners.get('Extra Trees', 0))}}}
\\newcommand{{\\HGBTaskWins}}{{{int(dataset_winners.get('HistGradientBoosting', 0))}}}
\\newcommand{{\\RandomForestTaskWins}}{{{int(dataset_winners.get('Random Forest', 0))}}}
\\newcommand{{\\LogisticTaskWins}}{{{int(dataset_winners.get('Logistic Regression', 0))}}}
\\newcommand{{\\RandomForestWinsVsMLP}}{{{int(random_forest_vs_mlp['macro_f1_wins_vs_mlp'])}}}
\\newcommand{{\\ExtraTreesWinsVsMLP}}{{{int(extra_trees_vs_mlp['macro_f1_wins_vs_mlp'])}}}
\\newcommand{{\\HGBWinsVsMLP}}{{{int(hgb_vs_mlp['macro_f1_wins_vs_mlp'])}}}
\\newcommand{{\\LogisticWinsVsMLP}}{{{int(logistic_vs_mlp['macro_f1_wins_vs_mlp'])}}}
\\newcommand{{\\RandomForestMacroGainVsMLP}}{{{format_float(random_forest_vs_mlp['macro_f1_mean_diff_vs_mlp'])}}}
\\newcommand{{\\ExtraTreesMacroGainVsMLP}}{{{format_float(extra_trees_vs_mlp['macro_f1_mean_diff_vs_mlp'])}}}
\\newcommand{{\\HGBMacroGainVsMLP}}{{{format_float(hgb_vs_mlp['macro_f1_mean_diff_vs_mlp'])}}}
\\newcommand{{\\LogisticMacroGainVsMLP}}{{{format_float(logistic_vs_mlp['macro_f1_mean_diff_vs_mlp'])}}}
\\newcommand{{\\RandomForestHolmP}}{{{format_pvalue(random_forest_vs_mlp['holm_adjusted_p_macro_f1'])}}}
\\newcommand{{\\ExtraTreesHolmP}}{{{format_pvalue(extra_trees_vs_mlp['holm_adjusted_p_macro_f1'])}}}
\\newcommand{{\\HGBHolmP}}{{{format_pvalue(hgb_vs_mlp['holm_adjusted_p_macro_f1'])}}}
\\newcommand{{\\LogisticHolmP}}{{{format_pvalue(logistic_vs_mlp['holm_adjusted_p_macro_f1'])}}}
""".strip() + "\n"
    (output_dir / "results_macros.tex").write_text(macros)


if __name__ == "__main__":
    build_assets(Path("results"), Path("paper/generated"))
