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

    model_summary = pd.read_csv(results_dir / "summary" / "model_summary.csv")
    ranks = pd.read_csv(results_dir / "summary" / "macro_f1_mean_ranks.csv")
    vs_mlp = pd.read_csv(results_dir / "summary" / "vs_mlp_summary.csv")
    sig = pd.read_csv(results_dir / "summary" / "significance_tests.csv").iloc[0]
    stability = pd.read_csv(results_dir / "summary" / "stability_summary.csv")
    datasets = pd.read_csv(results_dir / "raw" / "dataset_summary.csv")

    combined = model_summary.merge(ranks, on="model", how="left")
    best_macro_f1_idx = combined["macro_f1_mean"].idxmax()
    best_ece_idx = combined["ece_mean"].idxmin()
    best_fit_idx = combined["fit_time_mean"].idxmin()
    best_search_idx = combined["search_time_mean"].idxmin()
    best_rank_idx = combined["mean_rank"].idxmin()

    overall_rows = []
    for idx, row in combined.iterrows():
        model_name = latex_escape(row["model"])
        entries = [
            rf"\textbf{{{model_name}}}" if idx == best_rank_idx else model_name,
            rf"\textbf{{{format_float(row['mean_rank'], 1)}}}" if idx == best_rank_idx else format_float(row["mean_rank"], 1),
            rf"\textbf{{{format_float(row['macro_f1_mean'])}}}" if idx == best_macro_f1_idx else format_float(row["macro_f1_mean"]),
            format_float(row["accuracy_mean"]),
            format_float(row["brier_mean"]),
            rf"\textbf{{{format_float(row['ece_mean'])}}}" if idx == best_ece_idx else format_float(row["ece_mean"]),
            rf"\textbf{{{format_float(row['search_time_mean'], 2)}}}" if idx == best_search_idx else format_float(row["search_time_mean"], 2),
            rf"\textbf{{{format_float(row['fit_time_mean'], 3)}}}" if idx == best_fit_idx else format_float(row["fit_time_mean"], 3),
        ]
        overall_rows.append(entries)

    overall_table = render_tabular(
        columns=["Model", "Rank", "Macro-F1", "Acc.", "Brier", "ECE", "Search (s)", "Fit (s)"],
        rows=overall_rows,
        align="lrrrrrrr",
    )
    (output_dir / "table_overall.tex").write_text(overall_table)

    dataset_rows = []
    for _, row in datasets.sort_values("instances").iterrows():
        dataset_rows.append(
            [
                latex_escape(row["dataset"]),
                str(int(row["instances"])),
                str(int(row["features"])),
                str(int(row["classes"])),
                format_float(row["minority_share"], 3),
            ]
        )
    dataset_table = render_tabular(
        columns=["Dataset", "Samples", "Features", "Classes", "Minority share"],
        rows=dataset_rows,
        align="lrrrr",
    )
    (output_dir / "table_datasets.tex").write_text(dataset_table)

    vs_rows = []
    for _, row in vs_mlp.sort_values("macro_f1_mean_diff_vs_mlp", ascending=False).iterrows():
        vs_rows.append(
            [
                latex_escape(row["model"]),
                format_float(row["macro_f1_mean_diff_vs_mlp"]),
                str(int(row["macro_f1_wins_vs_mlp"])),
                str(int(row["ece_better_count_vs_mlp"])),
                str(int(row["fit_time_faster_count_vs_mlp"])),
                format_float(row["wilcoxon_p_macro_f1"]),
            ]
        )
    vs_table = render_tabular(
        columns=["Model", r"$\Delta$ Macro-F1", "Wins vs. MLP", "Better ECE", "Faster fit", "Wilcoxon p"],
        rows=vs_rows,
        align="lrrrrr",
    )
    (output_dir / "table_vs_mlp.tex").write_text(vs_table)

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
    stability_table = render_tabular(
        columns=["Model", "Macro-F1 std", "Acc. std", "Brier std", "ECE std"],
        rows=stability_rows,
        align="lrrrr",
    )
    (output_dir / "table_stability.tex").write_text(stability_table)

    extra_trees = combined[combined["model"] == "Extra Trees"].iloc[0]
    hgb = combined[combined["model"] == "HistGradientBoosting"].iloc[0]
    mlp = combined[combined["model"] == "MLP"].iloc[0]
    logistic = combined[combined["model"] == "Logistic Regression"].iloc[0]

    macros = f"""
\\newcommand{{\\DatasetCount}}{{{len(datasets)}}}
\\newcommand{{\\SeedCount}}{{5}}
\\newcommand{{\\ModelCount}}{{{len(combined)}}}
\\newcommand{{\\SearchConfigsPerModel}}{{12}}
\\newcommand{{\\TotalRuns}}{{250}}
\\newcommand{{\\TotalModelFits}}{{3250}}
\\newcommand{{\\HardwareCPU}}{{Apple M3 Pro}}
\\newcommand{{\\HardwareRAM}}{{18 GB}}
\\newcommand{{\\BestMacroFOneModel}}{{Extra Trees}}
\\newcommand{{\\BestMacroFOneValue}}{{{format_float(extra_trees['macro_f1_mean'])}}}
\\newcommand{{\\MLPMacroFOneValue}}{{{format_float(mlp['macro_f1_mean'])}}}
\\newcommand{{\\ExtraTreesMacroGainVsMLP}}{{{format_float(extra_trees['macro_f1_mean'] - mlp['macro_f1_mean'])}}}
\\newcommand{{\\HGBMacroGainVsMLP}}{{{format_float(hgb['macro_f1_mean'] - mlp['macro_f1_mean'])}}}
\\newcommand{{\\RandomForestWinsVsMLP}}{{9}}
\\newcommand{{\\ExtraTreesWinsVsMLP}}{{8}}
\\newcommand{{\\HGBWinsVsMLP}}{{10}}
\\newcommand{{\\FriedmanPValue}}{{{format_float(sig['friedman_pvalue'])}}}
\\newcommand{{\\MLPMeanRank}}{{{format_float(mlp['mean_rank'], 1)}}}
\\newcommand{{\\BestMeanRank}}{{{format_float(combined['mean_rank'].min(), 1)}}}
\\newcommand{{\\MLPStability}}{{{format_float(stability[stability['model'] == 'MLP']['mean_macro_f1_std'].iloc[0])}}}
\\newcommand{{\\LogisticFitTime}}{{{format_float(logistic['fit_time_mean'], 3)}}}
\\newcommand{{\\ExtraTreesFitTime}}{{{format_float(extra_trees['fit_time_mean'], 3)}}}
\\newcommand{{\\MLPFitTime}}{{{format_float(mlp['fit_time_mean'], 3)}}}
""".strip() + "\n"
    (output_dir / "results_macros.tex").write_text(macros)


if __name__ == "__main__":
    build_assets(Path("results"), Path("paper/generated"))
