# R1

This repository contains a reproducible study on laptop-scale tabular classification:
classical baselines versus a modest multilayer perceptron (MLP) on a deterministic
subset of numerical tasks from the `OpenML-CC18` benchmark suite.

## Environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e .
```

## Run the benchmark

```bash
source .venv/bin/activate
python scripts/run_benchmark.py
python scripts/summarize_results.py
python scripts/build_manuscript_assets.py
```

Raw results are written to `results/raw/`, summaries to `results/summary/`, and
figures to `results/figures/`.

## Benchmark rule

Datasets are not hand-picked. The benchmark starts from the full `OpenML-CC18`
classification suite and keeps only active tasks that satisfy all of the following:

- `500 <= n_samples <= 20000`
- `4 <= n_predictor_features <= 60`
- `2 <= n_classes <= 10`
- no missing values in OpenML metadata
- at most one symbolic feature in metadata, then a final check that all predictors
  are numeric after removing the task target

Every dataset-model-seed combination receives exactly 12 fixed hyperparameter
configurations. All models use the same explicit `60/20/20` train/validation/test
split, and no model uses internal early stopping.

## Build the manuscript

```bash
source .venv/bin/activate
cd paper
tectonic main.tex
```

The manuscript source lives in `paper/main.tex`, generated LaTeX tables/macros
in `paper/generated/`, and the rendered PDF at `paper/main.pdf`.
