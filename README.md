# R1

This repository contains a reproducible study on laptop-scale tabular classification:
classical models versus a width-limited multilayer perceptron (MLP) across public
numerical benchmarks.

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
```

Raw results are written to `results/raw/`, summaries to `results/summary/`, and
figures to `results/figures/`.

## Build the manuscript

```bash
source .venv/bin/activate
python scripts/build_manuscript_assets.py
cd paper
tectonic main.tex
```

The manuscript source lives in `paper/main.tex`, generated LaTeX tables/macros
in `paper/generated/`, and the rendered PDF at `paper/main.pdf`.
