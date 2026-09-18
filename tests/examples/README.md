# MultimodalVA — usage examples

Three ways to run the exact same pipelines. Pick whichever fits how you work —
they all call `multimodalva.run()` under the hood and produce identical outputs
(`output_dir/final/`, `predictions/`, `hpo/`, plus `hpo_leaderboard.csv` /
`hpo_convergence.png` diagnostics when HPO runs).

**New here? Start with the notebooks** in [`notebooks/`](notebooks/) — they walk
through the whole workflow (load → EDA → preprocess → model → save → publish) and
are meant to be copied and adapted:

| Notebook | Covers |
|---|---|
| `notebooks/01_getting_started.ipynb` | unimodal **text** & **tabular**, inspecting results, model reuse, Hub publishing |
| `notebooks/02_multimodal_fusion.ipynb` | **feature fusion**, **data fusion**, **decision fusion** (voting & stacking) |

The script templates below show the same pipelines through each non-notebook
interface:

| File | Interface | Best for |
|---|---|---|
| `example_python_api.py` | `from multimodalva import run` | scripts, batch jobs |
| `example_cli.sh` | `multimodalva run --task ...` | SLURM/sbatch, non-coders |
| `example_yaml.sh` + `example_config_*.yaml` | `multimodalva run config.yaml` | reproducible, shareable experiments |

Each interface script covers the three pipeline families — **unimodal text**,
**unimodal tabular**, and **ensemble** (data fusion / feature fusion / voting /
stacking) — plus **Hugging Face Hub publishing** (available for the text-based
models: `text` and `data_fusion`).

## Quick start

```bash
# Python
python tests/examples/example_python_api.py tabular

# CLI (or: python -m multimodalva.cli ...)
bash tests/examples/example_cli.sh

# YAML
multimodalva run tests/examples/example_config_tabular.yaml
```

All examples use the built-in synthetic dataset (`data("va_sample")`) so they
run with no external files. To use your own data, point `data` at a CSV path
(or pass a preprocessed DataFrame to the Python API) and set `label`,
`text_col`, and `features` to your columns.

## Key arguments

- **`data`** — CSV/Parquet path, an in-memory DataFrame, or `(train_df, test_df)`.
- **`features`** — an explicit list, a regex (`"re:^i\\d{3}[a-zA-Z]$"`), or
  `"auto"` (every column except label/text/filter/split).
- **`filters`** — `{column: value | [values]}` row filter (omit for none).
- **`split`** — reuse a fixed train/test split for cross-experiment comparison:
  a column name, a `{"train_ids": [...], "test_ids": [...]}` dict or JSON path
  (with `id_col`), or an explicit `(train_df, test_df)`.
- **`optimize` / `n_trials` / `metric`** — HPO controls (`metric` defaults to
  `f1_macro`).
- Anything else is forwarded to the underlying classifier's `run()`
  (`use_lora`, `time_limit`, `push_to_hub`, ...).

## Smoke test

Verify the install works end-to-end (offline, no GPU):

```bash
python tests/smoke_test.py              # core checks
python tests/smoke_test.py --with-text  # also runs a tiny text model (~17MB download)
```
