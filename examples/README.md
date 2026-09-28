# MultimodalVA — examples

Every file here runs as-is on built-in synthetic data, so you can execute all of
them before you have prepared any data of your own.

## Which file do I open?

Read the three notebooks in order. They are the files that explain what is
happening; the rest are templates to copy once you know what you want to run.

| | File | What it shows |
|---|---|---|
| **1** | `01_start_here_text_and_tabular.ipynb` | The whole workflow, one step per section: load data → look at it → train a model on the **narrative** → train a model on the **symptom indicators** → compare → save and reuse → publish |
| **2** | `02_combine_text_and_tabular_fusion.ipynb` | Using the narrative and the indicators **together** — data fusion, feature fusion, and decision fusion (voting, stacking) — and comparing all four |
| **3** | `03_evaluate_and_compare_results.ipynb` | What to do with the output: training several models at once, the predicted cause distribution, leaderboards, confidence intervals, confusion matrices, CSMF, top-k, calibration — with a menu of every results function |

Then copy whichever template matches how you work. All three do the same thing;
none is more capable than the others.

| File | Copy this if | Note |
|---|---|---|
| `04_python_script_template.py` | You write Python, and your data needs cleaning in pandas first | `run()` accepts a DataFrame directly |
| `05_command_line_template.sh` | You would rather not write Python, or you submit cluster jobs | One command per run |
| `06_config_file_template.sh` + `config_*.yaml` | You want the settings themselves to be the record of what you ran | The only comfortable way to set up voting and stacking |

## What actually happens when you run one

```
your CSV ──▶ split into train / test
                     │
                     ├─▶ (optional) search for good settings   ← hyperparams=Optimize()
                     │
                     └─▶ train on the training rows
                                  │
                                  └─▶ predict the test rows
                                              │
                                     output_dir/
                                       final/          the trained model
                                       predictions/    predictions_top1.csv
                                                       predictions_full.csv
                                                       predictions_topk.csv
                                       hpo/            the search, if one ran
                                       *.log, *.json   settings, timing, metrics
```

Those three CSVs have the same name and the same columns for every pipeline, so
anything that reads one can read all of them.

The four multi-model tasks — `data_fusion`, `feature_fusion`, `voting` and
`stacking` — put all of that one level down, in a folder named after the method
(`output_dir/stacking/`, `output_dir/soft_voting/`, …), so several methods can
share one `output_dir` without colliding. `voting` and `stacking` also keep each
base model under `base_models/`, in the same layout again.

Every pipeline follows that same shape, and every one returns the same kind of
result object, so a model on the narrative and a stacked ensemble can be scored
and plotted with exactly the same code.

**Several models at once.** You are not limited to one model per run: loop over
`run()` with the same `split_seed` (or a fixed `split`), or use `task="voting"`
/ `task="stacking"`, which train every model you list in one call and share a
single split between them. Either way,
`predictions_frame({name: run_directory})` collects finished runs — by path,
weeks later, without reloading a model — into one leaderboard. Notebook 3 shows
both.

## Running on your own data

Whichever template you copy, these four settings are the ones that describe
*your* data. Nothing else has to change for a first run.

| Setting | What it is | Example |
|---|---|---|
| `data` | Your file, or a DataFrame you already loaded | `"deaths_2023.csv"` |
| `label_col` | The column holding the cause of death | `"cause_of_death"` |
| `text_col` | The column holding the free-text narrative (text-based pipelines only) | `"narrative"` |
| `features` | The columns holding the symptom indicators (tabular and fusion pipelines only) | `["i019a", "i022e"]`, or `"re:^i\d{3}[a-zA-Z]$"`, or `"auto"` |

`features` accepts three forms: an explicit list of column names; a regular
expression written as `"re:..."`, useful when the columns follow a naming
pattern such as InterVA i-codes; or `"auto"`, meaning every column that is not
the label, the narrative, or a column used for filtering or splitting.

Plus `output_dir` — a new folder for each run, so runs do not overwrite one
another.

### Which settings need attention, and when

The config files tag every line so this is visible while you edit them:

| Tag | Meaning |
|---|---|
| `[CHANGE]` | No default can be right for your data. Set it before you run. |
| `[KEEP]` | A sensible default. Leave it alone unless you have a reason. |
| `[TUNE]` | Works as-is, but the value is worth revisiting for your data. |

The `[TUNE]` settings, in roughly the order they are worth your time:

- **`model`** — which model is trained. Defaults are reasonable, but the best
  choice depends on the data; `multimodalva list-models` shows the options, and
  the [FAQ](../FAQ.md#which-model-should-i-pick) discusses picking one.
- **`hyperparams`** — where the model's settings come from. Leave it out and the
  model library's own defaults apply. Pass `Optimize(...)` to search for them,
  which costs roughly `n_trials` times a single run, so it is usually turned on
  only once the pipeline works. Pass a dict to reuse values you already found.
- **`Optimize(metric=...)`** — what the search treats as "better". `f1_macro`
  weighs all causes equally; `csmf_accuracy` scores the cause distribution of
  the population rather than individual deaths.
- **`filters`** — restrict to a subset of rows, such as adults only.
- **`split`** — fix one train/test split and reuse it, so that different models
  are compared on identical rows rather than on different draws.
- **`max_length`** (text pipelines) — how much of a narrative the model reads.
  Narratives longer than this are cut off, so it is worth checking against the
  word counts in your own data; the first notebook prints them.

Settings you can safely ignore at first: `test_size`, `split_seed`, `train_seed`,
`stratify`, `top_k`, `encode_categoricals`, `scale_numeric`, `save_diagnostics`.

A mistyped setting stops the run immediately and names the key it did not
recognise, so a typo cannot quietly change what was trained.

## Config file format: YAML or JSON

`multimodalva run <file>` accepts both. The format is chosen by the file
extension — `.yaml` / `.yml` is read as YAML, anything else as JSON — and the
two produce identical runs. `config_tabular_model.json` is the same run as
`config_tabular_model.yaml`, in the other format.

The examples are written in YAML for one reason: **JSON cannot contain
comments**, and most of what is in these files *is* comments — the explanation
of each setting and the `[CHANGE]` / `[KEEP]` / `[TUNE]` tags. YAML also needs
no braces, quotes or commas, so a setting can be changed without a punctuation
error, and commenting a line out to disable it is one `#`.

Prefer JSON if you already read and write it, or if the config is generated by
another program: it has exactly one way to write anything, whereas YAML will
read `no` and `off` as `false`, and a version number like `1.10` as the number
`1.1`. Quote anything that should stay text.

If YAML is unfamiliar, note that you do not need a config file at all — the
Python and command-line templates run the same pipelines.

## Quick start

```bash
# Notebooks
jupyter lab examples/01_start_here_text_and_tabular.ipynb

# Python
python examples/04_python_script_template.py tabular

# Command line (or: python -m multimodalva.cli ...)
bash examples/05_command_line_template.sh

# Config file
multimodalva run examples/config_tabular_model.yaml
multimodalva run examples/config_tabular_model.json
```

## Check your install

Both run offline, need no GPU, and use only the standard library plus the
package itself — nothing extra to install:

```bash
python tests/smoke_test.py              # is it working? ~10 s
python tests/smoke_test.py --with-text  # also runs a tiny text model (~17MB download)
python tests/diagnose.py                # which pipelines and models work here? ~1 min
```

`diagnose.py` reports a missing optional dependency as `SKIP` with the install
command, so it tells you what is unavailable and why.
