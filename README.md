# MultimodalVA

Cause-of-death classification from verbal autopsy (VA) records, in one Python
package.

VA data is multimodal: a free-text narrative plus structured symptom indicators.
MultimodalVA trains, tunes and compares text models, tabular models and
multimodal fusion at all three levels — **data**, **feature** and **decision** —
through a single interface.

## Installation

```bash
pip install git+https://github.com/y-chu/MultimodalVA.git
```

Python 3.12 or 3.13. Runs on CUDA, Apple Silicon (MPS) or CPU.

That covers five of the six pipelines, all nine tabular models, hyperparameter
search and evaluation. Three things need an extra:

```bash
# parallel search across CPUs, GPUs or cluster nodes — Optimize(backend="ray")
pip install "multimodalva[ray] @ git+https://github.com/y-chu/MultimodalVA.git"

# LoRA fine-tuning — use_lora=True
pip install "multimodalva[lora] @ git+https://github.com/y-chu/MultimodalVA.git"

# the feature_fusion pipeline — AutoGluon AutoMM
pip install "multimodalva[feature_fusion] @ git+https://github.com/y-chu/MultimodalVA.git"

# all three
pip install "multimodalva[all] @ git+https://github.com/y-chu/MultimodalVA.git"
```

⚠️ `[feature_fusion]` pins torch and transformers tighter than the rest of the
package — install it into a clean environment, and see the
[FAQ](FAQ.md#feature-fusion-fails-to-install-or-conflicts-with-my-torch-version)
if it conflicts.

## Quick start

```python
from multimodalva import run

run(task="text", data="clean.csv", label_col="cause", text_col="narrative",
    model="bioclinicalbert", output_dir="runs/text")
```

One call does the whole thing: split the records, prepare them for the model,
train, predict on the held-out test set, and write everything to `output_dir/` —
`final/` (model and label maps), `predictions/` (top-1, top-k and
full-probability CSVs), `hpo/` (search records when there was a search).

**Tune the hyperparameters** by passing `Optimize()` instead of nothing. Each
pipeline has a default search space adapted to your data, so the only thing you
have to decide is the budget:

```python
from multimodalva import Optimize

run(task="text", ..., hyperparams=Optimize(n_trials=30))
```

`Optimize(metric="csmf_accuracy")` changes what is optimised,
`Optimize(cv=False)` searches on a single holdout instead of 3-fold CV, and
`Optimize(backend="ray")` runs trials in parallel. Passing a plain dict instead
(`hyperparams={"learning_rate": 3e-5}`) uses those values and searches nothing.

The same from the command line, or from a config file:

```bash
multimodalva run --task text --data clean.csv --label-col cause \
  --text-col narrative --model bioclinicalbert --output-dir runs/text --optimize

multimodalva run experiment.yaml
multimodalva run experiment.yaml --dry-run    # check the config, train nothing
```

<details>
<summary><b>The same call for each pipeline</b></summary>

```python
from multimodalva import Optimize, run

# 1. Text — fine-tune a transformer on the narrative
run(task="text", data="clean.csv", label_col="cause", text_col="narrative",
    model="bioclinicalbert", output_dir="runs/text",
    hyperparams=Optimize(n_trials=30))

# 2. Tabular — an ML model on the indicator columns
run(task="tabular", data="clean.csv", label_col="cause",
    features="re:^i\d{3}[a-z]$",       # a list of names, "auto", or a regex
    model="lightgbm", output_dir="runs/tabular",
    hyperparams=Optimize(n_trials=50))

# 3. Data fusion — indicators rendered as sentences, appended to the narrative,
#    then one long-context model over the result
run(task="data_fusion", data="clean.csv", label_col="cause",
    text_col="narrative", features="auto", model="clinicallongformer",
    output_dir="runs/data_fusion", hyperparams=Optimize(n_trials=20))

# 4. Feature fusion — a joint text + tabular model (needs [feature_fusion])
run(task="feature_fusion", data="clean.csv", label_col="cause",
    text_col="narrative", features="auto", output_dir="runs/feature_fusion")

# 5. Stacking — out-of-fold predictions from several base models, then a
#    combiner fitted on them
run(task="stacking", data="clean.csv", label_col="cause",
    text_col="narrative", features="auto",
    text_models=[{"model_name": "bioclinicalbert"}],
    tabular_models=[{"model_name": "lightgbm"}, {"model_name": "catboost"}],
    output_dir="runs/stacking")
```

Soft voting is the same as stacking with `task="voting"` — it averages the base
models' probabilities instead of learning a combiner.

</details>

**📖 [User manual](https://y-chu.github.io/MultimodalVA/)** — every argument, with
examples. Start there once the quick start runs.

## Pipelines

| Pipeline | `task=` | What it does |
|---|---|---|
| Text | `text` | Fine-tune a transformer on the narrative |
| Tabular | `tabular` | Train an ML model on the structured indicators |
| Data fusion | `data_fusion` | Render indicators as sentences, append to the narrative, fine-tune a long-context model |
| Feature fusion | `feature_fusion` | Joint text + tabular model via AutoGluon AutoMM |
| Soft voting | `voting` | Average the probabilities of independently trained models |
| Stacking | `stacking` | Out-of-fold base predictions, then a combiner — meta-learner (default), simple average, class-aware voter, ensemble selection, or `"best"` (`combiner=`) |

## Models

**Text** — 13 backbone aliases (`bioclinicalbert`, `clinicallongformer`,
`biomedbert`, …), any Hugging Face Hub ID, or a local directory.
`multimodalva list-models` prints the list with descriptions.

**Tabular** — `lightgbm`, `xgboost`, `catboost`, `random_forest`, `gbdt`, `mlp`,
`svm`, `knn`, `naive_bayes`.

## Why use it

- **One interface across six pipelines**, so a text model, a tabular model and
  three kinds of fusion are comparable instead of scattered across scripts.
- **VA-specific evaluation** — CSMF accuracy and its chance-corrected form,
  cause-specific breakdowns, bootstrap intervals, calibration.
- **The same code on a laptop and a cluster** — Optuna by default, Ray for
  parallel search, resumable either way.
- **Runs are reproducible and self-describing**: seeds for splitting and for
  training, and artifacts that record what produced them.

## Why not use it

- **It learns from labelled records.** InterVA and InSilicoVA assign causes from
  a fixed symptom–cause model and need no training data; this package needs
  records whose cause is already known. With no labelled data, it is not the
  tool. (You can still evaluate their output alongside yours — see the FAQ.)
- **If you only need one model family**, scikit-learn or Transformers directly
  is simpler.
- **It expects VA-shaped data**: one row per death, a cause label, and a
  narrative, indicators, or both. It is not general-purpose AutoML.

## Examples

<details>
<summary>Notebooks, templates and a prediction demo — all run on built-in synthetic data</summary>

Everything below runs as-is on built-in synthetic data — no files needed. What
the package expects of your own data, and how to look at the synthetic shape, is
in the [FAQ](FAQ.md#what-data-does-the-package-expect).

| | |
|---|---|
| [`01_start_here_text_and_tabular.ipynb`](examples/01_start_here_text_and_tabular.ipynb) | load → train on the narrative → train on the indicators → compare → save |
| [`02_combine_text_and_tabular_fusion.ipynb`](examples/02_combine_text_and_tabular_fusion.ipynb) | data, feature and decision fusion, compared |
| [`03_evaluate_and_compare_results.ipynb`](examples/03_evaluate_and_compare_results.ipynb) | leaderboards, confidence intervals, confusion matrices, CSMF, top-k, calibration |
| [`08_predict_with_a_trained_model.py`](examples/08_predict_with_a_trained_model.py) | score new records with a run that has already finished |
| [`07_publish_to_hub.py`](examples/07_publish_to_hub.py) | publish a text or data-fusion model to the Hugging Face Hub |
| `04_python_script_template.py` · `05_command_line_template.sh` · `06_config_file_template.sh` | templates to copy |

Hyperparameter search and checking a config before submitting a job are covered
in the templates. See [`examples/README.md`](examples/README.md).

Check the install:

```bash
python tests/smoke_test.py      # ~45 s
python tests/diagnose.py        # which pipelines and models work here, ~3 min
```

</details>

## Documentation

- **[User manual](https://y-chu.github.io/MultimodalVA/)** — every argument, with examples
- [FAQ](FAQ.md) — common questions about using the package
- [Examples](examples/README.md)
- [CHANGELOG](CHANGELOG.md) — changes between releases

Questions and bug reports are welcome as
[GitHub issues](https://github.com/y-chu/MultimodalVA/issues); see
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Built on

| | |
|---|---|
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | text model fine-tuning and inference |
| [scikit-learn](https://scikit-learn.org) | tabular models, preprocessing, cross-validation |
| [LightGBM](https://github.com/microsoft/LightGBM), [XGBoost](https://github.com/dmlc/xgboost), [CatBoost](https://github.com/catboost/catboost) | gradient-boosting tabular models |
| [Optuna](https://optuna.org) and [Ray Tune](https://docs.ray.io/en/latest/tune/) | hyperparameter search |
| [AutoGluon AutoMM](https://github.com/autogluon/autogluon) (Apache-2.0) | joint text–tabular feature fusion |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapters |

If you use the feature-fusion pipeline, please also cite AutoMM:

> Tang Z, Fang H, Zhou S, Yang T, Zhong Z, Hu C, Kirchhoff K, Karypis G.
> *AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation
> Models.* AutoML Conference, PMLR 256, 2024.

## Citation

Please cite the software (GitHub reads `CITATION.cff`, so the "Cite this
repository" button gives the same metadata):

```bibtex
@software{chu_multimodalva,
  author  = {Chu, Yue},
  title   = {MultimodalVA: cause-of-death classification from verbal autopsy data},
  url     = {https://github.com/y-chu/MultimodalVA},
  version = {0.1.0},
  year    = {2026}
}
```

For the methods:

- Chu Y, Wu Z, McCormick T, Li R, Clark SJ. *Multimodal Artificial Intelligence for
  Cause-of-Death Assignment From Verbal Autopsy Records: A Nationally Representative
  Evaluation in South Africa.* 2026.

## License

MIT — see [LICENSE](LICENSE).

## Use of Generative AI Tools

Claude Code (Opus) and Codex (GPT-5.5) were used as software development assistants for code optimization, package engineering, debugging, generating documentation, synthesizing demo datasets, and preparing testing scripts. The tools did not determine the scientific content of the work. All research questions, methodological choices, analytical strategies, model development, parameter optimization, result validation, interpretation, and scientific conclusions were conceived, evaluated, and approved by the author. All AI-generated outputs were reviewed, tested, and verified by the author prior to use.
