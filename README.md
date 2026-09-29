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
pipeline has a default search space, rescaled for your training-set size and
cause count, so the only thing you have to decide is the budget:

```python
from multimodalva import Optimize

run(task="text", ..., hyperparams=Optimize(n_trials=30))
```

`Optimize(metric="csmf_accuracy")` changes what is optimised,
`Optimize(cv=False)` searches on a single holdout instead of 3-fold CV, and
`Optimize(backend="ray")` runs trials in parallel. Passing a plain dict instead
(`hyperparams={"learning_rate": 3e-5}`) uses those values and searches nothing;
passing neither trains with the model's library defaults. To set the ranges
yourself, see [How do I set the hyperparameter search
space?](FAQ.md#how-do-i-set-the-hyperparameter-search-space).

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

**Text backbones** — pass an alias, any Hugging Face Hub ID, or a local
directory. `multimodalva list-models` prints the same list with descriptions.

| Alias | Checkpoint | Code |
|---|---|---|
| `bioclinicalbert` | [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | [repo](https://github.com/EmilyAlsentzer/clinicalBERT) |
| `bert` | [bert-base-uncased](https://huggingface.co/bert-base-uncased) | [repo](https://github.com/google-research/bert) |
| `biobert` | [dmis-lab/biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) | [repo](https://github.com/dmis-lab/biobert) |
| `bluebert` | [bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12](https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12) | [repo](https://github.com/ncbi-nlp/bluebert) |
| `biomedbert` | [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |  |
| `clinicalbert` | [medicalai/ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) |  |
| `biomedroberta` | [allenai/biomed_roberta_base](https://huggingface.co/allenai/biomed_roberta_base) | [repo](https://github.com/allenai/dont-stop-pretraining) |
| `bioelectra` | [kamalkraj/bioelectra-base-discriminator-pubmed](https://huggingface.co/kamalkraj/bioelectra-base-discriminator-pubmed) | [repo](https://github.com/kamalkraj/BioELECTRA) |
| `longformer` | [allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) | [repo](https://github.com/allenai/longformer) |
| `clinicallongformer` | [yikuan8/Clinical-Longformer](https://huggingface.co/yikuan8/Clinical-Longformer) | [repo](https://github.com/luoyuanlab/Clinical-Longformer) |
| `bigbird` | [google/bigbird-roberta-base](https://huggingface.co/google/bigbird-roberta-base) | [repo](https://github.com/google-research/bigbird) |
| `clinicalbigbird` | [yikuan8/Clinical-BigBird](https://huggingface.co/yikuan8/Clinical-BigBird) | [repo](https://github.com/luoyuanlab/Clinical-Longformer) |
| `roberta-pm` | RoBERTa-base-PM-M3-Voc-distill-hf — not on the Hub; downloaded once to `~/.cache/multimodalva/` | [repo](https://github.com/facebookresearch/bio-lm) |

**Tabular models**

| Alias | Estimator |
|---|---|
| `lightgbm` | [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html) |
| `xgboost` | [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_intro.html) |
| `catboost` | [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-quickstart) |
| `random_forest` | [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| `gbdt` | [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) |
| `mlp` | [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) |
| `svm` | [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| `knn` | [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
| `naive_bayes` | [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) |

**Evaluation** — the `results` subpackage: multi-model leaderboard (accuracy,
balanced accuracy, macro/weighted F1, CSMF accuracy, chance-corrected CSMF
accuracy, top-*k*), cause-specific heatmaps, confusion matrices, CSMF scatter
plots, HPO diagnostics, bootstrap confidence intervals and calibration.

## Why use it

- **One interface across six pipelines**, so a text model, a tabular model and
  three kinds of fusion are comparable instead of scattered across scripts.
- **Nine tabular models and thirteen text backbones**, with or without
  hyperparameter search, all taking the same input and producing the same
  output.
- **VA-specific evaluation** — CSMF accuracy and its chance-corrected form,
  cause-specific breakdowns, bootstrap intervals, calibration.
- **The same code on a laptop and a cluster** — Optuna by default, Ray for
  parallel search, resumable either way.
- **Runs are reproducible and self-describing**: seeds for splitting and for
  training, and artifacts that record what produced them.

**Before you start: the text pipelines need real compute.** The nine tabular
models train on any laptop in seconds. Fine-tuning a transformer wants a CUDA
GPU or Apple Silicon — one text fit is minutes, and a 30-trial search with
3-fold cross-validation is hours to half a day depending on the machine and the
number of records. Long-context data fusion and feature fusion want 16 GB of GPU
memory. Sizes and timings per pipeline are in the FAQ:
[what you need](FAQ.md#what-computing-environment-do-i-need) ·
[how long it takes](FAQ.md#how-long-does-a-run-take) ·
[SLURM resources](FAQ.md#what-slurm-resources-should-i-request).

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
| [`04_python_script_template.py`](examples/04_python_script_template.py) | a Python script to modify, customise and run |
| [`05_command_line_template.sh`](examples/05_command_line_template.sh) | the same from the command line, for cluster jobs |
| [`06_config_file_template.sh`](examples/06_config_file_template.sh) + `config_*.yaml` | settings files, so the config *is* the record of the run |
| [`07_publish_to_hub.py`](examples/07_publish_to_hub.py) | publish a text or data-fusion model to the Hugging Face Hub |
| [`08_predict_with_a_trained_model.py`](examples/08_predict_with_a_trained_model.py) | score new records with a run that has already finished |

Hyperparameter search and checking a config before submitting a job are covered
in the templates. See [`examples/README.md`](examples/README.md).

For what a run needs to run on — hardware, GPU memory, run time, disk, SLURM
resources — see the [FAQ](FAQ.md#what-computing-environment-do-i-need).

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
- Chu Y. *Leveraging Language Models and Machine Learning in Verbal Autopsy Analysis.*
  PhD thesis, The Ohio State University, 2025.
  [arXiv:2508.19274](https://arxiv.org/abs/2508.19274)

## License

MIT — see [LICENSE](LICENSE).

## Use of Generative AI Tools

Claude Code (Opus) and Codex (GPT-5.5) were used as software development assistants for code optimization, package engineering, debugging, generating documentation, synthesizing demo datasets, and preparing testing scripts. The tools did not determine the scientific content of the work. All research questions, methodological choices, analytical strategies, model development, parameter optimization, result validation, interpretation, and scientific conclusions were conceived, evaluated, and approved by the author. All AI-generated outputs were reviewed, tested, and verified by the author prior to use.
