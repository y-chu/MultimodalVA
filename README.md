# MultimodalVA

Cause-of-death classification from verbal autopsy (VA) data, in one Python package.

VA data is multimodal: a free-text narrative plus structured symptom indicators.
MultimodalVA lets you train, tune and compare text models, tabular models and
multimodal fusion at all three levels — **data**, **feature** and **decision** —
through a single interface, so results are comparable across model families
instead of scattered across ad hoc scripts.

Every pipeline follows the same steps — **split → prepare → [HPO] → train →
predict** — and returns the same `PredictionResult`: `top1` (predicted label and
confidence), `full` (all class probabilities), `topk`, and `id2label`.

## Installation

The package is not on PyPI yet, so install from GitHub:

```bash
pip install git+https://github.com/y-chu/MultimodalVA.git
```

Requires Python 3.12 or 3.13. Runs on CUDA, Apple Silicon (MPS) or CPU.

### What the default install gives you

The base install covers five of the six pipelines and all nine tabular models,
including LightGBM, XGBoost and CatBoost:

| | Default install | Needs an extra |
|---|---|---|
| Text classification (`task="text"`) | ✅ | |
| Tabular, all 9 models (`task="tabular"`) | ✅ | |
| Data fusion (`task="data_fusion"`) | ✅ | |
| Soft voting (`task="voting"`) | ✅ | |
| Stacking (`task="stacking"`) | ✅ | |
| Hyperparameter search, all pipelines | ✅ | |
| **Parallel / distributed search** (`Optimize(backend="ray")`, unverified — see below) | ❌ | `[ray]` |
| Evaluation, plots, calibration, bootstrap CIs | ✅ | |
| Publishing models to the Hugging Face Hub | ✅ | |
| **LoRA fine-tuning** (`use_lora=True`) | ❌ | `[lora]` |
| **Feature fusion** (`task="feature_fusion"`) | ❌ | `[feature_fusion]` |

So: **the default install does not include LoRA.** Text models train with full
fine-tuning unless you install the extra and pass `use_lora=True`.

Hyperparameter search itself needs no extra — it runs on Optuna, one trial at a
time. `[ray]` only adds the option of running those trials **in parallel** across
the CPUs, GPUs or cluster nodes Ray can see. Without it, `Optimize(backend="auto")` resolves to Optuna and
says so; on a machine where Ray would have helped (a CUDA GPU, or a SLURM
allocation) it also warns that the extra is missing, so the fallback is never
silent.

The Ray backend is **implemented but not yet verified on a real multi-GPU or
multi-node run** — off a CUDA machine it hands the work back to Optuna by design,
which is the path every test of it has taken so far. `tests/hpc_ray_smoke.py`
(with a SLURM wrapper beside it) exercises it on a GPU node; run that before
trusting it with a long search. Optuna is the backend this package was developed
on and needs no extra.

### Extras

```bash
pip install "multimodalva[all] @ git+https://github.com/y-chu/MultimodalVA.git"             # recommended — adds every extra
pip install "multimodalva[lora] @ git+https://github.com/y-chu/MultimodalVA.git"            # LoRA fine-tuning (PEFT)
pip install "multimodalva[ray] @ git+https://github.com/y-chu/MultimodalVA.git"             # parallel search on a GPU box or cluster
pip install "multimodalva[feature_fusion] @ git+https://github.com/y-chu/MultimodalVA.git"  # AutoGluon AutoMM
```

**On an HPC cluster with more than one GPU, `[ray]` is the extra to consider.** It
is the one whose absence costs you nothing but time: the search still runs and the
results are the same, just sequentially. `[ray]` adds about 360 MB (Ray itself
plus pyarrow and grpcio), which is why it is not in the default install — a laptop
running the Optuna backend never needs any of it. Note `[feature_fusion]` pulls Ray in regardless,
because AutoGluon uses it internally.

Once the package is on PyPI the shorter form works — `pip install "multimodalva[all]"`.

⚠️ **`[feature_fusion]` narrows your environment.** AutoGluon AutoMM pins
`torch>=2.6,<2.10` and `transformers>=4.51,<4.58`, where the rest of the package
accepts `torch>=2.1` and `transformers>=4.38,<5`, and it pulls in a large
dependency tree of its own. Install it when you want feature fusion; skip it on a
cluster where the available torch is outside that window. The other five
pipelines are unaffected either way. See the
[FAQ](FAQ.md#feature-fusion-fails-to-install-or-conflicts-with-my-torch-version)
if the install conflicts.

InSilicoVA is not supported yet. `pyinsilicova` does not install on Python 3.12+,
which this package requires, and it cannot train a probbase against your own
cause list — only InSilicoVA's native causes. Run it separately (in R, or with
`pyinsilicova` if its native causes fit your study) and bring the assignments
back: the leaderboard, bootstrap intervals and plots all accept predictions from
any source, so an externally produced model is compared on the same footing.

## How to use

One call runs a whole pipeline — loading, splitting, HPO, training, prediction
and saving:

```python
from multimodalva import Optimize, run

run(task="text", data="clean.csv", label_col="cause", text_col="narrative",
    model="bioclinicalbert", output_dir="runs/text",
    hyperparams=Optimize(n_trials=30))
```

The same call from the command line, or from a config file:

```bash
multimodalva run --task text --data clean.csv --label-col cause \
  --text-col narrative --model bioclinicalbert --output-dir runs/text --optimize

multimodalva run experiment.yaml
```

Swap `task=` for any other pipeline (`tabular`, `data_fusion`, `feature_fusion`,
`voting`, `stacking`). Results land in `output_dir/`: `final/` (model and label
maps), `predictions/` (top-1, top-k and full-probability CSVs), `hpo/` (study,
best hyperparameters, diagnostics).

Three arguments every task shares and that are worth knowing early:
`id_col=` names a record-identifier column, so each prediction carries its `id`;
`split_seed=` fixes which deaths land in the test set; `train_seed=` fixes the
model's own randomness. Re-running with the same `output_dir` resumes an
interrupted run, and finished runs can be reloaded and combined without
retraining — see the [FAQ](FAQ.md).

### Predicting with a model that is already trained

A finished run — yours, a colleague's, or one published on the Hugging Face Hub —
scores new records without retraining. Give it a label column and you get
performance as well; leave it out and you simply get predictions:

```python
from multimodalva import predict_from_pretrained

res = predict_from_pretrained("runs/text", new_df, text_col="narrative")
res.top1.head()          # predicted_label, predicted_prob (+ id, + true_label)
res.checks               # what was detected about the model and your input
```

```bash
multimodalva predict --source runs/text --data new.csv \
  --text-col narrative --label-col cause --output-dir preds/

# What would happen, without scoring anything — worth a few seconds
# in front of a queued job
multimodalva predict --source runs/text --data new.csv --dry-run
```

One command covers every pipeline: what kind of run it is is read from the
artifact, so a voting or stacking run is combined the way that run combined it
(`predict_ensemble_from_pretrained()` in Python). Columns are matched by name,
never by position, and a column the model was trained on that is missing from
your data is an error rather than a silently confident guess. The
[FAQ](FAQ.md) covers which artifacts can be loaded and what each message means.

No data of your own yet? Everything works on the built-in synthetic datasets:

```python
from multimodalva import data
df = data("va_sample")
```

For finer control, the classifiers are available directly — `TextClassifier`,
`TabularClassifier` and `EnsembleClassifier`, each with a `.run()` method.
Stacking additionally exposes its stages (`train_base_models()`,
`train_meta_learner_stage()`, `predict_test()`) so a long run can be done in
steps. Several combiners can be computed from one set of out-of-fold
predictions and reported side by side. See the examples below.

## Pipelines and models

| Pipeline | `task=` | What it does |
|---|---|---|
| Text | `text` | Fine-tune a transformer on the narrative |
| Tabular | `tabular` | Train an ML model on the structured indicators |
| Data fusion | `data_fusion` | Render indicators as sentences, append to the narrative, fine-tune a long-context model |
| Feature fusion | `feature_fusion` | Joint text + tabular model via AutoGluon AutoMM |
| Soft voting | `voting` | Average the probabilities of independently trained models |
| Stacking | `stacking` | Out-of-fold base predictions, then a combiner — meta-learner (default), simple average, class-aware voter, ensemble selection, several at once, or `"best"` chosen on training data alone (`combiner=`) |

**Text backbones** — pass an alias, a Hugging Face Hub ID, or a local directory.
`multimodalva list-models` prints the list.

| Alias | Checkpoint |
|---|---|
| `bioclinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` |
| `bert` | `bert-base-uncased` |
| `biobert` | `dmis-lab/biobert-base-cased-v1.2` |
| `bluebert` | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` |
| `biomedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| `clinicalbert` | `medicalai/ClinicalBERT` |
| `biomedroberta` | `allenai/biomed_roberta_base` |
| `bioelectra` | `kamalkraj/bioelectra-base-discriminator-pubmed` |
| `longformer` | `allenai/longformer-base-4096` |
| `clinicallongformer` | `yikuan8/Clinical-Longformer` |
| `bigbird` | `google/bigbird-roberta-base` |
| `clinicalbigbird` | `yikuan8/Clinical-BigBird` |
| `roberta-pm` | `RoBERTa-base-PM-M3-Voc-distill-hf` (not on the Hub; downloaded once) |

`biomedroberta` and `roberta-pm` are **different models** with different
vocabularies — see the [FAQ](FAQ.md#which-model-should-i-pick).

**Tabular models** — `lightgbm`, `xgboost`, `catboost`, `random_forest`, `gbdt`,
`mlp`, `svm`, `knn`, `naive_bayes`.

**Evaluation** — the `results` subpackage provides a multi-model leaderboard
(accuracy, balanced accuracy, macro/weighted F1, CSMF accuracy, chance-corrected
CSMF accuracy, top-*k*), cause-specific heatmaps, confusion matrices, CSMF
scatter plots and HPO diagnostics.

## Examples

All examples live in [`examples/`](examples/README.md) and run on the built-in
synthetic data, so they need no external files.

**New here? Read the three notebooks in order** — they are the files that explain
what is happening, one step per section.

| Read first | |
|---|---|
| [`01_start_here_text_and_tabular.ipynb`](examples/01_start_here_text_and_tabular.ipynb) | the whole workflow: load → explore → train on the narrative → train on the indicators → compare → save and reuse |
| [`02_combine_text_and_tabular_fusion.ipynb`](examples/02_combine_text_and_tabular_fusion.ipynb) | using both together: data, feature and decision fusion (voting, stacking), compared |
| [`03_evaluate_and_compare_results.ipynb`](examples/03_evaluate_and_compare_results.ipynb) | what to do with the output: several models at once, leaderboards, confidence intervals, confusion matrices, CSMF, top-k, calibration |

| Then copy a template | |
|---|---|
| `04_python_script_template.py` | if you write Python |
| `05_command_line_template.sh` | if you would rather not, or you submit cluster jobs |
| `06_config_file_template.sh` + `config_*.yaml` | if you want the settings file itself to be the record of the run |

To run any of them on your own data you change four things — your file, and the
names of your label, narrative and indicator columns. The config files tag every
other setting as `[CHANGE]`, `[KEEP]` or `[TUNE]`, and
[`examples/README.md`](examples/README.md) explains which are worth your time.

Check your install end-to-end (offline, no GPU):

```bash
python tests/smoke_test.py      # is it working? ~45 s
python tests/diagnose.py        # which pipelines and models work here? ~3 min
```

Both use only the standard library and the package itself, and both ship in the
distribution, so they run straight after `pip install` with nothing else to add.
`diagnose.py` reports a missing optional dependency as `SKIP` with the install
command, so it doubles as an environment check.

If you will need to reproduce a run later, save the environment next to it
(`pip freeze > runs/text/environment.txt`): seeds fix everything this package
controls, but not what a dependency changes between its own releases. The
[FAQ](FAQ.md#how-do-i-make-a-run-reproducible) has the details and one measured
example.

## What is coming

Planned work, with the reasoning, is kept in [CHANGELOG.md](CHANGELOG.md) under
*Unreleased → Planned*. The next release, **0.2.0**, adds a converter that
builds data-fusion long text from a DataFrame plus a standardised questionnaire,
so a data-fusion model can be used without preparing that text by hand. Further
out: image as a third modality, log-probability inputs for the stacking
meta-learner, and InSilicoVA as a base learner. Versions follow semantic
versioning; `1.0.0` will mark the API as stable rather than any one feature.

## Documentation

- [FAQ](FAQ.md) — hardware, run time, disk use, reproducibility, resuming,
  publishing models to the Hugging Face Hub, common errors
- [API reference](docs/index.html)
- [Examples](examples/README.md)

## Questions and contributions

Questions, bug reports and feature requests are welcome as
[GitHub issues](https://github.com/y-chu/MultimodalVA/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Built on

`MultimodalVA` is a layer over existing tools rather than a reimplementation:

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

A software paper is in preparation. Until it appears, please cite the software
(GitHub reads `CITATION.cff`, so the "Cite this repository" button gives the same
metadata):

```bibtex
@software{chu_multimodalva,
  author  = {Chu, Yue},
  title   = {MultimodalVA: cause-of-death classification from verbal autopsy data},
  url     = {https://github.com/y-chu/MultimodalVA},
  version = {0.1.0},
  year    = {2026}
}
```

For the methods themselves, cite the research they come from:

- Chu Y, Wu Z, Li R, McCormick T, Clark SJ. *Multimodal Artificial Intelligence for
  Cause-of-Death Assignment From Verbal Autopsy Records: A Nationally Representative
  Evaluation in South Africa.* Manuscript submitted for publication, 2026.
- Chu Y. *Leveraging Language Models and Machine Learning in Verbal Autopsy Analysis.*
  PhD thesis, The Ohio State University, 2025. [arXiv:2508.19274](https://arxiv.org/abs/2508.19274)

## License

MIT — see [LICENSE](LICENSE).

## Use of Generative AI Tools

Claude Code (Opus) and Codex (GPT-5.5) were used as software development assistants for code optimization, package engineering, debugging, generating documentation, synthesizing demo datasets, and preparing testing scripts. The tools did not determine the scientific content of the work. All research questions, methodological choices, analytical strategies, model development, parameter optimization, result validation, interpretation, and scientific conclusions were conceived, evaluated, and approved by the author. All AI-generated outputs were reviewed, tested, and verified by the author prior to use.
