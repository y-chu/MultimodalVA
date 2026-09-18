# MultimodalVA

Cause-of-death classification from verbal autopsy (VA) data, in one Python package.

VA data is multimodal: a free-text narrative plus structured symptom indicators.
MultimodalVA lets you train, tune and compare text models, tabular models and
four multimodal fusion strategies through a single interface, so results are
comparable across model families instead of scattered across ad hoc scripts.

Every pipeline follows the same steps — **split → prepare → [HPO] → train →
predict** — and returns the same `PredictionResult`: `top1` (predicted label and
confidence), `full` (all class probabilities), `topk`, and `id2label`.

## Installation

```bash
pip install git+https://github.com/y-chu/MultimodalVA.git
```

Optional extras:

```bash
pip install "multimodalva[tabular]"         # LightGBM / XGBoost / CatBoost
pip install "multimodalva[lora]"            # LoRA fine-tuning (PEFT)
pip install "multimodalva[feature_fusion]"  # AutoGluon AutoMM
pip install "multimodalva[all]"             # everything above
```

Requires Python 3.12 or 3.13. Runs on CUDA, Apple Silicon (MPS) or CPU.

## How to use

One call runs a whole pipeline — loading, splitting, HPO, training, prediction
and saving:

```python
from multimodalva import run

run(task="text", data="clean.csv", label="cause", text_col="narrative",
    model="bioclinicalbert", output_dir="runs/text", optimize=True, n_trials=30)
```

The same call from the command line, or from a config file:

```bash
multimodalva run --task text --data clean.csv --label cause \
  --text-col narrative --model bioclinicalbert --output-dir runs/text --optimize

multimodalva run experiment.yaml
```

Swap `task=` for any other pipeline (`tabular`, `data_fusion`, `feature_fusion`,
`voting`, `stacking`). Results land in `output_dir/`: `final/` (model and label
maps), `predictions/` (top-1, top-k and full-probability CSVs), `hpo/` (study,
best hyperparameters, diagnostics).

No data of your own yet? Everything works on the built-in synthetic datasets:

```python
from multimodalva import data
df = data("va_sample")
```

For finer control, the classifiers are available directly — `TextClassifier`,
`TabularClassifier` and `EnsembleClassifier`, each with a `.run()` method.
Stacking additionally exposes its stages (`train_base_models()`,
`train_meta_learner_stage()`, `predict_test()`) so a long run can be done in
steps. See the examples below.

## Pipelines and models

| Pipeline | `task=` | What it does |
|---|---|---|
| Text | `text` | Fine-tune a transformer on the narrative |
| Tabular | `tabular` | Train an ML model on the structured indicators |
| Data fusion | `data_fusion` | Render indicators as sentences, append to the narrative, fine-tune a long-context model |
| Feature fusion | `feature_fusion` | Joint text + tabular model via AutoGluon AutoMM |
| Soft voting | `voting` | Average the probabilities of independently trained models |
| Stacking | `stacking` | Out-of-fold base predictions, then a meta-learner or a class-aware voter |

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

All examples live in [`tests/examples/`](tests/examples/README.md) and run on the
built-in synthetic data, so they need no external files.

| Start here | |
|---|---|
| `notebooks/01_getting_started.ipynb` | text and tabular, end to end: load → explore → train → inspect → reuse |
| `notebooks/02_multimodal_fusion.ipynb` | the four fusion strategies |
| `example_python_api.py`, `example_cli.sh`, `example_yaml.sh` | the same pipelines through each interface |

Check your install end-to-end (offline, no GPU): `python tests/smoke_test.py`.

## Documentation

- [FAQ](FAQ.md) — hardware, run time, disk use, reproducibility, resuming,
  publishing models to the Hugging Face Hub, common errors
- [API reference](docs/index.html)
- [Examples](tests/examples/README.md)

## Questions and contributions

Questions, bug reports and feature requests are welcome as
[GitHub issues](https://github.com/y-chu/MultimodalVA/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Citation

A software paper is in preparation. Until it appears, please cite the software
(GitHub reads `CITATION.cff`, so the "Cite this repository" button gives the same
metadata):

```bibtex
@software{chu_multimodalva,
  author  = {Chu, Yue},
  title   = {MultimodalVA: cause-of-death classification from verbal autopsy data},
  url     = {https://github.com/y-chu/MultimodalVA},
  version = {0.1.0}
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
