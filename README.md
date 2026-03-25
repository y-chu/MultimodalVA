# MultimodalVA

**Cause of death classification using verbal autopsy data.**

MultimodalVA provides a unified Python interface for building, tuning, and ensembling
text-based (transformer), tabular (sklearn/LightGBM), and multimodal classifiers on
verbal autopsy (VA) datasets.  All pipelines share a common `split → prepare → [HPO →] train → predict`
interface and produce a `PredictionResult` that includes top-1 labels, full probability matrices,
and top-k predictions.

---

## Requirements

- **Python 3.12** (tested on 3.12.2; Python 3.13+ not yet supported)

---

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/y-chu/MultimodalVA.git
```

This installs the core package with all required dependencies automatically, including:
`torch`, `transformers`, `accelerate`, `datasets`, `scikit-learn`, `pandas`, `numpy`,
`optuna`, `ray[tune]`, `matplotlib`, `seaborn`, `sentencepiece`, and more.
No additional manual dependency installation is needed for the text or tabular pipelines.

### Development install (editable)

```bash
git clone https://github.com/y-chu/MultimodalVA.git
cd MultimodalVA
pip install -e ".[dev]"
```

### Optional extras

Install extras only when you need a specific feature:

```bash
# Tabular gradient boosters (LightGBM, XGBoost, CatBoost)
pip install "multimodalva[tabular]"

# LoRA parameter-efficient fine-tuning
pip install "multimodalva[lora]"

# Feature-fusion ensemble (AutoGluon AutoMM)
pip install "multimodalva[feature_fusion]"

# InSilicoVA base model in stacking (requires Python <3.10 — separate environment)
pip install "multimodalva[insilicova]"

# All Python 3.12-compatible extras (excludes insilicova)
pip install "multimodalva[all]"
```

---

## Quick Start

### Text classification

```python
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="bioclinicalbert", output_dir="runs/text")
results = clf.run(
    df=df,
    text_col="narrative",
    label_col="cause",
    hyperparams={"epochs": 5, "batch_size": 16, "learning_rate": 2e-5},
)
print(results["predictions"].top1.head())
```

### Tabular classification

```python
from multimodalva.tabular import TabularClassifier

clf = TabularClassifier(model_name="lightgbm", output_dir="runs/tabular")
results = clf.run(
    df=df,
    feature_cols=[...],
    label_col="cause",
    use_optimize=True,    # Optuna HPO
    n_trials=50,
    optimize_metric="f1_macro",
)
print(results["predictions"].top1.head())
```

### Ensemble — soft voting

```python
from multimodalva.ensemble import EnsembleClassifier

clf = EnsembleClassifier(
    method="soft_voting",
    output_dir="runs/ensemble",
    text_models=[{"model_name": "bioclinicalbert",
                  "hyperparams": {"epochs": 5}}],
    tabular_models=[{"model_name": "lightgbm",
                     "hyperparams": {"n_estimators": 300}}],
)
results = clf.run(df=df, text_col="narrative",
                  feature_cols=[...], label_col="cause")
```

### Ensemble — stacking (stage-wise)

```python
from multimodalva.ensemble import EnsembleClassifier

clf = EnsembleClassifier(
    method="stacking",
    output_dir="runs/ensemble",
    text_models=[{"model_name": "bioclinicalbert"}],
    tabular_models=[{"model_name": "lightgbm"}],
    n_folds=5,
)

# Stage 1 — OOF loop (long; can be restarted; resume-safe)
clf.train_base_models(df=df, label_col="cause",
                      text_col="narrative", feature_cols=[...])

# Stage 2 — meta-learner selection
clf.train_meta_learner_stage(meta_cv_folds=3)

# Stage 3 — predict test set
predictions = clf.predict_test()
```

---

## Supported models

### Text (Transformers)

Pass the alias string or the full HuggingFace checkpoint ID as `model_name`.

| Alias | HuggingFace checkpoint |
|---|---|
| `"bert"` | `bert-base-uncased` |
| `"biobert"` | `dmis-lab/biobert-base-cased-v1.2` |
| `"bioclinicalbert"` | `emilyalsentzer/Bio_ClinicalBERT` |
| `"bluebert"` | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` |
| `"biomedbert"` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| `"clinicalbert"` | `medicalai/ClinicalBERT` |
| `"biomedroberta"` | `allenai/biomed_roberta_base` |
| `"bioelectra"` | `kamalkraj/bioelectra-base-discriminator-pubmed` |
| `"longformer"` | `allenai/longformer-base-4096` |
| `"clinicallongformer"` | `yikuan8/Clinical-Longformer` |
| `"bigbird"` | `google/bigbird-roberta-base` |
| `"clinicalbigbird"` | `yikuan8/Clinical-BigBird` |

**Remote model (not on HuggingFace Hub):**

| Key | Source |
|---|---|
| `"roberta-pm"` | RoBERTa-PM-M3 — download via `download_model("roberta-pm")` |

Any public HuggingFace Hub ID or local model path is also accepted directly as `model_name`.

### Tabular (scikit-learn compatible)

Requires `pip install "multimodalva[tabular]"` for the gradient boosters.

| Alias | Library |
|---|---|
| `"lightgbm"` | LightGBM |
| `"xgboost"` | XGBoost |
| `"catboost"` | CatBoost |
| `"random_forest"` | scikit-learn `RandomForestClassifier` |
| `"gbdt"` | scikit-learn `GradientBoostingClassifier` |
| `"mlp"` | scikit-learn `MLPClassifier` |
| `"svm"` | scikit-learn `SVC` |
| `"knn"` | scikit-learn `KNeighborsClassifier` |
| `"naive_bayes"` | scikit-learn `GaussianNB` |

### Ensemble strategies

| `method=` | Description | Extra required |
|---|---|---|
| `"data_fusion"` | Tabular fields → natural language → concat with narrative → LM fine-tune | — |
| `"feature_fusion"` | AutoGluon AutoMM joint text + tabular representation | `[feature_fusion]` |
| `"soft_voting"` | Independent base models + weighted probability averaging | — |
| `"stacking"` | k-fold OOF meta-features → meta-learner (super learner) | — |

---

## Demo scripts

All demos are in `tests/` and use synthetic data unless noted:

| Script | Sections | Description |
|---|---|---|
| `demo_text_classification.py` | A–H | TextClassifier, HPO, LoRA, Ray Tune, focal loss |
| `demo_tabular_classification.py` | A–F | TabularClassifier, HPO, model comparison |
| `demo_ensemble_data_fusion.py` | A–E | DataFusionClassifier, tabular-to-text conversion |
| `demo_ensemble_feature_fusion.py` | A–E | FeatureFusionClassifier, AutoMM fusion strategies |
| `demo_ensemble_voting.py` | A–G | SoftVotingClassifier, vote_from_results |
| `demo_ensemble_stacking.py` | A–I | StackingClassifier, stage-wise, InSilicoVA stacking |
| `demo_results.py` | A–I | All visualization functions (synthetic data) |

Run a single section:

```bash
python tests/demo_results.py G         # cause accuracy heatmap
python tests/demo_ensemble_stacking.py D  # tabular-only stacking
```

---

## Documentation

Full API reference: [`docs/index.html`](docs/index.html)

---

## Upcoming

- Add support for [InSilicoVA](https://github.com/verbal-autopsy-software/pyinsilicova) in ensemble pipelines. 


---

## License

MIT
