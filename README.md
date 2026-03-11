# MultimodalVA

**Cause of death classification using verbal autopsy data.**

MultimodalVA provides a unified Python interface for building, tuning, and ensembling
text-based (transformer), tabular (sklearn/LightGBM), and multimodal classifiers on
verbal autopsy (VA) datasets.  All pipelines share a common `split → prepare → [HPO →] train → predict`
interface and produce a `PredictionResult` that includes top-1 labels, full probability matrices,
and top-k predictions.

---

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/y-chu/MultimodalVA.git
```

### Development install (editable)

```bash
git clone https://github.com/y-chu/MultimodalVA.git
cd MultimodalVA
pip install -e ".[dev]"
```

### Dependencies

Install the core dependencies before running the text or ensemble pipelines:

```bash
# Core
pip install torch transformers datasets scikit-learn pandas optuna

# Tabular gradient boosters (optional — install only what you need)
pip install lightgbm xgboost catboost

# LoRA fine-tuning (optional)
pip install peft

# Feature-fusion ensemble (optional)
pip install autogluon.multimodal

# Distributed HPO (optional)
pip install ray[tune] optuna-integration

# Visualization
pip install matplotlib seaborn
```

---

## Quick Start

### Text classification

```python
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="emilyalsentzer/Bio_ClinicalBERT",
                     output_dir="runs/text")
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
    n_trials=30,
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
    text_models=[{"model_name": "emilyalsentzer/Bio_ClinicalBERT",
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
    text_models=[{"model_name": "emilyalsentzer/Bio_ClinicalBERT"}],
    tabular_models=[{"model_name": "lightgbm"}],
    n_folds=5,
)

# Stage 1 — OOF loop (long; can be restarted with resume=True)
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

| Alias | Model |
|---|---|
| `"bioclinicalbert"` | `emilyalsentzer/Bio_ClinicalBERT` |
| `"bert-base-uncased"` | `bert-base-uncased` |
| `"biobert"` | `dmis-lab/biobert-base-cased-v1.2` |
| `"pubmedbert"` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| `"roberta-base"` | `roberta-base` |
| `"longformer-base"` | `allenai/longformer-base-4096` |
| `"bigbird-roberta"` | `google/bigbird-roberta-base` |

### Tabular (scikit-learn compatible)

`"lightgbm"`, `"catboost"`, `"xgboost"`, `"random_forest"`, `"gbdt"`,
`"mlp"`, `"svm"`, `"knn"`, `"naive_bayes"`

### Ensemble strategies

| `method=` | Description |
|---|---|
| `"data_fusion"` | Tabular fields → natural language → concat with narrative → LM fine-tune |
| `"feature_fusion"` | AutoGluon AutoMM joint text + tabular representation |
| `"soft_voting"` | Independent base models + weighted probability averaging |
| `"stacking"` | k-fold OOF meta-features → meta-learner (super learner) |

---

## Demo scripts

All demos are in `tests/` and use synthetic data unless noted:

| Script | Description |
|---|---|
| `demo_text_classification.py` | Sections A–G: TextClassifier, HPO, LoRA, Ray Tune |
| `demo_tabular_classification.py` | Sections A–F: TabularClassifier, HPO, model comparison |
| `demo_ensemble_data_fusion.py` | Sections A–E: DataFusionClassifier, tabular-to-text conversion |
| `demo_ensemble_feature_fusion.py` | Sections A–E: FeatureFusionClassifier, AutoMM fusion strategies |
| `demo_ensemble_voting.py` | Sections A–G: SoftVotingClassifier, vote_from_results |
| `demo_ensemble_stacking.py` | Sections A–H: StackingClassifier, stage-wise execution |
| `demo_results.py` | Sections A–H: all visualization functions (synthetic data) |

Run a single section:

```bash
python tests/demo_results.py G         # cause accuracy heatmap
python tests/demo_ensemble_stacking.py D  # tabular-only stacking
```

---

## Documentation

Full API reference: [`docs/index.html`](docs/index.html)

---

## License

MIT
