# MultimodalVA

MultimodalVA is a Python package for cause-of-death classification from verbal autopsy data.
It supports:

- text-only transformer models
- tabular machine-learning models
- multimodal fusion and ensemble pipelines
- shared prediction outputs and visualization utilities

All main pipelines follow the same pattern:

```text
split -> prepare -> [HPO] -> train -> predict
```

and return a common `PredictionResult` with:

- `top1`: top-1 predictions
- `full`: full class probabilities
- `topk`: top-k predictions
- `id2label`: class-id mapping

## Installation

Recommended:

```bash
pip install git+https://github.com/y-chu/MultimodalVA.git
```

Development install:

```bash
git clone https://github.com/y-chu/MultimodalVA.git
cd MultimodalVA
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "multimodalva[tabular]"         # LightGBM / XGBoost / CatBoost
pip install "multimodalva[lora]"            # PEFT / LoRA
pip install "multimodalva[feature_fusion]"  # AutoGluon AutoMM
pip install "multimodalva[insilicova]"      # PyInSilicoVA support
pip install "multimodalva[all]"             # all Python-3.12-compatible extras
```

## Quick Start

### Text

```python
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="bioclinicalbert", output_dir="runs/text")
results = clf.run(
    df=df,
    text_col="narrative",
    label_col="cause",
    hyperparams={"epochs": 3, "batch_size": 16, "learning_rate": 2e-5},
)

print(results["predictions"].top1.head())
```

### Tabular

```python
from multimodalva.tabular import TabularClassifier

clf = TabularClassifier(model_name="lightgbm", output_dir="runs/tabular")
results = clf.run(
    df=df,
    feature_cols=["age", "sex", "fever", "cough"],
    label_col="cause",
    use_optimize=True,
    n_trials=20,
    optimize_metric="f1_macro",
)

print(results["best_hyperparams"])
print(results["predictions"].top1.head())
```

Tabular HPO defaults are adaptive:

- the package infers a search-space profile from `X_train.shape`
- the default profile is `search_space_profile="auto"`
- user `search_space` overrides are still merged on top

### Ensemble: Soft Voting

```python
from multimodalva.ensemble import EnsembleClassifier

clf = EnsembleClassifier(
    method="soft_voting",
    output_dir="runs/ensemble",
    text_models=[{"model_name": "bioclinicalbert"}],
    tabular_models=[{"model_name": "lightgbm"}],
)

results = clf.run(
    df=df,
    text_col="narrative",
    feature_cols=["age", "sex", "fever", "cough"],
    label_col="cause",
)
```

### Ensemble: Stacking

```python
from multimodalva.ensemble import EnsembleClassifier

clf = EnsembleClassifier(
    method="stacking",
    output_dir="runs/stacking",
    text_models=[{"model_name": "bioclinicalbert"}],
    tabular_models=[{"model_name": "lightgbm"}],
    n_folds=5,
)

# Stage 1
clf.train_base_models(
    df=df,
    text_col="narrative",
    feature_cols=["age", "sex", "fever", "cough"],
    label_col="cause",
)

# Stage 2A: meta-learner
clf.train_meta_learner_stage(meta_cv_folds=3)
meta_pred = clf.predict_test()

# Stage 2B: class-aware voting (alternative to meta-learner)
clf.train_class_voter_stage(metric="recall", alpha=1.0, shrinkage=0.25)
vote_pred = clf.predict_test_class_voter()
```

## Supported Pipelines

### Text

Use a package alias, a HuggingFace checkpoint, or a local model path.

Common aliases include:

- `bert`
- `biobert`
- `bioclinicalbert`
- `bluebert`
- `biomedbert`
- `clinicalbert`
- `biomedroberta`
- `bioelectra`
- `longformer`
- `clinicallongformer`
- `bigbird`
- `clinicalbigbird`

Remote model support:

- `roberta-pm`
  - downloaded automatically via `download_model("roberta-pm")`
  - also resolved automatically inside AutoMM feature fusion

### Tabular

Supported model aliases:

- `lightgbm`
- `xgboost`
- `catboost`
- `random_forest`
- `gbdt`
- `mlp`
- `svm`
- `knn`
- `naive_bayes`

### Ensemble

- `data_fusion`
  - render tabular features as text and concatenate with the narrative
- `feature_fusion`
  - AutoGluon AutoMM joint text + tabular model
- `soft_voting`
  - probability averaging over independently trained base models
- `stacking`
  - Stage 1 OOF base models, then either:
    - Stage 2 meta-learner
    - Stage 2 class-aware voting learned from OOF predictions

## Current Defaults and Behavior

### Text pipeline

- best-model selection uses `eval_macro_f1`
- LoRA / PEFT evaluation keeps labels visible via `label_names=["labels"]`
- early stopping remains active whenever an eval split exists
- `roberta-pm` nested archive extraction is handled automatically

### Tabular pipeline

- adaptive default HPO search spaces based on dataset shape
- LightGBM defaults now search:
  - `subsample`
  - `colsample_bytree`
  - `reg_alpha`
  - `reg_lambda`
- Ray HPO auto-redirects to Optuna on non-CUDA environments

### Stacking pipeline

- shared NumPy / joblib compatibility loader for old artifacts
- stage-wise API supports:
  - `train_base_models()`
  - `train_meta_learner_stage()`
  - `predict_test()`
  - `train_class_voter_stage()`
  - `predict_test_class_voter()`

## Demo Scripts

The demos in `tests/` are intentionally small and focused on core package usage.
They use synthetic data where possible.

- `demo_text_classification.py`
  - wrapper usage, HPO, low-level pipeline steps
- `demo_tabular_classification.py`
  - wrapper usage, adaptive HPO, low-level pipeline steps
- `demo_ensemble_data_fusion.py`
  - core data-fusion workflow
- `demo_ensemble_feature_fusion.py`
  - core AutoMM feature-fusion workflow
- `demo_ensemble_voting.py`
  - simple voting from saved-like predictions and end-to-end soft voting
- `demo_ensemble_stacking.py`
  - stage-wise stacking plus class-aware voting
- `demo_results.py`
  - leaderboard, top-k, confusion, and heatmap visualization helpers

## Documentation

- API docs: [docs/index.html](docs/index.html)
- rebuild docs:

```bash
python docs/build_docs.py
```

## License

MIT
