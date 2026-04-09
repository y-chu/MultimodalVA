# MultimodalVA

MultimodalVA is a Python package for cause-of-death classification from verbal autopsy data.
It supports text-only transformer models, tabular machine-learning models, multimodal fusion and ensemble pipelines, and shared prediction / visualization utilities.

All main pipelines follow the same pattern:

```text
split -> prepare -> [HPO] -> train -> predict
```

and return a common `PredictionResult` with:

| Field | Description |
|---|---|
| `top1` | Top-1 predictions |
| `full` | Full class probabilities |
| `topk` | Top-k predictions |
| `id2label` | Class ID to label mapping |

## Installation

Recommended:

```bash
pip install git+https://github.com/y-chu/MultimodalVA.git
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

### Text Classification

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

### Tabular Classification

```python
from multimodalva.tabular import TabularClassifier

clf = TabularClassifier(model_name="lightgbm", output_dir="runs/tabular")
results = clf.run(
    df=df,
    feature_cols=["age", "sex", "fever", "cough"],
    label_col="cause",
    use_optimize=True,
    n_trials=50,
    optimize_metric="f1_macro",
)

print(results["best_hyperparams"])
print(results["predictions"].top1.head())
```

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

# Stage 1: base models + OOF predictions
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

## Supported Models

### Text Models

Use either a package alias, a HuggingFace checkpoint, or a local model path.

| Alias | Checkpoint | Notes |
|---|---|---|
| `bioclinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | Common default |
| `bert` | `bert-base-uncased` | Generic baseline |
| `biobert` | `dmis-lab/biobert-base-cased-v1.2` | Biomedical BERT |
| `bluebert` | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | Clinical + PubMed |
| `biomedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | PubMed full-text |
| `clinicalbert` | `medicalai/ClinicalBERT` | Clinical-domain BERT |
| `biomedroberta` | `allenai/biomed_roberta_base` | Biomedical RoBERTa |
| `bioelectra` | `kamalkraj/bioelectra-base-discriminator-pubmed` | ELECTRA variant |
| `longformer` | `allenai/longformer-base-4096` | Long context |
| `clinicallongformer` | `yikuan8/Clinical-Longformer` | Long clinical narratives |
| `bigbird` | `google/bigbird-roberta-base` | Long context |
| `clinicalbigbird` | `yikuan8/Clinical-BigBird` | Long clinical narratives |
| `roberta-pm` | Facebook Bio-LM RoBERTa-base-PM-M3-Voc-distill | Downloaded automatically via `download_model("roberta-pm")`; also resolved automatically in AutoMM feature fusion |

### Tabular Models

| Alias | Backend |
|---|---|
| `lightgbm` | `lightgbm.LGBMClassifier` |
| `xgboost` | `xgboost.XGBClassifier` |
| `catboost` | `catboost.CatBoostClassifier` |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` |
| `gbdt` | `sklearn.ensemble.GradientBoostingClassifier` |
| `mlp` | `sklearn.neural_network.MLPClassifier` |
| `svm` | `sklearn.svm.SVC` |
| `knn` | `sklearn.neighbors.KNeighborsClassifier` |
| `naive_bayes` | `sklearn.naive_bayes.GaussianNB` |

### Ensemble Strategies

| Method | Description |
|---|---|
| `data_fusion` | Convert tabular features to text and concatenate with the narrative |
| `feature_fusion` | AutoGluon AutoMM joint text + tabular model |
| `soft_voting` | Probability averaging over independently trained base models |
| `stacking` | Stage 1 OOF base models, then either a Stage 2 meta-learner or Stage 2 class-aware voting |

## Demo Scripts

The demos in `tests/` are intentionally small, synthetic where possible, and focused on one core workflow at a time.

| Script | What it shows | Default mode |
|---|---|---|
| `demo_text_classification.py` | TextClassifier wrapper, small HPO example, low-level pipeline steps, `roberta-pm` resolution | `preview` |
| `demo_tabular_classification.py` | TabularClassifier wrapper, adaptive HPO, low-level tabular steps | `preview` |
| `demo_ensemble_data_fusion.py` | Tabular-to-text conversion and one end-to-end data-fusion run | `text` |
| `demo_ensemble_feature_fusion.py` | AutoMM feature fusion with a small fixed run | `preview` |
| `demo_ensemble_voting.py` | Synthetic soft voting, tabular-only voting, mixed voting | `synthetic` |
| `demo_ensemble_stacking.py` | Stage-wise stacking with meta-learner and class-aware voter | `meta` |
| `demo_results.py` | Leaderboard, top-k, confusion, and cause-specific heatmap utilities | `leaderboard` |

## Documentation

- API docs: [docs/index.html](docs/index.html)

## License

MIT
