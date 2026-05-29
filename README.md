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
pip install "multimodalva[feature_fusion]"  # AutoGluon AutoMM (tested on autogluon.multimodal 1.5.x)
pip install "multimodalva[all]"             # all Python-3.12-compatible extras
```

Compatibility notes:

- `feature_fusion` is pinned to the AutoGluon MultiModal 1.5.x stack, which in turn expects `torch>=2.6,<2.10`, `transformers>=4.51,<4.58`, and `accelerate>=0.34,<2.0`.
- Python InSilicoVA support is not currently installable under this package's supported Python range (`>=3.12,<3.14`). Use the R-based InSilicoVA workflow in `Analysis/*/unimodal_tabular/InSilicoVA.R`, or a separate legacy environment if `pyinsilicova` is required.

## Quick Start

### Text Classification

```python
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="bioclinicalbert", output_dir="runs/text")
results = clf.run(
    df=df,
    text_col="narrative",
    label_col="cause",
    set_seed=484,
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
    use_cv=True,
    n_cv_folds=3,
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
clf.train_class_voter_stage(metric="brier", shrinkage=0.5)
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

## Reproducibility And Runtime Logs

### One Seed For The Whole Run

For wrapper pipelines, one seed now controls split + HPO + training:

```python
from multimodalva.text import TextClassifier
from multimodalva.ensemble import DataFusionClassifier

# Text pipeline
text_clf = TextClassifier(model_name="bioclinicalbert", output_dir="runs/text")
text_clf.run(
    df=df,
    text_col="narrative",
    label_col="cause",
    set_seed=484,  # single run-level seed
    use_optimize=True,
    n_trials=20,
    use_cv=True,
    n_cv_folds=3,
)

# Data-fusion pipeline
fusion_clf = DataFusionClassifier(
    model_name="allenai/longformer-base-4096",
    output_dir="runs/fusion",
)
fusion_clf.run(
    df=df,
    text_col="narrative",
    feature_cols=["age", "sex", "fever", "cough"],
    label_col="cause",
    set_seed=484,  # single run-level seed
    use_optimize=True,
    n_trials=20,
    use_cv=True,
    n_cv_folds=3,
)
```

Notes:
- Default seed is `42`.
- `set_seed` overrides `random_state` when both are provided.
- Lightweight seeding improves repeatability without forcing strict deterministic kernels.
- Tabular HPO now supports CV scoring too: `TabularClassifier.run(..., use_cv=True, n_cv_folds=3)`.

### Runtime Reports (Timing + GPU/MPS)

Runtime reports are written under each run directory in `runtime/`:

- Wrapper-level pipeline report:
  - `output_dir/runtime/pipeline_runtime.json`
  - `output_dir/runtime/stage_timings.csv`
- HPO report:
  - `output_dir/hpo/runtime/hpo_runtime.json` (Optuna) or `ray_hpo_runtime.json` (Ray)
  - `output_dir/hpo/runtime/stage_timings.csv`
- Final training report:
  - `output_dir/final/runtime/train_runtime.json`
  - `output_dir/final/runtime/stage_timings.csv`
- Prediction report:
  - `output_dir/predictions/runtime/predict_runtime.json`
  - `output_dir/predictions/runtime/stage_timings.csv`

When GPU monitoring is enabled, `gpu_usage.csv` is also saved in the same `runtime/` folder.

GPU monitor controls:
- `MULTIMODALVA_ENABLE_GPU_MONITOR=0` to disable sampling
- `MULTIMODALVA_GPU_MONITOR_INTERVAL_SEC=5` to sample every 5 seconds

### Disk footprint (HPO + training)

Text HPO and training are storage-efficient out of the box. The relevant defaults:

| Setting | Default | Effect |
|---|---|---|
| `train(cleanup_checkpoints=...)` | `True` | Delete `checkpoint-*/` after a run completes successfully |
| `train(save_total_limit=...)` | `2` | Max `checkpoint-*/` kept *during* a run |
| `train(report_to=...)` | `"none"` | No `events.out.tfevents.*` (TensorBoard) files |
| `optimize(export_best_trial=...)` | `False` | HPO trials train in temp dirs; no per-trial weights kept |
| `optimize(cleanup_trials=...)` | `True` | Remove any persisted `trial_*/` (only present if `export_best_trial=True`) |

- **HPO trials persist no model weights.** Each Optuna trial/fold trains inside a temporary directory that is deleted as soon as its score is computed (crash-safe — nothing accumulates even if a sweep is interrupted). Only the scores survive, in the resumable JournalStorage `hpo_<model>.log` and `hpo_trials.csv`, plus `best_hyperparams.json`. Set `optimize(..., export_best_trial=True)` to also keep a reloadable copy of the best trial's model.
- **Final training cleans intermediate checkpoints.** `train(..., cleanup_checkpoints=True)` (the default) removes `checkpoint-*/` after a run completes successfully; the final model (`model.safetensors` + config + tokenizer + label maps) remains and reloads via `predict()`. Interrupted-run resume is unaffected — cleanup runs only on completion, so the latest checkpoint is always available to resume from mid-run. Pass `cleanup_checkpoints=False` to keep checkpoints for inspection.
- TensorBoard event files are off by default (`report_to="none"`; pass e.g. `"tensorboard"` to re-enable).

**How big is a checkpoint?** Each `checkpoint-*/` stores the model weights **plus** the AdamW optimizer state (two moment tensors per trainable parameter, ≈ 2× the weight size). Exact size depends on the model, the precision the weights/optimizer are saved in, and whether the optimizer state is written, but as a rough guide:

| Model (full fine-tune) | ~weights | ~per checkpoint (weights + optimizer) |
|---|---|---|
| BERT-base / BioClinicalBERT / RoBERTa-base (~110M) | ~0.4 GB | **~0.5–1.3 GB** |
| Longformer-base / BigBird-base (~150M) | ~0.6 GB | **~0.7–1.8 GB** |
| bert-tiny (~4M, demos/tests) | ~17 MB | ~30–50 MB |

(The downstream sweep that motivated this saw ~0.5–0.6 GB per checkpoint for a BERT-base-class model.) With **LoRA** only the adapter params are trainable, so the optimizer state is negligible and a checkpoint is roughly just the base-model weight size (the LoRA path also merges adapters into the base model before the final save). At the default `save_total_limit=2`, a completed run holds up to 2 of these *until* `cleanup_checkpoints` removes them — so an uncleaned BERT-base run could leave ~1–2.5 GB behind, and a 20-trial HPO sweep without temp dirs could leave tens of GB.

**Choosing `save_total_limit`:** `1` is the leanest (HuggingFace still protects the best checkpoint when `load_best_model_at_end=True`, so up to 2 exist transiently during a run); `2` (default) is a safe balance for resume + best-model selection; raise it only if you want to inspect several epochs' checkpoints. HPO trials internally use `save_total_limit=1` since they only need a score. These limits apply *during* a run — `cleanup_checkpoints=True` removes all of them afterward regardless.

## Publishing And Re-using Models (Hugging Face Hub)

Trained `TextClassifier` and `DataFusionClassifier` models are saved in the standard Hugging Face format (cause labels in `config.json`, LoRA adapters merged into the base weights), so they can be published to the Hub and reloaded by anyone.

### Publish during a run

Both wrappers accept `push_to_hub` / `hub_repo_id` (default off). Publishing runs on rank 0 only and auto-generates a model card with metrics and usage notes:

```python
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="bioclinicalbert", output_dir="runs/text")
results = clf.run(
    df=df, text_col="narrative", label_col="cause",
    push_to_hub=True,
    hub_repo_id="your-org/va-bert-cod",
    hub_private=True,          # default
)
print(results["hub_url"])
```

### Publish a past run (standalone util)

`push_to_hub` works on any saved model directory — point it at the run's `final/` folder:

```python
from multimodalva.utils import push_to_hub

push_to_hub("runs/text/final", "your-org/va-bert-cod", private=True)
```

Authenticate once with `huggingface-cli login` (or pass `token=...`). For data-fusion models pass `model_kind="data_fusion"` so the card notes the fused-input training format.

### Re-use a published model

```python
# 1. Classify a narrative directly — the model already has a cause-of-death head:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("your-org/va-bert-cod")
model = AutoModelForSequenceClassification.from_pretrained("your-org/va-bert-cod").eval()
probs = model(**tok("male adult with cough and fever for 9 days", return_tensors="pt")).logits.softmax(-1)
print(model.config.id2label[int(probs.argmax())])

# 2. Continue fine-tuning on your own VA data (encoder transfers; a fresh head is
#    initialised when your cause set differs, via ignore_mismatched_sizes):
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="your-org/va-bert-cod", output_dir="runs/finetuned")
clf.run(df=my_df, text_col="narrative", label_col="cause",
        hyperparams={"epochs": 3, "ignore_mismatched_sizes": True})
```

The model is an ordinary text classifier and accepts any string. Data-fusion models were fine-tuned on the narrative concatenated with structured fields rendered as sentences; for the closest match to training, rebuild that fused string with `multimodalva.ensemble.data_fusion.build_fused_text(...)` when you also have the structured fields.

## Sample Data

The quickest way to see the input the package expects is the built-in `data()` loader (R-style; synthetic, no real records):

```python
from multimodalva import data, list_datasets

list_datasets()                 # {name: description}
df = data("va_sample")          # InterVA i-code-style, InterVA i-codes  -> pairs with utils/qdesc.csv
df = data("va_who2016")         # WHO 2016 ODK Id10xxx               -> pairs with utils/qdesc_who2016.csv
df = data("va_sample", n_per_class=10, seed=0)   # kwargs forwarded to the generator
```

`import multimodalva` for `data()` is lightweight — it does not import torch/transformers.

The same datasets are also committed as CSVs under `tests/sample_data/` (no real records):

- `va_sample.csv` — class-balanced InterVA i-code style; `id`, `cause_of_death` (broad cause grouping labels), `narrative`, and **InterVA `i`-code indicator columns** (`i019a`/`i019b` sex, `i022x` age band, `i147o`/`i153o`… symptoms, `y`/`n`). Pairs with the **default** `multimodalva/utils/qdesc.csv` (auto-loaded by data fusion).
- `va_sample_text_only.csv` — `id`, `cause_of_death`, `narrative`
- `va_who2016_sample.csv` — larger, instrument-modeled; `id`, `cause_of_death` (broad cause grouping labels), `narrative`, `sex`, `age_group`, and **WHO 2016 ODK indicator columns (`Id10xxx`, yes/no)**. Pairs with `multimodalva/utils/qdesc_who2016.csv` for data fusion.

Regenerate:

```bash
python tests/demo_hub.py sample-data --output-dir tests/sample_data                 # simple
python tests/demo_hub.py sample-data --schema who2016 --output-dir tests/sample_data # WHO 2016 ODK
```

### Question-description tables (`qdesc`) for data fusion

Data fusion converts tabular indicators to sentences using a `qdesc` table keyed by indicator code:

- `multimodalva/utils/qdesc.csv` — InterVA **`i`-codes** (`i019a`, `i077o`...). Auto-loaded default.
- `multimodalva/utils/qdesc_who2016.csv` — WHO 2016 ODK **`Id10xxx`** codes. Pass explicitly:

```python
from multimodalva.ensemble.data_fusion import load_qdesc
qdesc = load_qdesc("multimodalva/utils/qdesc_who2016.csv")
# DataFusionClassifier.run(..., qdesc=qdesc)
```

Both `qdesc` files are curated, pre-defined data tables shipped with the package — edit the CSV directly to add or adjust indicators (columns: `indic, qdesc, sdesc, type, yes, no, desc`; a blank `yes`/`no` falls back to per-type default verbs).

## Demo Scripts

The demos in `tests/` are intentionally small, synthetic where possible, and focused on one core workflow at a time.

| Script | What it shows | Default mode |
|---|---|---|
| `demo_text_classification.py` | TextClassifier wrapper, small HPO example, low-level pipeline steps, `roberta-pm` resolution | `run` |
| `demo_tabular_classification.py` | TabularClassifier wrapper, adaptive HPO, low-level tabular steps | `run` |
| `demo_ensemble_data_fusion.py` | Tabular-to-text conversion, end-to-end data-fusion run, model-family timing benchmark | `text` |
| `demo_ensemble_feature_fusion.py` | AutoMM feature fusion with a small fixed run | `preview` |
| `demo_ensemble_voting.py` | Synthetic soft voting, tabular-only voting, mixed voting | `synthetic` |
| `demo_ensemble_stacking.py` | Stage-wise stacking with meta-learner and class-aware voter | `meta` |
| `demo_results.py` | Leaderboard, top-k, confusion, and cause-specific heatmap utilities | `leaderboard` |
| `demo_hub.py` | Publish to / re-use models from the Hugging Face Hub; write synthetic sample data | `sample-data` |

## Documentation

- API docs: [docs/index.html](docs/index.html)

## License

MIT

## Use of Generative AI Tools

Claude Code (Opus) and Codex (GPT-5.5) were used as software development assistants for code optimization, package engineering, debugging, generating documentation, synthesizing demo datasets, and preparing testing scripts. The tools did not determine the scientific content of the work. All research questions, methodological choices, analytical strategies, model development, parameter optimization, result validation, interpretation, and scientific conclusions were conceived, evaluated, and approved by the author. All AI-generated outputs were reviewed, tested, and verified by the author prior to use.