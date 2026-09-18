# MultimodalVA — FAQ

Practical questions about running the package: what hardware you need, how long
runs take, how much disk they use, how to keep a run reproducible, and how to
publish and re-use trained models.

- [What computing environment do I need?](#what-computing-environment-do-i-need)
- [How long does a run take?](#how-long-does-a-run-take)
- [How much disk space does a run use?](#how-much-disk-space-does-a-run-use)
- [How do I make a run reproducible?](#how-do-i-make-a-run-reproducible)
- [Where are the timing and GPU logs?](#where-are-the-timing-and-gpu-logs)
- [Can I stop a run and resume it later?](#can-i-stop-a-run-and-resume-it-later)
- [What data does the package expect?](#what-data-does-the-package-expect)
- [How do I publish and re-use a trained model?](#how-do-i-publish-and-re-use-a-trained-model)
- [Which model should I pick?](#which-model-should-i-pick)
- [Common errors](#common-errors)

---

## What computing environment do I need?

| Pipeline | Minimum | Comfortable |
|---|---|---|
| Tabular (LightGBM, CatBoost, random forest, …) | Any laptop, CPU only | Any laptop |
| Text (BERT-family, 512 tokens) | Laptop with Apple Silicon (MPS) or 8 GB GPU | One CUDA GPU, 16 GB |
| Data fusion (long-context, 1024–4096 tokens) | 16 GB GPU | One A100/H100, or an HPC allocation |
| Feature fusion (AutoGluon AutoMM) | 16 GB GPU | One CUDA GPU |

The package runs on CUDA, Apple Silicon (MPS) and CPU, and picks the device
automatically. Two Apple Silicon caveats:

- **Longformer and Clinical-Longformer need `PYTORCH_ENABLE_MPS_FALLBACK=1` on
  MPS**, because their attention uses operations Metal does not implement. With
  the fallback those operations run on CPU, which is often *slower* than running
  entirely on CPU. Use **BigBird or Clinical-BigBird** for long-context work on a
  Mac — the package switches them to Metal-native dense attention automatically.
- Multi-worker data loading is disabled on MPS on purpose (macOS process
  start-up costs more than it saves).

On SLURM, **1 GPU + 16 CPUs + 64 GB RAM** is a good default. Dataloader workers
scale from `SLURM_CPUS_PER_TASK`, so a much larger CPU request usually just
lowers your reported CPU efficiency without running faster.

## How long does a run take?

Very roughly, for a VA dataset of a few thousand narratives on one modern GPU:

| Job | Typical wall time |
|---|---|
| Tabular model, 50 HPO trials | minutes |
| Text model (512 tokens), single training run | under an hour |
| Text model, 30–50 HPO trials with 3-fold CV | several hours to a day |
| Long-context data fusion, single training run | a few hours |
| Feature fusion (AutoMM) | set explicitly with `time_limit` |

HPO cost scales with `n_trials × n_cv_folds`, so cross-validated search costs
about *k* times a single trial. Suggested trial counts: 20–25 with CV and no
LoRA, 15–20 with CV and LoRA, 30–40 without CV.

## How much disk space does a run use?

Storage-efficient defaults mean a finished run keeps the final model and the
scores, not the intermediate checkpoints:

| Setting | Default | Effect |
|---|---|---|
| `train(cleanup_checkpoints=…)` | `True` | Delete `checkpoint-*/` after a run completes |
| `train(save_total_limit=…)` | `2` | Most `checkpoint-*/` kept *during* a run |
| `train(report_to=…)` | `"none"` | No TensorBoard event files |
| `optimize(export_best_trial=…)` | `False` | HPO trials train in temp dirs; no per-trial weights kept |
| `optimize(cleanup_trials=…)` | `True` | Remove any persisted `trial_*/` |

**HPO trials keep no model weights.** Each trial and fold trains inside a
temporary directory that is deleted as soon as its score is recorded, so nothing
accumulates even if a sweep is interrupted. What survives is what you need to
resume or retrain: the JournalStorage log `hpo_<model>.log`, `hpo_trials.csv`
and `best_hyperparams.json`.

**How big is a checkpoint?** Each one stores the model weights *plus* the AdamW
optimizer state (roughly 2× the weights):

| Model (full fine-tune) | ~weights | ~per checkpoint |
|---|---|---|
| BERT-base / BioClinicalBERT / RoBERTa-base (~110M) | ~0.4 GB | ~0.5–1.3 GB |
| Longformer-base / BigBird-base (~150M) | ~0.6 GB | ~0.7–1.8 GB |
| bert-tiny (~4M, demos and tests) | ~17 MB | ~30–50 MB |

With LoRA only the adapter parameters are trainable, so the optimizer state is
negligible and a checkpoint is roughly the base-model weight size. Pass
`cleanup_checkpoints=False` if you want to keep checkpoints for inspection —
an uncleaned BERT-base run leaves about 1–2.5 GB behind.

## How do I make a run reproducible?

Pass one seed. `set_seed` drives the train/test split, HPO sampling and training:

```python
from multimodalva import run

run(task="text", data="clean.csv", label="cause", text_col="narrative",
    model="bioclinicalbert", output_dir="runs/text", random_state=484)
```

The default is `42`. For the classifier wrappers the same value is accepted as
`set_seed`, which overrides `random_state` when both are given.

To compare models on exactly the same cases, fix the split instead of the seed:
pass `split=` a column name, a `{"train_ids": [...], "test_ids": [...]}` mapping
(or a JSON file of that shape, with `id_col`), or an explicit
`(train_df, test_df)` tuple.

Seeding is deliberately lightweight — it does not force deterministic GPU
kernels, so small run-to-run differences on GPU are still possible.

## Where are the timing and GPU logs?

Every stage writes a runtime report under its own output directory:

| Stage | Files |
|---|---|
| Pipeline | `output_dir/runtime/pipeline_runtime.json`, `stage_timings.csv` |
| HPO | `output_dir/hpo/runtime/hpo_runtime.json` (or `ray_hpo_runtime.json`) |
| Training | `output_dir/final/runtime/train_runtime.json` |
| Prediction | `output_dir/predictions/runtime/predict_runtime.json` |

When GPU monitoring is on, `gpu_usage.csv` appears alongside them. Control it
with `MULTIMODALVA_ENABLE_GPU_MONITOR=0` to disable, or
`MULTIMODALVA_GPU_MONITOR_INTERVAL_SEC=5` to change the sampling interval.

## Can I stop a run and resume it later?

Yes — re-run the same command. Finished HPO trials, a trained model and existing
predictions are all detected and reused:

- **HPO** resumes from the append-only `hpo_<model>.log` study, and runs only the
  remaining trials.
- **Training** resumes from the latest `checkpoint-*/` (checkpoints are cleaned
  only after a run *completes*, so an interrupted run always has one to use).
- **Stacking** resumes per fold, via the saved out-of-fold prediction files.

## What data does the package expect?

One row per death, with a label column, a free-text narrative column, and/or
structured indicator columns. The fastest way to see the expected shape is the
built-in synthetic data (no real records):

```python
from multimodalva import data, list_datasets

list_datasets()                 # {name: description}
df = data("va_sample")          # InterVA i-codes    -> pairs with utils/qdesc.csv
df = data("va_who2016")         # WHO 2016 ODK codes -> pairs with utils/qdesc_who2016.csv
```

The same datasets are committed as CSVs under `tests/sample_data/`.

**Question-description tables (`qdesc`).** Data fusion turns indicator columns
into sentences using a lookup table keyed by indicator code. Two are shipped:
`utils/qdesc.csv` for InterVA `i`-codes (loaded by default), and
`utils/qdesc_who2016.csv` for WHO 2016 ODK `Id10xxx` codes:

```python
from multimodalva.ensemble.data_fusion import load_qdesc
qdesc = load_qdesc("multimodalva/utils/qdesc_who2016.csv")
```

Both are curated tables — to add or change an indicator, edit the CSV
(columns: `indic, qdesc, sdesc, type, yes, no, desc`).

## How do I publish and re-use a trained model?

Text and data-fusion models are saved in standard Hugging Face format, with the
cause labels in `config.json` and any LoRA adapters merged into the base
weights, so anyone can reload them.

**Publish during a run** (authenticate once with `huggingface-cli login`):

```python
results = clf.run(df=df, text_col="narrative", label_col="cause",
                  push_to_hub=True, hub_repo_id="your-org/va-bert-cod",
                  hub_private=True)
print(results["hub_url"])
```

**Publish a past run** — point at the run's `final/` folder:

```python
from multimodalva.utils import push_to_hub
push_to_hub("runs/text/final", "your-org/va-bert-cod", private=True)
```

A model card with the test metrics, cause list and usage snippets is generated
automatically. For data-fusion models pass `model_kind="data_fusion"` so the
card records the fused-input training format.

**Re-use a published model** — classify directly, or keep fine-tuning:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("your-org/va-bert-cod")
model = AutoModelForSequenceClassification.from_pretrained("your-org/va-bert-cod").eval()
probs = model(**tok("male adult with cough and fever for 9 days",
                    return_tensors="pt")).logits.softmax(-1)
print(model.config.id2label[int(probs.argmax())])
```

```python
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="your-org/va-bert-cod", output_dir="runs/finetuned")
clf.run(df=my_df, text_col="narrative", label_col="cause",
        hyperparams={"epochs": 3, "ignore_mismatched_sizes": True})
```

`ignore_mismatched_sizes=True` initialises a fresh classification head when your
cause set differs from the published model's.

Data-fusion models accept any string, but were trained on the narrative
concatenated with structured fields rendered as sentences. For the closest match
to training, rebuild that string with
`multimodalva.ensemble.data_fusion.build_fused_text(...)`.

## Which model should I pick?

- **Starting out:** `bioclinicalbert` for narratives, `lightgbm` for indicators.
- **Long narratives (over ~512 tokens) or data fusion:** `clinicalbigbird` or
  `clinicallongformer`. On Apple Silicon, prefer BigBird (see above).
- **Limited GPU memory:** enable LoRA (`use_lora=True`) — only the adapters
  train, which cuts optimizer memory sharply.
- **BioMed-RoBERTa vs RoBERTa-PM:** these are two different models. `biomedroberta`
  is `allenai/biomed_roberta_base` (vocabulary 50,265); `roberta-pm` is
  `RoBERTa-base-PM-M3-Voc-distill-hf` (vocabulary 50,008), which is not on the
  Hub and is downloaded once to `~/.cache/multimodalva/`. Ambiguous spellings such
  as `roberta_pm` are rejected with an error naming both.

## Why is there no confidence interval for ECE and MCE?

`calibration_summary()` returns a bootstrap interval for the Brier score but not
for ECE or MCE. That is deliberate: both are biased upward on small samples, and
a bootstrap resample is a small sample.

Drawing *n* cases with replacement from *n* cases — the standard bootstrap —
lands on only about 63% distinct rows, so each resample behaves like a smaller
dataset. Measured on 1000 held-out cases:

| metric | point estimate | bootstrap mean | shift |
|---|---|---|---|
| accuracy | 0.6940 | 0.6943 | +0.0% |
| Brier | 0.0224 | 0.0224 | −0.0% |
| CSMF accuracy | 0.9256 | 0.9157 | −1.1% |
| **ECE** | 0.0121 | 0.0152 | **+25%** |
| **MCE** | 0.6347 | 0.7218 | **+14%** |

Accuracy, Brier and CSMF accuracy come back essentially unbiased under the very
same resamples, so the resampling scheme is not the problem — these two metrics
are. ECE averages `|observed frequency − predicted probability|`, and an absolute
value cannot cancel noise, so a noisier observed frequency can only push the
score up. MCE takes the single worst bin, which is the sparsest and noisiest one.
Brier is a plain mean, so its noise cancels.

The shift is big enough that the interval can sit entirely above the point
estimate, which would misstate the uncertainty rather than describe it. So the
package reports ECE and MCE as point estimates and leaves the choice of
uncertainty method to you.

### Building one yourself

Everything needed is already in the run's outputs — the `full` predictions table
(`true_label` plus `prob_0`, `prob_1`, …) and `id2label.json`, both written to
`<output_dir>/predictions/` and `<output_dir>/final/`. The metric functions are
importable, so you can resample however your analysis calls for:

```python
import json
import numpy as np
import pandas as pd
from multimodalva.results.calibration import classwise_bin_data, ece_score

full = pd.read_csv("runs/text/predictions/predictions_full.csv")
id2label = {int(k): v for k, v in json.load(open("runs/text/final/id2label.json")).items()}

classes = [id2label[i] for i in sorted(id2label)]
y_prob = full[[f"prob_{i}" for i in sorted(id2label)]].to_numpy(float)
y_true = np.zeros_like(y_prob)
y_true[np.arange(len(full)), full["true_label"].map({c: i for i, c in enumerate(classes)})] = 1

def ece(idx):
    return ece_score(classwise_bin_data(y_true[idx], y_prob[idx], classes), len(idx))

point = ece(np.arange(len(full)))

rng = np.random.default_rng(42)
draws = np.array([ece(rng.integers(0, len(full), len(full))) for _ in range(2000)])

print(f"ECE {point:.4f}")
print(f"percentile interval {np.percentile(draws, [2.5, 97.5]).round(4)}")
print(f"bootstrap mean {draws.mean():.4f}  (shift {draws.mean() - point:+.4f})")
```

Whatever you report, state the method and show the shift, because the interval
and the point estimate are not describing the same sample size.

If you want a bias-corrected value rather than an interval, the shift is a
function of sample size, so you can estimate it: score subsamples of several
sizes *m* (drawn **without** replacement, to keep duplicates out of it), fit
`ECE(m) = a + C·m^(−α)`, and read off `a` as the large-sample limit. On the
reference predictions this gives α ≈ 0.65 and a limit roughly 1.4–1.65× below
the reported value across models. Treat that as a sensitivity check rather than
a headline number: it is an extrapolation, and it is fitted, not measured.

## Common errors

**`ImportError` for lightgbm / xgboost / catboost / peft / autogluon** — these
live in optional extras: `pip install "multimodalva[tabular]"`, `[lora]`,
`[feature_fusion]`, or `[all]`.

**`RuntimeError` about Metal/MPS operations with Longformer** — set
`PYTORCH_ENABLE_MPS_FALLBACK=1`, or switch to BigBird, or run on CPU.

**All HPO trials failed** — the package raises with an actionable message rather
than a cryptic Optuna error. The usual causes are GPU out-of-memory (lower
`batch_size`, or enable `gradient_checkpointing=True`) and file-descriptor limits
on macOS.

**HPO is slow to write on a cloud-synced folder** (Dropbox, iCloud, OneDrive) —
pass `storage_path="/tmp/hpo_<name>.log"` to keep the study log on local disk.

**Python InSilicoVA won't install** — `pyinsilicova` requires Python < 3.10,
outside this package's supported range. Use the R implementation, or a separate
legacy environment. InSilicoVA can still be used as a stacking base learner
through that path.
