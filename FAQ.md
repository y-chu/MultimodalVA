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

Ballpark only — enough to size a job request, not to quote. Wall time scales with
the number of deaths, the number of epochs, and the sequence length, so treat
these as order-of-magnitude.

| Pipeline | MPS ~1k | MPS ~5k | GPU ~1k | GPU ~5k |
|---|---|---|---|---|
| Tabular, one fit (any of the 9 models) | seconds | seconds | seconds | seconds |
| Text, one fit (512 tokens) | ~2 min | ~10 min | <1 min | ~3 min |
| Text, prediction only | seconds | <1 min | seconds | seconds |
| Text HPO, 30 trials, no CV | ~1 h | ~5 h | ~10 min | ~1 h |
| Text HPO, 30 trials, 3-fold CV | ~3 h | ~12 h | ~30 min | ~3 h |
| Data fusion, one fit (1,346 tokens, BigBird) | ~30 min | ~2 h | ~5 min | ~30 min |
| Data fusion, one fit (1,346 tokens, Longformer) | ~1.5 h | ~5 h | ~10 min | ~45 min |
| Stage-1 OOF for stacking (5 text models × 5 folds) | ~1 h | ~4 h | ~15 min | ~1 h |
| Feature fusion (AutoMM) | you set `time_limit` | | | |

Not listed because they finish in seconds to a couple of minutes on a laptop, at
any of these sizes: soft voting, the stacking meta-learner itself, class-aware
voting, bootstrap confidence intervals, calibration, and every plot. Once the
base models exist, combining them is cheap — the cost is all in Stage 1.

**LoRA does not make training much faster.** Measured on the same machine and
model, a training step took 1.32 s with LoRA and 1.31 s without: the forward and
backward passes still run through the full frozen network, and only the optimizer
state shrinks. Use it to fit a larger model in memory, not to save wall time.

**CPU is not a realistic option for the text pipelines.** The same step took
15.5 s on CPU against 1.3 s on Apple Silicon, about 12× slower, which turns a
10-minute fit into two hours and a CV search into weeks. Tabular pipelines are
fine on CPU; the transformer ones are not.

Long-context models are where hardware matters most. Longformer's attention has
no Metal implementation, so on Apple Silicon it runs with
`PYTORCH_ENABLE_MPS_FALLBACK=1` and those operations execute on CPU with a copy
each way every step. That penalty is not proportional to the hardware gap, which
is why the advice is BigBird locally and Longformer on a GPU.

*Basis and disclaimer: these figures were derived with AI assistance by
normalising the runtime logs the package writes itself (per death, per epoch) and
rounding to an order of magnitude. MPS figures come from an Apple M4 Max
(16-core, 128 GB); GPU figures from an NVIDIA H200. The GPU runs used LoRA and
smaller datasets than the MPS runs, so most of the GPU column is extrapolated
rather than measured at that size, and no cell was benchmarked at every
combination shown. Treat the whole table as indicative only — not as a benchmark,
and not as a basis for a performance claim. Measure on your own hardware before
planning a long job: `runtime/gpu_usage.csv` in any run directory records the
accelerator and `*_runtime.json` the wall time.*

## What SLURM resources should I request?

More CPUs is not better. The dataloader is the only part that scales with them,
and past 16 they sit idle:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

With 16 CPUs the package derives 14 dataloader workers, which keeps ~94% of the
allocation busy and saturates one GPU. Requesting 48 CPUs gives 24 workers and
about 52% efficiency — the same wall time, three times the CPU-hours, and a
longer queue. 24 CPUs is also fine (~96%) if memory headroom matters.

One GPU is the right request for every pipeline here. Multi-GPU only helps
through `torchrun`, and these datasets are too small for it to pay off.

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

`calibration_summary()` returns a bootstrap interval for Brier but not for ECE or
MCE, because both are biased upward on small samples — and a bootstrap resample
*is* a small sample, holding only ~63% distinct rows.

Measured on 1,000 held-out cases, same resamples for every metric:

| metric | point | bootstrap mean | shift |
|---|---|---|---|
| accuracy | 0.6940 | 0.6943 | +0.0% |
| Brier | 0.0224 | 0.0224 | −0.0% |
| **ECE** | 0.0121 | 0.0152 | **+25%** |
| **MCE** | 0.6347 | 0.7218 | **+14%** |

Accuracy and Brier are plain means, so their noise cancels. ECE averages
`|observed − predicted|`, and an absolute value cannot cancel noise; MCE takes
the sparsest, noisiest bin. The shift is large enough that the interval can sit
entirely above the point estimate, so quoting one would misstate the uncertainty.

### If you need one anyway

Everything required is in the run outputs — `predictions_full.csv` and
`id2label.json` — and the metric functions are importable, so you can resample
however your analysis calls for:

```python
import json, numpy as np, pandas as pd
from multimodalva.results.calibration import classwise_bin_data, ece_score

full = pd.read_csv("runs/text/predictions/predictions_full.csv")
id2label = {int(k): v for k, v in json.load(open("runs/text/final/id2label.json")).items()}

classes = [id2label[i] for i in sorted(id2label)]
y_prob = full[[f"prob_{i}" for i in sorted(id2label)]].to_numpy(float)
y_true = np.zeros_like(y_prob)
y_true[np.arange(len(full)), full["true_label"].map({c: i for i, c in enumerate(classes)})] = 1

ece = lambda idx: ece_score(classwise_bin_data(y_true[idx], y_prob[idx], classes), len(idx))
point = ece(np.arange(len(full)))

rng = np.random.default_rng(42)
draws = np.array([ece(rng.integers(0, len(full), len(full))) for _ in range(2000)])
print(f"ECE {point:.4f}  interval {np.percentile(draws, [2.5, 97.5]).round(4)}  shift {draws.mean()-point:+.4f}")
```

State the method and report the shift alongside it: the interval and the point
estimate describe different effective sample sizes.

## Feature fusion fails to install, or conflicts with my torch version

Feature fusion is the one pipeline that is not in the default install, because
AutoGluon AutoMM constrains the environment more tightly than the rest of the
package:

| | rest of the package | `[feature_fusion]` |
|---|---|---|
| torch | `>=2.1` | `>=2.6,<2.10` |
| transformers | `>=4.38,<5` | `>=4.51,<4.58` |
| accelerate | `>=0.26` | `>=0.34,<2.0` |

AutoGluon also brings about forty direct dependencies of its own. Those pins are
AutoGluon's, not ours — we state them explicitly so a failed resolve is legible
rather than mysterious.

Two situations come up:

**Your torch is outside 2.6–2.9.** Common on a cluster where the module system
fixes the torch build. Nothing can be done about the pin, but nothing needs to
be: the other five pipelines run on any torch from 2.1 up. Install without the
extra and use `task="stacking"` or `task="voting"` for multimodal work — both
combine text and tabular models, they just do it at the decision level instead
of jointly.

**pip cannot resolve the environment.** Install the extra into a clean virtual
environment rather than on top of an existing one, so pip is free to choose the
torch build AutoGluon wants:

```bash
python -m venv .venv-fusion && source .venv-fusion/bin/activate
pip install "multimodalva[feature_fusion] @ git+https://github.com/y-chu/MultimodalVA.git"
```

To check what you have, ask the package:

```bash
python tests/diagnose.py --group pipelines --with-text
```

Pipelines whose dependencies are missing are reported `SKIP` with the install
command, not `FAIL`.

## Common errors

**`ImportError` for peft or autogluon** — these live in optional extras:
`pip install "multimodalva[lora]"`, `[feature_fusion]`, or `[all]`. LightGBM,
XGBoost and CatBoost are in the default install and need no extra.

**`TypeError` about `group_by_length` when training a text model** — you are on
transformers 5.x, which removed that `TrainingArguments` argument. The package
requires `transformers>=4.38,<5`, so a normal `pip install` downgrades for you;
this only appears if the constraint was bypassed (`--no-deps`, a module-provided
build, or a conda-managed transformers). Install a 4.x build in your environment.
Support for 5.x is planned for a future release.

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
