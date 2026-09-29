# MultimodalVA — FAQ

Practical questions about running the package: what hardware you need, how long
runs take, how much disk they use, how to keep a run reproducible, and how to
publish and re-use trained models.

- [I have never used this before — where do I start?](#i-have-never-used-this-before--where-do-i-start)
- [What computing environment do I need?](#what-computing-environment-do-i-need)
- [How long does a run take?](#how-long-does-a-run-take)
- [What SLURM resources should I request?](#what-slurm-resources-should-i-request)
- [How much disk space does a run use?](#how-much-disk-space-does-a-run-use)
- [How do I make a run reproducible?](#how-do-i-make-a-run-reproducible)
- [Where are the timing and GPU logs?](#where-are-the-timing-and-gpu-logs)
- [Can I stop a run and resume it later?](#can-i-stop-a-run-and-resume-it-later)
- [Can I combine runs I already finished, without retraining?](#can-i-combine-runs-i-already-finished-without-retraining)
- [How do I keep track of which prediction belongs to which death?](#how-do-i-keep-track-of-which-prediction-belongs-to-which-death)
- [What data does the package expect?](#what-data-does-the-package-expect)
- [How are my indicator columns preprocessed?](#how-are-my-indicator-columns-preprocessed)
- [Can I check a config before I submit the job?](#can-i-check-a-config-before-i-submit-the-job)
- [Some of my causes are very rare — what does the package do?](#some-of-my-causes-are-very-rare--what-does-the-package-do)
- [YAML or JSON for config files?](#yaml-or-json-for-config-files)
- [How do I publish and re-use a trained model?](#how-do-i-publish-and-re-use-a-trained-model)
- [Which trained models can I load?](#which-trained-models-can-i-load)
- [How does it know what kind of model I pointed it at? (and what those errors mean)](#how-does-it-know-what-kind-of-model-i-pointed-it-at-and-what-those-errors-mean)
- [Optuna or Ray — which search backend, and does it change my results?](#optuna-or-ray--which-search-backend-and-does-it-change-my-results)
- [Which search space did my run actually use?](#which-search-space-did-my-run-actually-use)
- [Which model should I pick?](#which-model-should-i-pick)
- [How do I choose a representative model without touching the test set?](#how-do-i-choose-a-representative-model-without-touching-the-test-set)
- [Can I evaluate predictions from another tool (InSilicoVA, InterVA, a colleague's model)?](#can-i-evaluate-predictions-from-another-tool-insilicova-interva-a-colleagues-model)
- [Which stacking combiner should I use?](#which-stacking-combiner-should-i-use)
- [Which meta-learner does stacking use, and can I use my own?](#which-meta-learner-does-stacking-use-and-can-i-use-my-own)
- [Why is there no confidence interval for ECE and MCE?](#why-is-there-no-confidence-interval-for-ece-and-mce)
- [Feature fusion fails to install, or conflicts with my torch version](#feature-fusion-fails-to-install-or-conflicts-with-my-torch-version)
- [Common errors](#common-errors)

---

## I have never used this before — where do I start?

Open [`examples/01_start_here_text_and_tabular.ipynb`](examples/01_start_here_text_and_tabular.ipynb).
It runs on built-in synthetic data, so you can execute every cell before you
have prepared any data of your own, and it explains each step as it goes:
load the data, look at it, train a model on the narrative, train a model on the
symptom indicators, compare them, save the result.

Then [`examples/02_combine_text_and_tabular_fusion.ipynb`](examples/02_combine_text_and_tabular_fusion.ipynb),
which uses the narrative and the indicators together.

After that, copy one of the three templates in
[`examples/`](examples/README.md) — a Python script, a shell command, or a
settings file — and point it at your own data. That means changing four things:
your file, and the names of your label, narrative and indicator columns.
Everything else has a working default, and the settings files tag each line
`[CHANGE]`, `[KEEP]` or `[TUNE]` so it is clear which ones repay attention.

If a run fails, `python tests/diagnose.py` reports which pipelines work in your
environment and what is missing for the rest; see [Common errors](#common-errors).

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
voting, ensemble selection, the stacking combiner comparison, bootstrap
confidence intervals, calibration, and every plot. Once the
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

### What a finished run leaves behind

Measured on the built-in demo data (330 deaths, 11 causes, 66 test rows) with
BERT-base-sized text models. The **number of rows barely matters** — model
weights dominate everything else.

| Run | Size | What dominates |
|---|---|---|
| `tabular`, no search | **2.8 MB** | the fitted model |
| `tabular` + search (3-fold CV) | **23 MB** | the best trial's fold models (~17 MB) |
| `text`, no search | **419 MB** | `model.safetensors` (418 MB) |
| `text` + search | **419 MB** | same — the search itself adds ~0.1 MB |
| `data_fusion` | **419 MB** | one long-context model |
| `voting`, 3 tabular models | **8.7 MB** | three fitted models |
| `stacking`, 1 text + 2 tabular | **426 MB** | the text base model |

A rough planning rule: **count the models you are training and multiply.** One
text model is ~0.4 GB (BERT-base) or ~0.6 GB (Longformer, BigBird); one tabular
model is single-digit MB for boosted trees, more for a large random forest.
Everything else — predictions, search records, metadata — is typically under
1 % of the total.

### Predictions scale with deaths × causes

The three CSVs are small and grow predictably: roughly **22 bytes per death per
cause** for `predictions_full.csv`, ~53 bytes per death for
`predictions_top1.csv`, and ~130 bytes per death for `predictions_topk.csv` at
`top_k=3`.

For 10 000 test deaths and 30 causes that is about 6.6 MB, 0.5 MB and 1.3 MB —
about 8 MB in total, against hundreds of MB for a single text model.

### What the search keeps

Text and tabular differ here, and the difference is the reason a tabular search
costs more than a text one:

- **Text** (`optimize_text`) keeps **no model weights**. Every trial and fold
  trains in a temporary directory that is deleted once its score is recorded, so
  an interrupted sweep accumulates nothing. What survives is the JournalStorage
  log `hpo_<model>.log`, `hpo_trials.csv`, `hpo_leaderboard.csv`,
  `hpo_convergence.png` and `best_hyperparams.json` — about 100 KB in total.
  Pass `export_best_trial=True` to also keep the best trial's weights.
- **Tabular** (`optimize_tabular`) keeps **the best trial's model for every CV
  fold**, in `hpo/best_trial/fold_*/`. With `n_cv_folds=3` that is three models,
  which above is larger than the final model itself. The other trials are
  removed by `cleanup_trials=True`. There is no switch to turn this off; delete
  `hpo/best_trial/` afterwards if you do not want it.

In an ensemble, each base model gets its own `hpo/` folder under
`base_models/`, so the search cost multiplies by the number of models searched.

### What is kept while a run is in progress

Defaults keep the final model and the scores, not the intermediate checkpoints:

| Setting | Default | Effect |
|---|---|---|
| `train_text(cleanup_checkpoints=…)` | `True` | Delete `checkpoint-*/` once the run completes |
| `train_text(save_total_limit=…)` | `2` | At most two `checkpoint-*/` kept *during* a run |
| `train_text(report_to=…)` | `"none"` | No TensorBoard event files |
| `optimize_text(export_best_trial=…)` | `False` | No per-trial weights kept |
| `optimize_*(cleanup_trials=…)` | `True` | Remove persisted `trial_*/` |

**Peak usage is higher than the final size.** A checkpoint holds the weights
*plus* the AdamW optimizer state, roughly 2–3× the weights:

| Model (full fine-tune) | ~weights | ~per checkpoint |
|---|---|---|
| BERT-base / BioClinicalBERT / RoBERTa-base (~110M) | ~0.4 GB | ~0.5–1.3 GB |
| Longformer-base / BigBird-base (~150M) | ~0.6 GB | ~0.7–1.8 GB |
| bert-tiny (~4M, demos and tests) | ~17 MB | ~30–50 MB |

So budget roughly **3× the final model size** as headroom while a text model
trains. With LoRA only the adapter parameters are trainable, so the optimizer
state is negligible and a checkpoint is about the base-model weight size.
`cleanup_checkpoints=False` keeps checkpoints for inspection — an uncleaned
BERT-base run leaves about 1–2.5 GB behind.

## How do I make a run reproducible?

A run has two separate sources of randomness, and there is one seed for each:

| Seed | Controls | Vary it to measure |
|---|---|---|
| `split_seed` | **which rows go where** — the train/test split, each search's cross-validation folds, the early-stopping validation slice, stacking's out-of-fold partition | sampling uncertainty: how much the result depends on which deaths landed in the test set |
| `train_seed` | **what the model does with its rows** — weight initialisation, dropout, batch order, the Optuna sampler, each estimator's `random_state`, AutoMM's trainer | model stochasticity |

Both default to `42`, so a run is reproducible without passing either:

```python
from multimodalva import run

run(task="text", data="clean.csv", label_col="cause", text_col="narrative",
    model="bioclinicalbert", output_dir="runs/text",
    split_seed=42, train_seed=42)
```

**What varying each one tells you** depends on how you pass `hyperparams`:

| You vary | with `hyperparams=` | You measure |
|---|---|---|
| `split_seed` only | anything | total sensitivity to how the data was partitioned |
| `train_seed` only | a fixed `{...}` | pure model stochasticity — folds and hyperparameters held constant |
| `train_seed` only | `Optimize(...)` | variance of the whole training procedure, search included |

The cross-validation folds follow `split_seed`, not `train_seed`, on purpose. If
they followed `train_seed`, reseeding the model would also change which
hyperparameters the search picked, so "model noise" replicates would quietly
differ in their hyperparameters too — and models on a validation leaderboard
could only be compared if they had been scored on the same folds. Whether the
search is stable against fold composition is answered more directly by the
`cv_std_<metric>` and `fold_<i>_<metric>` columns each search records per trial.

Stacking runs in stages, and stages 2 and 3 can run in a fresh session. They
read the seeds stage 1 used from `oof/oof_metadata.json`, so you do not have to
restate them; an explicit `split_seed=` / `train_seed=` still overrides.

`bootstrap_ci(random_state=)` is separate again. It resamples the finished test
predictions to build confidence intervals and has nothing to do with training.

**To compare models on exactly the same cases,** keep `split_seed` the same for
all of them, or fix the split outright: pass `split=` a column name, a
`{"train_ids": [...], "test_ids": [...]}` mapping (or a JSON file of that shape,
with `id_col`), or an explicit `(train_df, test_df)` tuple. A fixed `split=`
also survives changes to the data's row order, which a seed does not.

`random_state`, `set_seed` and `automm_seed` were replaced by these two. Passing
one raises an error naming its replacement, rather than being silently ignored.

### How exact is "reproducible"?

By default, the same two seeds on the same machine give **the same predicted
labels**. The probabilities can differ in the last few digits — around 1e-16 on
CPU, from the order floating-point sums are taken in, and up to about 1e-7 on an
Ampere-or-newer GPU, where TF32 matrix multiply is on for speed. No metric this
package reports is affected at that size.

For bit-for-bit identical probabilities, pass `deterministic=True`. It costs:

- **Speed.** TF32 is switched off, so matrix multiply on Ampere+ GPUs runs
  roughly two to three times slower.
- **Robustness.** Torch is told to use only deterministic kernels. An operation
  that has none **raises `RuntimeError`** instead of falling back, so a model
  that trains fine by default can fail outright.
- **Scope.** It holds on one machine with one set of library versions. Nothing
  makes results bit-identical across GPU models or library versions.

Use it for the run behind a published number, not for day-to-day work.

**Library versions are part of the run, and the package does not pin them for
you.** Seeds fix what this package controls; they cannot fix what a dependency
changes between releases. Gradient-boosting libraries are the clearest case: with
`subsample` or `colsample_bytree` below 1 — the regime a tuned model usually lands
in — xgboost's tree construction is not stable across minor releases. Fitting the
same rows, folds, seed and hyperparameters under 3.0.2 and under 3.1.3 was
measured to move predicted probabilities by up to 0.30, enough to change 6% of
top-1 labels, while lightgbm, catboost, the MLP and the text models all reproduced
exactly. The dependency ranges here are deliberately wide enough to install
alongside other software, so they do not prevent this.

So if a run is one you will need to reproduce, record the environment next to it
and reinstall that:

```bash
pip freeze > runs/text/environment.txt
```

A run records the `multimodalva_version` that produced it in
`training_metadata.json`, but not its dependencies' versions — `pip freeze` is
what closes that gap today. A `pip install` that silently resolves a
gradient-boosting library to a different minor version is enough to stop old
artifacts reproducing, and it does not announce itself.

## Where are the timing and GPU logs?

Every stage writes a runtime report under its own output directory:

| Stage | Files |
|---|---|
| Pipeline | `output_dir/runtime/pipeline_runtime.json`, `stage_timings.csv` |
| HPO | `output_dir/hpo/runtime/hpo_runtime.json` |
| Training | `output_dir/final/runtime/train_runtime.json` |
| Prediction | `output_dir/predictions/runtime/predict_runtime.json` |

The HPO report has the same name whichever search backend ran; which one it was
is a field inside it (`pipeline`), not part of the file name.

When GPU monitoring is on, `gpu_usage.csv` appears alongside them. Control it
with `MULTIMODALVA_ENABLE_GPU_MONITOR=0` to disable, or
`MULTIMODALVA_GPU_MONITOR_INTERVAL_SEC=5` to change the sampling interval.

## Can I stop a run and resume it later?

Yes — re-run the same call with the same `output_dir`. What gets picked up
depends on the pipeline:

| Pipeline | Search (`hyperparams=Optimize(...)`) | Training | Finished work |
|---|---|---|---|
| `text`, `data_fusion` | continues the study; only the remaining trials run | continues from the latest `checkpoint-*/` | — |
| `tabular` | continues the study | re-fits (seconds to minutes) | — |
| `feature_fusion` | not resumable — AutoMM runs its own search | not resumable mid-fit | a finished model or finished predictions are reused |
| `voting` | continues each base model's study | — | **each finished base model is reused**; only unfinished ones train |
| `stacking` | continues each base model's study | — | each finished fold, final model and meta-learner is reused |

Checkpoints are deleted only after a run *completes*, so an interrupted run
always has one to resume from. Pass `Optimize(resume=False)` to restart a search
from scratch. To start the whole run over, pass `resume=False` — the same
argument on every task (`--no-resume` on the command line). A search follows it
unless told otherwise with `Optimize(resume=...)`. Stacking holds it on the
object, `StackingClassifier(resume=False)`, because one object spans several
stage calls; `run(task="stacking", resume=False)` puts it there for you.

A voting base model counts as finished only when both its trained model and its
prediction files exist, so one interrupted during prediction is redone rather
than half-reused.

Resume is deliberately strict. Each training pipeline writes
`resume_manifest.json`, fingerprinting the consumed train/test rows and the
settings that affect its artifacts. With `resume=True`, existing artifacts are
reused only when that signature matches exactly. A changed split, feature set,
model spec, hyperparameter configuration or fused text raises and tells you to
use a new `output_dir` or move the old artifacts. Pass `resume=False` only when
you intend to rebuild the directory's artifacts from the current call.

**Artifacts made before manifests existed: `resume="adopt"`.** A directory with
artifacts but no `resume_manifest.json` is refused by default, because there is
no signature to check. `resume="adopt"` (`--resume-adopt`) reuses it anyway and
records the signature of the current call, so later runs are checked normally:

```bash
multimodalva run job.yaml --resume-adopt      # once, for a pre-manifest directory
multimodalva run job.yaml                     # from here on, checked as usual
```

It warns every time, because nothing verified that those artifacts came from this
data, split and settings — that is your judgement, not a check. It covers only
the case where there is **no** signature: a manifest that is present and
*disagrees* still raises, since a disagreement is evidence of a real difference
rather than a missing check. Runs the current pipeline produces need no adoption;
they carry a signature from the start.

## Can I combine runs I already finished, without retraining?

Yes. `load_predictions()` reads any run's saved predictions back, and
`vote_from_results()` soft-votes over them:

```python
from multimodalva.utils.predictions import load_predictions
from multimodalva.ensemble import vote_from_results

bert = load_predictions("runs/bert")          # a run directory ...
lgbm = load_predictions("runs/lgbm/predictions")  # ... or its predictions/ folder
voted = vote_from_results([bert, lgbm], id2label=bert.id2label)
```

The runs must share one train/test split — the same `split_seed`, or the same
fixed `split=`. When they were run with `id_col`, `vote_from_results()` checks
that the ids line up row for row and refuses to vote otherwise. Without ids it
cannot check, so matching the split is up to you.

To compare finished runs rather than combine them, pass the directories to
`predictions_frame()` and then `performance_leaderboard()`; see
`examples/03_evaluate_and_compare_results.ipynb`.

## How do I keep track of which prediction belongs to which death?

Pass `id_col=` — the name of a column that identifies each record. Every task
accepts it, and every prediction table (`top1`, `full`, `topk`) then starts with
an `id` column, so results can be joined back to the source data and matched
across pipelines by id rather than by row position.

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

`data()` builds these from a fixed seed, so it needs no files. The same datasets
are also committed as CSVs under `tests/sample_data/` in the repository, for
looking at outside Python.

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

## How are my indicator columns preprocessed?

Two settings, both on `run()` (and on `prepare_tabular_dataset()` underneath).
Column types are auto-detected from the training split — `object`/`category`
dtype becomes categorical, numeric dtype becomes numeric — or name them yourself
with `cat_cols=` / `num_cols=`. The transformer is fitted on the training rows
only, applied to both splits, and saved inside `model.joblib` so inference on new
data repeats it exactly.

**`encode_categoricals`** — four accepted values:

| Value | What it does |
|---|---|
| `"ordinal"` (default) | `OrdinalEncoder`. Unknown categories become `-1`, and **missing stays missing** (`encoded_missing_value=np.nan`), so CatBoost, LightGBM and XGBoost can learn from missingness itself |
| `"auto"` | **Identical to `"ordinal"`** — not a cleverer policy, just the tree-safe default under another name |
| `"onehot"` | `OneHotEncoder(handle_unknown="ignore")`. Recommended for MLP and linear models, which would otherwise read the ordinal codes as a magnitude. Costs width: 353 indicators with three values each become roughly a thousand columns |
| `None` | No encoding. The columns must already be numeric |

**`scale_numeric`** — `StandardScaler` on the numeric columns; default `False`,
because tree models do not need it. **It does nothing when there are no numeric
columns.** That is the usual case for WHO-style indicator data: if the indicators
are strings (`"Y"` / `"N"` / `"."`), every feature is categorical, the scaler is
never added to the pipeline, and `scale_numeric=True` and `False` produce
identical matrices. The line to check in your own log is:

```
Features: 0 numeric, 353 categorical. encode_categoricals=ordinal, scale_numeric=False.
```

**Per base model, in an ensemble.** Voting prepares each base model separately,
so a spec may set its own:

```python
tabular_models=[
    {"model_name": "lightgbm", "encode_categoricals": "ordinal"},
    {"model_name": "mlp", "encode_categoricals": "onehot", "scale_numeric": True},
]
```

Stacking cannot: it builds **one** shared feature matrix that the out-of-fold
loop and the saved `X_test` both assume, so per-spec preprocessing is impossible
there. A stacking spec that sets either key to something other than the run-level
value now raises and says so, rather than being ignored. If the base models need
different preprocessing, use voting, or train them as separate tabular runs and
combine the saved predictions.

## Can I check a config before I submit the job?

Yes — `--dry-run` runs every step that happens before a model is built (argument
checks, loading, filtering, the missing-row drop, the train/test split, feature
resolution), reports what the run would do, and trains nothing.

```
$ multimodalva run job.yaml --dry-run
task             text
pipeline         TextClassifier
model            bioclinicalbert -> emilyalsentzer/Bio_ClinicalBERT
rows             4213 after filtering and the missing-row drop
classes          11
smallest class   17 row(s) ('Assault')
split            random: test_size=0.2, stratify=True, split_seed=42
hyperparameters  search: metric=csmf_accuracy, n_trials=40
hpo backend      ray
search space     the package default, adapted to this data
search split     3-fold cross-validation

error: max_lenght= is not accepted by TextClassifier. Did you mean max_length=?
error: text column 'narrativ' is not in the data. Available: ['id', 'cause', ...]

Dry run: 2 problem(s); the run would not get past them.
```

It exits 2 when anything would stop the run, so a job script can gate on it:

```bash
multimodalva run job.yaml --dry-run || exit 1
multimodalva run job.yaml
```

One report names every problem it finds rather than only the first, so a config
with three mistakes takes one look instead of three submissions.

A config file is where this earns its keep. On the command line `argparse`
already rejects a flag it does not know, but a config file's keys are passed
straight through, so `max_lenght: 512` in YAML is accepted by the loader and
only fails once the pipeline is reached. The check compares each key `run()`
does not name against the signature of the class the task dispatches to, and the
same comparison covers `init_kwargs` keys, model names and base-model specs.

Among the things it reports before training:

- more search folds than the rarest class has rows in the actual training
  split — this is an error, because every fold must contain every class;
- `init_kwargs={"text_models": ...}` alongside `text_models=`, where
  `init_kwargs` wins and the other argument has no effect;
- a filter value that matches no row, which otherwise surfaces as "no rows
  remain after dropping missing label/text" — naming the step that noticed
  rather than the value responsible.

It performs the real split, so the reported train/test counts and HPO
feasibility checks are based on the rows the model will actually see. It also
rejects unknown base-model spec keys, one-model voting, constructor-only
settings passed as run keywords, and modality inputs that do not match a
feature-fusion strategy.

**What a clean dry run does not promise.** It checks that the configuration is
consistent with the data, not that the run will succeed. Nothing trains, so it
cannot tell you whether a batch size fits in the GPU, whether the trial budget is
enough for the search space, or whether an `Optimize(extra=...)` key means
anything to the backend. It also cannot verify a Hugging Face Hub ID without
network access, and it will not download a remote model to check it — a preflight
that pulls half a gigabyte is no longer a preflight.

From Python the same check is `preflight(**config)`, which takes exactly what
`run()` takes and returns a report instead of printing one:

```python
from multimodalva import preflight

report = preflight(task="tabular", data="clean.csv", label_col="cause",
                   features="re:^i\d{3}[a-zA-Z]$", model="lightgbm")
if not report["ok"]:
    raise SystemExit("\n".join(report["problems"]))
```

`report["facts"]` holds the label → value pairs shown above, `report["notes"]`
the silent-behaviour warnings, and `report["log"]` the lines the package itself
emitted while checking.

## Some of my causes are very rare — what does the package do?

Three separate thresholds decide what "too rare" means, and they do different
things:

| Threshold | Default | What happens |
|---|---|---|
| A class has fewer rows than `cv_folds` in the **training** rows | `cv_folds=3` | A search **raises**: a stratified fold cannot hold a class with fewer rows than there are folds. Lower `Optimize(cv_folds=...)`, or use `"auto"` (below) |
| A class has fewer than 2 rows | fixed by scikit-learn | A stratified train/test split **raises**. `stratify=False` is the only way past it, and it may leave a class out of training |
| A class has fewer than `SMALL_CLASS_MIN` rows on one side of a split | 5 | A **warning** only. Nothing is dropped, merged or changed |

`Optimize(cv_folds="auto")` is the setting for data whose rarest cause you do not
know in advance: it uses as many folds as the rarest training class supports, up
to the default 3, and says in the log how many it chose and why. It still raises
if a class has a single row, because no fold count is valid for that.

```python
mv.run(task="text", data="clean.csv", label_col="cause", text_col="narrative",
       hyperparams=Optimize(n_trials=30, cv_folds="auto"))
```

Two things to know before using it. Trial scores average over however many folds
it picked, so a 2-fold search is noisier than a 3-fold one and **the two are not
comparable** — do not put them on one leaderboard. And the fold count that
actually ran is what the run records, in its search-name signature and its
per-fold trial columns, so an artifact never claims `"auto"`.

`--dry-run` reports the resolved fold count before the job starts.



It warns, and leaves the data alone.

Every split is stratified, so a rare cause keeps its share on both sides — but a
share of a small number is still a small number. After the train/test split, and
after a validation split, the run logs a line for every cause left with **fewer
than five rows**, naming the causes and their counts:

```
Small sample size: 2 of 14 classes have fewer than 5 rows in the test set
(Meningitis=1, Maternal=3). Scores for these classes carry large uncertainty,
and a stratified split cannot hold their proportions; ...
```

A cause the split left out of one side entirely is reported as `=0`. This is the
case worth catching early: a per-cause F1 of 0.0 for a cause with no test rows is
an artefact of the split, not a finding about the model.

Nothing is dropped, merged, or resampled on the package's initiative, and no
threshold changes a default — what to do about a rare cause is your decision.
The usual options:

| Situation | What to consider |
|---|---|
| A few causes with a handful of deaths each | Group them into a broader category (e.g. an "other" group) before calling `run()`, and say so in the paper. |
| A cause absent from the test set | A larger `test_size`, or report that cause's metrics as not estimable rather than as zero. |
| Many rare causes, few total records | Prefer `f1_macro` / `balanced_accuracy` / `csmf_accuracy` over accuracy, and read per-cause numbers as indicative. Bootstrap confidence intervals (`bootstrap_ci()`) make the uncertainty explicit. |

Two things the package does handle by itself, because the alternative is a crash:
a validation slice thinner than the number of causes is enlarged to one row per
cause, and a cause with only one training row makes that split unstratified. Both
are logged when they happen.

## YAML or JSON for config files?

Both work. `multimodalva run <file>` picks the parser from the extension —
`.yaml` and `.yml` are read as YAML, anything else as JSON — and the two give
identical runs. `examples/config_tabular_model.json` is the same run as
`examples/config_tabular_model.yaml`.

The shipped examples are YAML because **JSON cannot contain comments**, and the
explanation of each setting is most of what those files are for. YAML also needs
no braces, quotes or commas, so a value can be edited without a punctuation
error, and a setting is disabled by putting `#` in front of it.

Use JSON if you already work in it, or if the config is written by another
program. It has one unambiguous way to write any value, whereas YAML follows the
1.1 rules here: `no` and `off` become `false`, and `1.10` becomes the number
`1.1`. Quote anything that must stay text.

Neither is required — the Python and command-line templates in `examples/` run
the same pipelines with no config file at all.

A key that is not a `run()` argument stops the run immediately and names the
key, in either format, so a typo cannot silently change what is trained.

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

## Which trained models can I load?

With `predict_from_pretrained()`. Point it at a run's `output_dir` (or its
`final/`), a Hugging Face Hub repo id, or a Hub URL, and pass the new data —
with a label column for evaluation, without one for plain prediction:

```python
import multimodalva as mv

res = mv.predict_from_pretrained("runs/text_run", new_df, text_col="narrative")
res = mv.predict_from_pretrained("runs/tabular_run", new_df, label_col="cause")
res.checks        # what was detected about the model and the input
```

Served: text, data fusion (pass the already-converted long text as `text_col`),
tabular and feature fusion.

Voting and stacking runs have their own entry point, because they hold several
base models plus the rule that combines them:

```python
res = mv.predict_ensemble_from_pretrained("runs/stacking", new_df,
                                          text_col="narrative")
res = mv.predict_ensemble_from_pretrained("runs/stacking", new_df,
                                          combiner="class_aware_voting")
```

Each base model is loaded from the run directory and scored, then combined
exactly as the run combined it: the saved weights for soft voting, and for
stacking whichever combiner that run fitted — the meta-learner, a simple
average, the class-aware voter, ensemble selection, or a meta-learner named in
`combiner=`. Left unset it reuses the combiner the run delivered. Stacking
records absolute paths to its base models, so a run that has been moved or
opened on another machine is re-anchored to the directory you point at. If you
pass the `output_dir` you gave `run()` and the run lives in a subdirectory of it
(`soft_voting/`, `stacking/`), that subdirectory is found for you; two runs
under one directory is an error, not a guess.

Anything MultimodalVA saved, and models trained elsewhere, provided the artifact
carries two things: **what the classes are**, and — for tabular models — **what
the feature columns are, by name**.

| Artifact | Classes | Feature names | Works |
|---|---|---|---|
| MultimodalVA text / data-fusion run | `config.json` + `id2label.json` | n/a | yes |
| MultimodalVA tabular run | `id2label.json` | read from the bundled preprocessor | yes |
| Hugging Face sequence classifier | `config.json` `id2label` | n/a | yes, if `id2label` is real |
| sklearn / CatBoost / LightGBM / XGBoost **fitted on a DataFrame** | you supply, or `id2label.json` | read from the model | yes |
| the same **fitted on a bare NumPy array** | as above | not recorded | only if you pass `feature_cols=[...]` |

Two traps, both of which look like valid metadata:

- A Hugging Face model fine-tuned without setting `id2label` still writes
  `{0: "LABEL_0", 1: "LABEL_1", ...}`. The names are placeholders.
- CatBoost fitted on a NumPy array writes `feature_names_ = ["0", "1", "2", ...]`
  — positions, not columns.

Both are treated as missing. Missing class names degrade to integer class ids
with a warning. Missing tabular feature names are an error: columns are matched
**by name**, never by position, because a silent mis-alignment produces confident
nonsense. Pass `feature_cols=[...]` to proceed anyway.

A trained-on column that is absent from your data is an error that names the
column; `missing_feature_method="fill_na"` fills it with NA instead. That is a
different thing from NA *values* inside a column that is present, which are
normal in verbal autopsy data and never warn.

A foreign tabular model brings no preprocessor, so its input must already be in
the form it was trained on. MultimodalVA artifacts carry their own
`ColumnTransformer` and handle encoding for you.

**On the labels themselves.** The ICD-10 mapping this package uses is published,
but whether a cause label from another site, instrument or coding round means the
same thing as yours is a judgement only you can make. Predictions are reported in
the label vocabulary of the artifact you loaded. Check that vocabulary against
your own definitions before you use or compare the output.

### The one case that cannot be re-scored: a vote over saved predictions

Scoring new data means running the base models again, so the run has to *have*
them. Which combiner was used is not the issue — where the models live is:

| Run | Re-scorable? | Why |
|---|---|---|
| `run(task="voting")` — the soft-voting pipeline | yes | it trains its base models and keeps them under `base_models/` |
| `run(task="stacking")`, any combiner — meta-learner, simple average, **class-aware voting**, ensemble selection | yes | the refitted base models sit in `final/`, and each fitted combiner (`meta_learner/`, `class_voter/class_weights.npy`, `ensemble_selection/`) is stored beside them |
| `vote_from_results()` / `soft_vote()` — a vote computed over predictions you already had | **no** | the output is the vote; no model is stored, and nothing records where the predictions came from |

So class-aware voting is fine: it is a stacking stage-2 combiner learned from the
out-of-fold matrix, and it is applied to base models the stacking run kept.
Asking for a combiner the run never fitted is refused by name:

```
error: combiner must be one this run fitted — ['meta_learner',
'class_aware_voting'] — or one of [...]
```

For the last row, score each base model with `predict_from_pretrained()` and
combine those results with `vote_from_results()`, or re-run the voting pipeline
so it trains and keeps its base models.

## How does it know what kind of model I pointed it at? (and what those errors mean)

You do not have to say. Both `predict_from_pretrained()` and the
`multimodalva predict` command read it from the artifact before loading
anything, and print what they concluded:

```
$ multimodalva predict --source runs/stacking --data new.csv --text-col narrative
Detected: stacking (from the run's own metadata)
```

The line always says **where** the answer came from, because the two sources are
not equally trustworthy:

| Wording | Meaning |
|---|---|
| `from the run's own metadata` | `training_metadata.json` records the pipeline. Reliable. |
| `from the files present — not recorded, so check it` | A guess from what is in the directory: `config.json` → a text model, `model.joblib` → tabular, `automm_model/` → feature fusion. Normal for a plain Hugging Face model or a run from an older version. |
| `from --task` | You said so. |

You can override with `--task` (`task=` in Python), and detection still runs.
What happens when the two disagree depends on which source was involved:

**The run's own metadata contradicts you → error, nothing is scored.**

```
error: task='text' contradicts this run: its own training_metadata.json says
'tabular'. Predicting anyway would spend a full inference pass producing 'text'
predictions from a 'tabular' run. Check that runs/lgbm/final is the directory
you meant; pass force=True (CLI: --force) to override a recorded pipeline you
know to be wrong.
```

Almost always this means the `--source` path is not the run you had in mind.
Fixing the path costs a minute; being warned instead and carrying on would cost
the whole job and hand back predictions that *look* fine. If you are certain the
recorded metadata is wrong, `--force` proceeds and warns loudly.

**A guess contradicts you → warning, and your `--task` wins.** Structure cannot
tell a data-fusion model from a plain text model — they are the same kind of
file — so this is exactly what `--task data_fusion` is for.

**Nothing could be determined:**

```
error: Cannot tell what kind of model is in <dir>: no training_metadata.json
'pipeline', no config.json (Hugging Face), no model.joblib, no automm_model/.
Pass task=... if you know it.
```

The directory holds no model this package recognises — check the path, or name
the kind with `--task`. Voting and stacking runs have no structural fallback (a
combiner is not a file that identifies itself), so an ensemble run from an older
version needs `--task voting` / `--task stacking`.

**Several runs under one directory:**

```
error: <dir> holds several runs (['soft_voting', 'stacking']). Name the one you
mean, e.g. source=<output_dir>/stacking.
```

`EnsembleClassifier` writes each method into its own subdirectory, so pointing
at the parent is ambiguous when you ran more than one. Add the subdirectory.

### Check before you submit a job

`--dry-run` does all of the above plus the input checks, then stops without
loading a single batch:

```
$ multimodalva predict --source runs/stacking --data new.csv \
      --text-col narrative --label-col cause --dry-run
Detected: stacking (from the run's own metadata)
Records: 4213
Base models: 10 (5 tabular, 5 text)
Combiner: meta_learner
Text column 'narrative': present
Label column 'cause': present, performance will be reported
Dry run: nothing was scored or written.
```

Worth its few seconds in front of anything queued on a GPU: a wrong path, a
missing column or the wrong combiner all surface here instead of at the end of
the run.

## Optuna or Ray — which search backend, and does it change my results?

`Optimize(backend=...)` picks the engine that runs the search's trials:

| `backend` | What it does | Needs |
|---|---|---|
| `"auto"` (default) | Ray when a CUDA GPU or a SLURM allocation is visible **and** Ray is installed; Optuna otherwise | — |
| `"optuna"` | one process, trials in sequence, resumable from its journal file | nothing extra |
| `"ray"` | trials in parallel across the CPUs, GPUs or cluster nodes Ray can see | `[ray]` extra |

**Status.** The Ray backend was verified on a two-GPU SLURM node on 2026-09-28,
all twelve checks of `tests/hpc_ray_smoke.py` passing: both families searched with
cross-validation, trials observed running concurrently in each, the run contract
intact, and a stacking base model searched on Ray. It has been exercised on one
cluster with one Ray version, so start with the smoke test below on yours — a Ray
misconfiguration usually shows up as trials that queue forever or run one at a
time, neither of which looks like an error. Optuna is what `"auto"` picks off a
CUDA machine and is the backend this package was developed on.

**It does not change what a trial means.** Both backends build the same search
space, score each configuration the same way — k-fold CV over the training split
by default, `Optimize(cv=False)` for a single holdout — and draw their folds from
`split_seed`. The files the run contract names are identical, down to
`hpo/runtime/hpo_runtime.json`; Ray additionally keeps its own experiment
directory under `output_dir/ray_experiment/`, which is where its trial state and
restore data live and which Optuna has no equivalent for. Ray changes *how many
trials run at once*, not what any one of them computes.

**When it is worth it.** Ray pays off when you have more than one GPU, or a
multi-node allocation. On one GPU or on Apple Silicon it does not: the Ray entry
points detect that and hand the work back to Optuna rather than pretending, since
several concurrent CPU trials on a laptop is a way to run out of RAM, not a
speed-up. That is why `"auto"` exists — leave it, and a config that runs on your
laptop also runs on the cluster.

**Differences that do exist, and are deliberate:**

- **Pruning.** `Optimize(pruning=...)` is Optuna's `MedianPruner`. Ray has no
  equivalent and ignores it with a warning; use
  `Optimize(extra={"use_asha": True})` there, which terminates unpromising trials
  by successive halving instead. Ray's CV path does no inter-fold pruning at all
  — reporting a result per fold means an API that hangs outside a Ray Train
  session, and a hang on a cluster costs the whole allocation.
- **Trial budget.** `n_trials=None` means the family default, and Ray's is higher
  when ASHA is on (it prunes many trials early, so more are needed to saturate
  the search). Pass an explicit `n_trials` when you want the two comparable.
- **Resume.** Optuna reattaches to a content-derived study name in its journal
  file; Ray restores a content-derived experiment directory. Changing the data
  or effective search configuration therefore starts a distinct state instead
  of merging incompatible trials. An explicitly supplied study/experiment name
  remains the caller's responsibility. Both follow the run's `resume=`. Ray's side
  needs a `RunConfig`, which also fixes where the experiment is written; if the
  log says `Could not create RunConfig`, resume is off for that run **and** Ray
  falls back to `~/ray_results`, which on a cluster with a small home quota fills
  up mid-search.

**Check the backend works on your cluster before trusting a long run.**
`tests/hpc_ray_smoke.py` runs the Ray path end to end on a GPU node — twelve
checks on synthetic data and a tiny model, ~15-30 min, or `--quick` for the
tabular checks alone. `tests/hpc_ray_smoke.sbatch` is a SLURM wrapper with the
cluster-specific lines marked `EDIT`. It is worth one submission before a real
search: a Ray misconfiguration usually shows up as trials that queue forever or
run one at a time, neither of which looks like an error.

**Cluster settings have no arguments of their own** — they go through `extra`,
which is passed to the underlying function unchanged:

```python
mv.run(task="text", ..., hyperparams=mv.Optimize(
    backend="ray",
    extra={"ray_address": "auto", "num_gpus_per_trial": 0.5,
           "max_concurrent_trials": 4, "use_asha": True},
))
```

An `extra` key the resolved backend does not accept raises, on purpose — a
mis-typed cluster setting should stop the run, not be ignored for six hours.

**Which backend ran?** The run log says so:
`Hyperparameters FROM SEARCH — … backend=ray, pruning=family default`, and the
HPO runtime report records it too. No result file is renamed after the engine, so
to keep two searches side by side, give them different `output_dir`s — the
directory name carries the distinction and the contents stay comparable
file-for-file.

## Which search space did my run actually use?

Precedence, lowest to highest: the balanced reference space, then the adaptive
adjustment for your training-set size and cause count, then LoRA or focal
additions if those are on, then `Optimize(space=...)`.

**A key you name is searched over exactly the range you wrote.** The adaptive
machinery runs on the defaults *before* your keys are applied, so nothing narrows,
rescales or re-profiles a range you set. **A key you do not name keeps its adaptive
value and is still searched** — a partial dict never silently shrinks the search.

The run log states both halves:

```
Tabular HPO adaptive search space: profile=wide, class_tier=few (n_samples=176, n_features=57, n_classes=11)
search_space= gave 2 key(s), used exactly as passed: learning_rate, n_estimators.
The other 7 key(s) come from the adaptive space for this data and are searched too:
colsample_bytree, max_depth, min_child_samples, num_leaves, reg_alpha, reg_lambda,
subsample. Name them in search_space= to set them yourself.
```

So the space that actually ran is yours where you spoke and adaptive where you did
not, and the log says which is which.

**To hold one hyperparameter fixed while searching the others, give it a
single-value range.** `hyperparams=` is one argument with one meaning at a time —
a dict of fixed values means no search at all — so a search that pins a value does
it inside the space:

```python
hyperparams=Optimize(space={
    "learning_rate": ("float_log", 1e-5, 1e-3),   # searched
    "batch_size":    ("categorical", [8]),        # pinned at 8
})
```

Every trial then gets `batch_size=8`. Omitting a key does something different: it
keeps that key's adaptive range and searches it.

**A malformed space is refused when you build the `Optimize`, not when the search
starts.** The searches run with per-trial errors caught so one bad configuration
cannot kill a long search, which means a bad *space* would otherwise fail every
trial quietly and reach the end of its budget with nothing completed — after the
queue wait, on a cluster. So `None` in place of a range, a NaN bound, an empty
choice list, a misspelled kind and a low above its high are all named up front:

```
ValueError: Optimize(space=...) is malformed:
  learning_rate: low=nan is not finite; a NaN bound reaches the sampler as an
  unreadable OverflowError
  batch_size: ('categorical', []) — needs at least one choice. A single choice
  pins the value, which is how a hyperparameter is held fixed while others are
  searched.
```

This matters most for data fusion, where `batch_size` and
`gradient_accumulation_steps` are calibrated so a long-context model fits on a GPU:
a partial space keeps them rather than dropping them, which is what stops a
two-key space from running out of memory.

The run also says where the hyperparameters came from at all:

```
Hyperparameters FROM SEARCH — metric=csmf_accuracy, n_trials=50, cv=True/3 folds, space overrides 1 key(s): learning_rate.
Hyperparameters FROM CALLER — 6 key(s): batch_size, epochs, ...
No hyperparameters given and no search requested — training lightgbm with its library defaults.
```

The last of those is a warning, because it is the case most easily reached by
accident.

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

## How do I choose a representative model without touching the test set?

Choosing the best text backbone, or the best fusion strategy, on **test**
scores leaks the test set into the choice, and the winner's test score is then
optimistic. Choose on validation scores instead, which every run records in
`validation.json`:

```python
from multimodalva.results import validation_leaderboard

board = validation_leaderboard({
    "biomedbert":   "runs/text/biomedbert",
    "clinicalbert": "runs/text/clinicalbert",
    "bluebert":     "runs/text/bluebert",
})
```

Pick the top row, then report that model's **test** score. Where each score
comes from is in the `source` column:

| `source` | The score is |
|---|---|
| `hpo_cv` | the best search trial's cross-validated score (`hyperparams=Optimize(...)`) |
| `early_stopping` | the best epoch on the early-stopping slice (text, fixed hyperparameters) |
| `automm_holdout` | AutoMM's best validation score on its holdout (feature fusion) |
| `meta_cv` | the chosen stacking meta-learner's cross-validated score |
| `none` | nothing — no validation data was used. Left missing, never filled from the test set |

Compare runs that share a `source`. A best-epoch score on one holdout runs
higher than a cross-validated search score for the same model, so a table
mixing them logs a warning. And none of these is a performance estimate to
report: each is the best of several trials, epochs or candidates.

Runs made before `validation.json` existed are read from the files they left
(`hpo_trials.csv`, the AutoMM training log, the training history), so the
leaderboard works on runs already on disk.

## Can I evaluate predictions from another tool (InSilicoVA, InterVA, a colleague's model)?

Yes. The `results` subpackage never asks where a prediction came from — a table
of predicted labels is enough, and more detail unlocks more metrics:

| What you can supply | What you get |
|---|---|
| **Top-1 label per death** (one column of cause names) | accuracy, balanced accuracy, macro/weighted F1, precision and recall, CSMF accuracy, chance-corrected CSMF accuracy; confusion matrix, cause-accuracy heatmap, CSMF scatter; bootstrap and paired-bootstrap confidence intervals |
| **+ a top-*k* table** | `top2_accuracy` … `top{k}_accuracy` columns, and the top-*k* accuracy curve |
| **+ the full probability matrix** | log loss, and calibration — expected and maximum calibration error, and the multi-class Brier score, each per cause |

Top-1 alone already covers what most comparisons report; you never need to
supply more than you have.

```python
from multimodalva.results import performance_leaderboard, bootstrap_ci

df = pd.DataFrame({
    "true_label":   truth,          # the gold-standard cause
    "mmva_text":    ours,           # a model trained here
    "insilicova_R": theirs,         # anything else — same rows, same order
})
performance_leaderboard(df, true_col="true_label",
                        model_cols=["mmva_text", "insilicova_R"])
bootstrap_ci(df, true_col="true_label", model_cols=["mmva_text", "insilicova_R"])
```

For probabilities, pass `prob_dfs={"insilicova_R": full}` together with
`id2label=`, where `full` has a `true_label` column plus `prob_0 … prob_{K-1}` —
one column per cause, **in the order of your `id2label`**. `topk_dfs=` adds the
top-*k* columns, and `topk_from_full()` builds that table from `full`. Cause
names have to match the gold standard's exactly: map them before passing them
in, because a name the label map does not know raises rather than scoring zero.

### What about InSilicoVA specifically?

Running it *from* this package is a future item, not a feature. `pyinsilicova` does
not install on Python 3.12+, which this package requires, and it cannot train a
probbase against your own cause list — it works with InSilicoVA's native causes
only. A stacking spec with `model_name="insilicova"` is rejected immediately;
there is no partial hook that can appear to work.

Run it where it runs — `pyinsilicova` if its native cause list fits your study,
otherwise the R implementation — and bring the assignments back through the
table above. That is the supported path, and it is how InSilicoVA and InterVA
should be compared against the models trained here.

## Which stacking combiner should I use?

Stacking combines its base models in Stage 2. `combiner=` picks how:

| `combiner` | What it learns |
|---|---|
| `meta_learner` (default) | a classifier over all base models' probabilities — a multinomial logistic regression unless you pass `meta_learners=` (see below) |
| `simple_average` | nothing — equal weights over the same base models |
| `class_aware_voting` | a model × cause weight matrix, shrunk toward equal weights for rare causes |
| `ensemble_selection` | one weight per model, by greedy selection with replacement (Caruana et al. 2004) |
| a meta-learner name, e.g. `lightgbm` | that one meta-learner, with no choosing; its hyperparameters come from the matching `meta_learners` entry |
| `best` | whichever of the above scores best on the out-of-fold rows — see below |

Stage 1 — the out-of-fold predictions — is the expensive part, and all of them
read the same one. So there is no need to choose before you run: pass a list,
and every combiner is computed from one Stage 1. The first one listed is the
main result in `predictions/`; each also writes its own folder:

```python
run(task="stacking", ..., combiner=["meta_learner", "class_aware_voting",
                                    "ensemble_selection", "simple_average"])
```

To add a combiner to a stacking run that has already finished, point
`oof_from=` at it. Its Stage 1 is reused and nothing is retrained — and because
the run it points at supplies the rows, the split and the seeds, this call needs
no `data`:

```python
run(task="stacking", output_dir="runs/stacking_cv",
    combiner="class_aware_voting", oof_from="runs/stacking/stacking")
```

You can also stop after Stage 1, which is the half that costs GPU hours, and
fit combiners later or elsewhere:

```python
# on the GPU box — trains every base model over every fold, then stops
run(task="stacking", data=df, label_col="cause", features=feats,
    tabular_models=specs, output_dir="runs/stacking", oof_only=True)

# afterwards, on a laptop — seconds each, as often as you like
run(task="stacking", output_dir="runs/meta", oof_from="runs/stacking/stacking",
    combiner=["meta_learner", "simple_average"])
```

On the command line: `--oof-only`, then `--oof-from <that run's output_dir>`.

Each combiner is fitted on the out-of-fold predictions only, so the test set is
untouched by all of them and their test scores can be compared directly — each
can be reported as a model in its own right.

**If you want just one model, and want it chosen without the test set**, use
`combiner="best"`. Every combiner is fitted on four fifths of the out-of-fold
rows and scored on the rest, on the same folds for all of them, and only the
winner is then trained and applied to the test set:

```python
out = run(task="stacking", ..., combiner="best")
out["combiner_chosen"]      # e.g. "ensemble_selection"
out["combiner_comparison"]  # the table, also in combiner_comparison/
```

Choosing the best of several by their *test* scores would make the reported
number optimistic; choosing here does not, because the choice never sees the
test set. The same table is available on its own as
`clf.compare_combiners_stage(n_combiner_folds=5)`. Note that scores computed on
the very rows a combiner was fitted on are in-sample and flatter the more
flexible combiners; this table avoids that. Each combiner is scored as it is
actually trained — the class voter including its fallback to equal weights, each
meta-learner candidate on its own row.

The meta-learner is chosen by cross-validation among fixed candidates and is not
tuned by a search: its input is only `n_models × n_classes` columns, where a few
cheap fixed candidates suffice. The chosen one, and every candidate's score, are
in `meta_learner/meta_learner_metadata.json` and in what `run()` returns. The
ensemble-selection weights for each base model are in
`ensemble_selection/ensemble_weights.json`.

## Which meta-learner does stacking use, and can I use my own?

Four layers, from "do nothing" to "bring your own model".

**1. Do nothing — one multinomial logistic regression.** `max_iter=1000, C=1.0`,
fitted on the out-of-fold probability matrix. It is still *scored* by CV, so
`meta_scores.json` records how well it did even though it had no rival, and
selection never touches the test set.

LR is the default deliberately. The meta-learner sees probabilities that already
came from strong models, so Stage 2 is a re-weighting job on roughly
`n_models × n_causes` features, and a linear model is the conservative choice at
the row counts VA studies have. **Whether to compare several candidates is a
decision about your study, so the package does not make it for you** — a run that
asks for nothing should not quietly do something wider than it was asked for.

**2. Choose from the ten built in.** Pass `meta_learners=` with the candidates you
want compared:

```python
mv.run(task="stacking", ..., meta_learners=[
    {"model_name": "logistic_regression",
     "hyperparams": {"max_iter": 2000, "C": 1.0, "solver": "lbfgs"}},
    {"model_name": "lightgbm",
     "hyperparams": {"n_estimators": 300, "learning_rate": 0.05,
                     "objective": "multiclass"}},
    {"model_name": "mlp",
     "hyperparams": {"hidden_layer_sizes": (64,), "max_iter": 500}},
])
```

From the command line, the same thing in a config file:

```yaml
task: stacking
meta_learners:
  - model_name: logistic_regression
    hyperparams: {max_iter: 2000, C: 1.0}
  - model_name: lightgbm
    hyperparams: {n_estimators: 300, objective: multiclass}
```

All ten are on the same footing — none is privileged beyond being the default:
`logistic_regression`, `lightgbm`, `xgboost`, `gbdt`, `catboost`,
`random_forest`, `mlp`, `svm`, `naive_bayes`, `knn`.

Each candidate is scored by CV on the OOF matrix and the winner is refit on all of
it; `meta_scores.json` records every score and `meta_learner_metadata.json` names
the winner. Comparing several instead of one is cheap — Stage 2 fits a combiner on
a saved matrix in seconds, where Stage 1 trained every base model over every fold.

One dict instead of a list is fine, and means "use this one, no comparison".
Names must be unique — candidates are reported by `model_name`. The meta-learner
is **not** hyperparameter-searched: candidates are fixed specs, which keeps Stage 2
cheap and its selection honest. Pass the settings you want.

**3. A model that is not one of the ten — fit it yourself on the OOF matrix.**
Naming one that stacking cannot build is an error, raised **before Stage 1 starts**
rather than after every base model has been trained, and it tells you this:

```
meta_learners names 'ridge_classifier', which stacking cannot build.
Accepted: logistic_regression, catboost, lightgbm, …
Stacking can only fit a meta-learner it knows how to build. For any other model,
compute stage 1 and fit the second stage yourself: …
```

Run Stage 1 alone, take the matrix, and do whatever you like with it:

```python
out = mv.run(task="stacking", data=df, label_col="cause", text_models=[...],
             tabular_models=[...], output_dir="runs/stack", oof_only=True)

X, y = out["oof_meta_X"], out["oof_y"]     # (n_train, n_models * n_causes), (n_train,)
my_model.fit(X, y)                          # anything with fit / predict_proba
```

`oof_only=True` stops after the out-of-fold predictions and returns
`oof_meta_X`, `oof_y`, `oof_dir` and `oof_metadata`; `predictions` is `None`.
`oof_metadata` points at `oof/oof_metadata.json`, whose `meta_feature_names` is
one name per column — `["tab_0_lightgbm_prob_0", …, "tab_1_random_forest_prob_10"]`
— so every column maps back to the model and cause it came from.
`model_sources` gives the same thing as blocks (`spec.model_name`, `col_start`,
`n_cols`, `final_dir`). This is the supported escape hatch, and it is the same one
that makes Stage 2 repeatable: `oof_from="runs/stack"` starts a later run from
that matrix without retraining anything.

Verified on the synthetic dataset with two tabular base models and 11 causes:
`oof_meta_X` came back `(28, 22)` — 28 training rows by 2 models × 11 causes —
and a `RidgeClassifier`, which is deliberately *not* one of the ten, fitted on it
directly.

To score your own model on the test set, get the base models' test
probabilities the same way Stage 3 does — `predict_ensemble_from_pretrained()` on
the finished run — and apply your fitted model to them.

**Three settings mention an out-of-fold matrix, and they do different things.**
They are easy to mix up, so side by side:

| Setting | Stage 1 (base models) | Stage 2 (combiner) | Use it when |
|---|---|---|---|
| `run(..., oof_only=True)` | trained, then **stop** | **yours**, outside the package | your combiner is not one of the ten — the escape hatch above |
| `run(..., oof_from=path)` | **not** trained — read from `path` | the package's, on that matrix | you want another combiner, or the same one again, without paying for stage 1 twice |
| `StackingClassifier(extend_oof_from=path)` | **only the models you list here** are trained; `path`'s columns are inherited | the package's, on `[inherited \| new]` | you want to **add base models** to a stacking run that is already finished |

`oof_only` and `oof_from` are the two halves of one split-it-up workflow — run
stage 1 on a GPU box, fit combiners later and repeatedly somewhere cheap.
`extend_oof_from` is the separate case of widening the matrix: it trains the new
base models, reuses the source run's train/test split so the rows still line up,
and `predict_test()` then loads inherited models from the source run and new ones
from this run's `final/`. It must be given the source run's `n_folds` and
`split_seed` — the inherited columns were computed over those folds and the new
ones are computed over yours, so a mismatch is refused with both values named
rather than combined into a matrix whose columns mean different things. Both are
recorded in the source run's `oof/oof_metadata.json`. `extend_oof_from` and
`n_folds` are constructor arguments, so through `run()` they travel in
`init_kwargs=`.

**4. Not a meta-learner at all.** `combiner=` also takes `simple_average`,
`class_aware_voting` and `ensemble_selection`, none of which fits a classifier
over the probabilities. See *Which stacking combiner should I use?* above; a list
runs several from one Stage 1.

Comparing candidates is your call, made by passing them — the default compares
nothing because which models are worth comparing depends on your data.

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

**`ImportError` for peft, ray or autogluon** — these live in optional extras:
`pip install "multimodalva[lora]"`, `[ray]`, `[feature_fusion]`, or `[all]`.
LightGBM, XGBoost and CatBoost are in the default install and need no extra.

**"backend='auto' resolved to Optuna because Ray is not installed"** — you are on
a machine with a CUDA GPU or a SLURM allocation, where Ray would have run the
search's trials in parallel, but the `[ray]` extra is not installed. The search
runs anyway, one trial at a time, and the results are identical — only slower.
Install `pip install "multimodalva[ray]"` to use it, or pass
`Optimize(backend="optuna")` to say you meant it and silence the warning.
Asking for `backend="ray"` explicitly without the extra raises instead, because
an explicit request deserves an explicit failure rather than a quiet
substitution.

**`TypeError` about `group_by_length` when training a text model** — you are on
transformers 5.x, which removed that `TrainingArguments` argument. The package
requires `transformers>=4.38,<5`, so a normal `pip install` downgrades for you;
this only appears if the constraint was bypassed (`--no-deps`, a module-provided
build, or a conda-managed transformers). Install a 4.x build in your environment.
Support for 5.x is planned for a future release.

**On macOS or Windows, a script runs itself again and again** — you set
`n_jobs` above 0 for a text model, so it starts data-loader worker processes,
and on macOS and Windows each new process re-runs your script from the top.
Put the script's work under a main guard:

```python
if __name__ == "__main__":
    run(task="text", ...)
```

Notebooks are not affected, and neither is the default (`n_jobs=None`), which
starts no workers on Apple Silicon.

**`RuntimeError` about Metal/MPS operations with Longformer** — set
`PYTORCH_ENABLE_MPS_FALLBACK=1`, or switch to BigBird, or run on CPU.

**All HPO trials failed** — the package raises with an actionable message rather
than a cryptic Optuna error. The usual causes are GPU out-of-memory (lower
`batch_size`, or enable `gradient_checkpointing=True`) and file-descriptor limits
on macOS.

**HPO is slow to write on a cloud-synced folder** (Dropbox, iCloud, OneDrive) —
pass `storage_path="/tmp/hpo_<name>.log"` to keep the study log on local disk.

**InSilicoVA** — not runnable from this package (a v1 item), and `pyinsilicova`
does not install on Python 3.12+. Run it separately and evaluate its output
here: [Can I evaluate predictions from another tool?](#can-i-evaluate-predictions-from-another-tool-insilicova-interva-a-colleagues-model)
