# MultimodalVA — FAQ

Practical questions, in the order you are likely to hit them.

**Getting started**

- [I have never used this before — where do I start?](#i-have-never-used-this-before--where-do-i-start)
- [What data does the package expect?](#what-data-does-the-package-expect)
- [How are my indicator columns preprocessed?](#how-are-my-indicator-columns-preprocessed)
- [YAML or JSON for config files?](#yaml-or-json-for-config-files)

**Running a job**

- [What computing environment do I need?](#what-computing-environment-do-i-need)
- [What SLURM resources should I request?](#what-slurm-resources-should-i-request)
- [How long does a run take?](#how-long-does-a-run-take)
- [How much disk space does a run use?](#how-much-disk-space-does-a-run-use)
- [Can I check a config before I submit the job?](#can-i-check-a-config-before-i-submit-the-job)
- [Can I stop a run and resume it later?](#can-i-stop-a-run-and-resume-it-later)
- [Can I see how long it took and how much GPU it used?](#can-i-see-how-long-it-took-and-how-much-gpu-it-used)
- [How do I make a run reproducible?](#how-do-i-make-a-run-reproducible)

**Choosing and tuning models**

- [Which model should I pick?](#which-model-should-i-pick)
- [How do I set the hyperparameter search space?](#how-do-i-set-the-hyperparameter-search-space)
- [Optuna or Ray — which search backend, and does it change my results?](#optuna-or-ray--which-search-backend-and-does-it-change-my-results)
- [Some of my causes are very rare — what does the package do?](#some-of-my-causes-are-very-rare--what-does-the-package-do)
- [How does stacking combine its base models, and can I change it?](#how-does-stacking-combine-its-base-models-and-can-i-change-it)
- [How do I choose a representative model without touching the test set?](#how-do-i-choose-a-representative-model-without-touching-the-test-set)

**Using a trained model**

- [I have no labelled data — can I just use a trained model to predict?](#i-have-no-labelled-data--can-i-just-use-a-trained-model-to-predict)
- [How do I publish and re-use a trained model?](#how-do-i-publish-and-re-use-a-trained-model)

**Evaluation**

- [Can I evaluate predictions from another tool (InSilicoVA, InterVA, a colleague's model)?](#can-i-evaluate-predictions-from-another-tool-insilicova-interva-a-colleagues-model)
- [Why is there no confidence interval for ECE and MCE?](#why-is-there-no-confidence-interval-for-ece-and-mce)

**When something goes wrong**

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

## How much disk space does a run use?

Measured on the synthetic dataset (330 records, 11 causes):

| Run | Size | What dominates |
|---|---|---|
| `tabular`, no search | 2.8 MB | the fitted model |
| `tabular` + search (3-fold CV) | 23 MB | the best trial's fold models |
| `text`, with or without search | 419 MB | `model.safetensors` |
| `data_fusion` | 419 MB | one long-context model |
| `voting`, 3 tabular models | 8.7 MB | three fitted models |
| `stacking`, 1 text + 2 tabular | 426 MB | the text base model |

**Planning rule: count the models you train and multiply.** One text model is
~0.4 GB (BERT-base) or ~0.6 GB (Longformer, BigBird); a tabular model is
single-digit MB. Everything else — predictions, search records, metadata — is
usually under 1 % of the total.

Predictions scale with deaths × causes: the full-probability CSV is roughly
`n_test × n_causes × 20 bytes` (50,000 deaths × 30 causes ≈ 30 MB).

A search keeps the best trial's fold models and deletes the rest as it goes, so
peak usage during a run is higher than the total afterwards — for a text search,
budget for two models rather than one.

## Can I check a config before I submit the job?

`--dry-run` runs everything that happens before a model is built — argument
checks, loading, filtering, the missing-row drop, the split, feature resolution —
reports what the run would do, and trains nothing. It exits non-zero if anything
would stop the run, so a job script can gate on it:

```bash
multimodalva run job.yaml --dry-run || exit 1
multimodalva run job.yaml
```

One report names every problem it finds, not just the first. A config file is
where it earns its keep: the command line rejects an unknown flag, but a config
file's keys are passed straight through, so `max_lenght: 512` in YAML is only
caught once the pipeline is reached. The check compares each key against the
arguments the pipeline actually accepts, and does the same for `init_kwargs`,
model names and base-model specs.

It also reports three things a real run would accept in silence: a class too rare
to appear in every search fold, `init_kwargs` overriding an argument passed
beside it, and a filter value that matches no row.

**A clean dry run means the configuration is consistent with the data, not that
the run will succeed.** Nothing trains, so it cannot see a batch size that will
not fit, a trial budget too small for the search space, or whether a Hugging Face
Hub ID exists.

From Python the same check is `preflight(**config)`, which takes what `run()`
takes and returns a report instead of printing one — `report["ok"]`,
`report["problems"]`, `report["notes"]`, `report["facts"]`.

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

## Can I see how long it took and how much GPU it used?

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

## How do I make a run reproducible?

Two seeds, and they do different things:

| Seed | Controls | Vary it to measure |
|---|---|---|
| `split_seed=` | every row-partitioning decision — the train/test split, each search's CV folds, the early-stopping slice, stacking's out-of-fold partition | sampling uncertainty |
| `train_seed=` | everything a model does with the rows it is given — initialisation, dropout, batch order, the Optuna sampler, each estimator's `random_state`, AutoMM's trainer | model stochasticity |

Both default to 42. Vary one at a time, with the other and `hyperparams` fixed,
or the two sources of variation are confounded.

**What the seeds do not fix.** Predicted labels reproduce exactly; probabilities
can differ in the last decimals (about 1e-16 on CPU, up to 1e-7 on an
Ampere-or-newer GPU), and inference `batch_size` shifts them at the same scale
because each batch is padded to its own longest sequence. Nothing makes results
bit-identical across GPU models, driver versions or library versions.

**If you need exact reproduction:** pass `deterministic=True` (slower, and it
raises on any operation with no deterministic kernel), keep `batch_size` the
same, and save the environment next to the run —
`pip freeze > runs/text/environment.txt`. Seeds fix what this package controls,
not what a dependency changes between its own releases.

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

## How do I set the hyperparameter search space?

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

## Optuna or Ray — which search backend, and does it change my results?

`Optimize(backend=...)`:

| `backend` | What it does | Needs |
|---|---|---|
| `"auto"` (default) | Ray when a CUDA GPU or a SLURM allocation is visible **and** Ray is installed; Optuna otherwise | — |
| `"optuna"` | one process, trials in sequence, resumable from its journal file | nothing extra |
| `"ray"` | trials in parallel across the CPUs, GPUs or cluster nodes Ray can see | `[ray]` extra |

**It does not change what a trial means.** Same search space, same metric, same
folds, same seeds. Ray runs trials at the same time; it does not change what each
one computes. Trial *order* differs, so with a small budget the two can land on
different hyperparameters — both are valid draws from the same space.

Ray pays off with more than one GPU or a multi-node allocation. On a laptop it
only adds overhead, which is why `"auto"` picks Optuna there.

Cluster settings go through `extra`, unchanged:

```python
mv.run(task="text", ..., hyperparams=mv.Optimize(
    backend="ray",
    extra={"ray_address": "auto", "num_gpus_per_trial": 0.5,
           "max_concurrent_trials": 4, "use_asha": True},
))
```

An `extra` key the backend does not accept raises — a mis-typed cluster setting
should stop the run, not be ignored for six hours.

The run log says which backend ran, and the HPO runtime report records it. No
file is renamed after the engine, so keep two searches in different
`output_dir`s if you want them side by side.

Run `tests/hpc_ray_smoke.py` on your cluster before a long Ray search: a
misconfiguration usually looks like trials queueing forever or running one at a
time, neither of which raises.

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

## How does stacking combine its base models, and can I change it?

Stage 1 trains the base models out of fold; Stage 2 combines them. `combiner=`
picks how:

| `combiner` | What it learns |
|---|---|
| `meta_learner` (default) | a classifier over all base models' probabilities — multinomial logistic regression unless you pass `meta_learners=` |
| `simple_average` | nothing — equal weights |
| `class_aware_voting` | a model × cause weight matrix, shrunk toward equal weights for rare causes |
| `ensemble_selection` | one weight per model, greedy selection with replacement (Caruana et al. 2004) |
| a meta-learner name, e.g. `lightgbm` | that one, with no choosing |
| `best` | whichever scores best on held-out out-of-fold rows |

**You do not have to choose before running.** Stage 1 is the expensive part and
every combiner reads the same one, so pass a list and get them all. The first is
the main result in `predictions/`; each also writes its own folder.

```python
run(task="stacking", ..., combiner=["meta_learner", "class_aware_voting",
                                    "ensemble_selection", "simple_average"])
```

All of them are fitted on out-of-fold predictions only, so the test set is
untouched and their test scores are directly comparable — each can be reported
as a model in its own right.

**If you need exactly one, chosen without the test set:** `combiner="best"`
fits every combiner on four fifths of the out-of-fold rows, scores it on the
rest, and applies only the winner to the test set. `out["combiner_chosen"]` and
`out["combiner_comparison"]` record what happened. Picking a winner by *test*
score would make the reported number optimistic; this does not.

### Changing the meta-learner

The default is one multinomial logistic regression (`max_iter=1000, C=1.0`).
Pass candidates to compare them — each is scored by CV on the out-of-fold matrix
and the winner refit on all of it:

```python
mv.run(task="stacking", ..., init_kwargs={"meta_learners": [
    {"model_name": "logistic_regression"},
    {"model_name": "lightgbm", "hyperparams": {"n_estimators": 300}},
]})
```

`logistic_regression`, `lightgbm`, `xgboost`, `gbdt`, `catboost`,
`random_forest`, `mlp`, `svm`, `naive_bayes`, `knn`. One dict instead of a list
means "use this one". A name outside the ten raises **before Stage 1 starts**,
not after the GPU hours. The chosen candidate and every score are in
`meta_learner/meta_learner_metadata.json`.

Meta-learners are chosen by cross-validation among fixed candidates, not tuned
by a search — the input is only `n_models × n_classes` columns.

### Something else entirely, or Stage 2 later

Stop after Stage 1 and fit what you like:

```python
out = mv.run(task="stacking", ..., oof_only=True)
X, y = out["oof_meta_X"], out["oof_y"]      # (n_train, n_models * n_causes)
```

Or reuse a finished Stage 1 — it supplies the rows, split and seeds, so this
call needs no `data`:

```python
run(task="stacking", output_dir="runs/meta", oof_from="runs/stacking/stacking",
    combiner=["meta_learner", "simple_average"])
```

On the command line: `--oof-only`, then `--oof-from <that run's output_dir>`.

**Three settings mention an out-of-fold matrix and are easy to mix up:**

| Setting | Stage 1 | Stage 2 | Use it when |
|---|---|---|---|
| `oof_only=True` | trained, then **stop** | **yours**, outside the package | your combiner is not one of the ten |
| `oof_from=path` | **not** trained — read from `path` | the package's | you want another combiner without paying for Stage 1 twice |
| `extend_oof_from=path` | **only** the models you list are trained | the package's, on `[inherited \| new]` | you want to **add** base models to a finished run |

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

## I have no labelled data — can I just use a trained model to predict?

Yes — that is what `predict_from_pretrained()` is for. Pass a label column only
if you have one and want performance reported too; leave it out and you get
predictions. Runnable code for every pipeline:
[`examples/08_predict_with_a_trained_model.py`](examples/08_predict_with_a_trained_model.py).

### What the run directory must contain

Point at the run's `output_dir` (or its `final/`); the rest is found from there.

| Pipeline | Entry point | Must be present | You must pass |
|---|---|---|---|
| `text`, `data_fusion` | `predict_from_pretrained()` | `final/config.json` + weights + tokenizer, `id2label.json` | `text_col=` |
| `tabular` | `predict_from_pretrained()` | `final/model.joblib`, `id2label.json`, `feature_names.json` | `feature_cols=` only if the model was fitted on a bare array |
| `feature_fusion` | `predict_from_pretrained()` | `final/automm_model/` | `text_col=` and `feature_cols=` |
| `voting` | `predict_ensemble_from_pretrained()` | `base_models/*/final/` for each base model | whatever the bases need |
| `stacking` | `predict_ensemble_from_pretrained()` | `final/<base>/` for each base, plus the fitted combiner (`meta_learner/`, `class_voter/`, `ensemble_selection/`) | whatever the bases need |

`training_metadata.json` records which pipeline trained the run, so the right
path is chosen for you. Without it the kind is guessed from the files present
(`config.json` → text, `model.joblib` → tabular, `automm_model/` → feature
fusion) and the guess is printed — pass `task=` if it is wrong. See
[Common errors](#common-errors) if it cannot tell.

Models trained elsewhere load too: a Hugging Face sequence classifier (if its
`id2label` is real, not `LABEL_0`), and sklearn / CatBoost / LightGBM / XGBoost
fitted on a **DataFrame**. One fitted on a bare NumPy array records no column
names, so pass `feature_cols=[...]`. Columns are matched **by name**, never by
position.

Predictions come back in the label vocabulary of the artifact you loaded.
Whether a cause label from another site, instrument or coding round means the
same as yours is a judgement only you can make.

### Combining runs you already finished

```python
from multimodalva.utils.predictions import load_predictions
from multimodalva.ensemble import vote_from_results

bert, lgbm = load_predictions("runs/bert"), load_predictions("runs/lgbm")
voted = vote_from_results([bert, lgbm], id2label=bert.id2label)
```

They must share one train/test split. With `id_col` the ids are checked row for
row; without ids, matching the split is up to you.

A vote produced this way **cannot be re-scored later** — the output *is* the
vote, and no model is stored. `run(task="voting")` and `run(task="stacking")`
both can, because they keep their base models.

## How do I publish and re-use a trained model?

Text and data-fusion runs are saved in standard Hugging Face format — cause
labels in `config.json`, LoRA adapters merged into the base weights — so they
reload with plain `transformers`. Tabular, voting and stacking runs are not
Hub-native; share those as a run directory.

Publish during a run, or afterwards from its `final/` folder:

```python
run(task="text", ..., push_to_hub=True, hub_repo_id="your-org/va-bert-cod")

from multimodalva.utils import push_to_hub
push_to_hub("runs/text/final", "your-org/va-bert-cod", private=True)
```

A model card with the test metrics, cause list and usage snippets is generated
for you. Pass `model_kind="data_fusion"` for a fusion model, so the card records
the fused-input format.

Re-use it as an ordinary Hugging Face model, or keep fine-tuning:

```python
from multimodalva.text import TextClassifier

clf = TextClassifier(model_name="your-org/va-bert-cod", output_dir="runs/tuned")
clf.run(df=my_df, text_col="narrative", label_col="cause",
        hyperparams={"epochs": 3, "ignore_mismatched_sizes": True})
```

`ignore_mismatched_sizes=True` starts a fresh classification head when your
cause set differs from the published model's.

Runnable:
[`examples/07_publish_to_hub.py`](examples/07_publish_to_hub.py) checks what
would be uploaded before it uploads anything;
[`examples/08_predict_with_a_trained_model.py`](examples/08_predict_with_a_trained_model.py)
covers loading runs back.

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

**"Cannot determine the pipeline" when predicting** — the run has no
`training_metadata.json` and none of `config.json`, `model.joblib` or
`automm_model/` is present, so there is nothing to read the kind from. Pass
`task=` (`--task`) yourself.

**"the run's own metadata says X, you asked for Y"** — the artifact records which
pipeline trained it and you asked for a different one. Nothing is scored, because
running the wrong pipeline produces confident nonsense rather than an error. Add
`--force` if you are certain the recorded value is wrong.

**"several runs under this directory"** — you pointed at a parent holding more
than one run. Point at one of the subdirectories it names.

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
