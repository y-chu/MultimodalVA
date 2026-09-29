# Changelog

All notable changes to MultimodalVA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

Versions follow [semantic versioning](https://semver.org): backwards-compatible
features land in `0.x` minor releases, and `1.0.0` is reserved for the point at
which the public API is declared stable — not for any single feature.

**Next (0.2.0)**

- A converter that builds data-fusion long text from a DataFrame plus a
  standardised instrument (e.g. a WHO 2016 questionnaire export), so data-fusion
  prediction needs no hand preparation. Today the fused text has to be built
  before calling in, and `predict_from_pretrained()` expects that column ready.

**Later, no release assigned**

- Image as a third modality. The AutoMM branches for it already exist.
- Log-probability inputs for the stacking meta-learner. A linear meta-learner
  combining log-probabilities pools the base models multiplicatively, which is
  the more natural way to combine probabilities; today it receives raw
  probabilities. Not the default until it has been compared on real
  out-of-fold data, since changing it would change published stacking results.
- InSilicoVA as a stacking base learner. It is not exposed in this release:
  `pyinsilicova` does not install on Python 3.12+, and it cannot train a
  probbase against a study's own cause list, only its native causes. Until then,
  run InSilicoVA separately (R, or `pyinsilicova` with its native causes) and
  evaluate its assignments alongside the rest — `results` accepts predictions
  from any source.
- `logistic_regression` as a tabular base model. It is available as a stacking
  meta-learner but not as a base model, where it performed poorly in testing.
- Publishing feature-fusion (AutoMM) and ensemble models to the Hugging Face Hub.
  A trained AutoMM model can already be shared as a directory; ensembles need a
  card that points at each base model's repository.

## [0.1.0] — 2026-09-28

First public release.

### Added

**Pipelines.** One entry point, `run(task=...)`, covers six tasks that share the
same `split → prepare → [optimise] → train → predict` shape:

- `text` — fine-tunes a BERT-family model on the narrative. Twelve backbones,
  including BioClinicalBERT, BlueBERT, BiomedBERT, Clinical-Longformer and
  Clinical-BigBird.
- `tabular` — nine scikit-learn / gradient-boosting models on the questionnaire
  indicators.
- `data_fusion` — converts the indicators to text, appends the narrative and
  fine-tunes a long-context model.
- `feature_fusion` — joint text and tabular training through AutoGluon AutoMM,
  with six named fusion strategies.
- `voting` — probability averaging over trained base models (soft voting).
- `stacking` — k-fold out-of-fold predictions, then a combiner (`combiner=`): a
  meta-learner (default), a simple average, a class-aware voter, ensemble
  selection (Caruana et al. 2004), or one named meta-learner — several of them
  from one set of out-of-fold predictions, or `"best"` to have the best one
  chosen on the out-of-fold rows alone. `oof_from=` reuses a finished run's
  stage 1 without retraining anything.

Every task is also available as a classifier object, as a CLI subcommand, and
from a YAML config.

**Hyperparameter search.** One argument says where a model's hyperparameters
come from, and nothing else does:

```python
hyperparams={"learning_rate": 3e-5}   # use exactly these
hyperparams="default"                 # the model library's own, no search
hyperparams=Optimize(n_trials=50, metric="csmf_accuracy")   # search for them
```

`Optimize` carries the search settings with it — the metric, the trial count, a
per-key override of the adaptive space, the cross-validation settings — so they
travel as one value instead of as loose arguments that mean nothing unless a
flag is set. An ensemble base-model spec therefore holds one `hyperparams` key
rather than six, and a pipeline forwarding base-model settings passes one object
instead of re-exporting each knob by hand.

Optuna and Ray Tune backends sit behind that one interface. Search spaces adapt
to the data: text by training-set size and class count, tabular additionally by
feature count. `Optimize(space=...)` overrides the adaptive default per key, and
the run log records which keys it replaced. Studies persist to an append-only
journal, so an interrupted search resumes. Trials keep no model weights, except
that a tabular search keeps the best trial's per-fold models.

The run log always says where the hyperparameters came from — the caller, a
search that just ran, or nothing at all, in which case the model library's own
defaults apply and a warning says so.

Command lines and config files are flat and cannot carry an object, so they
spell the same thing their own way: `--optimize --n-trials 50` on the CLI, and
in a config file

```yaml
hyperparams:
  optimize:
    metric: csmf_accuracy
    n_trials: 50
```

**Comparing models.** `predictions_frame()` gathers any number of finished runs
into one table — from `run()` results, `PredictionResult` objects, `top1`
DataFrames, or just the paths of run directories, in any mixture — and refuses
to build the table if two models were scored on different test sets.
`performance_leaderboard()`, `bootstrap_ci()` and `paired_bootstrap_ci()` read
that table, so a leaderboard can be rebuilt from disk long after training,
without reloading a model. `voting` and `stacking` train every model listed in
one call and share a single split between them, saving each base model
separately under `base_models/`.

**Choosing between combiners and models without touching the test set.**
`compare_combiners_stage()` scores simple averaging, the class-aware voter,
ensemble selection and every meta-learner candidate by nested cross-validation
over the out-of-fold rows, on the same folds for all of them, each as its
training stage fits it; `combiner="best"` uses it to pick one and trains only
the winner. `cross_validate_combiner()` does the same for any object with
`fit` / `predict`. `validation_leaderboard()`
ranks finished runs on their validation scores — the best search trial's
cross-validated score, the early-stopping slice, AutoMM's own holdout, or the
stacking meta-learner's cross-validated score — so a representative model can be
chosen without looking at test results. Runs without validation data are left
missing rather than filled from the test set.

**One contract for every run.** Whichever pipeline made it, a run's `output_dir`
holds `predictions/predictions_{top1,full,topk}.csv`, `training_metadata.json`
(which names the pipeline and package version), `validation.json`, and
`runtime/` timing and accelerator reports, plus `hpo/best_hyperparams.json` when
it searched. `run()` returns the same core keys for all six: `predictions`,
`output_dir`, `label2id`, `id2label`, `train_metadata`, `best_hyperparams`,
`validation`, `runtime_report`, `runtime_stage_csv`. Every task takes
`split_seed`, `train_seed`, `deterministic`, `resume` and `id_col`.

**Reusing finished runs.** `load_predictions()` reads any run's saved
predictions back into a `PredictionResult`, and `vote_from_results()` — now
exported from `multimodalva.ensemble` — soft-votes over them, so runs finished
separately can be combined without retraining. An interrupted `voting` run
reuses each base model that already finished. Every task accepts `id_col`, which
puts a leading `id` column on every prediction table so results join back to the
source records and match across pipelines by id.

**Sharing.** A trained text model can be pushed to the Hugging Face Hub in
standard format, with a generated model card, and reloaded for classification or
further fine-tuning.

**Reproducibility.** Two seeds, one per source of randomness. `split_seed`
decides which rows go where — the train/test split, each search's
cross-validation folds, the early-stopping slice and stacking's out-of-fold
partition — so varying it measures sampling uncertainty. `train_seed` drives
everything a model does with its rows — initialisation, dropout, batch order,
the Optuna sampler, each estimator's `random_state` and AutoMM's trainer — so
varying it with fixed `hyperparams` measures model stochasticity. Folds follow
`split_seed` so that reseeding a model never changes which hyperparameters the
search picks. Stacking's later stages inherit stage 1's seeds from
`oof/oof_metadata.json`. `deterministic=True` makes probabilities bit-for-bit
repeatable at a cost in speed and robustness. A split can instead be fixed by
column, by an explicit train/test id mapping, or by passing the two frames
directly. Runtime and GPU use are recorded per stage.

**Small samples are called out, not silently handled.** After a train/test split,
a run warns for every cause left with fewer than five rows on either side, naming
the causes and their counts, and reports a cause the split left out entirely as
zero. The same warning covers the training rows left after a validation split.
Nothing is dropped, merged or resampled on the package's initiative — what to do
with a rare cause is the analyst's decision. A validation slice thinner than the
number of causes is enlarged to one row per cause instead of failing, and a cause
with a single row falls back to an unstratified split; both are logged. Before
this, a search on very small data could fail every trial with an error that
blamed the model, not the split.

**Landing late in the 0.1.0 cycle**, after the pipelines above were in use:

- `predict_from_pretrained()` — load a trained model from a local run, a Hugging
  Face Hub repo id or a Hub URL, and predict on new data, with or without a
  label column. Serves text, data fusion, tabular and feature fusion, including
  compatible models trained elsewhere. Returns the usual `PredictionResult`,
  which gains a `checks` field (`None` from the training pipelines) reporting
  what was detected about the model and the input, and every warning raised.
- Stacking's two stages can be run separately: `oof_only=True` stops after the
  out-of-fold predictions, and `oof_from=` starts from a finished run's stage 1
  (that call now needs no `data`, since the run it points at supplies the rows,
  the split and the seeds). Stage 1 costs GPU hours and stage 2 costs seconds, so
  combiners can be fitted, compared and refitted without retraining base models.
  `--oof-only` / `--oof-from` on the command line.
- `predict_ensemble_from_pretrained()` — the same for a trained voting or
  stacking run: every base model is scored and combined the way that run
  combined them (soft-voting weights; for stacking the fitted meta-learner,
  simple average, class-aware voter or ensemble-selection weights, chosen with
  `combiner=`, defaulting to the combiner the run delivered).

- `multimodalva predict` — the command-line form of both, with the kind of run
  read from the artifact so one command serves single models and ensembles.
  `--dry-run` reports what would happen (kind of run, combiner, columns present
  and missing) without scoring anything, and `--task` overrides detection:
  contradicting a pipeline the run itself recorded is an error rather than an
  hour of inference down the wrong path, unless `--force` says otherwise.

- `preflight()` and `multimodalva run --dry-run` — check a configuration against
  the data without training anything. Every step that happens before a model is
  built runs (argument policing, loading, filtering, the missing-row drop, the
  train/test split, feature-column resolution), then it stops and reports what
  the run would do. Keywords `run()` does not name are checked against the
  signature of the class the task dispatches to, so a misspelt one is named with
  a suggestion instead of reaching the cluster; so are model names, base-model
  specs and `init_kwargs` keys. One report names every problem rather than only
  the first, and the exit status is 2 when any of them would stop the run, so a
  job script can gate on it:

  ```bash
  multimodalva run job.yaml --dry-run || exit 1
  multimodalva run job.yaml
  ```

  Three checks report what the run would otherwise accept in silence: a class too
  rare to appear in every search fold (now a hard error before training),
  `init_kwargs` overriding an argument passed alongside it, and a filter
  value that matches no row — which otherwise surfaces as "no rows remain",
  naming the step that noticed rather than the value responsible. A missing
  `text_col` is also caught here: the missing-row drop only looks at that column
  when it is present, so an absent one otherwise reaches the pipeline first.

  A clean dry run says the configuration is consistent with the data, not that
  the run will succeed: nothing trains, so it cannot see a batch size that will
  not fit or a trial budget too small for the search space.

- `Optimize(backend=...)` — choose the search engine from one call:
  `"ray"` runs trials in parallel across the CPUs, GPUs or cluster nodes Ray can
  see, `"optuna"` runs them in one resumable process, and `"auto"` (the default)
  picks Ray on a CUDA or SLURM machine and Optuna otherwise. `--hpo-backend` on
  the command line, and `backend:` inside a config file's `hyperparams.optimize:`
  block.

  The Ray backends existed and worked but nothing public reached them: every one
  of the seven search call sites named the Optuna function directly, so an HPC
  user had to drop to `optimize_text_ray()` / `optimize_tabular_ray()` and rebuild
  the training and prediction steps by hand — and **ensembles could not use Ray at
  all**, because voting's and stacking's base-model searches are internal. Both
  families now dispatch through one function (`search_text`, `search_tabular`),
  which is also where `Optimize` is mapped onto either backend, so a base-model
  spec can carry `hyperparams=Optimize(backend="ray")` like any other setting.
  Ray's cluster settings need no new fields: `Optimize(extra=...)` already reaches
  the underlying function verbatim.
- `optimize_text_ray()` scores trials by k-fold CV, with `use_cv=True` /
  `n_cv_folds=3` as on `optimize_text()` and `optimize_tabular_ray()`. Folds are
  pre-built once and shared by every trial and worker, so fold-assignment variance
  cannot be mistaken for a hyperparameter effect, and per-fold scores plus the
  across-fold standard deviation are recorded per trial as the Optuna backend
  records them. There is no inter-fold pruning on the Ray path, matching
  `optimize_tabular_ray()`: reporting an intermediate result per fold means
  `ray.train.report()`, which raises or hangs outside a Ray Train session, and a
  hang on a cluster costs the whole allocation rather than one trial. Use
  `extra={"use_asha": True}` for early termination across trials.
- `optimize_tabular_ray()` takes `split_seed=`, as `optimize_tabular()` does.
  Without it, switching backend silently redrew the CV folds of an otherwise
  identical search.
- `FAQ.md` documents the four ways to pick stacking's second stage: the default
  logistic regression; any of the ten built-in names via `meta_learners=`, which
  can be compared in one run; a model outside the ten by
  running `oof_only=True` and fitting it yourself on the returned `oof_meta_X` /
  `oof_y`; or no meta-learner at all via `combiner=`. The `oof_only` route was
  verified end to end with a `RidgeClassifier`, which is deliberately not one of
  the ten.
- `tests/hpc_ray_smoke.py` and `tests/hpc_ray_smoke.sbatch` — a way to verify the
  Ray backend on a real GPU node, which nothing else can do: without a CUDA GPU both
  Ray entry points redirect to Optuna, so the ordinary test suite exercises the
  redirect rather than Ray. Twelve checks on synthetic data and a tiny model.
- A `[ray]` extra: `pip install "multimodalva[ray]"`. See **Changed** — Ray moved
  out of the core dependencies, so this is now how you get the parallel backend.

### Fixed

Recorded because copies of the package were in use before this release; none of
these ever reached a tagged version.

- **`resume="adopt"` for artifacts made before resume manifests existed.** The
  manifest guard below refuses a directory that holds artifacts but no
  `resume_manifest.json`, because there is no signature to check — which would
  otherwise strand every run directory created before it. `resume="adopt"`
  (`--resume-adopt`) reuses such a directory and records the signature of the
  current call, so later runs are checked normally. It warns each time, since
  nothing verified those artifacts came from this data and configuration. It
  covers only the missing-signature case: a manifest that is present and
  disagrees still raises, because a disagreement is evidence of a real
  difference rather than an absent check. `--dry-run` reports which of the three
  states a directory is in before the job starts.

- **`Optimize(cv_folds="auto")` fits the fold count to the data.** An integer
  `cv_folds` raises when the rarest class in the training rows has fewer rows
  than there are folds, since a stratified fold cannot then hold every class.
  `"auto"` instead uses as many folds as the data supports, up to the default 3,
  and logs which it chose and why. It still raises for a class with a single row,
  where no fold count is valid. The resolved count — never the word `"auto"` — is
  what feeds the search-name signature, the run's reported configuration and the
  per-fold trial columns, so an artifact always says how many folds ran. All four
  search paths share one resolver, so the searches and `--dry-run` cannot
  disagree about what a dataset supports.

- **Resume can no longer mix incompatible runs.** Text, tabular, data fusion,
  feature fusion, voting and stacking write a content manifest for the consumed
  split and effective configuration before resumable work starts. Existing
  artifacts without a matching manifest are refused instead of being inferred
  compatible from filenames. Default Optuna study names and Ray experiment
  directories also include a data/configuration signature, so a shared journal
  or output directory cannot merge trials from a different search. Stacking
  additionally validates cached OOF shapes, labels, finiteness, probability
  ranges and row sums.
- Ray trial exceptions now propagate to Ray as failed trials. They were caught
  inside the worker and reported as zero/empty scores, so Ray could count a
  crashed configuration as a successful result. The GPU smoke contract now
  directly tests this failure path.
- **Text-only stacking now works.** Its OOF splitter was passed an empty label
  array unless a tabular base model had also prepared `y_train`. It now takes
  labels from the text dataset. The class-aware voter's implementation default
  is Brier score, matching its documented and recorded design.
- Feature fusion enforces the selected modalities. `text_only` no longer asks
  for tabular features; `tabular_only` no longer asks for text, resolves a text
  checkpoint, downloads NLTK data or accepts a text checkpoint setting it will
  not use. NLTK download failures now raise actionable errors and never disable
  TLS verification process-wide.
- Preflight performs the real split and checks HPO folds on the actual training
  rows. It rejects excess folds, one-model voting, unknown base-model spec keys,
  constructor-only settings passed to `.run()`, and modality mismatches. Flat
  single-model `init_kwargs` are now applied instead of being accepted and
  ignored. `Optimize.metric` is no longer silently repurposed as AutoMM's
  `eval_metric`; feature fusion names which `Optimize` fields AutoMM ignores.
- Row identity and imported-result alignment are strict. ID-based splits reject
  overlaps, absent requested ids, missing ids and duplicate ids. Voting over
  finished results verifies label maps, truth rows and ids before averaging.
  Malformed training metadata and unknown tabular inference backend settings
  raise rather than being ignored.
- The unfinished InSilicoVA stacking path was removed and its model name is
  rejected immediately. The supported path remains running it externally and
  comparing its predictions through `results`.
- Tabular single-holdout HPO now uses the same resilient split helper as text.
  Text Optuna CV now rejects more folds than the rarest class instead of letting
  sklearn run folds with missing classes.
- Bootstrap confidence intervals generate resample indices in bounded batches
  instead of allocating an `n_boot × n_rows` matrix (about 8 GB for 10,000 ×
  100,000). Fused-text generation avoids process startup on small automatic
  jobs and serializes plain row records for larger jobs; worker/data exceptions
  now surface once instead of being swallowed and recomputed sequentially.
- A meta-learner name the package cannot build is refused **before stage 1**,
  with an error naming the ten it accepts and pointing at `oof_only=True` for
  anything else. Every candidate is instantiated in stage 2, so an unknown name
  used to surface only after every base model had been trained over every fold —
  on a submitted job, hours of compute spent reaching a failure that was knowable
  at the start. Verified: zero base models are trained before it raises.
- Every meta-learner name the package advertises now works. `_meta_learner_names()`
  returns ten — `logistic_regression` plus all nine tabular aliases — but
  `naive_bayes` and `knn` raised `TypeError` the moment anyone passed them, and
  `catboost` ran **unseeded**: `_resolve_meta_learner` re-implemented the
  tabular pipeline's parameter handling and injected `random_state` into
  everything except catboost, which is backwards — those two constructors accept
  no `random_state`, and CatBoost's does. It now calls
  `tabular.train._build_model`, which checks the constructor signature, so a
  model behaves the same as a meta-learner as it does as a base model. No effect
  on the four candidates the published stacking results used (logistic
  regression, LightGBM, XGBoost, random forest) — all four already worked.
- `Optimize(pruning=...)` reaches the search. It was documented and then dropped
  at every call site, so `Optimize(pruning=False)` on a text model still pruned.
  `pruning=None` (the default) hid this, because it happens to equal each
  family's own default — on for text, off for tabular, now named as
  `TEXT_PRUNING_DEFAULT` / `TABULAR_PRUNING_DEFAULT` so the two cannot drift
  apart. `Optimize.describe()` reports both the resolved backend and pruning, so
  a run log says what actually ran.
- The Ray backends write `best_hyperparams.json` and `hpo_trials.csv`, the same
  names the Optuna backends write. The old `_ray` suffixes were harmless while
  Ray was only reachable from its own function and nothing looked for them; they
  stopped being harmless the moment the backend became a switch, because the run
  contract names `hpo/best_hyperparams.json`, `results/validation.py` reads
  `hpo_trials.csv` to produce `validation.json`'s `hpo_cv` score, and the LOPO
  and sample-size scripts reload `best_hyperparams.json` and feed it straight
  back as `hyperparams=`. A Ray-backed run was missing all three. Readers still
  accept the legacy `hpo_trials_ray.csv` for runs already on disk.
- `extend_oof_from=` refuses a source run whose out-of-fold predictions were
  computed with a different `n_folds` or `split_seed`, naming both values. The
  inherited columns come from the source run's folds and the new ones from this
  run's, so a mismatch makes the two halves of `[inherited | new]` incomparable —
  each column's out-of-fold predictions would come from models that saw a
  different amount of training data, or different fold members. Nothing
  downstream could detect it: stage 2 simply fitted a combiner on a matrix that
  was internally inconsistent. The source run records both values, so the
  mismatch was always knowable. A source run too old to record them warns instead.
- **The Ray search backend ran for real for the first time (2026-09-28, one H200
  node) and `tests/hpc_ray_smoke.py` found four defects, all fixed here.** None
  of them touches the Optuna backend, which is the default off a CUDA machine. The
  smoke test passes 11 of its 12 checks on a two-GPU node, the twelfth reporting
  that the text search's trials were too short to measure overlap. Two further
  fixes came out of the same runs:
  - `tests/hpc_ray_smoke.sbatch` picks a short directory for Ray's session state.
    Ray's Unix socket path must fit in 107 bytes and Ray appends about 65
    characters of its own, so a `$TMPDIR` on project storage — the normal case on
    a cluster — overflowed it and Ray never started
    (`AF_UNIX path length cannot exceed 107 bytes`). The wrapper now refuses to
    submit rather than letting Ray fail with an error that looks unrelated.
  - The trials table carries `datetime_start` and `datetime_complete` on the Ray
    path, reconstructed from Ray's completion timestamp and trial duration.
    Optuna writes both, and they are how a reader tells whether trials overlapped.
  - NumPy RNGs became unpicklable process-wide once a saved tabular model had
    been loaded. `ensure_numpy_rng_compat()` replaces
    `numpy.random._pickle._bit_generator_ctor` and never restores it, and the
    replacement was a nested function — pickle references functions by qualified
    name, and `…<locals>._compat_ctor` cannot be looked up, so every later pickle
    of an RNG raised `AttributeError: Can't get local object`. The replacement is
    a module-level function now. Ray Tune's searcher checkpointing was the first
    thing to hit this; the blast radius was every caller of `load_joblib_compat()`.
  - `hpo/hpo_trials.csv` from a Ray search now matches the shape its readers
    expect. The Ray path wrote Ray's own `ResultGrid` dataframe into a file with
    the right name — bare metric columns, `config/<name>`, `trial_id`, `logdir` —
    where `results/validation.py` keys on `value` and `user_attrs_*`. The result
    was `validation.json` recording `source="none"` for a Ray-searched run, and
    no fold stability anywhere. The 2026-09-27 cleanup had aligned the file
    *names*; this aligns the columns.
  - The Ray trial functions keep their per-fold scores. Tabular averaged
    `fold_scores` away before returning, so `fold_<i>_<metric>` and
    `cv_std_<metric>` could not reach the trials table however it was written.
  - A `RunConfig` is created wherever one can be. Ray ≥ ~2.49 rejects
    `CheckpointConfig(checkpoint_at_end=...)` outright, while older Ray Train v2
    requires it pinned to `False`; only an `ImportError` used to fall through to
    the plain variant, so on ray 2.52 the run got no `RunConfig` at all. That
    disabled resume *and* dropped `storage_path`, sending the experiment to
    `~/ray_results` — which exhausted a cluster home quota mid-search
    (`OSError: [Errno 122] Disk quota exceeded`, repeatedly, from Ray's syncer).
  - `max_concurrent_trials` is no longer passed to `TuneConfig` as well as to the
    `ConcurrencyLimiter` around `search_alg`. The cap always took effect through
    the limiter, but the duplicate made Ray log that it was ignoring the setting,
    which reads like it did nothing.
- **A setting the caller passed that a pipeline cannot act on is now named,
  rather than dropped in silence.** One `Optimize` object is accepted by every
  task, but not every task honours every field, and a deliberately passed setting
  vanishing is among the hardest things for a user to notice — especially in a
  queued cluster job. `warn_unused_settings()` reports the ones that will not be
  used, naming the pipeline and why, and says nothing about fields left at their
  defaults. Two places had been silent:
  - **Feature fusion honours only `n_trials` and `space`.** AutoGluon AutoMM runs
    the search itself, with its own scheduler and searcher (`hpo_scheduler=` /
    `hpo_searcher=`), its own holdout, and selection on the classifier's
    `eval_metric=` — so `backend=`, `metric=`, `cv=`, `cv_folds=`, `pruning=`,
    `resume=`, `space_profile=` and `extra=` reached nothing.
  - **`space_profile=` sizes the tabular search space only.** The text spaces
    adapt to training-set size and class count and have no profiles. The
    docstring said "Tabular only"; the behaviour said nothing.
- **The run log now states both halves of how `Optimize(space=...)` is applied.**
  The behaviour is unchanged and was already correct — a key you name is searched
  over exactly the range you wrote, because the adaptive adjustments run on the
  defaults before your keys are applied, and a key you do not name keeps its
  adaptive value and is still searched. What was missing was saying so: the log
  reported which default keys had been *replaced*, leaving a reader to work out
  what the rest of the space was. It now names the caller's keys as "used exactly
  as passed" and lists the keys that came from the adaptive space and are searched
  alongside them.
- **A malformed `Optimize(space=...)` is refused when the `Optimize` is built.**
  The searches run `study.optimize(..., catch=(Exception,))` so that one bad
  configuration cannot kill a long search — which meant a bad *space* failed every
  trial silently and reached the end of its budget with nothing completed, after
  the queue wait on a cluster. The errors that did escape the sampler were
  unreadable: a NaN bound arrived as `OverflowError: Range exceeds valid bounds`
  out of NumPy, a `None` bound as a `TypeError` about comparing `NoneType`. Every
  malformed entry is now named at once, with what the entry should look like —
  `None` in place of a range, a non-numeric or non-finite bound, a low above its
  high, a misspelled kind, an empty choice list, or the wrong number of values.
- **How to hold one hyperparameter fixed while searching the others is documented**:
  give it a single-value range, `("categorical", [8])`. `hyperparams=` carries one
  meaning at a time — a dict of fixed values means no search — so pinning happens
  inside the space. Omitting a key does the opposite: it keeps that key's adaptive
  range and searches it.
- The HPO runtime report is `hpo/runtime/hpo_runtime.json` on both backends. It
  was the one artifact the cleanup above missed: the Ray text search wrote
  `ray_hpo_runtime.json`, so a timing report had to be looked for under two names
  depending on which engine had run. Which engine ran is a field inside the file
  (`pipeline`), where it does not fragment the run contract.

### Changed

Same: relative to the pre-release copies described at the end of this entry, not
to any published version.

- `DEFAULT_META_LEARNER` (a dict) became `DEFAULT_META_LEARNERS` (a list of one),
  so adding a candidate is an edit to a list rather than a change of type. The
  default itself is unchanged: one multinomial logistic regression,
  `max_iter=1000, C=1.0`. Comparing several candidates stays something a caller
  asks for by passing `meta_learners=` — it is a decision about a particular
  study, and a run that asks for nothing should not quietly do something wider.
- **Ray is an optional extra, not a core dependency.** `ray[tune]`,
  `optuna-integration`, `grpcio` and `protobuf` moved from the core dependencies
  into `[ray]` (and are included in `[all]`). Measured before moving them: the
  group is **363 MB** installed — ray 192 MB, pyarrow 131 MB, grpcio 34 MB —
  where torch is 330 MB, and every user was paying it for a backend most never
  run. It also takes the `protobuf>=3.20.3,<5` ceiling out of core; that pin is
  this project's and not Ray's, which asks only for `protobuf>=3.15.3`, and it is
  what made the install unresolvable when the `opentelemetry-*` pins were
  present. Nothing imports ray at module level, so a Ray-less install runs every
  pipeline and every search on the Optuna backend, which is fully featured and is
  what the package was developed against.

  `Optimize(backend="auto")` now checks whether ray is importable and resolves to
  Optuna when it is not. When the machine looks like one where Ray would have
  paid off — a CUDA GPU, or a SLURM allocation — it **warns** that the extra is
  missing, because a silent fallback would be worse than an error: a submitted
  GPU job would run its trials one at a time, produce correct results, and go
  unnoticed until the wall clock showed it. `backend="ray"` asked for explicitly
  is never downgraded; the backend raises an `ImportError` naming the extra.

  `ray[tune]>=2.9,<2.53` — the floor is unchanged, and the ceiling matches
  `autogluon.core[raytune]`'s own window so `[ray]` and `[feature_fusion]`
  resolve together. `[feature_fusion]` pulls Ray in regardless, via
  `autogluon.multimodal`, so the saving reaches everyone *except* feature-fusion
  users.
- An `ImportError` from the Ray backend points at `multimodalva[ray]`, which now
  exists. It used to name an extra that did not.
- `xgboost>=3.0`, with no upper bound. It was `>=3.1,<3.2`, a window chosen to
  hold one environment still: xgboost's tree construction is not stable across
  minor releases when `subsample` or `colsample_bytree` is below 1, and a refit
  under a neighbouring version was measured to move 6% of top-1 labels. That is a
  property of xgboost, and a dependency range is the wrong place to answer it — a
  range wide enough to install permits some drift, and one narrow enough to
  prevent it makes the package uninstallable as soon as the next release lands.
  The floor stays at 3.0 so that saved boosters load in every later 3.x, which
  `predict_from_pretrained()` depends on, and the FAQ's reproducibility section
  now says to record the environment (`pip freeze`) beside any run that has to be
  reproduced.
- `pyyaml>=5.1` is declared. The CLI reads YAML config files, a documented entry
  point, and it was relying on transformers / datasets / optuna / accelerate to
  bring PyYAML in. They still do; the dependency is no longer implicit.
- The source distribution ships the three scripts under `tests/` that are not
  named `test_*.py`, so setuptools does not collect them by itself:
  `smoke_test.py` and `diagnose.py`, which the README and CONTRIBUTING both tell
  the reader to run straight after installing, and `hpc_ray_smoke.py`, which was
  shipping without its own `.sbatch` wrapper's other half. `tests/sample_data/`
  stays out — 204 KB of CSV that nothing in the distribution reads.

### Known limitations

- Stacking records absolute paths in `model_sources`, so a stacking run cannot be
  moved between machines without editing them.
- Feature fusion needs `autogluon.multimodal`, which pins torch and transformers
  more tightly than the rest of the package. Install it as the
  `[feature_fusion]` extra into a clean environment.
- By default a fixed pair of seeds reproduces the predicted labels exactly, but
  probabilities can differ at around 1e-16 on CPU and up to 1e-7 on an
  Ampere-or-newer GPU. `deterministic=True` removes that on one machine, but runs
  slower and raises on any operation without a deterministic kernel. Nothing
  makes results bit-identical across GPU models or library versions. Inference
  `batch_size` also shifts probabilities at float32 rounding level, because each
  batch is padded to its own longest sequence; reproducing a saved probability
  file byte-for-byte needs the same `batch_size`.
- A dry run checks the configuration against the data, not the run. It cannot see
  a batch size that will not fit, a trial budget too small for the search space,
  an `Optimize(extra=...)` key the backend ignores, or whether a Hugging Face Hub
  ID exists.
- The Ray search backend passes all twelve checks of `tests/hpc_ray_smoke.py` on a
  two-GPU SLURM node — including trials observed running concurrently in both the
  tabular and the text search — verified twice: once on the original
  implementation and again on the shipped code, with Ray 2.52.1 and torch 2.14.
  That is one cluster and two Ray versions, which is what gives any confidence
  that the `RunConfig` compatibility fallback works across them.

  What those twelve checks do **not** cover: a trial that actually fails (they
  only run trials that succeed, so the strict re-raise path is exercised by unit
  tests, not on the cluster), and `cv_folds="auto"` on the Ray path. Off a CUDA
  machine both entry points redirect to Optuna by design, so the ordinary test
  suite cannot reach the Ray path at all — it covers the trial functions in
  process only. Run the smoke test on your own cluster before relying on Ray for a
  long search.

### Notes for anyone who used a pre-release copy

The public names were made unique before this release. Both pipelines used to
export `train`, `predict`, `prepare_dataset`, `optimize`, `SUPPORTED_MODELS` and
`DEFAULT_SEARCH_SPACE`, which collided on import and read ambiguously:

| Was | Now |
|---|---|
| `text.dataset.prepare_dataset` / `tabular.dataset.prepare_dataset` | `prepare_text_dataset` / `prepare_tabular_dataset` |
| `text.train.train` / `tabular.train.train` | `train_text` / `train_tabular` |
| `text.predict.predict` / `tabular.predict.predict` | `predict_text` / `predict_tabular` |
| `optimize`, `optimize_ray` | `optimize_text`, `optimize_text_ray`, `optimize_tabular`, `optimize_tabular_ray` |
| `get_default_search_space` | `get_text_default_search_space` / `get_tabular_default_search_space` |
| `SUPPORTED_MODELS` | `TEXT_MODELS` / `TABULAR_MODELS` |
| `DEFAULT_HYPERPARAMS` | `TEXT_DEFAULT_HYPERPARAMS` / `TABULAR_DEFAULT_HYPERPARAMS` |
| `DEFAULT_SEARCH_SPACE(S)` | `TEXT_` / `TABULAR_` / `DATA_FUSION_DEFAULT_SEARCH_SPACE(S)` |
| `run(label=, optimize=, metric=)` | `run(label_col=, hyperparams=)` |
| `use_optimize=True, n_trials=N, optimize_metric=M, search_space=S` | `hyperparams=Optimize(n_trials=N, metric=M, space=S)` |
| `use_cv=`, `n_cv_folds=`, `resume_hpo=` on a classifier | `Optimize(cv=, cv_folds=, resume=)` |
| `FeatureFusionClassifier.run(hyperparameters=, hpo_search_space=)` | `hyperparams=` / `Optimize(space=)` |
| CLI `--label` / `--optimize` / `--metric` | `--label-col` / `--optimize` / `--optimize-metric` |
| `FeatureFusionClassifier.run(use_hpo=, n_hpo_trials=)` | `hyperparams=Optimize(n_trials=…)` |
| `random_state=` on `run()` or any classifier | `split_seed=` and/or `train_seed=` |
| `set_seed=` (text, data fusion) | `train_seed=` |
| `StackingClassifier(load_oof_from=)` | `extend_oof_from=` — it trains the new base models and widens the OOF matrix, which `load_` did not say, and which `run(oof_from=)`, a step away by one prefix, does not do |
| `FeatureFusionClassifier.run(automm_seed=)` | `train_seed=` |
| `train_class_voter_stage(fallback_random_state=)` | `split_seed=` |
| `train_meta_learner_stage(random_state=)` | `split_seed=` / `train_seed=`, both defaulting to stage 1's |
| CLI `--random-state` | `--split-seed` / `--train-seed`; new `--deterministic` |
| `StackingClassifier.run(meta_hyperparams=)` | removed; pass fixed candidates in `meta_learners=` |
| `resume_training=` (text, data fusion) | `resume=`, the name every task uses; `--no-resume` on the CLI |
| `FeatureFusionClassifier` result key and file `best_hpo_config(.json)` | `best_hyperparams`, written to `hpo/best_hyperparams.json` like every other search |

The low-level functions — `split`, `train_text`, `train_tabular`,
`optimize_text`, `optimize_tabular`, `generate_oof_predictions` — keep
`random_state` as the model seed and gain an optional `split_seed` that defaults
to it, so existing calls behave as before.

The same change removed `text.train.get_device` (import it from `utils.runtime`)
and the internal `_assemble_prediction_result` wrappers (call
`utils.predictions.assemble_predictions`).

**Results from a pre-release copy can differ, because three seeds were being
ignored.** Each produced a normal-looking run whose seed simply had no effect:

- **xgboost never received a seed.** Its constructor takes `**kwargs`, which the
  seed-injection check did not recognise, so every seed gave byte-identical
  models. A seed-variation study over xgboost would have shown zero variance.
- **Text base models in `voting` and `stacking` ignored the run's seed.**
  `train_text()` was called without it and trained at its own default, so the
  split and the search followed the seed but the training did not.
- **`feature_fusion` did not seed AutoMM by default,** and its search branch
  passed no seed at all, so the run's seed reached only the split. `train_seed`
  is now passed to `fit(seed=)` in both branches (AutoMM's own default is 0).

Other pipelines and models are unaffected. `tests/test_seed_propagation.py`
checks every training call site and trains each seedable model twice, so a seed
cannot be silently dropped again.

Other fixes a pre-release user may notice:

- **Log loss was wrong when the class ids were not in alphabetical order.**
  scikit-learn aligns the columns of a probability matrix to lexicographically
  sorted labels whatever order `labels=` is passed in, and only warns; the
  columns here are in class-id order, so the two disagreed and the score came
  out silently wrong. Predictions produced by this package were never affected,
  because its label encoders sort, but a probability table brought in from
  another tool could be. The columns are now reordered before scoring.

- **`id_col` was silently dropped** by `feature_fusion`, `voting` and `stacking`,
  and by `data_fusion` when called through `run()` — the argument was accepted
  and the `id` column simply never appeared. All six tasks now carry it.
- **`voting` could not resume.** An interrupted run retrained every base model;
  finished ones are now reused.
- **The stacking meta-learner search was removed.** It had its own search-space
  syntax that rejected the `"float_log"` spelling every other search uses, so a
  space written as documented failed on every trial, with an error blaming
  missing dependencies. The meta-learner sees only `n_models × n_classes`
  columns; a few cheap fixed candidates, compared by cross-validation, are
  enough. Every candidate — a lone one included — is now scored, and the chosen
  one and its score are returned by `run()` and written to
  `training_metadata.json`.
- **Early stopping inside ensembles now matches a standalone text model.**
  `voting` and `stacking` defaulted `early_stopping_patience` to 3 while
  `TextClassifier` used 4, so a text base model trained differently inside an
  ensemble than on its own. All default to 4.
- **`encode_categoricals` / `scale_numeric` given to `run()` never reached
  `voting` or `stacking`**, and stacking also ignored them inside a base-model
  spec, which voting honours. `run()` now passes them on; voting takes them as the
  default for every spec, a spec's own value overriding; stacking, which prepares
  one feature matrix for all tabular base models, raises if a spec asks for
  something else.
- **Feature fusion's validation holdout stays AutoMM's to make.** AutoMM splits
  it with the same seed it trains with, so in this one pipeline `train_seed` also
  decides which rows are held out, and `split_seed` governs only the train/test
  split. That is AutoMM's design; the package passes `val_size` through as its
  public `holdout_frac` and does not carve the holdout itself, so a feature-fusion
  run is AutoMM's own procedure end to end.
- **A search on a small dataset could fail every trial.** Each trial carves an
  early-stopping slice from its own sub-split; with fewer slice rows than
  causes the stratified split raised, and the summary blamed GPU memory. The
  slice now grows to one row per cause, or is drawn at random where a cause has
  a single row. Data that split fine before is unaffected.
- **The text data-loader worker count was fixed at import time**, so setting
  `MULTIMODALVA_DATALOADER_WORKERS` after importing the package had no effect on
  training. It is now decided when training starts.
- `TabularClassifier.run(search_space_profile=)` was removed. It had no effect
  once a search was configured with `Optimize`; use
  `Optimize(space_profile=...)`.

Prediction output was not uniform across pipelines, and two pipelines did not
write predictions at all. Every pipeline now writes the same three files to the
same place — `<output_dir>/predictions/predictions_{top1,full,topk}.csv`:

| Pipeline | Was | Now |
|---|---|---|
| text, tabular, data fusion | `predictions/predictions_*.csv` | unchanged |
| soft voting | *nothing written* — the voted result existed only in the returned object | `predictions/predictions_*.csv` |
| feature fusion | *nothing written* — and its `resume` path read files it never produced, so resuming never worked | `predictions/predictions_*.csv` |
| stacking | `predictions/top1.csv`, `full.csv`, `topk.csv` | `predictions/predictions_*.csv` |

The class-aware voter stage of stacking moved the same way, from
`class_voter/predictions/top1.csv` to
`class_voter/predictions/predictions_top1.csv`.

Anything that read a stacking run's `predictions/top1.csv`, or a feature-fusion
run's hand-written `predictions_top1.csv` in the run root, needs the new path.
Runs already on disk keep working with `predictions_frame()`, which still reads
the older spellings, and feature fusion still resumes from a run-root copy.

The default hyperparameter-search objective is now `f1_macro` for all six
pipelines. It was `accuracy` on `TextClassifier.run`, `TabularClassifier.run`,
`optimize_text` and `optimize_tabular`, and `csmf_accuracy` on
`DataFusionClassifier.run`, while `run()`, feature fusion and the ensemble
base-model specs already used `f1_macro` — so the same model searched against a
different objective depending on how it was called. Accuracy rewards getting the
common causes right, which is the wrong default when the rare causes are the
point.

LoRA is now off by default. `TextClassifier.run`, `optimize_text`,
`optimize_text_ray` and `DataFusionClassifier.run` defaulted to `use_lora=True`
while `train_text` — the function they all end up calling — defaulted to
`False`, so what you got depended on which door you came in through. All of them
are `False` now, and full fine-tuning is the default everywhere.

A stacking base model is the same model a single-model run trains, but stacking
drives the training functions directly — it needs one split, then k folds, then
a full-data refit, which the classifier `run()` methods cannot express because
each owns its own split. Every setting therefore has to be re-exported through
the base-model spec, and some had drifted or were missing: a text base model
searched with LoRA off where a single-model run used it on, against a different
objective and trial budget, and `use_focal`, `use_cv`, `n_cv_folds`, `use_fast`,
`resume_hpo` and `search_space_profile` could not be set at all. The spec now
defaults to the single-model values and passes all of them through. The search
*space* was always shared, so nothing about the ranges changed.

The run log now says where each base model's hyperparameters came from — the
caller's spec, a search that just ran, or nothing at all, in which case the
underlying library's own defaults apply and a warning says so. Fixed values and a
search can no longer be requested at once: both are spellings of the one
`hyperparams=` argument.

`run(use_optimize=True, n_trials=…, optimize_metric=…)` never reached the base
models of `voting` and `stacking`, because those tasks configure the search per
model inside the spec dicts. With one `hyperparams` value that problem does not
arise: a run-level setting applies to every spec that does not state its own,
and a spec that carries explicit values keeps them. The search diagnostics —
`hpo_leaderboard.csv` and `hpo_convergence.png` — were also written for at most
one search per run; they are now written next to every trials file, so each base
model in an ensemble gets its own.

Also fixed on the way: `cause_accuracy_heatmap()` accepted and documented
`group_sizes` but silently ignored it, so its column-group annotations were never
drawn; `utils/metrics.py` and `text/predict.py` had string annotations naming
`pandas` without importing it, which made `typing.get_type_hints()` raise; and a
documented example passed an `alpha` argument that `train_class_voter_stage()`
does not accept.

The six `opentelemetry-*` dependencies were dropped. They were listed as
transitive dependencies of `ray[tune]`, which they are not, and they made
`pip install multimodalva` fail because the current `opentelemetry-proto`
requires a protobuf major version this project caps below.
