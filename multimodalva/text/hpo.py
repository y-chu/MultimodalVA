"""
Step 5: Hyperparameter optimization.

Two backends are provided; both accept the same search_space format and return
``(best_hyperparams, backend_object)`` so callers can switch backends without changing call sites:

    optimize_text()      — Optuna (TPE sampler, JournalStorage persistence)
                      Best for single-machine sequential or lightly parallel HPO.

    optimize_text_ray()  — Ray Tune (OptunaSearch / TPE, distributed runtime)
                      Best for multi-GPU machines or multi-node clusters.
                      Runs N trials concurrently, one GPU per trial (configurable).

Uses train_text() and predict_text() internally.
Input:  train_dataset, label2id, id2label  from prepare_text_dataset()
Output: best_hyperparams dict and backend study / ResultGrid object
"""

from __future__ import annotations

import json
import inspect
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna
    import ray

    from ..utils.optimize_config import Optimize

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

# from ..utils.metrics import (  # noqa: F401
#     csmf_accuracy, score_predictions, sample_hyperparams,
#     log_loss_from_full, METRIC_DIRECTION,
# )
# from .predict import predict
# from .train import _get_dataset_labels, train

import sys

# -----------------------
# Package root resolution
# -----------------------
# __file__ = multimodalva/text/hpo.py  →  parent.parent.parent = project root (MultimodalVA/)
# Must point to the directory that *contains* the multimodalva/ package folder so that
# "from multimodalva.xxx import ..." resolves correctly when scripts are run directly.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # MultimodalVA/
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# -----------------------
# Internal imports
# -----------------------
from multimodalva.utils.metrics import (
    csmf_accuracy,
    score_predictions,
    sample_hyperparams,
    decode_hyperparams_for_space,
    decode_trials_dataframe_for_space,
    log_loss_from_full,
    METRIC_DIRECTION,
    CV_METRICS,
    HPO_METRICS,
)
from multimodalva.utils.runtime import RuntimeTracker, empty_accelerator_cache
from multimodalva.utils.split import stratified_indices
from multimodalva.utils.ray_compat import to_ray_space
from multimodalva.text.models import resolve_model_name
from multimodalva.text.train import train_text, _get_dataset_labels
from multimodalva.text.predict import predict_text

logger = logging.getLogger(__name__)

from .search_spaces import (  # noqa: E402
    TEXT_DEFAULT_SEARCH_SPACE,
    FOCAL_SEARCH_SPACE,
    LORA_SEARCH_SPACE,
    _sample_tier,
    get_text_default_search_space,
)
from ..utils.hpo_defaults import get_class_tier, merge_search_space  # noqa: E402


#: ``enable_pruning``'s default for this family, named so that
#: ``Optimize(pruning=None)`` ("whatever text normally does") resolves to the
#: same value the low-level function defaults to. A text trial costs minutes, so
#: stopping a hopeless configuration early is worth the risk of stopping a slow
#: starter.
TEXT_PRUNING_DEFAULT = True


@contextmanager
def _trial_workdir(persistent: Path | None):
    """Yield a directory for one trial/fold's train_text()+predict_text() artifacts.

    HPO trials only need the resulting *score* — never the model weights.  When
    ``persistent`` is None (the default path), training runs inside a
    ``tempfile.TemporaryDirectory`` that is removed as soon as the score has been
    computed, so no checkpoints / optimizer state / model weights survive on disk,
    even if the sweep is interrupted mid-run.  When ``persistent`` is a Path (only
    when ``export_best_trial=True``), that directory is used and left in place so
    the best trial can be exported afterwards.
    """
    if persistent is not None:
        persistent.mkdir(parents=True, exist_ok=True)
        yield persistent
    else:
        with tempfile.TemporaryDirectory(prefix="mmva_hpo_trial_") as td:
            yield Path(td)


def optimize_text(
    train_dataset,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    # ── Trial budget ──────────────────────────────────────────────────────────
    # Recommended n_trials (with use_cv=True, n_cv_folds=3 — each trial runs k full trains):
    #   Without LoRA (~7 HPs): 20–25 trials  →  60–75 total training runs
    #   With    LoRA (~10 HPs): 15–20 trials  →  45–60 total training runs
    # Recommended n_trials (with use_cv=False — each trial runs 1 train):
    #   Without LoRA: 30–40 trials
    #   With    LoRA: 25–30 trials
    # TPE needs ~15–20 random-exploration trials before it can exploit correlations,
    # so the minimum meaningful budget is 20 trials regardless of CV or LoRA.
    n_trials: int = 30,
    metric: str = "f1_macro",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    split_seed: int | None = None,
    study_name: str = "text_hpo",
    use_lora: bool = False,
    use_focal: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = 4,
    storage_path: str | None = None,
    load_if_exists: bool = True,
    enable_pruning: bool = TEXT_PRUNING_DEFAULT,
    save_trials_csv: bool = True,
    cleanup_trials: bool = True,
    export_best_trial: bool = False,
    use_fast: bool = True,
    # ── CV options ────────────────────────────────────────────────────────────
    use_cv: bool = True,
    n_cv_folds: int = 3,
) -> tuple[dict, optuna.Study]:
    """Run Optuna hyperparameter search using train_text() and predict_text().

    Internally splits train_dataset into an (1 - val_size) / val_size
    opt-train / opt-eval split. Each trial trains on opt-train and is scored
    on opt-eval. The held-out test set from prepare_text_dataset() is never used here.

    Storage: each trial/fold is trained inside a ``tempfile.TemporaryDirectory``
    that is deleted as soon as its score is computed: no model weights, optimizer
    state, or checkpoints survive per trial (the historical bloat: 500+ trial dirs
    × ~hundreds of MB each).  Only the *scores* persist, via the JournalStorage
    ``hpo_<model>.log`` (also drives resume) and the trials CSV.  best_hyperparams
    is saved to best_hyperparams.json.  The returned best_hyperparams can be passed
    directly to train(hyperparams=best_hyperparams) for a final run on the full
    training set — this is what the TextClassifier wrapper does, so the best trial's
    weights are never needed.  Set ``export_best_trial=True`` to additionally keep a
    reloadable copy of the best trial's model in output_dir/best_trial/.

    Args:
        train_dataset: Tokenized ClassificationDataset from prepare_text_dataset().
        label2id: Label-to-integer mapping from prepare_text_dataset().
        id2label: Integer-to-label mapping from prepare_text_dataset().
        model_name: HuggingFace model name or local path.
        output_dir: Root directory for all trial outputs.
        n_trials: Total number of Optuna trials to run. Default 20.
        metric: Metric to maximise — "accuracy", "balanced_accuracy",
                "f1_macro", "f1_weighted", "csmf_accuracy". Default "f1_macro".
                Use "balanced_accuracy", "csmf_accuracy", or "f1_macro" for imbalanced VA data.
        search_space: Parameter ranges. Overrides the default per key; keys you
                      omit keep their default. ``None`` uses
                      get_text_default_search_space(n_samples, n_classes), plus
                      LORA_SEARCH_SPACE / FOCAL_SEARCH_SPACE when enabled.
        val_size: Fraction of train_dataset held out for trial evaluation. Default 0.2.
        random_state: Seed for the Optuna sampler and for model training.
        split_seed:   Seed for the row partitioning this function does
                      internally — the cross-validation folds and the internal validation split. ``None`` (the default) reuses
                      ``random_state``, so existing callers are unaffected.
                      Pass it separately to hold the partition fixed while
                      ``random_state`` reseeds the model.
                      Default 42.
        study_name: Name for the Optuna study. Default "text_hpo".
        use_lora: Apply LoRA adapters during each trial. Default False.
        use_focal: Use focal loss during each trial.  When True, merges
                   FOCAL_SEARCH_SPACE (focal_gamma, class_weights) and fixes
                   loss_type="focal" in every trial's hyperparams.
                   Recommended for severe class imbalance; pair with
                   metric="f1_macro" or "balanced_accuracy". Default False.
        gradient_checkpointing: Enable gradient checkpointing per trial. Default False.
        early_stopping_patience: Early stopping patience passed to each trial's train_text().
                                  Set to None to disable. Default 3.
        storage_path: Path to the JournalStorage log file for persisting the study
                      (enables resume across sessions).
                      Defaults to output_dir/hpo_<model_name>.log.
                      Accepts a bare path (.log) or legacy sqlite:///… / .db paths
                      — .db paths are auto-redirected to .log with a warning.
        load_if_exists: Resume an existing study if storage_path already contains one.
                        Default True.
        enable_pruning: Use Optuna's MedianPruner to stop unpromising trials early.
                        Default True.
        use_cv: Use stratified k-fold cross-validation to score each trial instead
                of a fixed 80/20 split.  Each fold trains a fresh model on (k-1)/k
                of train_dataset and scores on the remaining 1/k; the k scores are
                averaged into the Optuna objective.  Reduces HPO metric variance for
                rare VA causes (1–2 val samples per class with a fixed split give a
                noisy binary signal; averaging k independent estimates improves it).
                Inter-trial pruning still applies after each fold.  Default True.
        n_cv_folds: Number of CV folds.  k=3 balances cost (3× trials) against
                    variance reduction (√3 ≈ 1.7×).  Use k=5 for very small datasets
                    (<1 500 samples) where a 20% fold already has reliable class
                    coverage.  Ignored when use_cv=False.  Default 3.
        save_trials_csv: Save all trial results to output_dir/hpo_trials.csv after
                         optimization completes. Includes hyperparameters, objective
                         score, all user_attrs_* metrics, state, and duration.
                         Default True.
        cleanup_trials: Only relevant when ``export_best_trial=True`` (otherwise trials
                        leave no directories to clean — see ``export_best_trial``).
                        When True, deletes the non-best persisted ``trial_*/`` dirs
                        after the best is exported.  The JournalStorage log,
                        ``best_hyperparams.json``, ``best_trial/``, and the CSV are kept.
                        Default True.
        export_best_trial: Keep a reloadable copy of the best trial's model at
                           ``output_dir/best_trial/``.  Default False — trials run in
                           temporary directories (no weights persisted), which is the
                           storage fix.  Set True to persist trials in ``trial_*/`` dirs
                           (each holding only the final merged model, no checkpoints or
                           optimizer state) so the best one can be exported afterwards.
                           The wrapper retrains ``final/`` from best_hyperparams, so this
                           is rarely needed; best_hyperparams.json + the .log already
                           capture everything required to retrain.

    Returns:
        best_hyperparams: Dict of hyperparameter values from the best trial.
                          Can be passed directly to train(hyperparams=best_hyperparams).
        study: The completed Optuna study object (for further analysis / visualisation).
    """
    # Raise the open-file-descriptor limit before the HPO loop.
    # On macOS the default soft limit is 256; Longformer HPO can exhaust this
    # (model weights, tokenizer files, checkpoint dirs, Dropbox daemon) causing
    # file I/O errors in JournalStorage or tokenizer loading, which make
    # Optuna crash with AssertionError when it can't record a trial failure.
    try:
        import resource as _resource
        _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
        _target = min(max(_soft, 8192), _hard) if _hard > 0 else max(_soft, 8192)
        if _soft < _target:
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
            logger.info("Raised open-file-descriptor limit: %d → %d.", _soft, _target)
    except Exception:
        pass  # resource module not available on Windows; best-effort only

    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise ImportError(
            "optuna is required for optimize_text(). "
            "Install with:  pip install 'optuna>=3.4'"
        ) from exc

    _VALID_METRICS = HPO_METRICS
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid metrics: {sorted(_VALID_METRICS)}"
        )

    # Resolve once so every trial loads the same checkpoint without re-downloading.
    model_name = resolve_model_name(model_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_tracker = RuntimeTracker(
        output_dir,
        report_name="hpo_runtime.json",
        metadata={
            "pipeline": "text_hpo_optuna",
            "model_name": model_name,
            "metric": metric,
            "n_trials_requested": n_trials,
            "use_lora": use_lora,
            "use_focal": use_focal,
            "gradient_checkpointing": gradient_checkpointing,
            "load_if_exists": load_if_exists,
            "use_cv": use_cv,
            "n_cv_folds": n_cv_folds if use_cv else None,
            "early_stopping_patience": early_stopping_patience,
        },
        logger_=logger,
    )

    # Build effective search space: data-adaptive base → LoRA/focal merges → caller overrides
    n_samples = len(train_dataset)
    n_classes = len(id2label)
    active_space = get_text_default_search_space(n_samples, n_classes)
    logger.info(
        "Text HPO adaptive search space: sample_tier=%s, class_tier=%s "
        "(n_samples=%d, n_classes=%d).",
        _sample_tier(n_samples), get_class_tier(n_classes), n_samples, n_classes,
    )
    if use_lora:
        active_space.update(LORA_SEARCH_SPACE)
    if use_focal:
        active_space.update(FOCAL_SEARCH_SPACE)
    merge_search_space(active_space, search_space, logger)

    # Drop hyperparameters not supported by this model architecture.
    # classifier_dropout: only BERT/RoBERTa/BigBird/ELECTRA configs accept it;
    # Longformer does not — remove from search space to avoid wasted trials.
    if "classifier_dropout" in active_space:
        import inspect as _inspect
        from transformers import AutoConfig as _AutoConfig
        _cfg_cls = type(_AutoConfig.from_pretrained(model_name))
        if "classifier_dropout" not in _inspect.signature(_cfg_cls.__init__).parameters:
            active_space.pop("classifier_dropout")
            logger.info(
                "Removed 'classifier_dropout' from HPO search space: "
                "%s (%s) does not support this parameter.",
                model_name, _cfg_cls.__name__,
            )

    # JournalStorage (append-only log + fcntl locking) replaces SQLite as the
    # default backend.  It tolerates cloud-sync folders (Dropbox, iCloud) and
    # network filesystems where SQLite's page-lock protocol tends to break.
    if storage_path is None:
        # A local checkpoint directory would otherwise produce a file name made
        # of the whole path, which also breaks resume if the directory moves.
        safe_name = Path(model_name).name if "/" in model_name and Path(model_name).is_dir() \
            else model_name.replace("/", "_")
        storage_path = str(output_dir.resolve() / f"hpo_{safe_name}.log")
    else:
        # Normalize legacy sqlite:/// URLs and auto-redirect .db → .log.
        _sp = storage_path
        if _sp.startswith("sqlite:///"):
            _sp = _sp[len("sqlite:///"):]
        _sp = str(Path(_sp).resolve())
        if _sp.endswith(".db"):
            _sp = _sp[:-3] + ".log"
            logger.warning(
                "storage_path points to a SQLite .db file; the HPO backend is now "
                "JournalStorage (.log).  Redirecting to %s.  "
                "To migrate existing trials: "
                "python Analysis/utils/migrate_hpo_storage.py <dir>",
                _sp,
            )
        storage_path = _sp
    try:
        from optuna.storages.journal import JournalFileBackend as _JBackend
    except ImportError:
        from optuna.storages import JournalFileStorage as _JBackend  # type: ignore
    storage_obj = optuna.storages.JournalStorage(_JBackend(storage_path))

    # --- Data split setup (fixed split only needed for use_cv=False) ---
    _all_labels = _get_dataset_labels(train_dataset)
    _indices = list(range(len(train_dataset)))
    _split_seed = split_seed if split_seed is not None else random_state
    if not use_cv:
        # Same resilience as train_text()'s internal split: on very small data a
        # val_size share can be thinner than the number of classes, and an
        # unguarded stratified split then fails every trial.
        opt_train_idx, opt_val_idx = stratified_indices(
            _all_labels,
            val_size,
            _split_seed,
            what="validation",
            label_names=id2label,
            log=logger,
        )
        opt_train = Subset(train_dataset, opt_train_idx)
        opt_val = Subset(train_dataset, opt_val_idx)

    # Pre-build CV folds once (outside objective) so all trials share the same splits.
    # This removes fold-assignment variance from the HPO signal — trial differences
    # reflect hyperparameters only, not different random fold draws.
    if use_cv:
        _skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=_split_seed)
        _cv_splits = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in _skf.split(_indices, _all_labels)
        ]
        logger.info(
            "CV HPO: %d-fold stratified splits pre-built "
            "(n_samples=%d, n_classes=%d).  Each trial runs %d training jobs.",
            n_cv_folds, len(_indices), len(id2label), n_cv_folds,
        )

    # --- Objective ---
    def objective(trial: optuna.Trial) -> float:
        import torch

        trial_started = time.perf_counter()
        hp = sample_hyperparams(trial, active_space)
        if use_focal:
            hp["loss_type"] = "focal"
        # Persist this trial's dir only when we intend to export the best trial;
        # otherwise train into an auto-deleted tempdir (see _trial_workdir).
        trial_dir = (output_dir / f"trial_{trial.number}") if export_best_trial else None
        score = float("nan")
        failed = False

        try:
            if use_cv:
                # ── k-fold CV path ──────────────────────────────────────────
                # Each fold trains a fresh model; no model state carries across folds.
                # val_size=None + early_stopping_patience=None: train the full sampled
                # epoch count so the `epochs` HP is meaningful and all folds are
                # comparable.  load_best_model_at_end is inactive without a val split,
                # so the final epoch checkpoint is used — acceptable for HPO scoring.
                _fold_scores: dict[str, list[float]] = {
                    m: [] for m in CV_METRICS
                }
                _fold_log_losses: list[float] = []

                for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(_cv_splits):
                    fold_train = Subset(train_dataset, cv_train_idx)
                    fold_val   = Subset(train_dataset, cv_val_idx)
                    fold_persistent = (trial_dir / f"fold_{fold_idx}") if trial_dir is not None else None

                    # Train + score inside the workdir; weights are discarded as soon
                    # as the fold score is read (only the score is needed for HPO).
                    with _trial_workdir(fold_persistent) as fold_dir:
                        train_text(
                            train_dataset=fold_train,
                            label2id=label2id,
                            id2label=id2label,
                            model_name=model_name,
                            output_dir=fold_dir,
                            hyperparams=hp,
                            val_size=None,               # full fold-train used; no internal split
                            use_lora=use_lora,
                            gradient_checkpointing=gradient_checkpointing,
                            early_stopping_patience=None,  # no early stopping within fold
                            resume=False,
                            cleanup_checkpoints=True,      # trials never need checkpoints
                            save_total_limit=1,            # minimise peak disk per fold
                            use_fast=use_fast,
                            random_state=random_state,
                        )
                        fold_result = predict_text(fold_dir, fold_val, use_fast=use_fast)
                        for m in CV_METRICS:
                            _fold_scores[m].append(score_predictions(fold_result.top1, m))
                        _fold_log_losses.append(log_loss_from_full(fold_result.full, fold_result.id2label))

                    # Store per-fold score for the target metric
                    trial.set_user_attr(f"fold_{fold_idx}_{metric}", _fold_scores[metric][-1])

                    # Inter-trial pruning based on running mean after each fold.
                    # MedianPruner compares this to the median of completed trials.
                    _running_mean = float(np.mean(_fold_scores[metric]))
                    if enable_pruning:
                        trial.report(_running_mean, fold_idx)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                    empty_accelerator_cache()

                all_scores = {m: float(np.mean(_fold_scores[m])) for m in _fold_scores}
                all_scores["log_loss"] = float(np.mean(_fold_log_losses))
                # CV standard deviation for the target metric — useful for stability analysis
                trial.set_user_attr(f"cv_std_{metric}", float(np.std(_fold_scores[metric])))

            else:
                # ── Single fixed-split path (original behaviour) ────────────
                # val_size=0.1: train_text() carves an internal eval split from opt_train
                # for early stopping / best-checkpoint selection; separate from opt_val.
                with _trial_workdir(trial_dir) as tdir:
                    train_text(
                        train_dataset=opt_train,
                        label2id=label2id,
                        id2label=id2label,
                        model_name=model_name,
                        output_dir=tdir,
                        hyperparams=hp,
                        val_size=0.1,
                        use_lora=use_lora,
                        gradient_checkpointing=gradient_checkpointing,
                        early_stopping_patience=early_stopping_patience,
                        resume=False,
                        cleanup_checkpoints=True,      # trials never need checkpoints
                        save_total_limit=1,            # minimise peak disk per trial
                        use_fast=use_fast,
                        random_state=random_state,
                    )
                    result = predict_text(tdir, opt_val, use_fast=use_fast)
                    all_scores = {
                        m: score_predictions(result.top1, m)
                        for m in CV_METRICS
                    }
                    all_scores["log_loss"] = log_loss_from_full(result.full, result.id2label)

            # Store all metrics as user attributes for traceability (both paths).
            for name, val in all_scores.items():
                trial.set_user_attr(name, val)
            score = all_scores[metric]
            return score

        except optuna.exceptions.TrialPruned:
            # Re-raise cleanly so Optuna marks the trial as PRUNED, not FAILED.
            raise
        except Exception:
            failed = True
            raise
        finally:
            trial_elapsed = round(time.perf_counter() - trial_started, 3)
            trial.set_user_attr("elapsed_seconds", trial_elapsed)
            trial.set_user_attr(
                "trial_dir",
                str(trial_dir) if trial_dir is not None else "<tempdir (not persisted)>",
            )
            logger.info(
                "Optuna trial %d %s in %.2fs%s",
                trial.number,
                "failed" if failed else "completed",
                trial_elapsed,
                (
                    f" — {metric}={score:.4f}"
                    if not failed and score == score
                    else ""
                ),
            )
            # Release GPU/MPS memory after each trial to avoid OOM on subsequent trials.
            empty_accelerator_cache()

    # --- Create / resume study ---
    pruner = (
        MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if enable_pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction=METRIC_DIRECTION.get(metric, "maximize"),
        sampler=TPESampler(seed=random_state),
        pruner=pruner,
        study_name=study_name,
        storage=storage_obj,
        load_if_exists=load_if_exists,
    )

    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    remaining = n_trials - completed
    runtime_tracker.update_metadata(
        storage_path=storage_path,
        completed_trials_before_resume=completed,
        remaining_trials=remaining,
    )
    with runtime_tracker.stage(
        "optuna_optimize",
        details={
            "remaining_trials": remaining,
            "val_size": val_size,
            "study_name": study_name,
        },
        monitor_gpu=True,
    ):
        if remaining > 0:
            # catch=(Exception,) lets Optuna mark a failed trial as FAILED and continue
            # with remaining trials instead of crashing the whole HPO run (e.g. on OOM).
            study.optimize(objective, n_trials=remaining, catch=(Exception,))
        else:
            logger.info("All %d trials already completed. Skipping optimization.", n_trials)

    # --- HPO Health Check ---
    _all_states = [t.state for t in study.trials]
    _n_total    = len(_all_states)
    _n_complete = sum(1 for s in _all_states if s == optuna.trial.TrialState.COMPLETE)
    _n_pruned   = sum(1 for s in _all_states if s == optuna.trial.TrialState.PRUNED)
    _n_failed   = sum(1 for s in _all_states if s == optuna.trial.TrialState.FAIL)
    _success_rt = (_n_complete / _n_total * 100) if _n_total > 0 else 0.0
    _best_val   = study.best_value if _n_complete > 0 else float("nan")
    logger.info(
        "\n%s\n  HPO HEALTH SUMMARY (Optuna)\n%s\n"
        "  Total trials launched : %d\n"
        "  Completed             : %d\n"
        "  Pruned                : %d\n"
        "  Failed / errored      : %d\n"
        "  Success rate          : %.1f%%\n"
        "  Best %-20s: %.4f\n%s",
        "=" * 42, "=" * 42,
        _n_total, _n_complete, _n_pruned, _n_failed, _success_rt,
        metric, _best_val, "=" * 42,
    )
    if _success_rt < 80.0 and _n_total > 0:
        logger.warning(
            "High HPO failure rate (%.1f%% success, %d/%d trials).  "
            "Likely causes: GPU OOM, file descriptor exhaustion, NaN loss, or "
            "checkpoint errors.  Check trial logs above for details.  "
            "Tip: pass storage_path='/tmp/hpo_<name>.log' if output_dir is on Dropbox/cloud.",
            _success_rt, _n_complete, _n_total,
        )

    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    if n_completed == 0:
        raise RuntimeError(
            f"All {len(study.trials)} Optuna trials failed — no completed trial to "
            "select best hyperparameters from.  Check the trial exception logs above "
            "(typically GPU OOM, tokenizer errors, or NaN loss).  "
            f"Trial artifacts are in: {output_dir}"
        )

    best_hyperparams = decode_hyperparams_for_space(study.best_params, active_space)

    # --- Copy best trial artifacts (opt-in) ---
    # Default path: trials ran in tempdirs, so there is nothing to copy — the
    # best trial's config lives in best_hyperparams.json + the JournalStorage log,
    # and the wrapper retrains final/ from it.  Only when export_best_trial=True did
    # trials persist in trial_*/ dirs, so copy the best one out for direct reuse.
    if export_best_trial:
        best_trial_dir = output_dir / f"trial_{study.best_trial.number}"
        best_output_dir = output_dir / "best_trial"
        if best_trial_dir.exists():
            # Remove stale artifacts from any previous run before copying to avoid
            # silently mixing weights from two different trials when resuming.
            if best_output_dir.exists():
                shutil.rmtree(best_output_dir)
            shutil.copytree(best_trial_dir, best_output_dir)
        else:
            logger.warning(
                "Best trial directory not found at %s — best_trial/ not updated.", best_trial_dir
            )

    with open(output_dir / "best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f, indent=2)

    logger.info("Best hyperparams: %s", best_hyperparams)
    logger.info("Best %s: %.4f", metric, study.best_value)

    # --- Save all trial results as CSV ---
    if save_trials_csv:
        trials_csv_path = output_dir / "hpo_trials.csv"
        decode_trials_dataframe_for_space(
            study.trials_dataframe(), active_space
        ).to_csv(trials_csv_path, index=False)
        logger.info("Saved trial results to %s", trials_csv_path)

    # --- Remove per-trial directories (only present when export_best_trial=True) ---
    # In the default path trials ran in tempdirs, so no trial_*/ dirs exist here and
    # this is a no-op.  When export_best_trial=True the best is already copied to
    # best_trial/, so the remaining trial_*/ dirs serve no further purpose.
    # The JournalStorage log is always kept for resumability / further Optuna analysis.
    if cleanup_trials:
        removed = 0
        for trial_dir in sorted(output_dir.iterdir()):
            if (
                trial_dir.is_dir()
                and trial_dir.name.startswith("trial_")
                and trial_dir.name.split("_")[-1].isdigit()
            ):
                shutil.rmtree(trial_dir)
                removed += 1
        if removed:
            logger.info("Removed %d trial directories from %s.", removed, output_dir)

    runtime_tracker.update_metadata(
        total_trials=_n_total,
        completed_trials=_n_complete,
        pruned_trials=_n_pruned,
        failed_trials=_n_failed,
        best_value=_best_val,
        best_hyperparams=best_hyperparams,
        best_trial_number=study.best_trial.number,
        trials_csv_path=(str(output_dir / "hpo_trials.csv") if save_trials_csv else None),
        best_hyperparams_path=str(output_dir / "best_hyperparams.json"),
        runtime_report=str(runtime_tracker.report_path),
        runtime_stage_csv=str(runtime_tracker.stage_csv_path),
    )

    return best_hyperparams, study


# ---------------------------------------------------------------------------
# Ray Tune — distributed HPO across multiple GPUs / cluster nodes
# ---------------------------------------------------------------------------

# -----------------------
# Ray Trainable: Single trial
# -----------------------
def _ray_trial_fn(
    config: dict,
    *,
    label2id: dict,
    id2label: dict,
    model_name: str,
    use_lora: bool,
    use_focal: bool,
    gradient_checkpointing: bool,
    early_stopping_patience: int | None,
    random_state: int,
    use_fast: bool = True,
    opt_train=None,
    opt_val=None,
    train_dataset=None,
    cv_splits: "list[tuple[list[int], list[int]]] | None" = None,
    metric_key: str = "f1_macro",
) -> dict:
    """Single trial for Ray Tune — fully compatible with Ray 2.x / Python 3.13.

    Returns a metrics dict, which Ray Tune records as the trial's final result.
    Returning a dict is the portable way to report metrics from a function
    trainable — it works across all Ray 2.x versions without requiring a
    Ray Train session (``ray.train.report()`` is the Train API and raises /
    hangs when called outside a proper Train context such as TorchTrainer).

    Two scoring modes, bound by :func:`optimize_text_ray` through
    ``tune.with_parameters()`` — the same two modes
    :func:`_tabular_ray_trial_fn` supports:

    - **fixed holdout** (``opt_train``, ``opt_val``): one training run per trial.
    - **k-fold CV** (``train_dataset``, ``cv_splits``): one training run per fold,
      the trial's score being the mean across folds. Per-fold scores and the
      across-fold standard deviation are returned as extra keys so they reach
      the trials CSV, matching the columns the Optuna backend writes.

    Unlike the Optuna CV path there is **no inter-fold pruning** here. Reporting
    an intermediate result per fold would mean calling ``ray.train.report()``,
    which raises or hangs outside a Ray Train session (see the note above), and
    a hang on a cluster costs the whole allocation rather than one trial. Use
    ``use_asha=True`` for early termination across trials instead;
    ``optimize_tabular_ray``'s CV path makes the same trade.

    Imports of ``multimodalva`` sub-modules are deferred to function body so
    that Ray worker processes resolve them correctly regardless of how the
    package was launched (installed wheel, editable install, or direct script).
    """
    import logging as _logging
    import torch

    _trial_logger = _logging.getLogger(__name__)

    # --- Ensure multimodalva is importable in the worker process ---
    import sys as _sys
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    from torch.utils.data import Subset

    from multimodalva.text.train import train_text
    from multimodalva.text.predict import predict_text
    from multimodalva.text.hpo import _trial_workdir
    from multimodalva.utils.metrics import score_predictions, log_loss_from_full
    from multimodalva.utils.ray_compat import get_ray_trial_dir

    use_cv_mode = bool(cv_splits)
    if use_cv_mode:
        if train_dataset is None:
            raise ValueError("cv_splits mode requires train_dataset.")
    elif opt_train is None or opt_val is None:
        raise ValueError(
            "Holdout mode requires opt_train and opt_val; pass cv_splits + "
            "train_dataset for k-fold scoring."
        )

    # --- Trial artifact directory (Ray Tune API compatibility shim) ---
    trial_dir = get_ray_trial_dir()
    _trial_logger.info("Resolved Ray trial artifact directory: %s", trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    if use_focal:
        config = {**config, "loss_type": "focal"}

    all_scores: dict[str, float] = {
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "csmf_accuracy": 0.0,
        "log_loss": float("inf"),
    }
    trial_started = time.perf_counter()
    trial_failed = False

    try:
        if use_cv_mode:
            # ── k-fold CV path ──────────────────────────────────────────────
            # Each fold trains a fresh model; no state carries across folds.
            # val_size=None + early_stopping_patience=None: train the full
            # sampled epoch count so the `epochs` HP is meaningful and folds are
            # comparable.  Identical to the Optuna CV path, minus the inter-fold
            # pruning (see the docstring).
            _fold_scores: dict[str, list[float]] = {m: [] for m in CV_METRICS}
            _fold_log_losses: list[float] = []

            for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(cv_splits):
                fold_train = Subset(train_dataset, cv_train_idx)
                fold_val   = Subset(train_dataset, cv_val_idx)

                # Train into an auto-deleted tempdir: only the score is needed,
                # and k model copies per trial under ray_experiment/ would dwarf
                # everything else on disk.
                with _trial_workdir(None) as fold_dir:
                    train_text(
                        train_dataset=fold_train,
                        label2id=label2id,
                        id2label=id2label,
                        model_name=model_name,
                        output_dir=fold_dir,
                        hyperparams=config,
                        val_size=None,                 # full fold-train; no internal split
                        use_lora=use_lora,
                        gradient_checkpointing=gradient_checkpointing,
                        early_stopping_patience=None,  # no early stopping within a fold
                        resume=False,
                        cleanup_checkpoints=True,
                        save_total_limit=1,
                        use_fast=use_fast,
                        random_state=random_state,
                    )
                    fold_result = predict_text(fold_dir, fold_val, use_fast=use_fast)
                    for m in CV_METRICS:
                        _fold_scores[m].append(score_predictions(fold_result.top1, m))
                    _fold_log_losses.append(
                        log_loss_from_full(fold_result.full, fold_result.id2label)
                    )

                # Per-fold score for the target metric — returned as its own key
                # so it lands in the trials CSV, as the Optuna backend's
                # fold_<i>_<metric> user attrs do.
                all_scores[f"fold_{fold_idx}_{metric_key}"] = _fold_scores[metric_key][-1]
                empty_accelerator_cache()

            all_scores.update(
                {m: float(np.mean(v)) for m, v in _fold_scores.items()}
            )
            all_scores["log_loss"] = float(np.mean(_fold_log_losses))
            all_scores[f"cv_std_{metric_key}"] = float(np.std(_fold_scores[metric_key]))
            all_scores["n_cv_folds"] = len(cv_splits)

        else:
            # ── Fixed holdout path ─────────────────────────────────────────
            train_text(
                train_dataset=opt_train,
                label2id=label2id,
                id2label=id2label,
                model_name=model_name,
                output_dir=trial_dir,
                hyperparams=config,
                val_size=0.1,
                use_lora=use_lora,
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                resume=False,
                cleanup_checkpoints=True,  # trials only need a score, not checkpoints
                save_total_limit=1,        # minimise per-trial disk under ray_experiment/
                use_fast=use_fast,
                random_state=random_state,
            )

            # --- Evaluate ---
            result = predict_text(trial_dir, opt_val, use_fast=use_fast)
            all_scores.update(
                {
                    m: score_predictions(result.top1, m)
                    for m in CV_METRICS
                }
            )
            all_scores["log_loss"] = log_loss_from_full(result.full, result.id2label)

    except Exception:
        trial_failed = True
        _trial_logger.exception(
            "Ray trial failed (config=%s); reporting zero/inf scores.", config
        )

    finally:
        # Free GPU / MPS memory before the next trial.
        empty_accelerator_cache()

    # Return the metrics dict — Ray Tune treats the return value of a function
    # trainable as the trial's final reported result.  This avoids calling
    # ray.train.report(), which is the Ray Train API and raises or hangs when
    # invoked outside a Train session (e.g. TorchTrainer).
    all_scores["elapsed_seconds"] = round(time.perf_counter() - trial_started, 3)
    _trial_logger.info(
        "Ray trial %s in %.2fs%s",
        "failed" if trial_failed else "completed",
        all_scores["elapsed_seconds"],
        (
            f" — accuracy={all_scores['accuracy']:.4f}, {config}"
            if not trial_failed
            else f" — {config}"
        ),
    )
    return all_scores


# -----------------------
# Main Ray HPO function
# -----------------------
def optimize_text_ray(
    train_dataset,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int | None = None,
    metric: str = "f1_macro",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    split_seed: int | None = None,
    use_lora: bool = False,
    use_focal: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = 2,
    ray_address: str | None = None,
    num_gpus_per_trial: float = 0.0,
    num_cpus_per_trial: int = 4,
    max_concurrent_trials: int | None = None,
    resume: bool = True,
    experiment_name: str | None = None,
    save_trials_csv: bool = True,
    cleanup_trials: bool = True,
    export_best_trial: bool = False,
    use_asha: bool = False,
    use_fast: bool = True,
    # ── CV options ────────────────────────────────────────────────────────────
    use_cv: bool = True,
    n_cv_folds: int = 3,
) -> tuple[dict, Any]:
    """Distributed HPO using Ray Tune.

    Builds its search space exactly as :func:`optimize_text` does, and scores
    trials the same two ways: ``use_cv=True`` (default) runs stratified k-fold CV
    per trial, ``use_cv=False`` scores on one holdout split.

    Extra args vs the original signature:

    Args:
        use_cv:          Score each trial by stratified k-fold CV over
                         ``train_dataset`` rather than one holdout split.
                         Steadier, and costs ``n_cv_folds`` times more per
                         trial — which is what the cluster is for. Default True,
                         matching :func:`optimize_text` and
                         :func:`optimize_tabular_ray`. Folds are pre-built once
                         so every trial sees the same splits, so trial
                         differences reflect hyperparameters only.
                         Unlike the Optuna backend there is no inter-fold
                         pruning; use ``use_asha=True`` for early termination
                         across trials (see :func:`_ray_trial_fn`).
        n_cv_folds:      Folds used when ``use_cv=True``. Default 3.
        split_seed:      Seed for the validation split this function carves,
                         or for the CV fold draw when ``use_cv=True``, as in
                         :func:`optimize_text`. ``None`` (default) falls
                         back to ``random_state``.
        n_trials:        Total number of trials. Defaults to 60 when
                         ``use_asha=True`` (ASHA prunes many trials early so
                         more are needed to saturate the search), or 30 otherwise.
                         Override by passing an explicit integer.
        resume:          If ``True`` (default) and a previous experiment exists
                         at ``output_dir/ray_experiment/<experiment_name>``,
                         restore it and **restart** any errored trials from
                         scratch with the same hyperparameter configs.
                         Unfinished (interrupted) trials are also continued.
                         Set to ``False`` to always start a fresh search.
        experiment_name: Name used as the Ray experiment directory inside
                         ``output_dir/ray_experiment/``.  Defaults to
                         ``"ray_hpo_<model_name>"``.  Use a fixed name to
                         reliably find the experiment on the next run.
        use_asha:        Use the ASHA (Asynchronous Successive Halving) scheduler
                         to terminate unpromising trials early based on epoch
                         count.  Increases throughput when many trials are run.
                         ``grace_period=2`` ensures every trial trains at least
                         2 epochs before being eligible for pruning.
                         Default False (OptunaSearch with no scheduler).
                         Note: ASHA prunes trials based on reported intermediate
                         results; since trials report only a final result (not
                         per-epoch), ASHA acts as successive halving over
                         completed trial scores rather than within-trial early
                         stopping.
    """
    _VALID_METRICS = HPO_METRICS
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid metrics: {sorted(_VALID_METRICS)}"
        )

    try:
        import ray
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.search import ConcurrencyLimiter
    except ImportError as exc:
        raise ImportError(
            "ray[tune] and optuna-integration are required for optimize_text_ray(). "
            "They live in an optional extra, so a normal install does not have them:\n"
            "    pip install 'multimodalva[ray]'\n"
            "Or search on the Optuna backend, which needs nothing extra:\n"
            "    Optimize(backend=\"optuna\")   # or backend=\"auto\""
        ) from exc

    # --- Resolve n_trials default (depends on use_asha) ---
    if n_trials is None:
        n_trials = 60 if use_asha else 30

    # Resolve once so every Ray worker loads the same checkpoint without re-downloading.
    model_name = resolve_model_name(model_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_tracker = RuntimeTracker(
        output_dir,
        report_name="hpo_runtime.json",
        metadata={
            "pipeline": "text_hpo_ray",
            "model_name": model_name,
            "metric": metric,
            "n_trials_requested": n_trials,
            "use_lora": use_lora,
            "use_focal": use_focal,
            "gradient_checkpointing": gradient_checkpointing,
            "resume": resume,
            "num_gpus_per_trial": num_gpus_per_trial,
            "num_cpus_per_trial": num_cpus_per_trial,
            "max_concurrent_trials": max_concurrent_trials,
        },
        logger_=logger,
    )

    # --- Auto-redirect to optimize_text() on MPS/CPU-only ---
    # Ray Train v2 is incompatible with function trainables on non-CUDA devices:
    # RunConfig auto-injects checkpoint_at_end=True which raises ValueError,
    # forcing a no-RunConfig fallback where trial state goes to ~/ray_results
    # and cannot be restored → resume always restarts from trial 1.
    # On MPS/CPU, optimize_text() (Optuna + JournalStorage) is the correct backend.
    try:
        import torch as _torch
        _cuda_available = _torch.cuda.device_count() > 0
    except ImportError:
        _cuda_available = False
    if not _cuda_available:
        _safe = model_name.replace("/", "_")
        logger.info(
            "optimize_text_ray(): no CUDA GPUs detected — redirecting to optimize_text() "
            "(Optuna sequential, JournalStorage-backed). Ray Tune requires CUDA for "
            "reliable experiment persistence and resume. "
            "Journal log: %s/hpo_%s.log",
            output_dir, _safe,
        )
        return optimize_text(
            train_dataset=train_dataset,
            label2id=label2id,
            id2label=id2label,
            model_name=model_name,
            output_dir=output_dir,
            n_trials=n_trials,
            metric=metric,
            search_space=search_space,
            val_size=val_size,
            random_state=random_state,
            use_lora=use_lora,
            use_focal=use_focal,
            gradient_checkpointing=gradient_checkpointing,
            early_stopping_patience=early_stopping_patience,
            storage_path=str(output_dir / f"hpo_{_safe}.log"),
            load_if_exists=resume,
            save_trials_csv=save_trials_csv,
            cleanup_trials=cleanup_trials,
            use_fast=use_fast,
        )

    # --- Experiment storage path (for resume) ---
    safe_name = model_name.replace("/", "_")
    exp_name    = experiment_name or f"ray_hpo_{safe_name}"
    exp_storage = output_dir / "ray_experiment"
    exp_path    = exp_storage / exp_name

    # --- Build effective search space: data-adaptive base → LoRA/focal merges → caller overrides ---
    n_samples = len(train_dataset)
    n_classes = len(id2label)
    active_space = get_text_default_search_space(n_samples, n_classes)
    logger.info(
        "Text HPO adaptive search space: sample_tier=%s, class_tier=%s "
        "(n_samples=%d, n_classes=%d).",
        _sample_tier(n_samples), get_class_tier(n_classes), n_samples, n_classes,
    )
    if use_lora:
        active_space.update(LORA_SEARCH_SPACE)
    if use_focal:
        active_space.update(FOCAL_SEARCH_SPACE)
    merge_search_space(active_space, search_space, logger)

    # Drop classifier_dropout for architectures that don't support it (e.g. Longformer).
    if "classifier_dropout" in active_space:
        import inspect as _inspect
        from transformers import AutoConfig as _AutoConfig
        _cfg_cls = type(_AutoConfig.from_pretrained(model_name))
        if "classifier_dropout" not in _inspect.signature(_cfg_cls.__init__).parameters:
            active_space.pop("classifier_dropout")
            logger.info(
                "Removed 'classifier_dropout' from Ray HPO search space: "
                "%s (%s) does not support this parameter.",
                model_name, _cfg_cls.__name__,
            )

    ray_space = to_ray_space(active_space)

    # --- Data split setup ---
    if use_cv and n_cv_folds < 2:
        raise ValueError("n_cv_folds must be >= 2 when use_cv=True.")

    all_labels = _get_dataset_labels(train_dataset)
    _split_seed = split_seed if split_seed is not None else random_state
    opt_train = opt_val = None
    cv_splits: list[tuple[list[int], list[int]]] | None = None

    if use_cv:
        # Pre-build the folds once, exactly as optimize_text() does, so every
        # trial — and every worker — shares the same splits. Fold-assignment
        # variance would otherwise be indistinguishable from a hyperparameter
        # effect in the search signal.
        _, _class_counts = np.unique(all_labels, return_counts=True)
        _min_class_count = int(_class_counts.min())
        if n_cv_folds > _min_class_count:
            raise ValueError(
                f"n_cv_folds={n_cv_folds} is greater than the minimum class count "
                f"({_min_class_count}) in train_dataset. Reduce n_cv_folds or "
                "rebalance the data."
            )
        _skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=_split_seed)
        cv_splits = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in _skf.split(list(range(len(train_dataset))), all_labels)
        ]
        logger.info(
            "CV HPO (Ray): %d-fold stratified splits pre-built "
            "(n_samples=%d, n_classes=%d).  Each trial runs %d training jobs.",
            n_cv_folds, len(all_labels), len(id2label), n_cv_folds,
        )
    else:
        opt_train_idx, opt_val_idx = stratified_indices(
            all_labels,
            val_size,
            _split_seed,
            what="validation",
            label_names=id2label,
            log=logger,
        )
        opt_train = Subset(train_dataset, opt_train_idx)
        opt_val = Subset(train_dataset, opt_val_idx)

    # --- Initialise Ray ---
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)
        logger.info("Ray initialised. Cluster resources: %s", ray.cluster_resources())
    elif ray_address is not None:
        logger.warning("Ray already initialised; ignoring ray_address=%r", ray_address)

    available_gpus = ray.cluster_resources().get("GPU", 0)
    runtime_tracker.update_metadata(available_gpus=available_gpus, ray_address=ray_address)
    if num_gpus_per_trial > 0 and available_gpus == 0:
        logger.warning(
            "num_gpus_per_trial=%.1f but no GPUs detected; trials will queue indefinitely.",
            num_gpus_per_trial,
        )

    # Warn early on MPS/CPU-only: Ray Train v2 is incompatible with function
    # trainables when RunConfig is used, causing checkpoint_at_end ValueError.
    # This means experiment state cannot be persisted → resume is impossible.
    # The fallback (tuner_no_rc) saves to ~/ray_results which we cannot restore.
    # Users on Apple Silicon or CPU-only machines should use optimize_text() instead.
    if available_gpus == 0:
        logger.warning(
            "optimize_text_ray(): no CUDA GPUs available (MPS/CPU-only environment). "
            "Ray Train v2 is incompatible with function trainables on non-CUDA "
            "devices — experiment state cannot be persisted and resume will not "
            "work across restarts. "
            "RECOMMENDATION: use optimize_text() (Optuna backend) instead — it uses "
            "JournalStorage for reliable crash recovery and resume on MPS/CPU machines.",
        )

    _ray_mode = "min" if METRIC_DIRECTION.get(metric, "maximize") == "minimize" else "max"

    optuna_search_kwargs = {
        "metric": metric,
        "mode": _ray_mode,
        "seed": random_state,
    }
    # Ray Tune changed OptunaSearch's constructor across releases.
    # Older cluster environments reject `storage`, while newer ones accept it.
    if "storage" in inspect.signature(OptunaSearch.__init__).parameters:
        optuna_search_kwargs["storage"] = None
    search_alg = OptunaSearch(**optuna_search_kwargs)
    if max_concurrent_trials is not None:
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent_trials)

    # --- ASHA scheduler (optional) ---
    # ASHAScheduler terminates unpromising trials based on successive halving.
    # max_t is set to the maximum epochs value in the active search space so
    # ASHA's budget matches the longest possible trial.
    # Note: since trials report only a final score (not per-epoch intermediate
    # results), ASHA acts as successive halving over completed trial scores
    # rather than within-trial early stopping.
    _scheduler = None
    if use_asha:
        from ray.tune.schedulers import ASHAScheduler
        _epochs_spec = active_space.get("epochs", ("categorical", [3, 5, 8]))
        _max_t = max(_epochs_spec[1])  # e.g. max([3, 5, 8]) = 8
        _scheduler = ASHAScheduler(
            max_t=_max_t,
            grace_period=2,       # every trial trains at least 2 epochs before pruning
            reduction_factor=2,   # keep top 50% at each halving bracket
            brackets=1,
        )
        logger.info("ASHA scheduler enabled (max_t=%d epochs, grace_period=2).", _max_t)

    # --- Trainable ---
    # Datasets are bound through the Ray object store by with_parameters(); in CV
    # mode the whole train_dataset is stored once and each worker Subsets it from
    # the shared fold indices, rather than k datasets being shipped per trial.
    _bound: dict = dict(
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        use_lora=use_lora,
        use_focal=use_focal,
        gradient_checkpointing=gradient_checkpointing,
        early_stopping_patience=early_stopping_patience,
        random_state=random_state,
        use_fast=use_fast,
        metric_key=metric,
    )
    if use_cv:
        _bound.update(train_dataset=train_dataset, cv_splits=cv_splits)
    else:
        _bound.update(opt_train=opt_train, opt_val=opt_val)
    trainable = tune.with_parameters(_ray_trial_fn, **_bound)
    trainable = tune.with_resources(
        trainable, resources={"cpu": num_cpus_per_trial, "gpu": num_gpus_per_trial}
    )

    # --- RunConfig for experiment persistence (enables Tuner.restore()) ---
    # Wrapped in try/except: some Ray versions raise checkpoint_at_end ValueError
    # when RunConfig is used with function trainables.  If that happens, we fall
    # back to no run_config (losing resume capability for this run).
    # We try two variants:
    #   1. RunConfig + CheckpointConfig(checkpoint_at_end=False) — prevents the
    #      auto-injection of checkpoint_at_end=True in Ray Train v2 which would
    #      raise ValueError("checkpoint_at_end=True is not supported for function
    #      trainables").  This is the preferred path.
    #   2. Plain RunConfig — fallback for older Ray versions that don't have
    #      CheckpointConfig or don't inject checkpoint_at_end.
    _run_config = None
    try:
        from ray.train import RunConfig, CheckpointConfig
        exp_storage.mkdir(parents=True, exist_ok=True)
        _run_config = RunConfig(
            storage_path=str(exp_storage.resolve()),
            name=exp_name,
            checkpoint_config=CheckpointConfig(checkpoint_at_end=False),
        )
    except ImportError:
        # CheckpointConfig not available — try plain RunConfig
        try:
            from ray.train import RunConfig  # noqa: F811
            exp_storage.mkdir(parents=True, exist_ok=True)
            _run_config = RunConfig(
                storage_path=str(exp_storage.resolve()),
                name=exp_name,
            )
        except Exception as _rc_err2:
            logger.warning(
                "Could not create RunConfig — experiment state will not be persisted "
                "(resume disabled for this run): %s", _rc_err2,
            )
    except Exception as _rc_err:
        logger.warning(
            "Could not create RunConfig — experiment state will not be persisted "
            "(resume disabled for this run): %s", _rc_err,
        )

    _tune_config = tune.TuneConfig(
        metric=metric,
        mode=_ray_mode,
        num_samples=n_trials,
        search_alg=search_alg,
        max_concurrent_trials=max_concurrent_trials,
        scheduler=_scheduler,
    )

    # --- Build or restore Tuner ---
    def _fresh_tuner() -> "tune.Tuner":
        kwargs: dict = dict(param_space=ray_space, tune_config=_tune_config)
        if _run_config is not None:
            kwargs["run_config"] = _run_config
        return tune.Tuner(trainable, **kwargs)

    if resume and exp_path.exists():
        logger.info(
            "resume=True and experiment found at %s — restoring "
            "(errored trials will be restarted).", exp_path,
        )
        try:
            tuner = tune.Tuner.restore(
                str(exp_path),
                trainable=trainable,
                restart_errored=True,
                resume_unfinished=True,
                param_space=ray_space,
            )
            logger.info("Experiment restored successfully.")
        except Exception as _restore_err:
            logger.warning(
                "Failed to restore experiment (%s); starting fresh.", _restore_err,
            )
            tuner = _fresh_tuner()
    else:
        if resume:
            logger.info(
                "resume=True but no experiment found at %s — starting fresh "
                "(re-run with resume=True after an interruption to continue).", exp_path,
            )
        tuner = _fresh_tuner()

    logger.info(
        "Launching %d trials, metric=%s, scoring=%s, gpus/trial=%.1f, cpus/trial=%d "
        "(%d training job(s) per trial)",
        n_trials, metric,
        f"{n_cv_folds}-fold CV" if use_cv else f"holdout {val_size:.0%}",
        num_gpus_per_trial, num_cpus_per_trial,
        n_cv_folds if use_cv else 1,
    )

    # AutoGluon sets RAY_AIR_NEW_OUTPUT=1, which changes Ray's verbose handling.
    # Ray internally passes verbose="auto" (a string) but the new output path
    # in get_air_verbosity() expects int or AirVerbosity enum → AttributeError.
    # Force RAY_AIR_NEW_OUTPUT=0 for our HPO run to use the stable output path,
    # then restore the caller's value so AutoGluon's own runs are unaffected.
    # RAY_AIR_LOCAL_CACHE_DIR is deprecated in newer Ray versions; if it is set
    # in the HPC environment, Ray raises a DeprecationWarning *as an exception*
    # inside tune.run(), aborting the run.  Unset it for our call and restore.
    _prev_ray_output   = os.environ.get("RAY_AIR_NEW_OUTPUT")
    _prev_ray_cache    = os.environ.pop("RAY_AIR_LOCAL_CACHE_DIR", None)
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    try:
        with runtime_tracker.stage(
            "ray_tuner_fit",
            details={
                "experiment_name": exp_name,
                "resume": resume,
                "n_trials": n_trials,
                "metric": metric,
                "use_cv": use_cv,
                "n_cv_folds": n_cv_folds if use_cv else None,
            },
            monitor_gpu=True,
        ):
            try:
                results = tuner.fit()
            except ValueError as _ve:
                if "checkpoint_at_end" in str(_ve):
                    # Ray Train v2 incompatibility: RunConfig auto-injects
                    # checkpoint_at_end=True for function trainables, which is
                    # unsupported and raises ValueError.  This typically occurs on
                    # MPS/CPU-only machines where CheckpointConfig(checkpoint_at_end=
                    # False) is not respected by the installed Ray version.
                    # In this fallback mode, trials run via tuner_no_rc and their
                    # state is saved to ~/ray_results — NOT to our exp_path — so
                    # resume via Tuner.restore() is impossible across restarts.
                    # On MPS/Apple Silicon: switch to optimize_text() (Optuna) for
                    # reliable resume.  optimize_text() uses JournalStorage and resumes
                    # correctly after crashes or SLURM preemptions.
                    logger.warning(
                        "checkpoint_at_end error from Ray Train v2 (%s). "
                        "Retrying without RunConfig — this run's trial state will be "
                        "saved to ~/ray_results (not %s) and CANNOT be resumed. "
                        "If you need resume support, use optimize_text() (Optuna backend) "
                        "which persists state in JournalStorage and resumes correctly on "
                        "MPS/CPU-only machines.", _ve, exp_path,
                    )
                    # Clean up the stale exp_path: tune.Tuner() constructor wrote
                    # tuner.pkl and .validate_storage_marker to exp_path before
                    # fit() raised.  If left in place, a subsequent resume=True run
                    # finds exp_path, calls Tuner.restore(), sees 0 completed trials
                    # (they ran via tuner_no_rc to ~/ray_results), and silently
                    # re-runs all N trials from scratch — identical to "starts from
                    # trial 1".  Deleting exp_path makes the next run start honestly
                    # fresh rather than triggering this misleading restore loop.
                    if exp_path.exists():
                        shutil.rmtree(exp_path, ignore_errors=True)
                        logger.warning(
                            "Removed stale exp_path %s (tuner.pkl written before "
                            "fit() raised — no trial state was persisted there).",
                            exp_path,
                        )
                    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
                    tuner_no_rc = tune.Tuner(
                        trainable, param_space=ray_space, tune_config=_tune_config,
                    )
                    results = tuner_no_rc.fit()
                else:
                    raise
            except Exception as _fit_err:
                _fit_err_text = str(_fit_err)
                _is_optuna_restore_mismatch = (
                    resume
                    and exp_path.exists()
                    and "optuna_search.py" in _fit_err_text
                    and "KeyError" in _fit_err_text
                    and "_ot_trials" in _fit_err_text
                )
                if _is_optuna_restore_mismatch:
                    logger.warning(
                        "Detected incompatible/corrupted Ray Tune restore state for "
                        "OptunaSearch (%s). Removing stale experiment dir %s and "
                        "restarting this Ray HPO run fresh. Prior in-progress Ray "
                        "resume state cannot be recovered on this run.",
                        _fit_err, exp_path,
                    )
                    shutil.rmtree(exp_path, ignore_errors=True)
                    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
                    results = _fresh_tuner().fit()
                else:
                    raise
    finally:
        if _prev_ray_output is None:
            os.environ.pop("RAY_AIR_NEW_OUTPUT", None)
        else:
            os.environ["RAY_AIR_NEW_OUTPUT"] = _prev_ray_output
        if _prev_ray_cache is not None:
            os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = _prev_ray_cache

    # --- HPO Health Check (Ray) ---
    _ray_df     = results.get_dataframe()
    _n_total_r  = len(_ray_df)
    _n_errors_r = int(_ray_df["error"].notnull().sum()) if "error" in _ray_df.columns else 0
    _n_ok_r     = _n_total_r - _n_errors_r
    _succ_rt_r  = (_n_ok_r / _n_total_r * 100) if _n_total_r > 0 else 0.0
    logger.info(
        "\n%s\n  HPO HEALTH SUMMARY (Ray Tune)\n%s\n"
        "  Total trials launched : %d\n"
        "  Successful / pruned   : %d\n"
        "  System errors         : %d\n"
        "  Success rate          : %.1f%%\n%s",
        "=" * 42, "=" * 42,
        _n_total_r, _n_ok_r, _n_errors_r, _succ_rt_r, "=" * 42,
    )
    if _succ_rt_r < 80.0 and _n_total_r > 0:
        logger.warning(
            "High HPO failure rate (%.1f%% success, %d/%d trials).  "
            "Likely causes: GPU OOM, file descriptor limit, or Ray worker errors.  "
            "Consider: reduce batch_size, set num_gpus_per_trial=0 for CPU-only, "
            "or pass storage_path='/tmp/hpo_<name>.log' to move the journal off cloud storage.",
            _succ_rt_r, _n_ok_r, _n_total_r,
        )

    _n_ok_ray = sum(1 for r in results if not r.error)
    if _n_ok_ray == 0:
        raise RuntimeError(
            f"All {len(results)} Ray Tune trials errored — no completed trial to "
            "select best hyperparameters from.  Check the trial logs above "
            "(typically GPU OOM, tokenizer errors, or NaN loss).  "
            f"Trial artifacts are in: {exp_storage}"
        )

    best_result = results.get_best_result(metric=metric, mode=_ray_mode)
    best_hyperparams = best_result.config

    # Canonical names, the same ones optimize_text() writes. They used to carry a
    # "_ray" suffix, which was harmless while Ray was only reachable from this
    # function and no reader looked for it. It stopped being harmless when
    # Optimize(backend="ray") made the backend a switch: the run contract names
    # hpo/best_hyperparams.json, results/validation.py reads hpo_trials.csv to
    # produce validation.json's hpo_cv score, and the LOPO / sample-size scripts
    # load best_hyperparams.json and feed it straight back as hyperparams=. With
    # the suffix, switching backend silently produced a run missing all three.
    # Readers still accept the legacy "_ray" spellings for runs already on disk.
    with open(output_dir / "best_hyperparams.json", "w") as fh:
        json.dump(best_hyperparams, fh, indent=2)

    # --- Copy best trial artifacts (opt-in) ---
    # best_hyperparams.json + the trials CSV already capture everything needed
    # to retrain; only persist the best trial's weights when explicitly requested.
    best_trial_path = Path(best_result.path)
    if export_best_trial:
        best_output_dir = output_dir / "best_trial"
        if best_trial_path.exists():
            # Remove stale artifacts from any previous run before copying.
            if best_output_dir.exists():
                shutil.rmtree(best_output_dir)
            shutil.copytree(best_trial_path, best_output_dir)
        else:
            logger.warning(
                "Best trial path not found at %s — best_trial/ not updated.", best_trial_path
            )

    # --- Save trial CSV ---
    if save_trials_csv:
        results.get_dataframe().to_csv(output_dir / "hpo_trials.csv", index=False)

    # --- Remove Ray experiment directory (optional) ---
    # The ray_experiment/ tree holds Ray's per-trial artifacts (model weights,
    # optimizer state, Ray metadata).  After a successful HPO run the best trial
    # is already in best_trial/ and all metrics are in hpo_trials.csv, so
    # the experiment tree is no longer needed and can be several GB for text models.
    if cleanup_trials and exp_storage.exists():
        shutil.rmtree(exp_storage)
        logger.info("Removed Ray experiment dir: %s", exp_storage)

    runtime_tracker.update_metadata(
        total_trials=_n_total_r,
        successful_or_pruned_trials=_n_ok_r,
        error_trials=_n_errors_r,
        success_rate=_succ_rt_r,
        best_hyperparams=best_hyperparams,
        best_trial_path=str(best_trial_path),
        best_hyperparams_path=str(output_dir / "best_hyperparams.json"),
        trials_csv_path=(str(output_dir / "hpo_trials.csv") if save_trials_csv else None),
        runtime_report=str(runtime_tracker.report_path),
        runtime_stage_csv=str(runtime_tracker.stage_csv_path),
    )

    return best_hyperparams, results


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------
def search_text(
    search: "Optimize",
    *,
    resume: bool,
    **common: Any,
) -> tuple[dict, Any]:
    """Run one text hyperparameter search on whichever backend ``search`` names.

    The single place the :class:`~multimodalva.utils.optimize_config.Optimize`
    fields are mapped onto :func:`optimize_text` / :func:`optimize_text_ray`, so
    the classifiers and the ensembles do not each carry their own copy of the
    mapping. Callers pass the arguments the two backends share; this function
    adds the ones they spell differently.

    ``Optimize(backend="auto")`` resolves through
    :func:`~multimodalva.utils.optimize_config.resolve_backend` — Ray on a
    CUDA/SLURM machine, Optuna otherwise. A resolved Ray call on a machine with
    no CUDA GPU still redirects itself back to :func:`optimize_text`, so
    ``backend="ray"`` is safe to leave set in a config that also runs on a
    laptop.

    Args:
        search:  The search settings.
        resume:  Already resolved by
                 :func:`~multimodalva.utils.optimize_config.resolve_search_resume`.
                 Spelled ``load_if_exists`` by the Optuna backend and ``resume``
                 by the Ray one.
        **common: Arguments both backends take under the same name
                 (``train_dataset``, ``label2id``, ``id2label``, ``model_name``,
                 ``output_dir``, ``random_state``, ``split_seed``, ``use_lora``,
                 ``use_focal``, ``gradient_checkpointing``,
                 ``early_stopping_patience``, ``use_fast``).

    Returns:
        ``(best_hyperparams, study_or_result_grid)`` — the backend object differs
        (``optuna.Study`` vs ``ray.tune.ResultGrid``), which is why callers store
        it without inspecting it.
    """
    from ..utils.optimize_config import resolve_backend, resolve_pruning

    backend = resolve_backend(search.backend)
    kwargs: dict[str, Any] = dict(
        metric=search.metric,
        search_space=search.space,
        use_cv=search.cv,
        n_cv_folds=search.cv_folds,
        **common,
    )
    # n_trials=None means "this family's default", and the two backends resolve
    # that to different numbers (Ray needs more trials when ASHA prunes). Let
    # each apply its own rather than freezing one here.
    if search.n_trials is not None:
        kwargs["n_trials"] = search.n_trials

    if backend == "ray":
        if search.pruning is not None:
            logger.warning(
                "Optimize(pruning=%s) is ignored by the Ray backend, which has no "
                "inter-trial pruner. Pass extra={'use_asha': True} to terminate "
                "unpromising trials by successive halving instead.",
                search.pruning,
            )
        kwargs["resume"] = resume
        kwargs.update(search.extra)
        return optimize_text_ray(**kwargs)

    kwargs["load_if_exists"] = resume
    kwargs["enable_pruning"] = resolve_pruning(search, TEXT_PRUNING_DEFAULT)
    kwargs.update(search.extra)
    return optimize_text(**kwargs)
