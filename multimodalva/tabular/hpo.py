"""
Step 5 (tabular pipeline): Hyperparameter optimization.

Two backends are provided; both accept the same search_space format and return
``(best_hyperparams, backend_object)`` so callers can switch backends without changing call sites:

    optimize_tabular()      — Optuna (TPE sampler, JournalStorage persistence)
                      Best for single-machine sequential or lightly parallel HPO.

    optimize_tabular_ray()  — Ray Tune (OptunaSearch / TPE, distributed runtime)
                      Best for multi-CPU/GPU machines or multi-node clusters.
                      Runs N trials concurrently; auto-caps n_jobs per trial
                      to prevent CPU contention between concurrent workers.
                      For GPU-accelerated models (catboost, lightgbm, xgboost),
                      set num_gpus_per_trial=1 and use_gpu=True.

Input:  X_train, y_train, label2id, id2label from prepare_tabular_dataset()
Output: best_hyperparams dict and backend study / ResultGrid object
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna
    import ray

    from ..utils.optimize_config import Optimize

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# -----------------------
# Package root resolution (mirrors text/hpo.py)
# -----------------------
# __file__ = multimodalva/tabular/hpo.py  →  parent.parent.parent = project root
# Inserted into sys.path so that "from multimodalva.xxx import ..." resolves
# correctly in direct-script and Ray worker contexts.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # MultimodalVA/
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ..utils.metrics import (  # noqa: F401
    csmf_accuracy, score_predictions, sample_hyperparams,
    log_loss_from_full, METRIC_DIRECTION, decode_hyperparams_for_space,
    decode_trials_dataframe_for_space, CV_METRICS, HPO_METRICS,
)
from ..utils.ray_compat import to_ray_space
from ..utils.hpo_defaults import get_class_tier, merge_search_space
from .predict import predict_tabular
from .train import TABULAR_MODELS, train_tabular

logger = logging.getLogger(__name__)

from .search_spaces import (  # noqa: E402
    TABULAR_DEFAULT_SEARCH_SPACES,
    SEARCH_SPACE_PROFILES,
    _int_spec,
    _float_spec,
    _float_log_spec,
    _categorical_spec,
    _apply_nclasses_adjustments,
    infer_search_space_profile,
    _resolve_search_space_profile,
    _max_features_choices,
    _build_profile_space,
    get_tabular_default_search_space,
)


#: ``enable_pruning``'s default for this family — off, unlike text. A tree-model
#: trial finishes in seconds, so the pruner mostly adds variance to the search
#: for no saving. ``Optimize(pruning=None)`` resolves to this.
TABULAR_PRUNING_DEFAULT = False


def optimize_tabular(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 50,
    metric: str = "f1_macro",
    search_space: dict | None = None,
    val_size: float = 0.2,
    use_cv: bool = True,
    n_cv_folds: int = 3,
    random_state: int = 42,
    split_seed: int | None = None,
    study_name: str = "tabular_hpo",
    storage_path: str | None = None,
    load_if_exists: bool = True,
    enable_pruning: bool = TABULAR_PRUNING_DEFAULT,
    save_trials_csv: bool = True,
    cleanup_trials: bool = True,
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    search_space_profile: str = "auto",
) -> tuple[dict, optuna.Study]:
    """Run Optuna hyperparameter search for a tabular model.

    Uses one of two internal evaluation strategies:
      - use_cv=True: stratified k-fold CV over X_train/y_train; each trial score
        is the mean across folds.
      - use_cv=False: single stratified opt-train / opt-val split.
    The held-out test set from prepare_tabular_dataset() is never used here.

    After all trials complete, the best trial's artifacts are copied to
    output_dir/best_trial/ and best hyperparams saved as best_hyperparams.json.

    Args:
        X_train:        Preprocessed feature matrix from prepare_tabular_dataset().
        y_train:        Integer label array from prepare_tabular_dataset().
        label2id:       Label-to-integer mapping from prepare_tabular_dataset().
        id2label:       Integer-to-label mapping from prepare_tabular_dataset().
        model_name:     Model alias — one of TABULAR_MODELS keys.
        output_dir:     Root directory for trial outputs and study database.
        n_trials:       Total Optuna trials. Default 50.
        metric:         Metric to optimise — "accuracy", "balanced_accuracy",
                        "f1_macro", "f1_weighted", "csmf_accuracy", or "log_loss".
                        Default "f1_macro". Use "balanced_accuracy",
                        "csmf_accuracy", or "f1_macro" for imbalanced VA data.
        search_space:   Custom search space dict. Merged over the adaptive
                        default search space selected for this dataset.
                        Overrides it per key; keys you omit keep
                        their default.
        search_space_profile:
                        One of "auto", "small", "balanced", "wide", "large".
                        "auto" infers a profile from X_train.shape.
        val_size:       Fraction of X_train for trial evaluation. Default 0.2.
                        Ignored when use_cv=True.
        use_cv:         Use stratified k-fold CV for trial scoring. Default True.
        n_cv_folds:     Number of CV folds when use_cv=True. Default 3.
        random_state:   Seed for the Optuna sampler and for each candidate fit.
                        Default 42.
        split_seed:     Seed for the cross-validation folds / internal
                        validation split this function draws. ``None`` (the
                        default) reuses ``random_state``, so existing callers
                        are unaffected. Pass it separately to hold the folds
                        fixed while ``random_state`` reseeds the models.
        study_name:     Optuna study name. Default "tabular_hpo".
        storage_path:   Path to the JournalStorage log file for study persistence.
                        Defaults to output_dir/hpo_<model_name>.log.
                        Accepts a bare path (.log) or legacy sqlite:///… / .db paths
                        — .db paths are auto-redirected to .log with a warning.
        load_if_exists: Resume an existing study. Default True.
        enable_pruning: MedianPruner for early trial termination. Default False
                        (pruning is less useful for fast sklearn fits).
        save_trials_csv: Save all trial results to hpo_trials.csv. Default True.
        cleanup_trials: Delete all per-trial directories (``trial_*/``) after HPO
                        completes.  The best trial is preserved at ``best_trial/``
                        before deletion.  The JournalStorage log, ``best_hyperparams.json``,
                        ``best_trial/``, and the CSV are all kept.  Default True.

    Returns:
        best_hyperparams: Dict from the best trial — pass to train(hyperparams=...).
        study:            Completed Optuna study object.

    Raises:
        ValueError: If model_name is not supported or metric is invalid.
    """
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise ImportError(
            "optuna is required for optimize_tabular(). "
            "Install with:  pip install 'optuna>=3.4'"
        ) from exc

    _VALID_METRICS = HPO_METRICS
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid metrics: {sorted(_VALID_METRICS)}"
        )
    if use_cv and n_cv_folds < 2:
        raise ValueError("n_cv_folds must be >= 2 when use_cv=True.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in TABULAR_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {list(TABULAR_MODELS)}."
        )

    # Build effective search space: data/class-adaptive base → caller overrides
    n_classes = len(id2label)
    resolved_profile = _resolve_search_space_profile(X_train, search_space_profile)
    active_space = get_tabular_default_search_space(
        model_name=model_name,
        X_train=X_train,
        search_space_profile=resolved_profile,
        n_classes=n_classes,
    )
    merge_search_space(active_space, search_space, logger)
    logger.info(
        "Tabular HPO adaptive search space: profile=%s, class_tier=%s "
        "(n_samples=%d, n_features=%d, n_classes=%d).",
        resolved_profile, get_class_tier(n_classes),
        X_train.shape[0], X_train.shape[1], n_classes,
    )

    # JournalStorage (append-only log + fcntl locking) replaces SQLite as the
    # default backend.  It tolerates cloud-sync folders (Dropbox, iCloud) and
    # network filesystems where SQLite's page-lock protocol tends to break.
    if storage_path is None:
        storage_path = str(output_dir.resolve() / f"hpo_{model_name}.log")
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

    # Trial evaluation setup — never uses the held-out test set
    cv_splits: list[tuple[list[int], list[int]]] | None = None
    X_opt_train = y_opt_train = X_opt_val = y_opt_val = None
    # Which rows form each fold is a partitioning decision, so it follows the
    # split seed rather than the seed that reseeds the models.
    _split_seed = split_seed if split_seed is not None else random_state
    if use_cv:
        _, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = int(class_counts.min())
        if n_cv_folds > min_class_count:
            raise ValueError(
                f"n_cv_folds={n_cv_folds} is greater than the minimum class count "
                f"({min_class_count}) in y_train. Reduce n_cv_folds or rebalance data."
            )
        indices = np.arange(len(y_train))
        skf = StratifiedKFold(
            n_splits=n_cv_folds,
            shuffle=True,
            random_state=_split_seed,
        )
        cv_splits = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in skf.split(indices, y_train)
        ]
        logger.info(
            "Tabular CV HPO: %d-fold stratified splits pre-built "
            "(n_samples=%d, n_classes=%d). Each trial runs %d fits.",
            n_cv_folds, len(y_train), len(id2label), n_cv_folds,
        )
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=_split_seed,
        )
        opt_train_idx, opt_val_idx = next(sss.split(X_train, y_train))
        X_opt_train = X_train[opt_train_idx]
        y_opt_train = y_train[opt_train_idx]
        X_opt_val = X_train[opt_val_idx]
        y_opt_val = y_train[opt_val_idx]

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hyperparams(trial, active_space)
        trial_dir = output_dir / f"trial_{trial.number}"
        if use_cv:
            if not cv_splits:
                raise RuntimeError("use_cv=True but no CV splits were prepared.")

            fold_scores: dict[str, list[float]] = {
                m: [] for m in CV_METRICS
            }
            fold_log_losses: list[float] = []
            for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(cv_splits):
                fold_dir = trial_dir / f"fold_{fold_idx}"
                X_fold_train = X_train[cv_train_idx]
                y_fold_train = y_train[cv_train_idx]
                X_fold_val = X_train[cv_val_idx]
                y_fold_val = y_train[cv_val_idx]

                train_tabular(
                    X_fold_train, y_fold_train,
                    label2id=label2id,
                    id2label=id2label,
                    model_name=model_name,
                    output_dir=fold_dir,
                    hyperparams=hp,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    use_gpu=use_gpu,
                )
                fold_result = predict_tabular(fold_dir, X_fold_val, y_fold_val)
                for m in CV_METRICS:
                    fold_scores[m].append(score_predictions(fold_result.top1, m))
                fold_log_losses.append(log_loss_from_full(fold_result.full, fold_result.id2label))
                trial.set_user_attr(f"fold_{fold_idx}_{metric}", fold_scores[metric][-1])

            all_scores = {m: float(np.mean(fold_scores[m])) for m in fold_scores}
            all_scores["log_loss"] = float(np.mean(fold_log_losses))
            trial.set_user_attr(f"cv_std_{metric}", float(np.std(fold_scores[metric])))
        else:
            if X_opt_train is None or y_opt_train is None or X_opt_val is None or y_opt_val is None:
                raise RuntimeError("Holdout split was not prepared for use_cv=False.")

            train_tabular(
                X_opt_train, y_opt_train,
                label2id=label2id,
                id2label=id2label,
                model_name=model_name,
                output_dir=trial_dir,
                hyperparams=hp,
                random_state=random_state,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
            result = predict_tabular(trial_dir, X_opt_val, y_opt_val)
            all_scores = {
                m: score_predictions(result.top1, m)
                for m in CV_METRICS
            }
            all_scores["log_loss"] = log_loss_from_full(result.full, result.id2label)

        for name, val in all_scores.items():
            trial.set_user_attr(name, val)

        return all_scores[metric]

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

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = n_trials - completed
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
            "Likely causes: model fit errors, invalid hyperparameter combinations, "
            "or OOM.  Check trial logs above for details.",
            _success_rt, _n_complete, _n_total,
        )

    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    if n_completed == 0:
        raise RuntimeError(
            f"All {len(study.trials)} Optuna trials failed — no completed trial to "
            "select best hyperparameters from.  Check the trial exception logs above "
            "(typically OOM, invalid hyperparameter combinations, or data errors).  "
            f"Trial artifacts are in: {output_dir}"
        )

    best_hyperparams = decode_hyperparams_for_space(study.best_params, active_space)

    # Copy best trial artifacts — remove stale artifacts first so resume never
    # silently merges weights from two different best trials.
    best_trial_dir = output_dir / f"trial_{study.best_trial.number}"
    best_output_dir = output_dir / "best_trial"
    if best_trial_dir.exists():
        if best_output_dir.exists():
            shutil.rmtree(best_output_dir)
        shutil.copytree(best_trial_dir, best_output_dir)
    else:
        logger.warning(
            "Best trial directory not found at %s — best_trial/ not updated.", best_trial_dir
        )

    with open(output_dir / "best_hyperparams.json", "w") as f:
        json.dump(
            best_hyperparams, f, indent=2,
            default=lambda o: list(o) if isinstance(o, tuple) else str(o),
        )

    logger.info("Best hyperparams: %s", best_hyperparams)
    logger.info("Best %s: %.4f", metric, study.best_value)

    if save_trials_csv:
        trials_csv_path = output_dir / "hpo_trials.csv"
        decode_trials_dataframe_for_space(
            study.trials_dataframe(), active_space
        ).to_csv(trials_csv_path, index=False)
        logger.info("Saved trial results to %s", trials_csv_path)

    # --- Remove per-trial directories (optional) ---
    # After HPO the best trial is in best_trial/ and hyperparams in best_hyperparams.json;
    # individual trial_N/ directories serve no further purpose.
    # The JournalStorage log is kept for resumability / further Optuna analysis.
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

    return best_hyperparams, study


# ---------------------------------------------------------------------------
# Ray Tune — distributed HPO across multiple CPUs / GPUs / cluster nodes
# ---------------------------------------------------------------------------

def _tabular_ray_trial_fn(
    config: dict,
    *,
    label2id: dict,
    id2label: dict,
    model_name: str,
    random_state: int,
    n_jobs: int,
    use_gpu: bool | None,
    X_opt_train: np.ndarray | None = None,
    y_opt_train: np.ndarray | None = None,
    X_opt_val: np.ndarray | None = None,
    y_opt_val: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    cv_splits: list[tuple[list[int], list[int]]] | None = None,
) -> dict:
    """Single-trial trainable for Ray Tune (tabular pipeline).

    Returns a metrics dict, which Ray Tune records as the trial's final result.
    Returning a dict is the portable way to report metrics from a function
    trainable — it works across all Ray 2.x versions without requiring a
    Ray Train session (``ray.train.report()`` is the Train API and raises /
    hangs when called outside a proper Train context such as TorchTrainer).

    Input arrays are passed via ``tune.with_parameters()`` and stored once in the
    Ray object store. Two modes are supported:
      - fixed holdout split (``X_opt_train``, ``X_opt_val``)
      - pre-built CV splits (``X_train``, ``y_train``, ``cv_splits``)

    This function must be defined at module level (not nested) so that Ray can
    serialise it by reference when dispatching to remote workers.
    """
    import logging as _logging

    _trial_logger = _logging.getLogger(__name__)

    # ----- Ensure multimodalva is importable in the worker process -----
    import sys as _sys
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    from multimodalva.tabular.train import train_tabular
    from multimodalva.tabular.predict import predict_tabular
    from multimodalva.utils.metrics import score_predictions, log_loss_from_full
    from multimodalva.utils.ray_compat import get_ray_trial_dir

    # ----- Trial artifact directory (Ray Tune API compatibility shim) -----
    trial_dir = get_ray_trial_dir()
    _trial_logger.info("Resolved Ray trial artifact directory: %s", trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    all_scores: dict = {
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "csmf_accuracy": 0.0,
        "log_loss": float("inf"),  # minimised — inf signals trial failure
    }
    try:
        if cv_splits:
            if X_train is None or y_train is None:
                raise ValueError("cv_splits mode requires X_train and y_train.")

            fold_scores: dict[str, list[float]] = {
                m: [] for m in CV_METRICS
            }
            fold_log_losses: list[float] = []
            for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(cv_splits):
                fold_dir = trial_dir / f"fold_{fold_idx}"
                X_fold_train = X_train[cv_train_idx]
                y_fold_train = y_train[cv_train_idx]
                X_fold_val = X_train[cv_val_idx]
                y_fold_val = y_train[cv_val_idx]

                train_tabular(
                    X_fold_train, y_fold_train,
                    label2id=label2id,
                    id2label=id2label,
                    model_name=model_name,
                    output_dir=fold_dir,
                    hyperparams=config,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    use_gpu=use_gpu,
                )
                fold_result = predict_tabular(fold_dir, X_fold_val, y_fold_val)
                for m in CV_METRICS:
                    fold_scores[m].append(score_predictions(fold_result.top1, m))
                fold_log_losses.append(log_loss_from_full(fold_result.full, fold_result.id2label))

            all_scores = {m: float(np.mean(fold_scores[m])) for m in fold_scores}
            all_scores["log_loss"] = float(np.mean(fold_log_losses))
        else:
            if X_opt_train is None or y_opt_train is None or X_opt_val is None or y_opt_val is None:
                raise ValueError(
                    "Holdout mode requires X_opt_train, y_opt_train, X_opt_val, and y_opt_val."
                )
            train_tabular(
                X_opt_train, y_opt_train,
                label2id=label2id,
                id2label=id2label,
                model_name=model_name,
                output_dir=trial_dir,
                hyperparams=config,
                random_state=random_state,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
            result = predict_tabular(trial_dir, X_opt_val, y_opt_val)
            all_scores = {
                m: score_predictions(result.top1, m)
                for m in CV_METRICS
            }
            all_scores["log_loss"] = log_loss_from_full(result.full, result.id2label)
    except Exception:
        _trial_logger.exception(
            "Ray trial failed (config=%s); reporting zero/inf scores.", config
        )

    # Return the metrics dict — Ray Tune treats the return value of a function
    # trainable as the trial's final reported result.  This avoids calling
    # ray.train.report(), which is the Ray Train API and raises or hangs when
    # invoked outside a Train session (e.g. TorchTrainer).
    return all_scores


def optimize_tabular_ray(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 50,
    metric: str = "f1_macro",
    search_space: dict | None = None,
    val_size: float = 0.2,
    use_cv: bool = True,
    n_cv_folds: int = 3,
    random_state: int = 42,
    split_seed: int | None = None,
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    # ---- Ray cluster / resource settings --------------------------------
    ray_address: str | None = None,
    num_gpus_per_trial: float = 0.0,
    num_cpus_per_trial: int = 4,
    max_concurrent_trials: int | None = None,
    # ---- Resume ---------------------------------------------------------
    resume: bool = True,
    experiment_name: str | None = None,
    # ---- Output ---------------------------------------------------------
    save_trials_csv: bool = True,
    cleanup_trials: bool = True,
    search_space_profile: str = "auto",
) -> tuple[dict, Any]:
    """Run distributed HPO with Ray Tune across multiple CPUs/GPUs or cluster nodes.

    Mirrors the interface of :func:`optimize` (Optuna backend) but executes
    trials **concurrently** across all available CPU/GPU resources using the
    Ray distributed runtime.

    When to use this over :func:`optimize`
    ---------------------------------------
    * You have a **multi-core machine** and want N parallel trials simultaneously
      (each trial uses ``num_cpus_per_trial`` cores).
    * You are on a **cluster** (SLURM, Kubernetes, AWS, GCP, …) and want to
      spread trials across nodes.
    * You need **fault tolerance** — Ray automatically restarts failed trials.

    CPU vs. GPU usage
    -----------------
    Most sklearn models are CPU-bound.  ``num_gpus_per_trial=0.0`` (default)
    is correct for ``gbdt``, ``mlp``, ``random_forest``, ``naive_bayes``,
    ``knn``, and ``svm``.

    For GPU-accelerated models, set both ``num_gpus_per_trial=1`` *and*
    ``use_gpu=True``:

    +------------+----------------------------------+
    | model_name | GPU backend                      |
    +============+==================================+
    | catboost   | ``task_type='GPU'``              |
    | lightgbm   | ``device_type='gpu'``            |
    | xgboost    | ``device='cuda'`` (XGBoost ≥2.0) |
    +------------+----------------------------------+

    CPU contention and n_jobs auto-cap
    ------------------------------------
    When multiple trials run in parallel, ``n_jobs=-1`` would cause each trial
    to attempt to use all CPU cores simultaneously, causing contention and
    degraded throughput.  ``optimize_tabular_ray()`` automatically caps ``n_jobs`` to
    ``num_cpus_per_trial`` to ensure each trial stays within its allocated
    resource budget:

    .. code-block:: text

        # 32-core node, num_cpus_per_trial=4 → 8 concurrent trials
        # n_jobs=-1 is silently replaced by n_jobs=4 per trial

    Cluster setup
    -------------
    .. code-block:: bash

        # On the head node:
        ray start --head --num-cpus=32

        # On each worker node:
        ray start --address=<HEAD_IP>:6379 --num-cpus=32

    Then pass ``ray_address="auto"``.  Set ``output_dir`` to a shared
    filesystem path (NFS / Lustre) or an S3/GCS URI so all nodes can write
    trial artifacts.

    Args:
        X_train:        Preprocessed feature matrix from prepare_tabular_dataset().
        y_train:        Integer label array from prepare_tabular_dataset().
        label2id:       Label-to-integer mapping from prepare_tabular_dataset().
        id2label:       Integer-to-label mapping from prepare_tabular_dataset().
        model_name:     Model alias — one of TABULAR_MODELS keys.
        output_dir:     Root directory for all Ray Tune artifacts.  Use a
                        shared filesystem path for multi-node clusters.
        n_trials:       Total number of trials. Default 50.
        metric:         Metric to optimise — ``"accuracy"``, ``"balanced_accuracy"``,
                        ``"f1_macro"``, ``"f1_weighted"``, ``"csmf_accuracy"``,
                        or ``"log_loss"``. Default ``"accuracy"``.
        search_space:   Custom search space dict (same format as
                        :func:`optimize_tabular`). Merged over the adaptive
                        default search space selected for this dataset.
                        Overrides it per key; keys you omit keep
                        their default.
        search_space_profile:
                        One of "auto", "small", "balanced", "wide", "large".
                        "auto" infers a profile from X_train.shape.
        val_size:       Fraction of X_train held out for trial evaluation.
                        Default 0.2.
                        Ignored when use_cv=True.
        use_cv:         Use stratified k-fold CV for trial scoring. Default True.
        n_cv_folds:     Number of CV folds when use_cv=True. Default 3.
        random_state:   Seed for each candidate fit and for the OptunaSearch
                        sampler. Default 42.
        split_seed:     Seed for the CV fold draw / internal validation split
                        this function makes, as in :func:`optimize_tabular`.
                        ``None`` (the default) reuses ``random_state``, so
                        existing callers are unaffected. Pass it separately to
                        hold the folds fixed while the model seed varies —
                        without it, switching backend would silently move the
                        fold boundaries of an otherwise identical search.
        n_jobs:         CPU parallelism *within* each trial.  Default -1
                        (all cores), but automatically capped to
                        ``num_cpus_per_trial`` when running in parallel to
                        prevent CPU contention.
        use_gpu:        Enable GPU acceleration for catboost / lightgbm /
                        xgboost.  ``None`` → auto-detect (same as
                        :func:`~tabular.train.train_tabular`).  Set to ``True``
                        together with ``num_gpus_per_trial=1``.
        ray_address:    Ray cluster address.
                        ``None``          — local Ray instance.
                        ``"auto"``        — auto-discover existing cluster.
                        ``"<ip>:<port>"`` — explicit head node.
                        Default None.
        num_gpus_per_trial: GPUs per trial. Default 0.0 (CPU-only).
                            Set to 1.0 for catboost / lightgbm / xgboost
                            with ``use_gpu=True``.
        num_cpus_per_trial: CPU cores per trial. Default 4.
                            This also becomes the effective ``n_jobs`` cap.
        max_concurrent_trials: Cap on simultaneously running trials.
                               ``None`` → Ray auto-determines. Default None.
        save_trials_csv: Save all trial metrics to
                         ``output_dir/hpo_trials.csv``. Default True.

    Returns:
        best_hyperparams: Dict from the best trial — pass to
                          ``train(hyperparams=best_hyperparams)``.
        results: ``ray.tune.ResultGrid`` — full results for analysis.
                 ``results.get_dataframe()`` → trial-level pandas DataFrame.
                 ``results.get_best_result().path`` → best trial artifact dir.

    Raises:
        ValueError:  If model_name is not in TABULAR_MODELS.
        ImportError: If ``ray[tune]`` is not installed.
    """
    _VALID_METRICS = HPO_METRICS
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid metrics: {sorted(_VALID_METRICS)}"
        )
    if use_cv and n_cv_folds < 2:
        raise ValueError("n_cv_folds must be >= 2 when use_cv=True.")

    try:
        import ray
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.search import ConcurrencyLimiter
    except ImportError as exc:
        raise ImportError(
            "Ray Tune is required for optimize_tabular_ray(). It lives in an "
            "optional extra, so a normal install does not have it:\n"
            "    pip install 'multimodalva[ray]'\n"
            "Or search on the Optuna backend, which needs nothing extra:\n"
            "    Optimize(backend=\"optuna\")   # or backend=\"auto\""
        ) from exc

    if model_name not in TABULAR_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {list(TABULAR_MODELS)}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Auto-redirect to optimize_tabular() on MPS/CPU-only ---
    # Ray Train v2 is incompatible with function trainables on non-CUDA devices:
    # RunConfig auto-injects checkpoint_at_end=True which raises ValueError,
    # forcing a no-RunConfig fallback where trial state goes to ~/ray_results
    # and cannot be restored → resume always restarts from trial 1.
    # On MPS/CPU, optimize_tabular() (Optuna + JournalStorage) is the correct backend.
    try:
        import torch as _torch
        _cuda_available = _torch.cuda.device_count() > 0
    except ImportError:
        _cuda_available = False
    if not _cuda_available:
        _safe = model_name.replace("/", "_")
        logger.info(
            "optimize_tabular_ray(): no CUDA GPUs detected — redirecting to optimize_tabular() "
            "(Optuna sequential, JournalStorage-backed). Ray Tune requires CUDA for "
            "reliable experiment persistence and resume. "
            "Journal log: %s/hpo_%s.log",
            output_dir, _safe,
        )
        return optimize_tabular(
            X_train=X_train,
            y_train=y_train,
            label2id=label2id,
            id2label=id2label,
            model_name=model_name,
            output_dir=output_dir,
            n_trials=n_trials,
            metric=metric,
            search_space=search_space,
            search_space_profile=search_space_profile,
            val_size=val_size,
            use_cv=use_cv,
            n_cv_folds=n_cv_folds,
            random_state=random_state,
            storage_path=str(output_dir / f"hpo_{_safe}.log"),
            load_if_exists=resume,
            save_trials_csv=save_trials_csv,
            cleanup_trials=cleanup_trials,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
        )

    # --- Experiment storage path (for resume) ---
    safe_model_name = model_name.replace("/", "_")
    exp_name    = experiment_name or f"ray_hpo_{safe_model_name}"
    exp_storage = output_dir / "ray_experiment"
    exp_path    = exp_storage / exp_name

    # --- Build effective search space: data/class-adaptive base → caller overrides ---
    n_classes = len(id2label)
    resolved_profile = _resolve_search_space_profile(X_train, search_space_profile)
    active_space = get_tabular_default_search_space(
        model_name=model_name,
        X_train=X_train,
        search_space_profile=resolved_profile,
        n_classes=n_classes,
    )
    merge_search_space(active_space, search_space, logger)
    ray_space = to_ray_space(active_space)
    logger.info(
        "Tabular HPO adaptive search space: profile=%s, class_tier=%s "
        "(n_samples=%d, n_features=%d, n_classes=%d).",
        resolved_profile, get_class_tier(n_classes),
        X_train.shape[0], X_train.shape[1], n_classes,
    )

    # --- Trial evaluation setup (mirrors optimize_tabular()) ---
    # split_seed governs which rows go where, random_state what the model does
    # with them. They are separate so that changing the model seed does not move
    # the fold boundaries; None falls back to random_state, as everywhere else.
    _split_seed = split_seed if split_seed is not None else random_state
    cv_splits: list[tuple[list[int], list[int]]] | None = None
    X_opt_train = y_opt_train = X_opt_val = y_opt_val = None
    if use_cv:
        _, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = int(class_counts.min())
        if n_cv_folds > min_class_count:
            raise ValueError(
                f"n_cv_folds={n_cv_folds} is greater than the minimum class count "
                f"({min_class_count}) in y_train. Reduce n_cv_folds or rebalance data."
            )
        indices = np.arange(len(y_train))
        skf = StratifiedKFold(
            n_splits=n_cv_folds,
            shuffle=True,
            random_state=_split_seed,
        )
        cv_splits = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in skf.split(indices, y_train)
        ]
        logger.info(
            "Tabular Ray CV HPO: %d-fold stratified splits pre-built "
            "(n_samples=%d, n_classes=%d). Each trial runs %d fits.",
            n_cv_folds, len(y_train), len(id2label), n_cv_folds,
        )
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=_split_seed
        )
        opt_train_idx, opt_val_idx = next(sss.split(X_train, y_train))
        X_opt_train = X_train[opt_train_idx]
        y_opt_train = y_train[opt_train_idx]
        X_opt_val = X_train[opt_val_idx]
        y_opt_val = y_train[opt_val_idx]

    # --- Auto-cap n_jobs to prevent CPU contention between parallel trials ---
    # When n_jobs=-1, each sklearn model would claim all available cores.
    # With N concurrent trials this causes contention and degrades throughput.
    # Capping to num_cpus_per_trial ensures each trial stays within its budget.
    effective_n_jobs = n_jobs
    if n_jobs == -1 and num_cpus_per_trial > 0:
        effective_n_jobs = num_cpus_per_trial
        logger.info(
            "n_jobs auto-capped from -1 to %d (num_cpus_per_trial) "
            "to prevent CPU contention between concurrent Ray trials.",
            num_cpus_per_trial,
        )

    # --- Initialise Ray (idempotent if already running) ---
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)
        logger.info(
            "Ray initialised.  Cluster resources: %s",
            ray.cluster_resources(),
        )
    elif ray_address is not None:
        logger.warning(
            "Ray is already initialised; ray_address=%r ignored.",
            ray_address,
        )

    # Log concurrency estimate and warn on GPU misconfigurations.
    available_cpus = ray.cluster_resources().get("CPU", 0)
    available_gpus = ray.cluster_resources().get("GPU", 0)
    if num_gpus_per_trial > 0 and available_gpus == 0:
        logger.warning(
            "num_gpus_per_trial=%.1f but the Ray cluster reports 0 GPUs. "
            "Trials will queue indefinitely. "
            "Set num_gpus_per_trial=0 for CPU-only mode.",
            num_gpus_per_trial,
        )

    # Warn early on MPS/CPU-only: Ray Train v2 is incompatible with function
    # trainables when RunConfig is used, causing checkpoint_at_end ValueError.
    # This means experiment state cannot be persisted → resume is impossible.
    if available_gpus == 0:
        logger.warning(
            "optimize_tabular_ray(): no CUDA GPUs available (MPS/CPU-only environment). "
            "Ray Train v2 is incompatible with function trainables on non-CUDA "
            "devices — experiment state cannot be persisted and resume will not "
            "work across restarts. "
            "RECOMMENDATION: use optimize_tabular() (Optuna backend) instead — it uses "
            "JournalStorage for reliable crash recovery and resume on MPS/CPU machines.",
        )
    else:
        max_parallel = int(available_cpus // num_cpus_per_trial) if num_cpus_per_trial > 0 else None
        logger.info(
            "Ray cluster: %.0f CPU(s), %.0f GPU(s) — up to %s concurrent "
            "trials at %d CPU(s)/trial.",
            available_cpus, available_gpus,
            max_parallel if max_parallel is not None else "unlimited",
            num_cpus_per_trial,
        )

    # --- Search algorithm: OptunaSearch (TPE) ---
    _ray_mode = "min" if METRIC_DIRECTION.get(metric, "maximize") == "minimize" else "max"
    search_alg: object = OptunaSearch(
        metric=metric,
        mode=_ray_mode,
        seed=random_state,
    )
    if max_concurrent_trials is not None:
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=max_concurrent_trials
        )

    # --- Trainable: bind fixed arguments via the Ray object store ---
    # Numpy arrays are stored in the object store using Apache Arrow shared
    # memory — zero-copy reads by workers on the same node.
    if use_cv:
        trainable = tune.with_parameters(
            _tabular_ray_trial_fn,
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            label2id=label2id,
            id2label=id2label,
            model_name=model_name,
            random_state=random_state,
            n_jobs=effective_n_jobs,
            use_gpu=use_gpu,
        )
    else:
        trainable = tune.with_parameters(
            _tabular_ray_trial_fn,
            X_opt_train=X_opt_train,
            y_opt_train=y_opt_train,
            X_opt_val=X_opt_val,
            y_opt_val=y_opt_val,
            label2id=label2id,
            id2label=id2label,
            model_name=model_name,
            random_state=random_state,
            n_jobs=effective_n_jobs,
            use_gpu=use_gpu,
        )
    trainable = tune.with_resources(
        trainable,
        resources={"cpu": num_cpus_per_trial, "gpu": num_gpus_per_trial},
    )

    # --- RunConfig for experiment persistence (enables Tuner.restore()) ---
    # Wrapped in try/except: some Ray versions raise checkpoint_at_end ValueError
    # when RunConfig is used with function trainables.  If that happens we fall
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
        "optimize_tabular_ray: launching %d trials  model=%s  metric=%s  "
        "cpus/trial=%d  gpus/trial=%.1f",
        n_trials, model_name, metric, num_cpus_per_trial, num_gpus_per_trial,
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
            # On MPS/Apple Silicon: switch to optimize_tabular() (Optuna) for
            # reliable resume.  optimize_tabular() uses JournalStorage and resumes
            # correctly after crashes or SLURM preemptions.
            logger.warning(
                "checkpoint_at_end error from Ray Train v2 (%s). "
                "Retrying without RunConfig — this run's trial state will be "
                "saved to ~/ray_results (not %s) and CANNOT be resumed. "
                "If you need resume support, use optimize_tabular() (Optuna backend) "
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
            "Likely causes: model fit errors, invalid hyperparameter combinations, "
            "or Ray worker errors.  Check logs above for details.",
            _succ_rt_r, _n_ok_r, _n_total_r,
        )

    # --- Extract best result ---
    _n_ok_ray = sum(1 for r in results if not r.error)
    if _n_ok_ray == 0:
        raise RuntimeError(
            f"All {len(results)} Ray Tune trials errored — no completed trial to "
            "select best hyperparameters from.  Check the trial logs above "
            "(typically OOM, invalid hyperparameter combinations, or Ray worker errors).  "
            f"Trial artifacts are in: {exp_storage}"
        )

    best_result = results.get_best_result(metric=metric, mode=_ray_mode)
    best_hyperparams = best_result.config

    # Canonical names, the same ones optimize_tabular() writes — see the note in
    # text/hpo.py: the "_ray" suffix broke the run contract, validation.json's
    # hpo_cv score and every script that reloads best_hyperparams.json, as soon as
    # Optimize(backend="ray") made the backend a switch rather than a separate
    # entry point. Readers still accept the legacy spellings for runs on disk.
    with open(output_dir / "best_hyperparams.json", "w") as fh:
        json.dump(
            best_hyperparams, fh, indent=2,
            default=lambda o: list(o) if isinstance(o, tuple) else str(o),
        )

    logger.info("Best hyperparams (Ray Tune): %s", best_hyperparams)
    logger.info(
        "Best %s (Ray Tune): %.4f",
        metric,
        best_result.metrics.get(metric, float("nan")),
    )
    logger.info("Best trial artifacts: %s", best_result.path)

    # --- Copy best trial artifacts to a predictable location ---
    # Remove stale artifacts first so resume never silently merges two trials.
    best_trial_path = Path(best_result.path)
    best_output_dir = output_dir / "best_trial"
    if best_trial_path.exists():
        if best_output_dir.exists():
            shutil.rmtree(best_output_dir)
        shutil.copytree(best_trial_path, best_output_dir)
        logger.info("Copied best trial artifacts to %s", best_output_dir)
    else:
        logger.warning(
            "Best trial path not found at %s — best_trial/ not updated.", best_trial_path
        )

    # --- Save all trial results as CSV ---
    if save_trials_csv:
        trials_csv_path = output_dir / "hpo_trials.csv"
        results.get_dataframe().to_csv(trials_csv_path, index=False)
        logger.info("Saved Ray trial results to %s", trials_csv_path)

    # --- Remove Ray experiment directory (optional) ---
    # After a successful HPO run the best trial is in best_trial/ and all metrics
    # are in hpo_trials.csv; the ray_experiment/ tree is no longer needed.
    if cleanup_trials and exp_storage.exists():
        shutil.rmtree(exp_storage)
        logger.info("Removed Ray experiment dir: %s", exp_storage)

    return best_hyperparams, results


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------
def search_tabular(
    search: "Optimize",
    *,
    resume: bool,
    **common: Any,
) -> tuple[dict, Any]:
    """Run one tabular hyperparameter search on whichever backend ``search`` names.

    The tabular twin of :func:`multimodalva.text.hpo.search_text`, and the single
    place :class:`~multimodalva.utils.optimize_config.Optimize` is mapped onto
    :func:`optimize_tabular` / :func:`optimize_tabular_ray`.

    Args:
        search:  The search settings.
        resume:  Already resolved by
                 :func:`~multimodalva.utils.optimize_config.resolve_search_resume`;
                 ``load_if_exists`` on the Optuna backend, ``resume`` on the Ray one.
        **common: Arguments both backends take under the same name
                 (``X_train``, ``y_train``, ``label2id``, ``id2label``,
                 ``model_name``, ``output_dir``, ``random_state``, ``split_seed``,
                 ``n_jobs``, ``use_gpu``).

    Returns:
        ``(best_hyperparams, study_or_result_grid)``.
    """
    from ..utils.optimize_config import resolve_backend, resolve_pruning

    backend = resolve_backend(search.backend)
    kwargs: dict[str, Any] = dict(
        metric=search.metric,
        search_space=search.space,
        search_space_profile=search.space_profile,
        use_cv=search.cv,
        n_cv_folds=search.cv_folds,
        **common,
    )
    if search.n_trials is not None:
        kwargs["n_trials"] = search.n_trials

    if backend == "ray":
        if search.pruning is not None:
            logger.warning(
                "Optimize(pruning=%s) is ignored by the Ray backend, which has no "
                "inter-trial pruner. Pass extra={'use_asha': True} instead.",
                search.pruning,
            )
        kwargs["resume"] = resume
        kwargs.update(search.extra)
        return optimize_tabular_ray(**kwargs)

    kwargs["load_if_exists"] = resume
    kwargs["enable_pruning"] = resolve_pruning(search, TABULAR_PRUNING_DEFAULT)
    kwargs.update(search.extra)
    return optimize_tabular(**kwargs)
