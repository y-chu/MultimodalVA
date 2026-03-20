"""
Step 5 (tabular pipeline): Hyperparameter optimization.

Two backends are provided; both accept the same search_space format and return
``(best_hyperparams, backend_object)`` so callers can switch seamlessly:

    optimize()      — Optuna (TPE sampler, SQLite persistence)
                      Best for single-machine sequential or lightly parallel HPO.

    optimize_ray()  — Ray Tune (OptunaSearch / TPE, distributed runtime)
                      Best for multi-CPU/GPU machines or multi-node clusters.
                      Runs N trials concurrently; auto-caps n_jobs per trial
                      to prevent CPU contention between concurrent workers.
                      For GPU-accelerated models (catboost, lightgbm, xgboost),
                      set num_gpus_per_trial=1 and use_gpu=True.

Input:  X_train, y_train, label2id, id2label from prepare_dataset()
Output: best_hyperparams dict and backend study / ResultGrid object
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna
    import ray

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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
    log_loss_from_full, METRIC_DIRECTION,
)
from .predict import predict
from .train import SUPPORTED_MODELS, train

logger = logging.getLogger(__name__)

# Default search spaces per model alias.
# Spec format mirrors text/hpo.py:
#   ("float_log", low, high)   — log-uniform float (best for learning rates)
#   ("float",     low, high)   — uniform float
#   ("int",       low, high)   — uniform int
#   ("categorical", [values])  — discrete choices
DEFAULT_SEARCH_SPACES: dict[str, dict] = {
    "catboost": {
        "iterations":    ("int",         100, 1000),    # number of trees; more = higher capacity but slower training
        "learning_rate": ("float_log",   1e-3, 0.3),   # shrinkage per tree; lower = better generalization but needs more iterations
        "depth":         ("int",         4, 10),        # tree depth; deeper captures more interactions but risks overfitting; >10 rarely helps
        "l2_leaf_reg":   ("float_log",   1e-2, 10.0),  # L2 penalty on leaf weights; higher = smoother predictions, less overfitting
        "boosting_type": ("categorical", ["Ordered", "Plain"]),  # Ordered = CatBoost permutation-based (better for small data); Plain = standard GBDT
    },
    "lightgbm": {
        "n_estimators":      ("int",         100, 1000),           # number of boosting rounds; more = better fit; too many = overfit without early stopping
        "learning_rate":     ("float_log",   1e-3, 0.3),          # smaller = more robust but needs proportionally more n_estimators
        "max_depth":         ("categorical", [-1, 6, 8, 10]),      # -1 = no limit (complexity governed by num_leaves instead)
        "num_leaves":        ("int",         20, 150),             # primary complexity knob for leaf-wise growth; more leaves = finer partitions, higher variance
        "min_child_samples": ("int",         5, 100),              # larger = coarser leaves, less overfitting on rare causes; critical for imbalanced VA data
    },
    "gbdt": {
        "n_estimators":      ("int",       50, 500),    # number of sequential trees; more = lower bias, risk of overfitting
        "learning_rate":     ("float_log", 1e-3, 0.3), # smaller = each tree contributes less, needs more trees but generalizes better
        "max_depth":         ("int",       2, 8),       # shallow trees (2-4) often best for boosting; deep trees overfit and slow training
        "min_samples_split": ("int",       2, 20),      # larger = fewer splits made, simpler trees, less overfitting to small subgroups
        "min_samples_leaf":  ("int",       1, 10),      # larger = smoother leaf predictions; prevents fitting leaves from very few samples
        "subsample":         ("float",     0.5, 1.0),   # row subsampling per tree; <1 adds randomness like bagging, reduces variance
    },
    "xgboost": {
        "n_estimators":      ("int",       100, 1000),  # number of trees; combine with learning_rate (lower lr → more trees needed)
        "learning_rate":     ("float_log", 1e-3, 0.3), # eta; smaller = more conservative updates, better generalization
        "max_depth":         ("int",       3, 10),      # deeper trees capture more complex patterns but overfit and are slower
        "subsample":         ("float",     0.5, 1.0),   # fraction of rows per tree; <1 introduces stochasticity, reduces overfitting
        "colsample_bytree":  ("float",     0.5, 1.0),   # fraction of features per tree; lower = more diverse trees, similar to random forest effect
        "gamma":             ("float",     0.0, 1.0),   # min loss reduction required to split a node; larger = fewer splits, more conservative trees
        "min_child_weight":  ("int",       1, 10),      # larger = prevents learning from small leaf groups; important for rare cause classes
    },
    "mlp": {
        "hidden_layer_sizes": ("categorical", [(64,), (128,), (128, 64), (256, 128), (256, 128, 64)]),  # network depth/width; wider or deeper = more capacity, more data needed to avoid overfitting
        "activation":         ("categorical", ["tanh", "relu"]),   # relu = sparse, fast, better for deep nets; tanh = smoother, bounded, better for shallow nets
        "learning_rate_init": ("float_log",   1e-4, 1e-1),        # too large = unstable training; too small = slow convergence
        "alpha":              ("float_log",   1e-5, 1e-1),         # L2 weight decay; larger = stronger regularization, smaller weights, better generalization
        "batch_size":         ("categorical", [32, 64, 128, 256]), # smaller = noisier gradients (can escape local minima); larger = faster but may overfit
    },
    "random_forest": {
        "n_estimators":      ("int",         50, 500),                   # more trees = more stable predictions; gains plateau around 200-300
        "max_depth":         ("categorical", [None, 5, 10, 20, 30]),     # None = fully grown (may overfit); limiting depth regularizes leaf predictions
        "min_samples_split": ("int",         2, 20),                     # larger = more conservative node splits; reduces overfitting on small subgroups
        "min_samples_leaf":  ("int",         1, 10),                     # larger = smoother predicted probabilities; prevents leaves from very few samples
        "max_features":      ("categorical", ["sqrt", "log2", 0.5]),     # fewer features per split = more diverse trees, stronger ensemble effect; sqrt is RF default for classification
    },
    "naive_bayes": {
        "var_smoothing": ("float_log", 1e-9, 1e-1),  # variance floor per feature; larger = more shrinkage toward uniform, useful when features have near-zero variance
    },
    "knn": {
        "n_neighbors": ("int",         3, 15),                                    # larger k = smoother boundary, less sensitive to noise; smaller k = complex boundary, overfits
        "weights":     ("categorical", ["uniform", "distance"]),                  # distance weighting emphasizes closest neighbors; useful when local structure matters
        "metric":      ("categorical", ["euclidean", "manhattan", "chebyshev"]),  # euclidean = L2 (sensitive to scale); manhattan = L1 (robust to outliers); chebyshev = max coordinate difference
    },
    "svm": {
        "C":      ("float_log",  1e-3, 1e3),                         # larger C = smaller margin, fits training data tightly (may overfit); smaller C = wider margin, more regularized
        "kernel": ("categorical", ["linear", "rbf", "poly"]),        # linear = fast for high-dim data; rbf = flexible nonlinear boundary; poly = polynomial feature interactions
        "gamma":  ("categorical", ["scale", "auto"]),                 # rbf bandwidth; "scale" = 1/(n_features * X.var()), adapts to feature spread; "auto" = 1/n_features
    },
}


def optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 50,
    metric: str = "accuracy",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    study_name: str = "tabular_hpo",
    storage_path: str | None = None,
    load_if_exists: bool = True,
    enable_pruning: bool = False,
    save_trials_csv: bool = True,
    cleanup_trials: bool = True,
    n_jobs: int = -1,
    use_gpu: bool | None = None,
) -> tuple[dict, optuna.Study]:
    """Run Optuna hyperparameter search for a tabular model.

    Internally splits (X_train, y_train) into opt-train / opt-val using a
    stratified split. Each trial trains on opt-train and is scored on opt-val.
    The held-out test set from prepare_dataset() is never used here.

    After all trials complete, the best trial's artifacts are copied to
    output_dir/best_trial/ and best hyperparams saved as best_hyperparams.json.

    Args:
        X_train:        Preprocessed feature matrix from prepare_dataset().
        y_train:        Integer label array from prepare_dataset().
        label2id:       Label-to-integer mapping from prepare_dataset().
        id2label:       Integer-to-label mapping from prepare_dataset().
        model_name:     Model alias — one of SUPPORTED_MODELS keys.
        output_dir:     Root directory for trial outputs and study database.
        n_trials:       Total Optuna trials. Default 50.
        metric:         Metric to maximise — "accuracy", "balanced_accuracy",
                        "f1_macro", "f1_weighted", "csmf_accuracy". Default "accuracy".
                        Use "balanced_accuracy", "csmf_accuracy", or "f1_macro" for imbalanced VA data.
        search_space:   Custom search space dict. Merged over DEFAULT_SEARCH_SPACES[model_name].
        val_size:       Fraction of X_train for trial evaluation. Default 0.2.
        random_state:   Seed for stratified split and Optuna sampler. Default 42.
        study_name:     Optuna study name. Default "tabular_hpo".
        storage_path:   SQLite URL for study persistence.
                        Defaults to output_dir/hpo_<model_name>.db.
        load_if_exists: Resume an existing study. Default True.
        enable_pruning: MedianPruner for early trial termination. Default False
                        (pruning is less useful for fast sklearn fits).
        save_trials_csv: Save all trial results to hpo_trials.csv. Default True.
        cleanup_trials: Delete all per-trial directories (``trial_*/``) after HPO
                        completes.  The best trial is preserved at ``best_trial/``
                        before deletion.  The SQLite study DB, ``best_hyperparams.json``,
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
            "optuna is required for optimize(). "
            "Install with:  pip install 'optuna>=3.4'"
        ) from exc

    _VALID_METRICS = {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy", "log_loss"}
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid metrics: {sorted(_VALID_METRICS)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {list(SUPPORTED_MODELS)}."
        )

    # Build effective search space
    active_space = dict(DEFAULT_SEARCH_SPACES.get(model_name, {}))
    if search_space:
        active_space.update(search_space)

    if storage_path is None:
        # Use resolved absolute path so the DB is always found regardless of CWD.
        storage_path = f"sqlite:///{output_dir.resolve()}/hpo_{model_name}.db"

    # Stratified internal split — never uses the held-out test set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    opt_train_idx, opt_val_idx = next(sss.split(X_train, y_train))
    X_opt_train = X_train[opt_train_idx]
    y_opt_train = y_train[opt_train_idx]
    X_opt_val = X_train[opt_val_idx]
    y_opt_val = y_train[opt_val_idx]

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hyperparams(trial, active_space)
        trial_dir = output_dir / f"trial_{trial.number}"

        train(
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
        result = predict(trial_dir, X_opt_val, y_opt_val)

        # Compute all metrics; store as user attributes for traceability.
        # Only `metric` drives Optuna's optimisation.
        all_scores = {
            m: score_predictions(result.top1, m)
            for m in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
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
        storage=storage_path,
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

    best_hyperparams = study.best_params

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
        study.trials_dataframe().to_csv(trials_csv_path, index=False)
        logger.info("Saved trial results to %s", trials_csv_path)

    # --- Remove per-trial directories (optional) ---
    # After HPO the best trial is in best_trial/ and hyperparams in best_hyperparams.json;
    # individual trial_N/ directories serve no further purpose.
    # The SQLite study DB is kept for resumability / further Optuna analysis.
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

def _to_ray_space(search_space: dict) -> dict:
    """Convert our ``(type, *args)`` search space format to Ray Tune's format.

    Mapping::

        ("float_log", low, high)  →  tune.loguniform(low, high)
        ("float",     low, high)  →  tune.uniform(low, high)
        ("int",       low, high)  →  tune.randint(low, high)
        ("categorical", [vals])   →  tune.choice([vals])

    Raises:
        ImportError: If ``ray[tune]`` is not installed.
        ValueError:  If an unknown type string is encountered.
    """
    try:
        from ray import tune
    except ImportError as exc:
        raise ImportError(
            "Ray Tune is required for optimize_ray(). "
            "Install with: pip install 'ray[tune]'"
        ) from exc

    ray_space: dict = {}
    for key, spec in search_space.items():
        kind, *args = spec
        if kind == "float_log":
            ray_space[key] = tune.loguniform(args[0], args[1])
        elif kind == "float":
            ray_space[key] = tune.uniform(args[0], args[1])
        elif kind == "int":
            ray_space[key] = tune.randint(args[0], args[1])
        elif kind == "categorical":
            ray_space[key] = tune.choice(args[0])
        else:
            raise ValueError(
                f"Unknown search space type {kind!r}. "
                "Valid types: 'float_log', 'float', 'int', 'categorical'."
            )
    return ray_space


def _tabular_ray_trial_fn(
    config: dict,
    *,
    X_opt_train: np.ndarray,
    y_opt_train: np.ndarray,
    X_opt_val: np.ndarray,
    y_opt_val: np.ndarray,
    label2id: dict,
    id2label: dict,
    model_name: str,
    random_state: int,
    n_jobs: int,
    use_gpu: bool | None,
) -> dict:
    """Single-trial trainable for Ray Tune (tabular pipeline).

    Returns a metrics dict, which Ray Tune records as the trial's final result.
    Returning a dict is the portable way to report metrics from a function
    trainable — it works across all Ray 2.x versions without requiring a
    Ray Train session (``ray.train.report()`` is the Train API and raises /
    hangs when called outside a proper Train context such as TorchTrainer).

    Numpy arrays (``X_opt_train``, ``y_opt_train``, ``X_opt_val``,
    ``y_opt_val``) are passed via ``tune.with_parameters()`` — stored once in
    the Ray object store and referenced by all workers without re-serialisation.

    This function must be defined at module level (not nested) so that Ray can
    serialise it by reference when dispatching to remote workers.
    """
    import logging as _logging
    from ray import tune as _ray_tune

    _trial_logger = _logging.getLogger(__name__)

    # ----- Ensure multimodalva is importable in the worker process -----
    import sys as _sys
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    from multimodalva.tabular.train import train as _train
    from multimodalva.tabular.predict import predict as _predict
    from multimodalva.utils.metrics import score_predictions, log_loss_from_full

    # ----- Trial artifact directory (Ray Tune API) -----
    ctx = _ray_tune.get_context()
    trial_dir = Path(ctx.get_trial_dir())
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
        _train(
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
        result = _predict(trial_dir, X_opt_val, y_opt_val)
        all_scores = {
            m: score_predictions(result.top1, m)
            for m in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
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


def optimize_ray(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 50,
    metric: str = "accuracy",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
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
) -> tuple[dict, "ray.tune.ResultGrid"]:
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
    degraded throughput.  ``optimize_ray()`` automatically caps ``n_jobs`` to
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
        X_train:        Preprocessed feature matrix from prepare_dataset().
        y_train:        Integer label array from prepare_dataset().
        label2id:       Label-to-integer mapping from prepare_dataset().
        id2label:       Integer-to-label mapping from prepare_dataset().
        model_name:     Model alias — one of SUPPORTED_MODELS keys.
        output_dir:     Root directory for all Ray Tune artifacts.  Use a
                        shared filesystem path for multi-node clusters.
        n_trials:       Total number of trials. Default 50.
        metric:         Metric to maximise — ``"accuracy"``, ``"balanced_accuracy"``,
                        ``"f1_macro"``, ``"f1_weighted"``, ``"csmf_accuracy"``. Default ``"accuracy"``.
        search_space:   Custom search space dict (same format as
                        :func:`optimize`). Merged over
                        ``DEFAULT_SEARCH_SPACES[model_name]``.
        val_size:       Fraction of X_train held out for trial evaluation.
                        Default 0.2.
        random_state:   Seed for stratified split and OptunaSearch sampler.
                        Default 42.
        n_jobs:         CPU parallelism *within* each trial.  Default -1
                        (all cores), but automatically capped to
                        ``num_cpus_per_trial`` when running in parallel to
                        prevent CPU contention.
        use_gpu:        Enable GPU acceleration for catboost / lightgbm /
                        xgboost.  ``None`` → auto-detect (same as
                        :func:`~tabular.train.train`).  Set to ``True``
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
                         ``output_dir/hpo_trials_ray.csv``. Default True.

    Returns:
        best_hyperparams: Dict from the best trial — pass to
                          ``train(hyperparams=best_hyperparams)``.
        results: ``ray.tune.ResultGrid`` — full results for analysis.
                 ``results.get_dataframe()`` → trial-level pandas DataFrame.
                 ``results.get_best_result().path`` → best trial artifact dir.

    Raises:
        ValueError:  If model_name is not in SUPPORTED_MODELS.
        ImportError: If ``ray[tune]`` is not installed.
    """
    _VALID_METRICS = {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy", "log_loss"}
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
            "Ray Tune and Optuna are required for optimize_ray(). "
            "Install with: pip install 'ray[tune]' optuna"
        ) from exc

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {list(SUPPORTED_MODELS)}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Experiment storage path (for resume) ---
    safe_model_name = model_name.replace("/", "_")
    exp_name    = experiment_name or f"ray_hpo_{safe_model_name}"
    exp_storage = output_dir / "ray_experiment"
    exp_path    = exp_storage / exp_name

    # --- Build effective search space ---
    active_space = dict(DEFAULT_SEARCH_SPACES.get(model_name, {}))
    if search_space:
        active_space.update(search_space)
    ray_space = _to_ray_space(active_space)

    # --- Stratified internal split (mirrors optimize()) ---
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    opt_train_idx, opt_val_idx = next(sss.split(X_train, y_train))
    X_opt_train = X_train[opt_train_idx]
    y_opt_train = y_train[opt_train_idx]
    X_opt_val   = X_train[opt_val_idx]
    y_opt_val   = y_train[opt_val_idx]

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
    _run_config = None
    try:
        from ray.train import RunConfig
        exp_storage.mkdir(parents=True, exist_ok=True)
        _run_config = RunConfig(
            storage_path=str(exp_storage.resolve()),
            name=exp_name,
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
        "optimize_ray: launching %d trials  model=%s  metric=%s  "
        "cpus/trial=%d  gpus/trial=%.1f",
        n_trials, model_name, metric, num_cpus_per_trial, num_gpus_per_trial,
    )
    # AutoGluon sets RAY_AIR_NEW_OUTPUT=1, which changes Ray's verbose handling.
    # Ray internally passes verbose="auto" (a string) but the new output path
    # in get_air_verbosity() expects int or AirVerbosity enum → AttributeError.
    # Force RAY_AIR_NEW_OUTPUT=0 for our HPO run to use the stable output path,
    # then restore the caller's value so AutoGluon's own runs are unaffected.
    _prev_ray_output = os.environ.get("RAY_AIR_NEW_OUTPUT")
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    try:
        results = tuner.fit()
    except ValueError as _ve:
        if "checkpoint_at_end" in str(_ve):
            # RunConfig caused the incompatibility — retry without it.
            logger.warning(
                "RunConfig triggered checkpoint_at_end error (%s). "
                "Retrying without RunConfig — experiment will not be persisted "
                "and resume will be unavailable for this run.", _ve,
            )
            os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
            tuner_no_rc = tune.Tuner(
                trainable, param_space=ray_space, tune_config=_tune_config,
            )
            results = tuner_no_rc.fit()
        else:
            raise
    finally:
        if _prev_ray_output is None:
            os.environ.pop("RAY_AIR_NEW_OUTPUT", None)
        else:
            os.environ["RAY_AIR_NEW_OUTPUT"] = _prev_ray_output

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
    best_result = results.get_best_result(metric=metric, mode=_ray_mode)
    best_hyperparams = best_result.config

    with open(output_dir / "best_hyperparams_ray.json", "w") as fh:
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
        trials_csv_path = output_dir / "hpo_trials_ray.csv"
        results.get_dataframe().to_csv(trials_csv_path, index=False)
        logger.info("Saved Ray trial results to %s", trials_csv_path)

    # --- Remove Ray experiment directory (optional) ---
    # After a successful HPO run the best trial is in best_trial/ and all metrics
    # are in hpo_trials_ray.csv; the ray_experiment/ tree is no longer needed.
    if cleanup_trials and exp_storage.exists():
        shutil.rmtree(exp_storage)
        logger.info("Removed Ray experiment dir: %s", exp_storage)

    return best_hyperparams, results
