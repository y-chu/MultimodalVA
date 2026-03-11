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
import shutil
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedShuffleSplit

from ..utils.metrics import csmf_accuracy, score_predictions, sample_hyperparams  # noqa: F401
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
    n_trials: int = 20,
    metric: str = "accuracy",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    study_name: str = "tabular_hpo",
    storage_path: str | None = None,
    load_if_exists: bool = True,
    enable_pruning: bool = False,
    save_trials_csv: bool = True,
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
        n_trials:       Total Optuna trials. Default 20.
        metric:         Metric to maximise — "accuracy", "f1_macro", "f1_weighted",
                        "csmf_accuracy". Default "accuracy".
                        Use "csmf_accuracy" or "f1_macro" for imbalanced VA data.
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

    Returns:
        best_hyperparams: Dict from the best trial — pass to train(hyperparams=...).
        study:            Completed Optuna study object.

    Raises:
        ValueError: If model_name is not supported or metric is invalid.
    """
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
        storage_path = f"sqlite:///{output_dir}/hpo_{model_name}.db"

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

        # Compute all 4 metrics; store as user attributes for traceability.
        # Only `metric` drives Optuna's optimisation.
        all_scores = {
            m: score_predictions(result.top1, m)
            for m in ("accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
        }
        for name, val in all_scores.items():
            trial.set_user_attr(name, val)

        return all_scores[metric]

    pruner = (
        MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if enable_pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=pruner,
        study_name=study_name,
        storage=storage_path,
        load_if_exists=load_if_exists,
    )

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = n_trials - completed
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("All %d trials already completed. Skipping optimization.", n_trials)

    best_hyperparams = study.best_params

    # Copy best trial artifacts
    best_trial_dir = output_dir / f"trial_{study.best_trial.number}"
    best_output_dir = output_dir / "best_trial"
    if best_trial_dir.exists():
        shutil.copytree(best_trial_dir, best_output_dir, dirs_exist_ok=True)

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
) -> None:
    """Single-trial trainable for Ray Tune (tabular pipeline).

    Called once per trial by a Ray worker process.  Runs ``train()`` followed
    by ``predict()`` and reports all four evaluation metrics to the Ray runtime
    via ``ray.train.report()``.

    Numpy arrays (``X_opt_train``, ``y_opt_train``, ``X_opt_val``,
    ``y_opt_val``) are passed via ``tune.with_parameters()`` — stored once in
    the Ray object store and referenced by all workers without re-serialisation.

    Unlike the text trainable, no GPU memory clearing is needed here because
    sklearn / tree-model fits do not allocate PyTorch tensors.  GPU-accelerated
    variants (CatBoost, LightGBM, XGBoost) manage their own device memory and
    release it when the model object goes out of scope.

    This function must be defined at module level (not nested) so that Ray can
    serialise it by reference when dispatching to remote workers.
    """
    from ray import train as _ray_train

    trial_dir = Path(_ray_train.get_context().get_trial_dir())

    all_scores: dict = {
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "csmf_accuracy": 0.0,
    }
    try:
        train(
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
        result = predict(trial_dir, X_opt_val, y_opt_val)
        all_scores = {
            m: score_predictions(result.top1, m)
            for m in ("accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
        }
    finally:
        # ray.train.report() must always be called so Ray records the outcome
        # rather than leaving the trial in RUNNING state indefinitely.
        _ray_train.report(all_scores)


def optimize_ray(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 20,
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
    # ---- Output ---------------------------------------------------------
    save_trials_csv: bool = True,
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
        n_trials:       Total number of trials. Default 20.
        metric:         Metric to maximise — ``"accuracy"``, ``"f1_macro"``,
                        ``"f1_weighted"``, ``"csmf_accuracy"``. Default ``"accuracy"``.
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
    try:
        import ray
        from ray import tune, air
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
    search_alg: object = OptunaSearch(
        metric=metric,
        mode="max",
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

    # --- Tuner ---
    tuner = tune.Tuner(
        trainable,
        param_space=ray_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode="max",
            num_samples=n_trials,
            search_alg=search_alg,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=air.RunConfig(
            storage_path=str(output_dir),
            name="tabular_hpo_ray",
            log_to_file=True,
        ),
    )

    logger.info(
        "optimize_ray: launching %d trials  model=%s  metric=%s  "
        "cpus/trial=%d  gpus/trial=%.1f",
        n_trials, model_name, metric, num_cpus_per_trial, num_gpus_per_trial,
    )
    results = tuner.fit()

    # --- Extract best result ---
    best_result = results.get_best_result(metric=metric, mode="max")
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
    best_trial_path = Path(best_result.path)
    best_output_dir = output_dir / "best_trial"
    if best_trial_path.exists():
        shutil.copytree(best_trial_path, best_output_dir, dirs_exist_ok=True)
        logger.info("Copied best trial artifacts to %s", best_output_dir)

    # --- Save all trial results as CSV ---
    if save_trials_csv:
        trials_csv_path = output_dir / "hpo_trials_ray.csv"
        results.get_dataframe().to_csv(trials_csv_path, index=False)
        logger.info("Saved Ray trial results to %s", trials_csv_path)

    return best_hyperparams, results
