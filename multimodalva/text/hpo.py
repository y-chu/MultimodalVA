"""
Step 5: Hyperparameter optimization.

Two backends are provided; both accept the same search_space format and return
``(best_hyperparams, backend_object)`` so callers can switch seamlessly:

    optimize()      — Optuna (TPE sampler, SQLite persistence)
                      Best for single-machine sequential or lightly parallel HPO.

    optimize_ray()  — Ray Tune (OptunaSearch / TPE, distributed runtime)
                      Best for multi-GPU machines or multi-node clusters.
                      Runs N trials concurrently, one GPU per trial (configurable).

Uses train() and predict() internally.
Input:  train_dataset, label2id, id2label  from prepare_dataset()
Output: best_hyperparams dict and backend study / ResultGrid object
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

from sklearn.model_selection import train_test_split
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
    log_loss_from_full,
    METRIC_DIRECTION,
)
from multimodalva.text.train import train, _get_dataset_labels
from multimodalva.text.predict import predict

logger = logging.getLogger(__name__)


# Default search space: (type, *args)
#   float_log  — log-uniform float:  (low, high)       best for scale-sensitive params like LR
#   float      — uniform float:       (low, high)
#   int        — uniform int:         (low, high)
#   categorical — discrete choice:   ([values])
DEFAULT_SEARCH_SPACE: dict = {
    # AdamW learning rate — most influential hyperparameter for BERT fine-tuning.
    # BERT paper recommends 2e-5 to 5e-5; log-uniform covers the relevant scale.
    # Wider range (1e-5, 1e-4) if the default range consistently hits a boundary.
    "learning_rate": ("float_log", 1e-5, 4e-5),

    # Per-device training batch size.
    # 16 is the standard for 16 GB GPUs; use 8 if hitting OOM, or add 32 for larger GPUs.
    # Effective batch size = batch_size × gradient_accumulation_steps × n_GPUs.
    "batch_size": ("categorical", [8, 16]),

    # Number of full passes over the training data.
    # 3–5 epochs is typical for BERT fine-tuning; small datasets may benefit from more (5–10).
    # Always included so that optimize() returns an epoch count for the final train() call.
    "epochs": ("categorical", [3, 5]),

    # L2 regularisation on non-bias / non-LayerNorm parameters (AdamW decoupled decay).
    # 0.01 is the HuggingFace default; 0.0–0.1 covers most reasonable settings.
    # Increase toward 0.1–0.3 if overfitting on small datasets.
    "weight_decay": ("float", 0.0, 0.1),

    # Fraction of total training steps used for linear LR warmup.
    # BERT paper uses 0.1 (10%); 0.0–0.06 is common in practice.
    # Larger values (0.1–0.2) can help stabilize training on noisy or imbalanced data.
    "warmup_ratio": ("float", 0.0, 0.1),

    # Number of gradient steps to accumulate before an optimizer update.
    # Simulates a larger effective batch size without extra GPU memory.
    # Use 4 or 8 when batch_size is forced low by memory constraints.
    "gradient_accumulation_steps": ("categorical", [1, 2]),

    # Number of encoder layers to freeze from the bottom (embeddings + N layers).
    # Freezing reduces trainable parameters and acts as regularisation, which is
    # especially useful for small or domain-specific datasets like VA narratives.
    # 0  — full fine-tuning; best when data is large or domain is very different from
    #       BERT's pre-training corpus.
    # 2  — light freeze (default); good starting point for most VA datasets.
    # 4  — moderate; useful when the dataset is small (<2 000 samples).
    # 6  — aggressive (half of BERT-base's 12 layers); use when heavily overfitting
    #       or as a final check that lower layers are not needed.
    # Note: values above 6 rarely help and reduce model capacity significantly.
    "freeze_layers": ("categorical", [0, 2, 4, 6]),
}

# Focal loss search space (merged in when use_focal=True).
# loss_type is fixed to "focal" — only the tunable focal parameters are searched.
# Recommended HPO metric when using focal: "f1_macro", "balanced_accuracy",
# or "csmf_accuracy".  Avoid "log_loss" (focal loss distorts calibration).
FOCAL_SEARCH_SPACE: dict = {
    # Focus strength γ — how aggressively easy examples are down-weighted.
    # γ=0 reduces to standard CE.  γ=2 is the RetinaNet default and covers
    # most VA imbalance settings.  γ>3 is rarely beneficial and can cause
    # gradient instability on very small classes.
    "focal_gamma": ("float", 0.5, 5.0),

    # Per-class alpha weights (α) — rebalances the gradient contribution
    # across classes before the focal term is applied.
    # "effective_n" (Cui et al. 2019) is recommended when any class has
    # fewer than ~10 training samples; it prevents weight blow-up.
    # "balanced" (sklearn) is adequate for moderate imbalance.
    "class_weights": ("categorical", ["balanced", "effective_n"]),
}

# Additional LoRA search space entries (merged in when use_lora=True)
LORA_SEARCH_SPACE: dict = {
    # Rank of the LoRA low-rank decomposition — controls adapter capacity.
    # Higher rank = more trainable parameters and expressiveness, but slower.
    # r=8 is the LoRA paper default; r=16–32 for complex tasks or large label sets.
    "lora_r": ("categorical", [4, 8, 16]),

    # LoRA scaling factor applied to the adapter output (output *= lora_alpha / lora_r).
    # Common heuristic: set lora_alpha = 2 × lora_r (e.g. r=8 → alpha=16).
    # Larger alpha amplifies adapter contributions; too large can destabilise training.
    "lora_alpha": ("categorical", [16, 32, 64]),

    # Dropout applied inside LoRA adapters for regularisation.
    # 0.05 is the LoRA paper default; 0.0 (no dropout) often works well for small adapters.
    "lora_dropout": ("float", 0.0, 0.1),
}


def optimize(
    train_dataset,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 20,
    metric: str = "accuracy",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    study_name: str = "text_hpo",
    use_lora: bool = False,
    use_focal: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = 3,
    storage_path: str | None = None,
    load_if_exists: bool = True,
    enable_pruning: bool = True,
    save_trials_csv: bool = True,
) -> tuple[dict, optuna.Study]:
    """Run Optuna hyperparameter search using train() and predict().

    Internally splits train_dataset into an (1 - val_size) / val_size
    opt-train / opt-eval split. Each trial trains on opt-train and is scored
    on opt-eval. The held-out test set from prepare_dataset() is never used here.

    After all trials complete, the best trial's artifacts are copied to
    output_dir/best_trial/ and best hyperparams are saved as best_hyperparams.json.
    The returned best_hyperparams can be passed directly to
    train(hyperparams=best_hyperparams) for a final run on the full training set.

    Args:
        train_dataset: Tokenized ClassificationDataset from prepare_dataset().
        label2id: Label-to-integer mapping from prepare_dataset().
        id2label: Integer-to-label mapping from prepare_dataset().
        model_name: HuggingFace model name or local path.
        output_dir: Root directory for all trial outputs.
        n_trials: Total number of Optuna trials to run. Default 20.
        metric: Metric to maximise — "accuracy", "balanced_accuracy",
                "f1_macro", "f1_weighted", "csmf_accuracy". Default "accuracy".
                Use "balanced_accuracy", "csmf_accuracy", or "f1_macro" for imbalanced VA data.
        search_space: Dict defining parameter ranges. Defaults to DEFAULT_SEARCH_SPACE
                      (plus LORA_SEARCH_SPACE when use_lora=True, plus
                      FOCAL_SEARCH_SPACE when use_focal=True).
        val_size: Fraction of train_dataset held out for trial evaluation. Default 0.2.
        random_state: Seed for the internal stratified split and Optuna sampler.
                      Default 42.
        study_name: Name for the Optuna study. Default "text_hpo".
        use_lora: Apply LoRA adapters during each trial. Default False.
        use_focal: Use focal loss during each trial.  When True, merges
                   FOCAL_SEARCH_SPACE (focal_gamma, class_weights) and fixes
                   loss_type="focal" in every trial's hyperparams.
                   Recommended for severe class imbalance; pair with
                   metric="f1_macro" or "balanced_accuracy". Default False.
        gradient_checkpointing: Enable gradient checkpointing per trial. Default False.
        early_stopping_patience: Early stopping patience passed to each trial's train().
                                  Set to None to disable. Default 3.
        storage_path: SQLite URL for persisting the study (enables resume across
                      sessions). Defaults to output_dir/hpo_<model_name>.db.
        load_if_exists: Resume an existing study if storage_path already contains one.
                        Default True.
        enable_pruning: Use Optuna's MedianPruner to stop unpromising trials early.
                        Default True.
        save_trials_csv: Save all trial results to output_dir/hpo_trials.csv after
                         optimization completes. Includes hyperparameters, objective
                         score, all user_attrs_* metrics, state, and duration.
                         Default True.

    Returns:
        best_hyperparams: Dict of hyperparameter values from the best trial.
                          Can be passed directly to train(hyperparams=best_hyperparams).
        study: The completed Optuna study object (for further analysis / visualisation).
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

    # Build effective search space
    active_space = dict(DEFAULT_SEARCH_SPACE)
    if use_lora:
        active_space.update(LORA_SEARCH_SPACE)
    if use_focal:
        active_space.update(FOCAL_SEARCH_SPACE)
    if search_space:
        active_space.update(search_space)

    if storage_path is None:
        safe_name = model_name.replace("/", "_")
        # Use resolved absolute path so the DB is always found regardless of CWD.
        storage_path = f"sqlite:///{output_dir.resolve()}/hpo_{safe_name}.db"

    # --- Stratified internal split ---
    all_labels = _get_dataset_labels(train_dataset)
    indices = list(range(len(train_dataset)))
    opt_train_idx, opt_val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=all_labels,
        random_state=random_state,
    )
    opt_train = Subset(train_dataset, opt_train_idx)
    opt_val = Subset(train_dataset, opt_val_idx)

    # --- Objective ---
    def objective(trial: optuna.Trial) -> float:
        import torch

        hp = sample_hyperparams(trial, active_space)
        if use_focal:
            hp["loss_type"] = "focal"
        trial_dir = output_dir / f"trial_{trial.number}"

        try:
            train(
                train_dataset=opt_train,
                label2id=label2id,
                id2label=id2label,
                model_name=model_name,
                output_dir=trial_dir,
                hyperparams=hp,
                # val_size=0.1: train() carves an internal eval split from opt_train
                # for early stopping and best-checkpoint selection. This is separate
                # from opt_val, which is used below to score the trial.
                val_size=0.1,
                use_lora=use_lora,
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                # Each trial gets a fresh directory — never resume from a previous trial.
                resume=False,
            )
            result = predict(trial_dir, opt_val)
            # Compute all metrics and store as user attributes for traceability.
            # Only `metric` drives Optuna's optimization; the rest are available
            # in study.trials_dataframe() as user_attrs_* columns.
            all_scores = {
                m: score_predictions(result.top1, m)
                for m in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
            }
            all_scores["log_loss"] = log_loss_from_full(result.full, result.id2label)
            for name, val in all_scores.items():
                trial.set_user_attr(name, val)
            score = all_scores[metric]
        finally:
            # Release GPU/MPS memory after each trial to avoid OOM on subsequent trials.
            torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return score

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
        storage=storage_path,
        load_if_exists=load_if_exists,
    )

    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    remaining = n_trials - completed
    if remaining > 0:
        # catch=(Exception,) lets Optuna mark a failed trial as FAILED and continue
        # with remaining trials instead of crashing the whole HPO run (e.g. on OOM).
        study.optimize(objective, n_trials=remaining, catch=(Exception,))
    else:
        logger.info("All %d trials already completed. Skipping optimization.", n_trials)

    best_hyperparams = study.best_params

    # --- Copy best trial artifacts ---
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
        study.trials_dataframe().to_csv(trials_csv_path, index=False)
        logger.info("Saved trial results to %s", trials_csv_path)

    return best_hyperparams, study


# ---------------------------------------------------------------------------
# Ray Tune — distributed HPO across multiple GPUs / cluster nodes
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

# -----------------------
# Ray Trainable: Single trial
# -----------------------
def _ray_trial_fn(
    config: dict,
    *,
    opt_train,
    opt_val,
    label2id: dict,
    id2label: dict,
    model_name: str,
    use_lora: bool,
    use_focal: bool,
    gradient_checkpointing: bool,
    early_stopping_patience: int | None,
) -> None:
    """Single trial for Ray Tune — fully compatible with Ray 2.x / Python 3.13.

    Ray 2.x API used:
      - ``ray.train.get_context().get_trial_dir()`` for the trial artifact directory
        (replaces deprecated ``tune.get_trial_dir()`` / ``ctx.logdir``).
      - ``ray.train.report(metrics_dict)`` for metric reporting
        (replaces deprecated ``tune.report(**kwargs)``).

    Imports of ``multimodalva`` sub-modules are deferred to function body so that
    Ray worker processes resolve them correctly regardless of how the package was
    launched (installed wheel, editable install, or direct script execution).
    """
    import torch
    # ray.tune.get_context() is the correct API for function trainables passed to
    # Ray Tune (ray.train.get_context() is deprecated in that context per
    # https://github.com/ray-project/ray/issues/49454).
    # ray.train.report() remains the canonical reporting call for both Tune and Train.
    from ray import tune as _ray_tune
    import ray.train as _ray_train

    # --- Ensure multimodalva is importable in the worker process ---
    # Each Ray worker is a fresh Python process; insert the project root into
    # sys.path so absolute imports resolve even without a formal pip install.
    import sys as _sys
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    # Import inside the function body to avoid the name collision that occurs when
    # ``from ray import train`` is used at the top of the file, which would shadow
    # ``multimodalva.text.train.train``.
    from multimodalva.text.train import train as _train
    from multimodalva.text.predict import predict as _predict
    from multimodalva.utils.metrics import score_predictions, log_loss_from_full

    # --- Trial artifact directory (Ray 2.x Tune API) ---
    ctx = _ray_tune.get_context()
    trial_dir = Path(ctx.get_trial_dir())
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

    try:
        # --- Train ---
        _train(
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
        )

        # --- Evaluate ---
        result = _predict(trial_dir, opt_val)
        all_scores.update(
            {
                m: score_predictions(result.top1, m)
                for m in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
            }
        )
        all_scores["log_loss"] = log_loss_from_full(result.full, result.id2label)

    finally:
        # --- Free GPU / MPS memory before the next trial ---
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Ray 2.x: report metrics as a plain dict — tune.report(**kwargs) is deprecated.
        _ray_train.report(all_scores)


# -----------------------
# Main Ray HPO function
# -----------------------
def optimize_ray(
    train_dataset,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 20,
    metric: str = "accuracy",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    use_lora: bool = False,
    use_focal: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = 3,
    ray_address: str | None = None,
    num_gpus_per_trial: float = 1.0,
    num_cpus_per_trial: int = 4,
    max_concurrent_trials: int | None = None,
    save_trials_csv: bool = True,
) -> tuple[dict, "ray.tune.ResultGrid"]:
    """Distributed HPO using Ray Tune."""
    _VALID_METRICS = {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy", "log_loss"}
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid metrics: {sorted(_VALID_METRICS)}"
        )

    import ray
    from ray import tune
    from ray.train import RunConfig  # ray.air.RunConfig is deprecated in Ray 2.7+
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import ConcurrencyLimiter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build effective search space ---
    active_space = dict(DEFAULT_SEARCH_SPACE)
    if use_lora:
        active_space.update(LORA_SEARCH_SPACE)
    if use_focal:
        active_space.update(FOCAL_SEARCH_SPACE)
    if search_space:
        active_space.update(search_space)
    ray_space = _to_ray_space(active_space)

    # --- Stratified split ---
    all_labels = _get_dataset_labels(train_dataset)
    indices = list(range(len(train_dataset)))
    opt_train_idx, opt_val_idx = train_test_split(
        indices, test_size=val_size, stratify=all_labels, random_state=random_state
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
    if num_gpus_per_trial > 0 and available_gpus == 0:
        logger.warning(
            "num_gpus_per_trial=%.1f but no GPUs detected; trials will queue indefinitely.",
            num_gpus_per_trial,
        )

    _ray_mode = "min" if METRIC_DIRECTION.get(metric, "maximize") == "minimize" else "max"

    search_alg = OptunaSearch(metric=metric, mode=_ray_mode, seed=random_state)
    if max_concurrent_trials is not None:
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent_trials)

    # --- Trainable ---
    trainable = tune.with_parameters(
        _ray_trial_fn,
        opt_train=opt_train,
        opt_val=opt_val,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        use_lora=use_lora,
        use_focal=use_focal,
        gradient_checkpointing=gradient_checkpointing,
        early_stopping_patience=early_stopping_patience,
        # output_dir is NOT passed here — _ray_trial_fn derives the trial directory
        # from ray.train.get_context().get_trial_dir() (Ray 2.x API).
    )
    trainable = tune.with_resources(
        trainable, resources={"cpu": num_cpus_per_trial, "gpu": num_gpus_per_trial}
    )

    # --- Tuner ---
    tuner = tune.Tuner(
        trainable,
        param_space=ray_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=_ray_mode,
            num_samples=n_trials,
            search_alg=search_alg,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=RunConfig(storage_path=str(output_dir), name="text_hpo_ray"),
    )

    logger.info("Launching %d trials, metric=%s, gpus/trial=%.1f, cpus/trial=%d",
                n_trials, metric, num_gpus_per_trial, num_cpus_per_trial)

    # AutoGluon sets RAY_AIR_NEW_OUTPUT=1, which changes Ray's verbose handling.
    # Ray internally passes verbose="auto" (a string) but the new output path
    # in get_air_verbosity() expects int or AirVerbosity enum → AttributeError.
    # Force RAY_AIR_NEW_OUTPUT=0 for our HPO run to use the stable output path,
    # then restore the caller's value so AutoGluon's own runs are unaffected.
    _prev_ray_output = os.environ.get("RAY_AIR_NEW_OUTPUT")
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    try:
        results = tuner.fit()
    finally:
        if _prev_ray_output is None:
            os.environ.pop("RAY_AIR_NEW_OUTPUT", None)
        else:
            os.environ["RAY_AIR_NEW_OUTPUT"] = _prev_ray_output

    best_result = results.get_best_result(metric=metric, mode=_ray_mode)
    best_hyperparams = best_result.config

    with open(output_dir / "best_hyperparams_ray.json", "w") as fh:
        json.dump(best_hyperparams, fh, indent=2)

    # --- Copy best trial artifacts ---
    best_trial_path = Path(best_result.path)
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
        results.get_dataframe().to_csv(output_dir / "hpo_trials_ray.csv", index=False)

    return best_hyperparams, results