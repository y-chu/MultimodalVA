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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna
    import ray

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


# Updated for ~20 classes and 3k-5k samples

# Default search space: (type, *args)
#   float_log  — log-uniform float:  (low, high)       best for scale-sensitive params like LR
#   float      — uniform float:       (low, high)
#   int        — uniform int:         (low, high)
#   categorical — discrete choice:   ([values])
DEFAULT_SEARCH_SPACE: dict = {
    # AdamW learning rate — most influential hyperparameter for BERT fine-tuning.
    # BERT paper recommends 2e-5 to 5e-5; log-uniform covers the relevant scale.
    # Wider range (1e-5, 1e-4) if the default range consistently hits a boundary.
    "learning_rate": ("float_log", 8e-6, 4e-5),

    # Per-device training batch size.
    # 16 is the standard for 16 GB GPUs; use 8 if hitting OOM, or add 32 for larger GPUs.
    # Effective batch size = batch_size × gradient_accumulation_steps × n_GPUs.
    "batch_size": ("categorical", [8, 16, 32]),

    # Number of full passes over the training data.
    # 3–5 epochs is typical for BERT fine-tuning; small datasets may benefit from more (5–10).
    # Always included so that optimize() returns an epoch count for the final train() call.
    "epochs": ("categorical", [3, 5, 7]),

    # L2 regularisation on non-bias / non-LayerNorm parameters (AdamW decoupled decay).
    # 0.01 is the HuggingFace default; 0.0–0.1 covers most reasonable settings.
    # Increase toward 0.1–0.3 if overfitting on small datasets.
    "weight_decay": ("float", 0.0, 0.15),

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
    "freeze_layers": ("categorical", [2, 4, 6]),

    # Classifier dropout (high impact for small/imbalanced data) 
    # strong regularizer for small datasets with 20-60 classes; 0.1–0.3 is a good range to explore. 0.0 (no dropout) can work well for larger datasets.
    "classifier_dropout": ("float", 0.1, 0.4),

    # Label smoothing (stabilizes multi-class training)
    "label_smoothing": ("float", 0.0, 0.1),

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
    "focal_gamma": ("float", 1.0, 4.0),

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
    "lora_alpha": ("categorical", [8, 16, 32, 64]),

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
    n_trials: int = 30,
    metric: str = "accuracy",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    study_name: str = "text_hpo",
    use_lora: bool = True,
    use_focal: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = 2,
    storage_path: str | None = None,
    load_if_exists: bool = True,
    enable_pruning: bool = True,
    save_trials_csv: bool = True,
    cleanup_trials: bool = True,
    use_fast: bool = False,
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
        cleanup_trials: Delete all per-trial directories (``trial_*/``) after HPO
                        completes.  The best trial is preserved at ``best_trial/``
                        before deletion.  Each trial directory contains a full model
                        checkpoint (≈400 MB for BERT-base); 20 trials can accumulate
                        8+ GB.  The SQLite study DB, ``best_hyperparams.json``,
                        ``best_trial/``, and the CSV are all kept.  Default True.

    Returns:
        best_hyperparams: Dict of hyperparameter values from the best trial.
                          Can be passed directly to train(hyperparams=best_hyperparams).
        study: The completed Optuna study object (for further analysis / visualisation).
    """
    # Raise the open-file-descriptor limit before the HPO loop.
    # On macOS the default soft limit is 256; Longformer HPO can exhaust this
    # (model weights, tokenizer files, checkpoint dirs, Dropbox daemon) causing
    # SQLite to fail with "unable to open database file" which in turn makes
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

    if storage_path is None:
        safe_name = model_name.replace("/", "_")
        # Use resolved absolute path so the DB is always found regardless of CWD.
        storage_path = f"sqlite:///{output_dir.resolve()}/hpo_{safe_name}.db"
        # Warn when the SQLite DB will live inside a cloud-sync folder.
        # Dropbox / iCloud / OneDrive hold their own file locks on .db files while
        # syncing, which can cause "unable to open database file" mid-trial.
        # Pass an explicit storage_path pointing to a local dir (e.g. /tmp/) to avoid this.
        _cloud_markers = ("Dropbox", "iCloudDrive", "OneDrive", "Google Drive", "Box")
        _db_path_str = str(output_dir.resolve())
        if any(m in _db_path_str for m in _cloud_markers):
            logger.warning(
                "Optuna SQLite DB is inside a cloud-sync folder (%s).  "
                "Cloud sync daemons can hold file locks that prevent SQLite writes "
                "mid-trial (→ 'unable to open database file').  "
                "Pass storage_path='/tmp/hpo_%s.db' (or any local path) to avoid this.",
                _db_path_str, safe_name,
            )
    elif "://" not in storage_path:
        # User passed a plain file path (e.g. "/tmp/hpo.db") without the SQLite URL
        # scheme.  Convert it so SQLAlchemy can parse it.
        storage_path = f"sqlite:///{Path(storage_path).resolve()}"
        logger.info("storage_path converted to SQLite URL: %s", storage_path)

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
                use_fast=use_fast,
            )
            result = predict(trial_dir, opt_val, use_fast=use_fast)
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
            "Tip: pass storage_path='/tmp/hpo_<name>.db' if output_dir is on Dropbox/cloud.",
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

    # --- Remove per-trial directories (optional) ---
    # Each trial dir contains a full model copy (≈400 MB for BERT-base).
    # After HPO the best trial is in best_trial/ and hyperparams in best_hyperparams.json;
    # the individual trial_N/ directories serve no further purpose.
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
    use_fast: bool = False,
) -> dict:
    """Single trial for Ray Tune — fully compatible with Ray 2.x / Python 3.13.

    Returns a metrics dict, which Ray Tune records as the trial's final result.
    Returning a dict is the portable way to report metrics from a function
    trainable — it works across all Ray 2.x versions without requiring a
    Ray Train session (``ray.train.report()`` is the Train API and raises /
    hangs when called outside a proper Train context such as TorchTrainer).

    Imports of ``multimodalva`` sub-modules are deferred to function body so
    that Ray worker processes resolve them correctly regardless of how the
    package was launched (installed wheel, editable install, or direct script).
    """
    import logging as _logging
    import torch
    from ray import tune as _ray_tune

    _trial_logger = _logging.getLogger(__name__)

    # --- Ensure multimodalva is importable in the worker process ---
    import sys as _sys
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    from multimodalva.text.train import train as _train
    from multimodalva.text.predict import predict as _predict
    from multimodalva.utils.metrics import score_predictions, log_loss_from_full

    # --- Trial artifact directory (Ray Tune API) ---
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
            use_fast=use_fast,
        )

        # --- Evaluate ---
        result = _predict(trial_dir, opt_val, use_fast=use_fast)
        all_scores.update(
            {
                m: score_predictions(result.top1, m)
                for m in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
            }
        )
        all_scores["log_loss"] = log_loss_from_full(result.full, result.id2label)

    except Exception:
        _trial_logger.exception(
            "Ray trial failed (config=%s); reporting zero/inf scores.", config
        )

    finally:
        # Free GPU / MPS memory before the next trial.
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Return the metrics dict — Ray Tune treats the return value of a function
    # trainable as the trial's final reported result.  This avoids calling
    # ray.train.report(), which is the Ray Train API and raises or hangs when
    # invoked outside a Train session (e.g. TorchTrainer).
    return all_scores


# -----------------------
# Main Ray HPO function
# -----------------------
def optimize_ray(
    train_dataset,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    n_trials: int | None = None,
    metric: str = "accuracy",
    search_space: dict | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    use_lora: bool = True,
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
    use_asha: bool = False,
    use_fast: bool = False,
) -> tuple[dict, Any]:
    """Distributed HPO using Ray Tune.

    Extra args vs the original signature:

    Args:
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
            "ray[tune] and optuna-integration are required for optimize_ray(). "
            "Install with:  pip install 'ray[tune]>=2.9' 'optuna-integration>=3.4'\n"
            "Or:            pip install multimodalva[ray]"
        ) from exc

    # --- Resolve n_trials default (depends on use_asha) ---
    if n_trials is None:
        n_trials = 60 if use_asha else 30

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Auto-redirect to optimize() on MPS/CPU-only ---
    # Ray Train v2 is incompatible with function trainables on non-CUDA devices:
    # RunConfig auto-injects checkpoint_at_end=True which raises ValueError,
    # forcing a no-RunConfig fallback where trial state goes to ~/ray_results
    # and cannot be restored → resume always restarts from trial 1.
    # On MPS/CPU, optimize() (Optuna + SQLite) is the correct backend.
    try:
        import torch as _torch
        _cuda_available = _torch.cuda.device_count() > 0
    except ImportError:
        _cuda_available = False
    if not _cuda_available:
        _safe = model_name.replace("/", "_")
        logger.info(
            "optimize_ray(): no CUDA GPUs detected — redirecting to optimize() "
            "(Optuna sequential, SQLite-backed). Ray Tune requires CUDA for "
            "reliable experiment persistence and resume. "
            "SQLite DB: %s/hpo_%s.db",
            output_dir, _safe,
        )
        return optimize(
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
            storage_path=str(output_dir / f"hpo_{_safe}.db"),
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

    # --- Build effective search space ---
    active_space = dict(DEFAULT_SEARCH_SPACE)
    if use_lora:
        active_space.update(LORA_SEARCH_SPACE)
    if use_focal:
        active_space.update(FOCAL_SEARCH_SPACE)
    if search_space:
        active_space.update(search_space)

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

    # Warn early on MPS/CPU-only: Ray Train v2 is incompatible with function
    # trainables when RunConfig is used, causing checkpoint_at_end ValueError.
    # This means experiment state cannot be persisted → resume is impossible.
    # The fallback (tuner_no_rc) saves to ~/ray_results which we cannot restore.
    # Users on Apple Silicon or CPU-only machines should use optimize() instead.
    if available_gpus == 0:
        logger.warning(
            "optimize_ray(): no CUDA GPUs available (MPS/CPU-only environment). "
            "Ray Train v2 is incompatible with function trainables on non-CUDA "
            "devices — experiment state cannot be persisted and resume will not "
            "work across restarts. "
            "RECOMMENDATION: use optimize() (Optuna backend) instead — it uses "
            "SQLite for reliable crash recovery and resume on MPS/CPU machines.",
        )

    _ray_mode = "min" if METRIC_DIRECTION.get(metric, "maximize") == "minimize" else "max"

    search_alg = OptunaSearch(
        metric=metric,
        mode=_ray_mode,
        seed=random_state,
        storage=None,   # explicit None ensures self._storage is always initialized
    )
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
        use_fast=use_fast,
    )
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
            # On MPS/Apple Silicon: switch to optimize() (Optuna) for
            # reliable resume.  optimize() uses SQLite and resumes correctly
            # after crashes or SLURM preemptions.
            logger.warning(
                "checkpoint_at_end error from Ray Train v2 (%s). "
                "Retrying without RunConfig — this run's trial state will be "
                "saved to ~/ray_results (not %s) and CANNOT be resumed. "
                "If you need resume support, use optimize() (Optuna backend) "
                "which persists state in SQLite and resumes correctly on "
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
            "Likely causes: GPU OOM, file descriptor limit, or Ray worker errors.  "
            "Consider: reduce batch_size, set num_gpus_per_trial=0 for CPU-only, "
            "or pass storage_path='/tmp/...' to move the SQLite DB off cloud storage.",
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

    # --- Remove Ray experiment directory (optional) ---
    # The ray_experiment/ tree holds Ray's per-trial artifacts (model weights,
    # optimizer state, Ray metadata).  After a successful HPO run the best trial
    # is already in best_trial/ and all metrics are in hpo_trials_ray.csv, so
    # the experiment tree is no longer needed and can be several GB for text models.
    if cleanup_trials and exp_storage.exists():
        shutil.rmtree(exp_storage)
        logger.info("Removed Ray experiment dir: %s", exp_storage)

    return best_hyperparams, results