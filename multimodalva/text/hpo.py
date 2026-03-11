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
import shutil
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from ..utils.metrics import csmf_accuracy, score_predictions, sample_hyperparams  # noqa: F401
from .predict import predict
from .train import _get_dataset_labels, train

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
        metric: Metric to maximise — "accuracy", "f1_macro", "f1_weighted",
                "csmf_accuracy". Default "accuracy".
                Use "csmf_accuracy" or "f1_macro" for imbalanced VA data.
        search_space: Dict defining parameter ranges. Defaults to DEFAULT_SEARCH_SPACE
                      (plus LORA_SEARCH_SPACE when use_lora=True).
        val_size: Fraction of train_dataset held out for trial evaluation. Default 0.2.
        random_state: Seed for the internal stratified split and Optuna sampler.
                      Default 42.
        study_name: Name for the Optuna study. Default "text_hpo".
        use_lora: Apply LoRA adapters during each trial. Default False.
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build effective search space
    active_space = dict(DEFAULT_SEARCH_SPACE)
    if use_lora:
        active_space.update(LORA_SEARCH_SPACE)
    if search_space:
        active_space.update(search_space)

    if storage_path is None:
        safe_name = model_name.replace("/", "_")
        storage_path = f"sqlite:///{output_dir}/hpo_{safe_name}.db"

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
                for m in ("accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
            }
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
        direction="maximize",
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
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("All %d trials already completed. Skipping optimization.", n_trials)

    best_hyperparams = study.best_params

    # --- Copy best trial artifacts ---
    best_trial_dir = output_dir / f"trial_{study.best_trial.number}"
    best_output_dir = output_dir / "best_trial"
    if best_trial_dir.exists():
        shutil.copytree(best_trial_dir, best_output_dir, dirs_exist_ok=True)

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


def _ray_trial_fn(
    config: dict,
    *,
    opt_train,
    opt_val,
    label2id: dict,
    id2label: dict,
    model_name: str,
    use_lora: bool,
    gradient_checkpointing: bool,
    early_stopping_patience: int | None,
) -> None:
    """Single-trial trainable for Ray Tune.

    Called once per trial by a Ray worker process.  Runs ``train()`` followed
    by ``predict()`` and reports all four evaluation metrics to the Ray runtime
    via ``ray.train.report()``.

    Large dataset objects (``opt_train``, ``opt_val``) are passed through the
    Ray object store via ``tune.with_parameters()`` — they are referenced by
    pointer, not re-serialised for every trial.

    ``config`` contains the hyperparameter values sampled by the search
    algorithm and is forwarded verbatim to ``train()`` as ``hyperparams``.

    This function must be defined at module level (not nested) so that Ray can
    serialise it by reference when dispatching to remote workers.  The package
    must be installed (``pip install -e .``) on every worker node so that the
    relative imports resolve correctly.
    """
    import torch
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
            train_dataset=opt_train,
            label2id=label2id,
            id2label=id2label,
            model_name=model_name,
            output_dir=trial_dir,
            hyperparams=config,
            # val_size=0.1: train() carves an internal eval split from opt_train
            # for early stopping / best-checkpoint selection.  This is separate
            # from opt_val, which scores the trial below.
            val_size=0.1,
            use_lora=use_lora,
            gradient_checkpointing=gradient_checkpointing,
            early_stopping_patience=early_stopping_patience,
            # Never resume from another trial's checkpoint.
            resume=False,
        )
        result = predict(trial_dir, opt_val)
        all_scores = {
            m: score_predictions(result.top1, m)
            for m in ("accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
        }
    finally:
        # Free GPU/MPS memory before Ray marks the trial slot as available.
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # ray.train.report() must always be called, even on failure, so Ray
        # records the trial outcome rather than leaving it in RUNNING state.
        _ray_train.report(all_scores)


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
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = 3,
    # ---- Ray cluster / resource settings --------------------------------
    ray_address: str | None = None,
    num_gpus_per_trial: float = 1.0,
    num_cpus_per_trial: int = 4,
    max_concurrent_trials: int | None = None,
    # ---- Output ---------------------------------------------------------
    save_trials_csv: bool = True,
) -> tuple[dict, "ray.tune.ResultGrid"]:
    """Run distributed HPO with Ray Tune across multiple GPUs or cluster nodes.

    Mirrors the interface of :func:`optimize` (Optuna backend) but executes
    trials **concurrently** across all available GPUs and/or cluster nodes
    using the Ray distributed runtime.

    When to use this over :func:`optimize`
    ---------------------------------------
    * You have **multiple GPUs** on one machine and want N parallel trials
      simultaneously (one GPU each).
    * You are on a **cluster** (SLURM, Kubernetes, AWS, GCP, …) and want to
      spread trials across nodes — pass ``ray_address="auto"`` after starting
      a Ray cluster with ``ray start --head``.
    * You need **fault tolerance** — Ray automatically restarts failed trials
      and checkpoints progress so a node failure does not restart everything.

    Trial-level vs. intra-trial parallelism
    ----------------------------------------
    ``num_gpus_per_trial=1.0`` (default) allocates one whole GPU per trial.
    Ray schedules as many concurrent trials as total GPUs permit.  For
    example, on a 4-GPU node you get 4 concurrent trials.

    To use multiple GPUs *within* a single trial (data-parallel training),
    set ``num_gpus_per_trial > 1`` and configure ``train()`` for distributed
    data parallelism (``torch.distributed`` / ``accelerate``).

    Fractional GPUs (e.g. ``num_gpus_per_trial=0.5``) allow packing two
    trials onto one GPU — only safe when GPU memory permits.

    Cluster setup
    -------------
    .. code-block:: bash

        # On the head node (4 GPUs):
        ray start --head --num-gpus=4

        # On each worker node (4 GPUs):
        ray start --address=<HEAD_IP>:6379 --num-gpus=4

    Then pass ``ray_address="auto"`` (or ``"<HEAD_IP>:6379"`` explicitly).

    For **shared-filesystem clusters** (NFS / Lustre), set ``output_dir`` to a
    path accessible by all nodes — Ray writes trial artifacts there.  For
    cloud storage (S3 / GCS), pass an ``s3://`` / ``gs://`` URI as
    ``output_dir`` (requires ``pyarrow`` and the relevant cloud SDK).

    The package must be installed on every worker node so remote workers can
    import ``multimodalva``.  On SLURM clusters, activate the same conda/venv
    environment on all nodes before launching the Ray cluster.

    Search algorithm
    ----------------
    Uses ``OptunaSearch`` (Tree-structured Parzen Estimator — the same sampler
    as :func:`optimize`).  The acquisition function is shared across all
    parallel workers, so the search is sample-efficient even at high
    concurrency.

    Note on intra-trial ASHA pruning
    ---------------------------------
    Ray Tune's ``ASHAScheduler`` can kill underperforming trials mid-training
    (epoch-level), freeing GPUs sooner.  This requires the training loop to
    call ``ray.train.report({...})`` after each epoch — i.e. the HuggingFace
    ``Trainer`` needs a ``TuneReportCheckpointCallback`` from
    ``ray.tune.integration.huggingface``.  Adding an ``extra_callbacks``
    parameter to ``train()`` would unlock this.  Without it, all trials run to
    completion (no intra-trial pruning), which is still fully parallel.

    Args:
        train_dataset: Tokenised ClassificationDataset from prepare_dataset().
        label2id: Label-to-integer mapping from prepare_dataset().
        id2label: Integer-to-label mapping from prepare_dataset().
        model_name: HuggingFace model name or local path.
        output_dir: Root directory for all Ray Tune artifacts.  Must be on a
                    shared filesystem when running on a multi-node cluster.
                    S3/GCS URIs are also accepted (requires ``pyarrow``).
        n_trials: Total number of trials. Default 20.
        metric: Metric to maximise — ``"accuracy"``, ``"f1_macro"``,
                ``"f1_weighted"``, ``"csmf_accuracy"``. Default ``"accuracy"``.
                Use ``"csmf_accuracy"`` or ``"f1_macro"`` for imbalanced VA data.
        search_space: Same ``(type, *args)`` format as :func:`optimize`.
                      Defaults to :data:`DEFAULT_SEARCH_SPACE` (plus
                      :data:`LORA_SEARCH_SPACE` when ``use_lora=True``).
        val_size: Fraction of train_dataset held out for trial evaluation.
                  Default 0.2.
        random_state: Seed for the stratified split and OptunaSearch sampler.
                      Default 42.
        use_lora: Apply LoRA adapters during each trial. Default False.
        gradient_checkpointing: Enable gradient checkpointing. Default False.
        early_stopping_patience: Early stopping patience per trial. Default 3.
        ray_address: Ray cluster address.
                     ``None``         — start / connect to a local Ray instance.
                     ``"auto"``       — auto-discover an existing cluster via the
                                        ``RAY_ADDRESS`` environment variable.
                     ``"<ip>:<port>"`` — connect to an explicit head node.
                     Default None.
        num_gpus_per_trial: GPUs allocated to each trial. Default 1.0.
                            ``0``   — CPU-only mode (no GPU required).
                            ``> 1`` — multi-GPU per trial (requires distributed
                                      training setup inside ``train()``).
                            ``0 < x < 1`` — fractional GPU (pack multiple
                                      trials onto one GPU; use with care).
        num_cpus_per_trial: CPU cores per trial. Default 4.
        max_concurrent_trials: Maximum trials running simultaneously.
                               ``None`` → Ray auto-determines from available
                               resources (recommended). Default None.
        save_trials_csv: Save all trial metrics to
                         ``output_dir/hpo_trials_ray.csv``. Default True.

    Returns:
        best_hyperparams: Dict of best trial hyperparameters.  Can be passed
                          directly to ``train(hyperparams=best_hyperparams)``.
        results: ``ray.tune.ResultGrid`` — full results for analysis.
                 ``results.get_dataframe()`` → trial-level pandas DataFrame.
                 ``results.get_best_result(metric, mode)`` → best trial.
                 ``results.get_best_result().path`` → best trial artifact dir.

    Raises:
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build effective search space (same logic as optimize()) ---
    active_space = dict(DEFAULT_SEARCH_SPACE)
    if use_lora:
        active_space.update(LORA_SEARCH_SPACE)
    if search_space:
        active_space.update(search_space)
    ray_space = _to_ray_space(active_space)

    # --- Stratified internal split (mirrors optimize()) ---
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

    # Warn early if no GPUs are available but trials request one.
    available_gpus = ray.cluster_resources().get("GPU", 0)
    if num_gpus_per_trial > 0 and available_gpus == 0:
        logger.warning(
            "num_gpus_per_trial=%.1f but the Ray cluster reports 0 GPUs. "
            "Trials will queue indefinitely. "
            "Set num_gpus_per_trial=0 for CPU-only mode.",
            num_gpus_per_trial,
        )
    else:
        max_parallel = int(available_gpus // num_gpus_per_trial) if num_gpus_per_trial > 0 else None
        logger.info(
            "Ray cluster: %.0f GPU(s) available — up to %s concurrent trials "
            "at %.1f GPU(s)/trial.",
            available_gpus,
            max_parallel if max_parallel is not None else "unlimited",
            num_gpus_per_trial,
        )

    # --- Search algorithm: OptunaSearch (TPE) ---
    # OptunaSearch shares the acquisition function across parallel workers so
    # the search remains sample-efficient even with many concurrent trials.
    search_alg: object = OptunaSearch(
        metric=metric,
        mode="max",
        seed=random_state,
    )
    if max_concurrent_trials is not None:
        # ConcurrencyLimiter prevents the search algorithm from suggesting too
        # many points before receiving feedback, keeping TPE effective.
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=max_concurrent_trials
        )

    # --- Trainable: bind fixed arguments via the Ray object store ---
    # tune.with_parameters() stores opt_train / opt_val in the Ray object
    # store once; workers receive a lightweight reference rather than a
    # serialised copy, which is critical for large tokenised datasets.
    trainable = tune.with_parameters(
        _ray_trial_fn,
        opt_train=opt_train,
        opt_val=opt_val,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        use_lora=use_lora,
        gradient_checkpointing=gradient_checkpointing,
        early_stopping_patience=early_stopping_patience,
    )
    # Attach resource requirements so the Ray scheduler knows how many GPUs /
    # CPUs to reserve before launching each trial.
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
            # max_concurrent_trials=None lets Ray fill all available GPU slots
            # automatically; set an explicit value to leave GPUs for other work.
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=air.RunConfig(
            # Ray writes trial subdirectories inside storage_path/name/.
            # Use a shared filesystem or cloud URI for multi-node clusters.
            storage_path=str(output_dir),
            name="text_hpo_ray",
            log_to_file=True,
        ),
    )

    logger.info(
        "optimize_ray: launching %d trials  metric=%s  "
        "gpus/trial=%.1f  cpus/trial=%d",
        n_trials, metric, num_gpus_per_trial, num_cpus_per_trial,
    )
    results = tuner.fit()

    # --- Extract best result ---
    best_result = results.get_best_result(metric=metric, mode="max")
    best_hyperparams = best_result.config

    with open(output_dir / "best_hyperparams_ray.json", "w") as fh:
        json.dump(best_hyperparams, fh, indent=2)

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
