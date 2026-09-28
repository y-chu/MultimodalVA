"""
Ensemble strategy 3 — Decision-level fusion via soft voting.

Two usage modes are supported:

    Mode 1 — vote over existing results (no training required)::

        from multimodalva.ensemble import vote_from_results
        from multimodalva.utils.predictions import load_predictions

        # Results still in memory from this session ...
        final = vote_from_results([bert_result, lgbm_result], id2label=id2label)

        # ... or runs finished days ago, read back off disk.
        bert = load_predictions("runs/bert")
        lgbm = load_predictions("runs/lgbm")
        final = vote_from_results([bert, lgbm], id2label=bert.id2label)

    Mode 2 — full pipeline: train base models from scratch, then vote::

        from multimodalva.ensemble import SoftVotingClassifier

        clf = SoftVotingClassifier(
            text_models=[
                {"model_name": "bioclinicalbert", "hyperparams": {"epochs": 5}},
            ],
            tabular_models=[
                {"model_name": "lightgbm", "hyperparams": {"n_estimators": 300}},
                {"model_name": "random_forest"},
            ],
            output_dir="runs/ensemble/voting",
        )
        results = clf.run(df, text_col="narrative", feature_cols=[...], label_col="cause")

Base model spec keys
---------------------
``"hyperparams"`` says where that base model's hyperparameters come from, and
is the only key that configures a search::

    "hyperparams": {"epochs": 5, "learning_rate": 2e-5}   # use exactly these
    "hyperparams": Optimize(n_trials=20)                  # search, 20 trials
    "hyperparams": "default"   (or omitted)               # library defaults

Text model spec::

    {
        "model_name":        "bioclinicalbert",  # HuggingFace ID or shorthand
        "hyperparams":       Optimize(metric="f1_macro", n_trials=20),
        "max_length":        512,
        "use_lora":          False,
        "use_focal":         False,
        "early_stopping_patience": 3,
    }

Tabular model spec::

    {
        "model_name":          "lightgbm",       # alias from tabular.train.TABULAR_MODELS
        "hyperparams":         {"n_estimators": 300},
        "encode_categoricals": "ordinal",
        "scale_numeric":       False,
    }

``n_trials``, ``optimize_metric``, ``search_space``, ``use_cv``, ``n_cv_folds``
and ``resume_hpo`` are no longer spec keys — they are fields of ``Optimize``.
A spec still carrying one raises, naming the replacement, rather than silently
not searching.

Public API
----------
    soft_vote(prob_list, weights)
        Weighted average of probability matrices.
    vote_from_results(results, id2label, weights, top_k)
        Mode 1 — soft vote over pre-computed PredictionResult objects.
    SoftVotingClassifier
        Mode 2 — end-to-end wrapper: split → train → predict → vote.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.types import PredictionResult
from ..utils.predictions import (
    assemble_predictions, load_predictions, resolve_test_ids, save_predictions,
)
from ..utils.hpo_defaults import TABULAR_SPEC_DEFAULTS, TEXT_SPEC_DEFAULTS
from ..utils.optimize_config import (
    _log_hp_source, resolve_spec_hyperparams, resolve_search_resume,
)
from ..utils.runtime import track_run
from ..utils.seeds import seed_everything, set_determinism

logger = logging.getLogger(__name__)

# Type alias for a base-model specification dict.
BaseModelSpec = dict


def _write_base_label_maps(model_dir: "Path", label2id: dict, id2label: dict) -> None:
    """Give a base model's directory its own label maps.

    Voting writes the maps once at the ensemble root, but a base model's
    predictions are also useful on their own — reloaded with
    :func:`~multimodalva.utils.predictions.load_predictions`, compared in a
    leaderboard, or voted over in a different combination. Without the maps
    beside them the class names have to be guessed from the tables.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "label2id.json", "w") as fh:
        json.dump(label2id, fh, indent=2)
    with open(model_dir / "id2label.json", "w") as fh:
        json.dump({str(k): v for k, v in id2label.items()}, fh, indent=2)


def _base_model_done(model_dir: "Path") -> bool:
    """Whether a voting base model finished and left everything voting needs.

    Two artifacts must both exist: the trained model's metadata, written last by
    the train step, and the prediction CSVs. Checking only the model directory
    would reuse a model whose predict step was interrupted, and checking only
    the predictions would accept a directory left behind by an aborted run.
    """
    return (
        (model_dir / "final" / "training_metadata.json").is_file()
        and (model_dir / "predictions" / "predictions_top1.csv").is_file()
    )


# ---------------------------------------------------------------------------
# Core voting function
# ---------------------------------------------------------------------------

def soft_vote(
    prob_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Average a list of probability matrices into a single combined matrix.

    Args:
        prob_list: List of probability matrices, each of shape
                   ``(n_samples, n_classes)``.  All matrices must have
                   identical shape.  Class ordering must be consistent across
                   matrices (guaranteed when all models share the same
                   ``label2id`` / ``id2label``).
        weights:   Non-negative floats, one per matrix.  Matrices are
                   combined as a weighted average.  ``None`` = equal weights
                   (simple average).  Default ``None``.

    Returns:
        Combined probability matrix of shape ``(n_samples, n_classes)``.

    Raises:
        ValueError: If ``prob_list`` is empty, shapes differ, or
                    ``weights`` length does not match ``prob_list``.
    """
    if not prob_list:
        raise ValueError("prob_list must contain at least one matrix.")

    ref_shape = prob_list[0].shape
    for i, m in enumerate(prob_list[1:], start=1):
        if m.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: prob_list[0] has shape {ref_shape}, "
                f"but prob_list[{i}] has shape {m.shape}.  "
                "Ensure all base models share the same label2id / id2label."
            )

    if weights is not None:
        if len(weights) != len(prob_list):
            raise ValueError(
                f"weights has {len(weights)} entries but prob_list has "
                f"{len(prob_list)} matrices."
            )
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative.")
        if w.sum() == 0:
            raise ValueError("weights must not all be zero.")
    else:
        w = None

    stacked = np.stack(prob_list, axis=0)            # (n_models, n_samples, n_classes)
    combined = np.average(stacked, axis=0, weights=w) # (n_samples, n_classes)
    return combined


# ---------------------------------------------------------------------------
# Internal helper: build PredictionResult from a combined probability matrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mode 1: vote over pre-computed PredictionResult objects
# ---------------------------------------------------------------------------

def vote_from_results(
    results: list,
    id2label: dict,
    weights: list[float] | None = None,
    top_k: int = 3,
) -> PredictionResult:
    """Soft vote over pre-computed ``PredictionResult`` objects.

    Use this when you already have predictions from separately trained models
    (e.g. a fine-tuned BERT and a trained LightGBM) and want to combine them
    without retraining anything.

    All results must have been produced using the **same** ``id2label`` mapping
    so that ``prob_0``, ``prob_1``, … refer to identical classes across models.
    This is guaranteed when all models share the same ``label2id`` (built once
    via a single ``prepare_text_dataset()`` / ``prepare_tabular_dataset()`` call on
    the same train/test split).

    Args:
        results:  List of :class:`~multimodalva.utils.types.PredictionResult`
                  objects.  Pass ``result.full`` probability matrices directly
                  from ``predict_text()`` or ``predict_tabular()`` outputs.
                  Must contain at least 2 elements for voting to be meaningful.
        id2label: Integer class ID → label string mapping shared by all models.
                  Pass ``result.id2label`` from any one result.
        weights:  Optional per-model weights (non-negative floats).  Length
                  must match ``len(results)``.  ``None`` = equal weights.
        top_k:    Number of top classes in the topk output.  Default 3.

    Returns:
        :class:`~multimodalva.utils.types.PredictionResult` with combined
        probabilities.

    Raises:
        ValueError: If ``results`` is empty, shapes differ, or ``id2label``
                    keys do not match the probability columns.

    Example::

        from multimodalva.ensemble.voting import vote_from_results

        # text + tabular predictions already obtained
        text_result    = text_clf.predictions
        tabular_result = tabular_clf.predictions

        # vote with equal weights
        final = vote_from_results(
            [text_result, tabular_result],
            id2label=text_result.id2label,
        )

        # vote with custom weights (trust BERT more)
        final = vote_from_results(
            [bert_result, lgbm_result, rf_result],
            id2label=id2label,
            weights=[0.5, 0.3, 0.2],
        )
    """
    if not results:
        raise ValueError("results must contain at least one PredictionResult.")

    sorted_ids = sorted(id2label.keys())
    prob_cols  = [f"prob_{cid}" for cid in sorted_ids]

    # Validate id2label coverage against first result
    first_full = results[0].full
    missing = [c for c in prob_cols if c not in first_full.columns]
    if missing:
        raise ValueError(
            f"id2label keys {[c[5:] for c in missing]} not found in result.full columns.  "
            "Ensure all results share the same id2label."
        )

    # Extract true labels from first result (same for all)
    true_labels = first_full["true_label"].tolist()

    # Build probability matrices in canonical sorted-id order
    prob_matrices = []
    for idx, r in enumerate(results):
        missing_in = [c for c in prob_cols if c not in r.full.columns]
        if missing_in:
            raise ValueError(
                f"results[{idx}].full is missing columns {missing_in}.  "
                "All results must share the same id2label."
            )
        prob_matrices.append(r.full[prob_cols].to_numpy(dtype=float))

    # Carry the row identifiers through when the base results have them, so the
    # voted result can be joined back to the source records like any other
    # pipeline's output. Voting averages row i of every matrix, so disagreeing
    # ids mean the inputs are not aligned and the vote would be meaningless.
    ids = None
    if "id" in first_full.columns:
        ids = first_full["id"].tolist()
        for idx, r in enumerate(results[1:], start=1):
            if "id" not in r.full.columns:
                raise ValueError(
                    f"results[0] carries an 'id' column but results[{idx}] does "
                    "not, so the two cannot be checked for alignment. Run every "
                    "base model with the same id_col, or none of them."
                )
            if r.full["id"].tolist() != ids:
                raise ValueError(
                    f"results[{idx}] has different row ids from results[0]. "
                    "Soft voting averages matching rows, so all results must "
                    "come from the same test split in the same order."
                )

    combined = soft_vote(prob_matrices, weights=weights)
    return assemble_predictions(
        combined, id2label,
        true_labels=true_labels, ids=ids, top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Mode 2: SoftVotingClassifier — train from scratch + vote
# ---------------------------------------------------------------------------

class SoftVotingClassifier:
    """End-to-end soft-voting ensemble classifier.

    Trains a configurable set of text and/or tabular base models independently,
    then combines their predicted probability distributions via weighted
    averaging at inference time.

    Requires at least 2 base models in total.

    Base model specs
    ----------------
    Each element of ``text_models`` / ``tabular_models`` is a dict.  See
    module-level docstring for full key reference.

    Text model spec::

        {
            "model_name":      "bioclinicalbert",
            "hyperparams":     {"epochs": 5, "learning_rate": 2e-5},
            "max_length":      512,
            "use_lora":        False,
            "hyperparams":     None,   # dict | Optimize(...) | None
            "n_trials":        20,
            "optimize_metric": "f1_macro",
            "search_space":    None,
        }

    Tabular model spec::

        {
            "model_name":          "lightgbm",
            "hyperparams":         {"n_estimators": 300},
            "encode_categoricals": "ordinal",
            "scale_numeric":       False,
            "hyperparams":         None,
            "n_trials":            20,
            "optimize_metric":     "f1_macro",
            "search_space":        None,
        }

    Attributes (populated after run())
    ------------------------------------
    train_df :          Training split DataFrame.
    test_df :           Test split DataFrame.
    label2id :          Label → integer ID mapping.
    id2label :          Integer ID → label mapping.
    predictions :       Final ``PredictionResult`` after voting.
    base_predictions :  List of per-model ``PredictionResult`` objects,
                        ordered as ``text_models + tabular_models``.
    """

    def __init__(
        self,
        text_models: list[BaseModelSpec] | None = None,
        tabular_models: list[BaseModelSpec] | None = None,
        output_dir: str | Path = "runs/ensemble/voting",
        weights: list[float] | None = None,
    ):
        """Initialise SoftVotingClassifier.

        Args:
            text_models:    List of text base-model spec dicts.  ``None``/``[]``
                            if no text models are desired.
            tabular_models: List of tabular base-model spec dicts.  ``None``/``[]``
                            if no tabular models are desired.
            output_dir:     Root output directory.  Each base model is saved to
                            ``output_dir/base_models/text_<i>/`` and
                            ``output_dir/base_models/tabular_<i>/``.
            weights:        Optional per-model weights for
                            :func:`soft_vote`.  Length must equal the total
                            number of base models.  ``None`` = equal weights.
        """
        text_models    = text_models    or []
        tabular_models = tabular_models or []

        n_models = len(text_models) + len(tabular_models)
        if n_models < 2:
            raise ValueError(
                f"SoftVotingClassifier requires at least 2 base models total; "
                f"got {n_models} ({len(text_models)} text, {len(tabular_models)} tabular)."
            )

        if weights is not None and len(weights) != n_models:
            raise ValueError(
                f"weights has {len(weights)} entries but there are {n_models} base models."
            )
        self.text_models    = text_models
        self.tabular_models = tabular_models
        self.output_dir     = Path(output_dir)
        self.weights        = weights

        # Populated after run()
        self.train_df:        pd.DataFrame | None = None
        self.test_df:         pd.DataFrame | None = None
        self.label2id:        dict | None          = None
        self.id2label:        dict | None          = None
        self.predictions:     PredictionResult | None = None
        self.base_predictions: list                = []

    # ------------------------------------------------------------------

    @track_run("voting", monitor_gpu=True)
    def run(
        self,
        df: pd.DataFrame,
        label_col: str,
        text_col: str | None = None,
        feature_cols: list[str] | None = None,
        # --- split ---
        test_size: float = 0.2,
        split_seed: int = 42,
        train_seed: int = 42,
        deterministic: bool = False,
        stratify: bool = True,
        split_col: str | None = None,
        # --- text training options (global defaults; override per-model in spec) ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 4,
        # --- tabular training options ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        # --- inference ---
        top_k: int = 3,
        batch_size: int = 32,
        resume: bool = True,
        id_col: str | None = None,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
    ) -> dict:
        """Run the full soft-voting ensemble pipeline.

        Steps:
            1. ``split()``
            2. For each text base model:
               a. ``prepare_text_dataset()``
               b. ``[search_text() →] train_text()``
               c. ``predict_text()``
            3. For each tabular base model:
               a. ``prepare_tabular_dataset()``
               b. ``[search_tabular() →] train_tabular()``
               c. ``predict_tabular()``
            4. ``soft_vote()`` over all base-model probability matrices.
            5. Assemble final ``PredictionResult`` + save metadata.

        Args:
            df:                    Input DataFrame.
            label_col:             Column containing cause-of-death labels.
            text_col:              Column containing free-text narratives.
                                   Required when ``text_models`` is non-empty.
            feature_cols:          Tabular feature columns.
                                   Required when ``tabular_models`` is non-empty.
            test_size:             Fraction held out for testing.  Default 0.2.
            split_seed:            Seed for every row-partitioning decision —
                                   the train/test split, each base model's
                                   search folds and early-stopping slice. Vary it
                                   to measure sampling uncertainty. Default 42.
            train_seed:            Seed for every base model's training and
                                   search sampler. Vary it, with each spec's
                                   ``hyperparams`` fixed, to measure model
                                   stochasticity. Default 42.
            deterministic:         Demand bit-for-bit repeatable kernels, at a
                                   cost in speed and robustness. Default False.
            stratify:              Stratified split.  Default True.
            val_size:              Internal val fraction for text training when
                                   no search runs.  Default 0.1.
                                   Set to ``None`` to train on all training data.
            gradient_checkpointing: Enable gradient checkpointing for text
                                   models.  Default False.
            early_stopping_patience: Text model early-stopping patience.
                                   Default 4, as TextClassifier.  Override
                                   per-model in spec.
            n_jobs:                CPU parallelism for tabular models.
                                   Default -1 (all cores).
            use_gpu:               GPU flag for tabular models.
                                   ``None`` = auto-detect.
            top_k:                 Number of top classes in topk output.
                                   Default 3.
            batch_size:            Inference batch size for text models.
                                   Default 32.
            resume:                Skip base models that already finished, reusing
                                   their saved predictions. Default True. Set False
                                   to retrain every base model from scratch.
            encode_categoricals:   Categorical encoding for tabular base models
                                   (``"ordinal"`` default). A spec's own
                                   ``encode_categoricals`` overrides it.
            scale_numeric:         Standardise numeric columns for tabular base
                                   models. A spec's own value overrides it.
            id_col:                Optional column holding a row identifier. When
                                   given, the voted predictions and every base
                                   model's predictions carry a leading ``id``
                                   column, so they can be joined back to the
                                   source records and compared across pipelines.

        Returns:
            dict with keys:

            ``"predictions"``
                Final :class:`~multimodalva.utils.types.PredictionResult`
                after soft voting.
            ``"base_predictions"``
                List of per-model ``PredictionResult`` objects
                (text models first, then tabular).
            ``"label2id"``
                Label → integer ID mapping.
            ``"id2label"``
                Integer ID → label mapping.
            ``"output_dir"``
                :class:`pathlib.Path` to the root output directory.

        Raises:
            ValueError: If ``text_col`` is None when text_models is non-empty,
                        or ``feature_cols`` is None when tabular_models is
                        non-empty.
            ImportError: If a required pipeline dependency is missing.
        """
        # --- validation -----------------------------------------------------
        if self.text_models and text_col is None:
            raise ValueError("text_col is required when text_models is non-empty.")
        if self.tabular_models and feature_cols is None:
            raise ValueError("feature_cols is required when tabular_models is non-empty.")

        # --- lazy imports (avoid top-level circular imports) ----------------
        from ..utils.split import split
        from ..text.dataset import prepare_text_dataset
        from ..text.train   import train_text
        from ..text.predict import predict_text
        from ..text.hpo     import search_text
        from ..tabular.dataset import prepare_tabular_dataset
        from ..tabular.train   import train_tabular
        from ..tabular.predict import predict_tabular
        from ..tabular.hpo     import search_tabular

        set_determinism(deterministic)
        seed_everything(train_seed)

        # --- Step 1: split --------------------------------------------------
        self.train_df, self.test_df = split(
            df,
            label_col=label_col,
            text_col=text_col,
            test_size=test_size,
            random_state=split_seed,
            stratify=stratify,
            split_col=split_col,
        )
        logger.info(
            "Split: %d train / %d test samples",
            len(self.train_df), len(self.test_df),
        )

        # Identifiers for the scored rows. prepare_*_dataset() keeps every test
        # row, so these line up with the predictions without further masking.
        test_ids = resolve_test_ids(self.test_df, id_col)

        base_dir = self.output_dir / "base_models"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_predictions = []
        label2id = id2label = None

        # --- Steps 2: text base models --------------------------------------
        for i, spec in enumerate(self.text_models):
            model_dir = base_dir / f"text_{i}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_name = spec["model_name"]
            max_length = spec.get("max_length", 512)
            use_lora   = spec.get("use_lora", False)
            esp        = spec.get("early_stopping_patience", early_stopping_patience)
            gc         = spec.get("gradient_checkpointing", gradient_checkpointing)

            logger.info("Training text base model %d/%d: %s", i + 1, len(self.text_models), model_name)

            train_ds, test_ds, lbl2id, id2lbl = prepare_text_dataset(
                self.train_df, self.test_df,
                text_col=text_col, label_col=label_col,
                model_name=model_name, max_length=max_length,
            )
            # Capture label maps from first model; all subsequent calls must match.
            if label2id is None:
                label2id, id2label = lbl2id, id2lbl

            # A finished base model is reusable as-is: voting only needs its
            # probability matrix, which is already on disk. Without this a run
            # interrupted after model 3 of 4 retrains all four.
            if resume and _base_model_done(model_dir):
                result = load_predictions(model_dir)
                logger.info(
                    "Base model text_%d (%s): already trained — reusing "
                    "predictions from %s (skipping training).",
                    i, model_name, model_dir / "predictions",
                )
                self.base_predictions.append(result)
                continue

            kind, fixed, search = resolve_spec_hyperparams(spec, TEXT_SPEC_DEFAULTS)
            if kind == "search":
                logger.info("Base model text_%d (%s): hyperparameters FROM SEARCH — %s.",
                            i, model_name, search.describe())
                best_hp, _ = search_text(
                    search,
                    resume=resolve_search_resume(search, resume),
                    train_dataset=train_ds,
                    label2id=label2id, id2label=id2label,
                    model_name=model_name,
                    output_dir=model_dir / "hpo",
                    random_state=train_seed,
                    split_seed=split_seed,
                    use_lora=use_lora,
                    use_focal=spec.get("use_focal", TEXT_SPEC_DEFAULTS["use_focal"]),
                    gradient_checkpointing=gc,
                    early_stopping_patience=esp,
                )
                final_val = None   # epochs already determined by the search
                logger.info("Base model text_%d (%s): search finished — best %s.",
                            i, model_name, best_hp)
            else:
                best_hp   = fixed or {}
                _log_hp_source(f"text_{i}", model_name, best_hp)
                final_val = val_size

            train_text(
                train_dataset=train_ds,
                label2id=label2id, id2label=id2label,
                model_name=model_name,
                output_dir=model_dir / "final",
                hyperparams=best_hp,
                val_size=final_val,
                use_lora=use_lora,
                gradient_checkpointing=gc,
                early_stopping_patience=esp,
                # Without this the text base trains at train_text's own default
                # seed, so the run-level seed reached the split and the search
                # but not the training — the base model came out the same
                # whatever seed the caller asked for.
                random_state=train_seed,
                split_seed=split_seed,
            )

            result = predict_text(
                output_dir=model_dir / "final",
                test_dataset=test_ds,
                batch_size=batch_size,
                top_k=top_k,
                save_dir=model_dir / "predictions",
                ids=test_ids,
            )
            _write_base_label_maps(model_dir, label2id, id2label)
            self.base_predictions.append(result)
            logger.info(
                "Text model %d top-1 acc: %.4f",
                i, (result.top1["true_label"] == result.top1["predicted_label"]).mean(),
            )

        # --- Steps 3: tabular base models -----------------------------------
        for i, spec in enumerate(self.tabular_models):
            model_dir = base_dir / f"tabular_{i}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_name = spec["model_name"]
            # Run-level setting, overridable per base model — voting prepares
            # each base model separately, so it can honour both.
            enc_cat    = spec.get("encode_categoricals", encode_categoricals)
            scale_num  = spec.get("scale_numeric", scale_numeric)

            logger.info("Training tabular base model %d/%d: %s", i + 1, len(self.tabular_models), model_name)

            (X_train, X_test, y_train, y_test,
             preprocessor, lbl2id, id2lbl, feature_names) = prepare_tabular_dataset(
                self.train_df, self.test_df,
                feature_cols=feature_cols, label_col=label_col,
                encode_categoricals=enc_cat, scale_numeric=scale_num,
            )
            if label2id is None:
                label2id, id2label = lbl2id, id2lbl

            if resume and _base_model_done(model_dir):
                result = load_predictions(model_dir)
                logger.info(
                    "Base model tabular_%d (%s): already trained — reusing "
                    "predictions from %s (skipping training).",
                    i, model_name, model_dir / "predictions",
                )
                self.base_predictions.append(result)
                continue

            kind, fixed, search = resolve_spec_hyperparams(spec, TABULAR_SPEC_DEFAULTS)
            if kind == "search":
                logger.info("Base model tabular_%d (%s): hyperparameters FROM SEARCH — %s.",
                            i, model_name, search.describe())
                best_hp, _ = search_tabular(
                    search,
                    resume=resolve_search_resume(search, resume),
                    X_train=X_train, y_train=y_train,
                    label2id=label2id, id2label=id2label,
                    model_name=model_name,
                    output_dir=model_dir / "hpo",
                    random_state=train_seed,
                    split_seed=split_seed,
                    n_jobs=n_jobs, use_gpu=use_gpu,
                )
                logger.info("Base model tabular_%d (%s): search finished — best %s.",
                            i, model_name, best_hp)
            else:
                best_hp = fixed
                _log_hp_source(f"tabular_{i}", model_name, best_hp)

            train_tabular(
                X_train=X_train, y_train=y_train,
                label2id=label2id, id2label=id2label,
                model_name=model_name,
                output_dir=model_dir / "final",
                hyperparams=best_hp,
                preprocessor=preprocessor, feature_names=feature_names,
                random_state=train_seed, n_jobs=n_jobs, use_gpu=use_gpu,
            )

            result = predict_tabular(
                output_dir=model_dir / "final",
                X_test=X_test, y_test=y_test,
                top_k=top_k,
                save_dir=model_dir / "predictions",
                ids=test_ids,
            )
            _write_base_label_maps(model_dir, label2id, id2label)
            self.base_predictions.append(result)
            logger.info(
                "Tabular model %d top-1 acc: %.4f",
                i, (result.top1["true_label"] == result.top1["predicted_label"]).mean(),
            )

        # --- Step 4: soft vote ----------------------------------------------
        logger.info(
            "Soft voting over %d base models (weights=%s)",
            len(self.base_predictions),
            self.weights,
        )
        self.predictions = vote_from_results(
            self.base_predictions,
            id2label=id2label,
            weights=self.weights,
            top_k=top_k,
        )
        self.label2id = label2id
        self.id2label = id2label

        voted_acc = (
            self.predictions.top1["true_label"] == self.predictions.top1["predicted_label"]
        ).mean()
        logger.info("Voted ensemble top-1 acc: %.4f", voted_acc)

        # --- Step 5: save predictions and metadata --------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Write the voted result in the same layout every other pipeline uses,
        # so a finished voting run can be read back from disk by name — e.g.
        # predictions_frame({"voting": output_dir}) — instead of only from the
        # object this call returns.
        save_predictions(self.predictions, self.output_dir / "predictions")
        with open(self.output_dir / "label2id.json", "w") as fh:
            json.dump(label2id, fh, indent=2)
        with open(self.output_dir / "id2label.json", "w") as fh:
            json.dump({str(k): v for k, v in id2label.items()}, fh, indent=2)

        metadata = {
            "output_dir":       str(self.output_dir),
            "n_text_models":    len(self.text_models),
            "n_tabular_models": len(self.tabular_models),
            "weights":          self.weights,
            "text_specs":       self.text_models,
            "tabular_specs":    self.tabular_models,
            "label2id":         label2id,
            "id2label":         {str(k): v for k, v in id2label.items()},
            "n_train":          len(self.train_df),
            "n_test":           len(self.test_df),
            "n_classes":        len(label2id),
        }
        with open(self.output_dir / "training_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        logger.info("Metadata saved to %s", self.output_dir)

        return {
            "predictions":      self.predictions,
            "base_predictions": self.base_predictions,
            "label2id":         label2id,
            "id2label":         id2label,
            "output_dir":       self.output_dir,
        }
