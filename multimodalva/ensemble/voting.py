"""
Ensemble strategy 3 — Decision-level fusion via soft voting.

Two usage modes are supported:

    Mode 1 — vote over existing results (no training required)::

        from multimodalva.ensemble.voting import vote_from_results

        # Pass PredictionResult objects already obtained from text/tabular pipelines.
        final = vote_from_results([bert_result, lgbm_result], id2label=id2label)

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
Text model spec::

    {
        "model_name":        "bioclinicalbert",  # HuggingFace ID or shorthand
        "hyperparams":       {"epochs": 5, "learning_rate": 2e-5},  # fixed HP
        "max_length":        512,
        "use_lora":          False,
        "use_optimize":      False,              # True → run Optuna HPO
        "n_trials":          20,
        "optimize_metric":   "f1_macro",
        "search_space":      None,               # None = DEFAULT_SEARCH_SPACE
    }

Tabular model spec::

    {
        "model_name":          "lightgbm",       # alias from tabular.train.SUPPORTED_MODELS
        "hyperparams":         {"n_estimators": 300},
        "encode_categoricals": "ordinal",
        "scale_numeric":       False,
        "use_optimize":        False,
        "n_trials":            20,
        "optimize_metric":     "f1_macro",
        "search_space":        None,
    }

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

logger = logging.getLogger(__name__)

# Type alias for a base-model specification dict.
BaseModelSpec = dict


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

def _assemble_prediction_result(
    combined_proba: np.ndarray,
    true_labels: list,
    id2label: dict,
    top_k: int,
) -> PredictionResult:
    """Build a PredictionResult from a combined probability matrix.

    Args:
        combined_proba: ``(n_samples, n_classes)`` probability matrix in
                        canonical id2label order (column j → class sorted_ids[j]).
        true_labels:    Ground-truth string labels (length n_samples).
        id2label:       Integer class ID → label string mapping.
        top_k:          Number of top classes in topk output.

    Returns:
        PredictionResult(top1, full, topk, id2label).
    """
    n = len(true_labels)
    sorted_ids = sorted(id2label.keys())
    n_classes  = len(sorted_ids)
    top_k      = min(top_k, n_classes)

    # --- top1 ---------------------------------------------------------------
    top1_pos   = np.argmax(combined_proba, axis=1)
    top1_ids   = [sorted_ids[p] for p in top1_pos]
    top1_df    = pd.DataFrame({
        "true_label":      true_labels,
        "predicted_label": [id2label[ci] for ci in top1_ids],
        "predicted_prob":  combined_proba[np.arange(n), top1_pos],
    })

    # --- full (one prob column per class, integer ID suffix) ----------------
    full_data = {"true_label": true_labels}
    for j, cid in enumerate(sorted_ids):
        full_data[f"prob_{cid}"] = combined_proba[:, j]
    full_df = pd.DataFrame(full_data)

    # --- topk ---------------------------------------------------------------
    top_indices = np.argsort(combined_proba, axis=1)[:, ::-1][:, :top_k]
    topk_data = {"true_label": true_labels}
    for j in range(top_k):
        col_pos   = top_indices[:, j]
        class_ids = [sorted_ids[p] for p in col_pos]
        topk_data[f"top{j+1}_label"] = [id2label[ci] for ci in class_ids]
        topk_data[f"top{j+1}_prob"]  = combined_proba[np.arange(n), col_pos]
    topk_df = pd.DataFrame(topk_data)

    return PredictionResult(top1=top1_df, full=full_df, topk=topk_df, id2label=id2label)


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
    via a single ``prepare_dataset()`` call on the same train/test split).

    Args:
        results:  List of :class:`~multimodalva.utils.types.PredictionResult`
                  objects.  Pass ``result.full`` probability matrices directly
                  from ``text.predict()`` or ``tabular.predict()`` outputs.
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

    combined = soft_vote(prob_matrices, weights=weights)
    return _assemble_prediction_result(combined, true_labels, id2label, top_k)


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
            "use_optimize":    False,  # True → Optuna HPO
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
            "use_optimize":        False,
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

    def run(
        self,
        df: pd.DataFrame,
        label_col: str,
        text_col: str | None = None,
        feature_cols: list[str] | None = None,
        # --- split ---
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        # --- text training options (global defaults; override per-model in spec) ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 3,
        # --- tabular training options ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        # --- inference ---
        top_k: int = 3,
        batch_size: int = 32,
    ) -> dict:
        """Run the full soft-voting ensemble pipeline.

        Steps:
            1. ``split()``
            2. For each text base model:
               a. ``text.prepare_dataset()``
               b. ``[text.optimize() →] text.train()``
               c. ``text.predict()``
            3. For each tabular base model:
               a. ``tabular.prepare_dataset()``
               b. ``[tabular.optimize() →] tabular.train()``
               c. ``tabular.predict()``
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
            random_state:          Random seed.  Default 42.
            stratify:              Stratified split.  Default True.
            val_size:              Internal val fraction for text training when
                                   ``use_optimize=False``.  Default 0.1.
                                   Set to ``None`` to train on all training data.
            gradient_checkpointing: Enable gradient checkpointing for text
                                   models.  Default False.
            early_stopping_patience: Text model early-stopping patience.
                                   Default 3.  Override per-model in spec.
            n_jobs:                CPU parallelism for tabular models.
                                   Default -1 (all cores).
            use_gpu:               GPU flag for tabular models.
                                   ``None`` = auto-detect.
            top_k:                 Number of top classes in topk output.
                                   Default 3.
            batch_size:            Inference batch size for text models.
                                   Default 32.

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
        from ..text.dataset import prepare_dataset as text_prepare_dataset
        from ..text.train   import train            as text_train
        from ..text.predict import predict          as text_predict
        from ..text.hpo     import optimize         as text_optimize
        from ..tabular.dataset import prepare_dataset as tabular_prepare_dataset
        from ..tabular.train   import train            as tabular_train
        from ..tabular.predict import predict          as tabular_predict
        from ..tabular.hpo     import optimize         as tabular_optimize

        # --- Step 1: split --------------------------------------------------
        self.train_df, self.test_df = split(
            df,
            label_col=label_col,
            text_col=text_col,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        logger.info(
            "Split: %d train / %d test samples",
            len(self.train_df), len(self.test_df),
        )

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

            train_ds, test_ds, lbl2id, id2lbl = text_prepare_dataset(
                self.train_df, self.test_df,
                text_col=text_col, label_col=label_col,
                model_name=model_name, max_length=max_length,
            )
            # Capture label maps from first model; all subsequent calls must match.
            if label2id is None:
                label2id, id2label = lbl2id, id2lbl

            if spec.get("use_optimize", False):
                best_hp, _ = text_optimize(
                    train_dataset=train_ds,
                    label2id=label2id, id2label=id2label,
                    model_name=model_name,
                    output_dir=model_dir / "hpo",
                    n_trials=spec.get("n_trials", 20),
                    metric=spec.get("optimize_metric", "f1_macro"),
                    search_space=spec.get("search_space"),
                    random_state=random_state,
                    use_lora=use_lora,
                    gradient_checkpointing=gc,
                    early_stopping_patience=esp,
                )
                final_val = None   # epochs already determined by HPO
                logger.info("Text HPO done — best_hp: %s", best_hp)
            else:
                best_hp   = spec.get("hyperparams") or {}
                final_val = val_size

            text_train(
                train_dataset=train_ds,
                label2id=label2id, id2label=id2label,
                model_name=model_name,
                output_dir=model_dir / "final",
                hyperparams=best_hp,
                val_size=final_val,
                use_lora=use_lora,
                gradient_checkpointing=gc,
                early_stopping_patience=esp,
            )

            result = text_predict(
                output_dir=model_dir / "final",
                test_dataset=test_ds,
                batch_size=batch_size,
                top_k=top_k,
                save_dir=model_dir / "predictions",
            )
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
            enc_cat    = spec.get("encode_categoricals", "ordinal")
            scale_num  = spec.get("scale_numeric", False)

            logger.info("Training tabular base model %d/%d: %s", i + 1, len(self.tabular_models), model_name)

            (X_train, X_test, y_train, y_test,
             preprocessor, lbl2id, id2lbl, feature_names) = tabular_prepare_dataset(
                self.train_df, self.test_df,
                feature_cols=feature_cols, label_col=label_col,
                encode_categoricals=enc_cat, scale_numeric=scale_num,
            )
            if label2id is None:
                label2id, id2label = lbl2id, id2lbl

            if spec.get("use_optimize", False):
                best_hp, _ = tabular_optimize(
                    X_train=X_train, y_train=y_train,
                    label2id=label2id, id2label=id2label,
                    model_name=model_name,
                    output_dir=model_dir / "hpo",
                    n_trials=spec.get("n_trials", 20),
                    metric=spec.get("optimize_metric", "f1_macro"),
                    search_space=spec.get("search_space"),
                    use_cv=spec.get("use_cv", True),
                    n_cv_folds=spec.get("n_cv_folds", 3),
                    random_state=random_state,
                    n_jobs=n_jobs, use_gpu=use_gpu,
                )
                logger.info("Tabular HPO done — best_hp: %s", best_hp)
            else:
                best_hp = spec.get("hyperparams")

            tabular_train(
                X_train=X_train, y_train=y_train,
                label2id=label2id, id2label=id2label,
                model_name=model_name,
                output_dir=model_dir / "final",
                hyperparams=best_hp,
                preprocessor=preprocessor, feature_names=feature_names,
                random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu,
            )

            result = tabular_predict(
                output_dir=model_dir / "final",
                X_test=X_test, y_test=y_test,
                top_k=top_k,
                save_dir=model_dir / "predictions",
            )
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

        # --- Step 5: save metadata ------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
