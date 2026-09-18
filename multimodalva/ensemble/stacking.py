"""
Ensemble strategy 4 — Decision-level fusion via stacking (super learner).

Two-stage pipeline:

    Stage 1 — train_base_models():
        Split → k-fold OOF loop over text + tabular base models →
        retrain final base models on full training set.
        All artifacts (OOF probs, model weights) saved to disk; fully resumable.

    Stage 2 — train_meta_learner_stage():
        Load OOF meta-features → train one or more meta-learner candidates →
        select best by metric via stratified k-fold CV → save fitted meta-learner.

    Stage 3 — predict_test():
        Reload final base models → predict on hold-out test set →
        stack test probabilities → meta-learner.predict_proba() →
        assemble PredictionResult.

    Convenience: run() calls all three stages in sequence.

Output directory layout::

    output_dir/
    ├── data/
    │   ├── train_df.csv          — training split
    │   ├── test_df.csv           — held-out test split
    │   ├── X_test.npy            — preprocessed tabular test features
    │   └── y_test.npy            — integer test labels
    ├── oof/
    │   ├── fold_0/
    │   │   ├── text_0/           — per-fold model artifacts (if save_fold_models=True)
    │   │   │   └── oof_probs.npy — OOF probability matrix for fold 0, text model 0
    │   │   ├── text_1/
    │   │   └── tabular_0/
    │   ├── fold_1/ …
    │   ├── oof_meta_X.npy        — assembled OOF meta-feature matrix (n_train × n_models*n_classes)
    │   ├── oof_y.npy             — integer labels aligned with OOF rows
    │   ├── meta_feature_names.json
    │   └── oof_metadata.json
    ├── hpo/
    │   ├── text_0/               — Optuna HPO artifacts (if use_optimize=True)
    │   └── tabular_0/
    ├── final/
    │   ├── text_0/               — base model retrained on full training set
    │   ├── text_1/
    │   └── tabular_0/
    ├── meta_learner/
    │   ├── meta_learner.joblib   — fitted best meta-learner
    │   ├── meta_scores.json      — CV scores for all candidate meta-learners
    │   └── meta_learner_metadata.json
    ├── predictions/
    ├── label2id.json
    ├── id2label.json
    └── training_metadata.json

HPO per base model:
    Add ``"use_optimize": True`` to any model spec.  HPO runs ONCE on the full
    training set (before the OOF loop), and the best HP are reused for every
    fold and for the final full-data model.  This is the standard approach —
    per-fold HPO is too expensive and risks fold leakage.

Resume:
    Interrupted runs auto-resume by default (``resume=True``).
    Fold completion is detected by the presence of ``oof_probs.npy``; final
    model completion by ``training_metadata.json``.  Re-run the same stage
    call to continue from where training stopped.

Multi-GPU / MPS:
    Text models: HuggingFace Trainer handles device placement automatically
    (single GPU, multi-GPU DDP, Apple MPS).  No extra configuration needed.
    Tabular models: ``n_jobs=-1`` uses all CPU cores; GPU-capable models
    (lightgbm, catboost, xgboost) activate CUDA when ``use_gpu=True``.

InSilicoVA base model:
    Any tabular model spec may use ``model_name="insilicova"`` to include
    PyInSilicoVA as a base model.  Key differences from sklearn tabular models:

    * No training phase — InSilicoVA applies a fixed Bayesian symptom-cause
      probability database (WHO 2016 or PHMRC), so ``use_optimize`` is ignored.
    * Raw DataFrame required — ``train_df`` must be supplied to
      :func:`generate_oof_predictions` (and is passed automatically by
      :class:`StackingClassifier`).  InSilicoVA needs the original VA indicator
      columns, not the preprocessed numpy arrays.
    * Cause mapping — InSilicoVA's output uses its own cause-name strings
      (e.g. ``"HIV/AIDS related death"``).  Supply ``cause_map`` in the spec
      to map your label names to InSilicoVA column names::

          {"HIV/AIDS": "HIV/AIDS related death", "Malaria": "Malaria", ...}

      Omit ``cause_map`` for automatic case-insensitive matching (a warning is
      logged for any unmatched causes, which receive probability 0).
    * Columns — specify ``va_cols`` in the spec to restrict which indicator
      columns are passed to InSilicoVA; defaults to ``feature_cols``.

    Spec example::

        {"model_name": "insilicova",
         "data_type":  "WHO2016",          # "WHO2016" (default) or "PHMRC"
         "va_cols":    [...],               # optional; defaults to feature_cols
         "cause_map":  {"HIV/AIDS": "HIV/AIDS related death", ...},  # optional
         "hyperparams": {"n_sim": 4000, "burnin": 2000, "thin": 10}}

    Requires:  ``pip install pyinsilicova``

Public API:
    generate_oof_predictions(...)
        Standalone k-fold OOF function for advanced users.
    StackingClassifier
        End-to-end wrapper with explicit stage methods and run().
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..utils.types import PredictionResult
from ..utils.numpy_compat import (
    load_joblib_compat,
    null_rng_pickler,
    prepare_estimator_for_joblib,
)

from ..utils.provenance import check_inputs, guard_inputs, record_inputs

logger = logging.getLogger(__name__)


BaseModelSpec = dict

DEFAULT_META_LEARNER: dict = {
    "model_name":  "logistic_regression",
    "hyperparams": {"max_iter": 1000, "C": 1.0},
}

# Default Optuna search spaces for meta-learner HPO (use_optimize=True).
# Keys are model aliases; values are dicts of param → tuple spec or fixed value.
# Tuple format: ("float", lo, hi), ("log_float", lo, hi), ("int", lo, hi),
#               ("categorical", [v1, v2, ...])
# Fixed values (non-tuple) are passed through unchanged — use to pin params
# that should not be searched (e.g. {"max_iter": 2000} alongside searched "C").
DEFAULT_META_SEARCH_SPACES: dict[str, dict] = {
    "logistic_regression": {
        "C": ("log_float", 1e-3, 1e2),
        "class_weight": ("categorical", [None, "balanced"]),
    },
    "lightgbm": {
        "n_estimators":      ("int",       50,  500),
        "learning_rate":     ("log_float", 0.01, 0.3),
        "max_depth":         ("int",        3,   10),
        "num_leaves":        ("int",       15,   63),
        "min_child_samples": ("int",        5,   50),
    },
    "random_forest": {
        "n_estimators":      ("int",  50, 400),
        "max_depth":         ("int",   3,  20),
        "min_samples_split": ("int",   2,  20),
    },
    "xgboost": {
        "n_estimators":      ("int",       50,  500),
        "learning_rate":     ("log_float", 0.01, 0.3),
        "max_depth":         ("int",        3,   10),
        "subsample":         ("float",      0.5,  1.0),
        "colsample_bytree":  ("float",      0.5,  1.0),
    },
}

# Alias used to identify InSilicoVA specs throughout this module
_INSILICOVA = "insilicova"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SubsetDataset:
    """Lightweight wrapper exposing a contiguous-index subset of a Dataset.

    Compatible with any ``torch.utils.data.Dataset`` via ``__len__`` /
    ``__getitem__``.  Used to feed fold-specific training / validation slices
    to ``text.train()`` and ``text.predict()`` without re-tokenising.
    """

    def __init__(self, full_ds, indices):
        self._full    = full_ds
        self._indices = np.asarray(indices, dtype=int)

    @property
    def labels(self):
        """Integer labels for the subset, in subset order.

        Delegates to the underlying dataset's .labels list so that
        text.train._get_dataset_labels() can handle _SubsetDataset the same
        way it handles ClassificationDataset and torch.utils.data.Subset.
        """
        return [self._full.labels[int(i)] for i in self._indices]

    @property
    def ids(self):
        """Row identifiers for the subset, in subset order (None if unused)."""
        full_ids = getattr(self._full, "ids", None)
        if full_ids is None:
            return None
        return [full_ids[int(i)] for i in self._indices]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._full[int(self._indices[i])]


def _extract_probs(result: PredictionResult, sorted_ids: list) -> np.ndarray:
    """Return (n_samples, n_classes) float array from a PredictionResult.full."""
    prob_cols = [f"prob_{cid}" for cid in sorted_ids]
    return result.full[prob_cols].to_numpy(dtype=float)


def _assemble_prediction_result(
    combined_proba: np.ndarray,
    true_labels: list,
    id2label: dict,
    top_k: int,
) -> PredictionResult:
    """Build PredictionResult from a (n_samples, n_classes) probability matrix."""
    n          = len(true_labels)
    sorted_ids = sorted(id2label.keys())
    n_classes  = len(sorted_ids)
    top_k      = min(top_k, n_classes)

    top1_pos = np.argmax(combined_proba, axis=1)
    top1_ids = [sorted_ids[p] for p in top1_pos]
    top1_df  = pd.DataFrame({
        "true_label":      true_labels,
        "predicted_label": [id2label[ci] for ci in top1_ids],
        "predicted_prob":  combined_proba[np.arange(n), top1_pos],
    })

    full_data = {"true_label": true_labels}
    for j, cid in enumerate(sorted_ids):
        full_data[f"prob_{cid}"] = combined_proba[:, j]
    full_df = pd.DataFrame(full_data)

    top_indices = np.argsort(combined_proba, axis=1)[:, ::-1][:, :top_k]
    topk_data   = {"true_label": true_labels}
    for j in range(top_k):
        col_pos   = top_indices[:, j]
        class_ids = [sorted_ids[p] for p in col_pos]
        topk_data[f"top{j+1}_label"] = [id2label[ci] for ci in class_ids]
        topk_data[f"top{j+1}_prob"]  = combined_proba[np.arange(n), col_pos]
    topk_df = pd.DataFrame(topk_data)

    return PredictionResult(top1=top1_df, full=full_df, topk=topk_df, id2label=id2label)


def _build_meta_feature_names(
    text_specs: list,
    tabular_specs: list,
    sorted_ids: list,
) -> list[str]:
    """Build a list of column names for the meta-feature matrix.

    Format: ``text_{i}_{model_name}_prob_{class_id}`` and
            ``tab_{i}_{model_name}_prob_{class_id}``.
    """
    names = []
    for i, spec in enumerate(text_specs):
        mn = spec["model_name"].replace("/", "_").replace("-", "_")
        for cid in sorted_ids:
            names.append(f"text_{i}_{mn}_prob_{cid}")
    for i, spec in enumerate(tabular_specs):
        mn = spec["model_name"].replace("/", "_").replace("-", "_")
        for cid in sorted_ids:
            names.append(f"tab_{i}_{mn}_prob_{cid}")
    return names


def _resolve_meta_learner(
    spec: dict,
    n_jobs: int = -1,
    random_state: int = 42,
    use_gpu: bool = False,
):
    """Instantiate a meta-learner from a spec dict.

    Special alias ``"logistic_regression"`` maps to
    ``sklearn.linear_model.LogisticRegression``.
    All tabular model aliases from ``tabular.train.SUPPORTED_MODELS`` are also
    accepted (lightgbm, catboost, xgboost, random_forest, mlp, …).

    Args:
        spec:         Dict with ``model_name`` and optional ``hyperparams``.
        n_jobs:       Parallelism for CPU-bound models.
        random_state: Seed for reproducibility.
        use_gpu:      GPU flag (catboost / lightgbm / xgboost).

    Returns:
        Instantiated sklearn-compatible classifier (unfitted).
    """
    model_name = spec["model_name"]
    hp         = dict(spec.get("hyperparams") or {})

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        hp.setdefault("max_iter",      1000)
        hp.setdefault("C",             1.0)
        hp.setdefault("solver",        "lbfgs")   # lbfgs supports multinomial multiclass natively
        hp.setdefault("n_jobs",        n_jobs)
        hp.setdefault("random_state",  random_state)
        return LogisticRegression(**hp)

    # All other models: delegate to tabular pipeline's SUPPORTED_MODELS registry
    from ..tabular.train import SUPPORTED_MODELS, _NJOBS_PARAM, _CUDA_PARAMS, _FORCED_PARAMS
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown meta-learner '{model_name}'.  "
            f"Use 'logistic_regression' or one of: {list(SUPPORTED_MODELS)}"
        )

    import importlib
    module_path, class_name = SUPPORTED_MODELS[model_name]
    cls = getattr(importlib.import_module(module_path), class_name)

    if model_name in _NJOBS_PARAM:
        hp.setdefault(_NJOBS_PARAM[model_name], n_jobs)
    if model_name in _FORCED_PARAMS:
        hp.update(_FORCED_PARAMS[model_name])
    if use_gpu and model_name in _CUDA_PARAMS:
        hp.update(_CUDA_PARAMS[model_name])

    # Silence verbose output in the meta-learner CV loop
    if model_name == "catboost":
        hp.setdefault("verbose", 0)
    elif model_name == "lightgbm":
        hp.setdefault("verbose", -1)
    elif model_name == "xgboost":
        hp.setdefault("verbosity", 0)

    if model_name not in {"catboost"}:
        hp.setdefault("random_state", random_state)

    return cls(**hp)


def _score_meta_candidate(
    meta_model,
    meta_X: np.ndarray,
    y_oof: np.ndarray,
    id2label: dict,
    metric: str,
    cv_folds: int,
    random_state: int,
) -> float:
    """Score a meta-learner via stratified k-fold CV on the OOF meta-features.

    Uses ``cross_val_predict`` to get second-level OOF predictions, then scores
    with the given metric.  All four metrics supported by ``score_predictions``
    are valid here.

    Args:
        meta_model:   Unfitted sklearn-compatible classifier.
        meta_X:       OOF meta-feature matrix (n_train × n_models*n_classes).
        y_oof:        Integer label array (n_train,).
        id2label:     Integer → label string mapping.
        metric:       One of ``accuracy``, ``f1_macro``, ``f1_weighted``,
                      ``csmf_accuracy``.
        cv_folds:     CV folds for meta-learner selection (default 3).
        random_state: Seed.

    Returns:
        Mean CV score (float).
    """
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from ..utils.metrics import score_predictions

    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_pred = cross_val_predict(meta_model, meta_X, y_oof, cv=cv, method="predict")

    sorted_ids = sorted(id2label.keys())
    true_str   = [id2label[int(y)] for y in y_oof]
    pred_str   = [id2label[sorted_ids[int(p)]] for p in oof_pred]

    top1_df = pd.DataFrame({"true_label": true_str, "predicted_label": pred_str})
    return score_predictions(top1_df, metric=metric)


def _learn_class_voter_weights(
    prob_matrices: list[np.ndarray],
    y_true: np.ndarray,
    id2label: dict,
    metric: str = "f1",
    shrinkage: float = 0.1,
    min_support_for_trust: float = 20.0,
) -> np.ndarray:
    """Learn a per-model × per-class weight matrix from OOF probabilities.

    Weighting rule
    --------------
    1. Compute a per-class score for each model from OOF *probabilities*
       (not hard decisions).  Default metric is Brier score, which rewards
       calibrated confidence and is reliable even when a class has very few
       OOF examples.
    2. Normalize within each class so model weights sum to 1 per class.
       A column-zero guard ensures classes that no model ever predicted still
       get uniform weights (1/n_models) rather than NaN.
    3. Blend toward uniform with *support-adaptive shrinkage*: classes with
       fewer OOF training examples are shrunk more aggressively toward equal
       weights than well-represented classes.

    Why no alpha
    ------------
    The previous additive-smoothing (alpha) parameter compresses real score
    differences before normalization and double-regularizes alongside
    shrinkage.  Zero-score columns are now handled by the column-zero guard;
    rare-class stability is handled by the support-adaptive shrinkage.
    Both mechanisms are more targeted than a global alpha offset.

    Args:
        prob_matrices:         List of (n_samples, n_classes) OOF probability
                               matrices, one per base model.
        y_true:                Integer class labels aligned with OOF rows.
        id2label:              Map from integer class ID to label string.
        metric:                Per-class scoring metric.  ``"brier"`` (default)
                               uses 1 − Brier score; operates on soft
                               probabilities and degrades gracefully for rare
                               classes.  ``"recall"``, ``"precision"``,
                               ``"f1"`` use hard-argmax decisions and are
                               provided for comparison only.
        shrinkage:             Base blend factor toward uniform per-class
                               weights.  This is the shrinkage applied to a
                               well-supported class (support ≥ ~3×
                               ``min_support_for_trust``).  Classes with fewer
                               OOF examples receive additional shrinkage on
                               top of this base.  ``0.0`` = trust learned OOF
                               weights for all classes; ``1.0`` = uniform
                               voting for all classes.  Default ``0.5``.
        min_support_for_trust: OOF sample count at which per-class scores
                               receive roughly ``(1 − shrinkage) × 63%`` of
                               their full weight (exponential decay constant).
                               Classes below this count are pulled strongly
                               toward uniform; classes well above it use the
                               base ``shrinkage``.  Default ``20``.
    """
    if not prob_matrices:
        raise ValueError("prob_matrices must contain at least one matrix.")
    if metric not in {"brier", "recall", "precision", "f1"}:
        raise ValueError("metric must be one of: 'brier', 'recall', 'precision', 'f1'.")
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be in [0, 1].")
    if min_support_for_trust <= 0:
        raise ValueError("min_support_for_trust must be > 0.")

    sorted_ids = sorted(id2label.keys())
    n_models = len(prob_matrices)
    n_classes = len(sorted_ids)
    y_int = np.asarray(y_true, dtype=int)

    # Map integer label positions: id2label keys may not be 0..n_classes-1.
    # Build a lookup from sorted position → integer class id.
    cid_arr = np.array(sorted_ids)  # position j → class id cid_arr[j]

    scores = np.zeros((n_models, n_classes), dtype=float)

    if metric == "brier":
        # Brier score per class: 1 − mean((p_c − 1{y==c})²).
        # Range: worst uncalibrated = ~0.75 for balanced classes; perfect = 1.0.
        # Degenerate edge cases (class entirely absent or present in OOF):
        # if y_bin is all-zero, Brier score = 1 − mean(p²) which is ≥ 0.
        # if y_bin is all-one, Brier score = 1 − mean((p−1)²) which is ≥ 0.
        # Both degenerate cases yield a valid non-negative score; the support-
        # adaptive shrinkage will force near-zero-support classes toward uniform
        # anyway.
        for i, probs in enumerate(prob_matrices):
            for j, cid in enumerate(sorted_ids):
                y_bin = (y_int == cid).astype(float)
                p_col = probs[:, j]
                scores[i, j] = 1.0 - float(np.mean((p_col - y_bin) ** 2))
    else:
        # Hard-argmax metrics kept for diagnostic / comparison use.
        labels = [id2label[cid] for cid in sorted_ids]
        true_labels = [id2label[int(y)] for y in y_true]
        for i, probs in enumerate(prob_matrices):
            pred_ids = np.argmax(probs, axis=1)
            pred_labels = [id2label[cid_arr[int(j)]] for j in pred_ids]
            for j, label in enumerate(labels):
                tp = sum((t == label) and (p == label) for t, p in zip(true_labels, pred_labels))
                fp = sum((t != label) and (p == label) for t, p in zip(true_labels, pred_labels))
                fn = sum((t == label) and (p != label) for t, p in zip(true_labels, pred_labels))
                if metric == "recall":
                    denom = tp + fn
                    scores[i, j] = tp / denom if denom else 0.0
                elif metric == "precision":
                    denom = tp + fp
                    scores[i, j] = tp / denom if denom else 0.0
                else:  # f1
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) else 0.0
                    scores[i, j] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    # Column-zero guard: if no model ever predicted a class in OOF data, every
    # score in that column is 0.  Dividing by 0 would produce NaN; instead we
    # replace the column sum with 1.0 so all scores stay 0, then shrinkage will
    # pull them to uniform (1/n_models).  This is the correct behaviour: no
    # signal → default to equal weight.
    col_sums = scores.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0.0] = 1.0
    scores = scores / col_sums  # each column sums to 1

    # Support-adaptive shrinkage: rare classes (few OOF examples) are pulled
    # harder toward uniform than well-represented ones.
    # Effective shrinkage for class c:
    #   s_c = shrinkage + (1 - shrinkage) × exp(-support_c / min_support_for_trust)
    # When support_c >> min_support_for_trust: s_c ≈ shrinkage (base level).
    # When support_c → 0:                      s_c → 1.0 (forced uniform).
    support = np.array([float(np.sum(y_int == cid)) for cid in sorted_ids])
    class_shrinkage = shrinkage + (1.0 - shrinkage) * np.exp(
        -support / min_support_for_trust
    )  # shape (n_classes,) — values in [shrinkage, 1.0]

    uniform = np.full((n_models, n_classes), 1.0 / n_models)
    # Broadcast class_shrinkage across models: shape (1, n_classes)
    scores = class_shrinkage[None, :] * uniform + (1.0 - class_shrinkage[None, :]) * scores
    # Re-normalize columns to exactly 1 after floating-point blend.
    scores = scores / scores.sum(axis=0, keepdims=True)
    return scores


def _apply_class_voter(
    prob_matrices: list[np.ndarray],
    class_weights: np.ndarray,
) -> np.ndarray:
    """Combine base-model probability matrices with class-aware soft-gating.

    Gate design
    -----------
    For each sample i, model m's gate is the class weight for the class model m
    predicts as most likely:

        gate[m, i] = class_weights[m, argmax_c prob_m[i, c]]

    This directly asks "is model m reliable for the class it is actually
    predicting for sample i?" — a targeted per-sample trust score that is
    informative even when the weight matrix has been shrunk close to uniform.

    The previous design used the *expected* class weight
    (Σ_c prob_m[i,c] × class_weights[m,c]), which is dominated by the all-class
    average when weights are near-uniform, producing gates that are nearly
    identical across models and samples and adding noise rather than signal.

    Normalize gates across models per sample to get convex combination
    coefficients:

        w[m, i] = gate[m, i] / Σ_{m'} gate[m', i]

    Then the combined output is a proper weighted average of valid probability
    vectors, so rows sum to exactly 1 with no renormalization needed:

        combined[i, :] = Σ_m  w[m, i] × prob_m[i, :]

    When class_weights are uniform (1/n_models for all m, c), every gate equals
    1/n_models and the output reduces to plain uniform soft voting.
    """
    if not prob_matrices:
        raise ValueError("prob_matrices must contain at least one matrix.")
    n_models = len(prob_matrices)
    n_samples, n_classes = prob_matrices[0].shape
    if class_weights.shape != (n_models, n_classes):
        raise ValueError(
            f"class_weights must have shape ({n_models}, {n_classes}), got {class_weights.shape}."
        )
    stacked = np.stack(prob_matrices, axis=0)  # (n_models, n_samples, n_classes)

    # gate[m, i] = class_weights[m, argmax_c prob_m[i, c]]
    # pred_classes: (n_models, n_samples) — top-1 predicted class index per model per sample
    pred_classes = np.argmax(stacked, axis=2)  # (n_models, n_samples)
    # Gather the weight for the predicted class: class_weights[m, pred_classes[m, i]]
    m_idx = np.arange(n_models)[:, None]       # (n_models, 1) — broadcast over samples
    gate = class_weights[m_idx, pred_classes]  # (n_models, n_samples)

    # Normalize across models per sample → convex combination coefficients.
    gate_sum = gate.sum(axis=0, keepdims=True)                # (1, n_samples)
    # If all models get zero gate on a sample, fall back to uniform voting for
    # that sample instead of leaving the row as all zeros.
    zero_cols = np.where(gate_sum[0] == 0.0)[0]
    if len(zero_cols) > 0:
        gate[:, zero_cols] = 1.0 / n_models
        gate_sum = gate.sum(axis=0, keepdims=True)
    gate = gate / gate_sum                                     # (n_models, n_samples)

    # Weighted average of full probability vectors.
    # gate[:, :, None] → (n_models, n_samples, 1)
    combined = (stacked * gate[:, :, None]).sum(axis=0)       # (n_samples, n_classes)
    # Rows sum to 1 exactly (convex combination of simplex elements).
    return combined


def _uniform_soft_vote(prob_matrices: list[np.ndarray]) -> np.ndarray:
    """Uniform soft voting over probability matrices."""
    if not prob_matrices:
        raise ValueError("prob_matrices must contain at least one matrix.")
    ref_shape = prob_matrices[0].shape
    for i, m in enumerate(prob_matrices[1:], start=1):
        if m.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: prob_matrices[0] has shape {ref_shape}, "
                f"but prob_matrices[{i}] has shape {m.shape}."
            )
    return np.stack(prob_matrices, axis=0).mean(axis=0)


def _score_prob_matrix(
    probs: np.ndarray,
    y_true: np.ndarray,
    id2label: dict,
    metric: str,
) -> float:
    """Score a probability matrix via top-1 labels and score_predictions()."""
    from ..utils.metrics import score_predictions

    true_labels = [id2label[int(y)] for y in y_true]
    result = _assemble_prediction_result(probs, true_labels, id2label, top_k=1)
    return float(score_predictions(result.top1, metric=metric))


def _compare_class_voter_vs_soft_cv(
    prob_matrices: list[np.ndarray],
    y_true: np.ndarray,
    id2label: dict,
    voter_metric: str,
    shrinkage: float,
    min_support_for_trust: float,
    selection_metric: str,
    cv_folds: int,
    random_state: int,
) -> dict:
    """CV comparison of learned class voter vs uniform soft vote on OOF data.

    Returns a report dict. If stratified CV is not feasible (too few samples in
    at least one class), ``enabled`` is False and no fallback decision should
    be made from this report.
    """
    from sklearn.model_selection import StratifiedKFold

    y_arr = np.asarray(y_true, dtype=int)
    sorted_ids = sorted(id2label.keys())
    supports = [int(np.sum(y_arr == cid)) for cid in sorted_ids]
    max_folds = min(supports) if supports else 0
    folds = min(int(cv_folds), int(max_folds))
    if folds < 2:
        return {
            "enabled": False,
            "reason": (
                "Insufficient per-class support for stratified fallback CV "
                f"(requested={cv_folds}, max_possible={max_folds})."
            ),
            "cv_folds_used": int(folds),
        }

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    voter_scores: list[float] = []
    soft_scores: list[float] = []

    idx = np.arange(len(y_arr))
    for tr_idx, va_idx in cv.split(idx, y_arr):
        tr_probs = [p[tr_idx] for p in prob_matrices]
        va_probs = [p[va_idx] for p in prob_matrices]
        tr_y = y_arr[tr_idx]
        va_y = y_arr[va_idx]

        w = _learn_class_voter_weights(
            prob_matrices=tr_probs,
            y_true=tr_y,
            id2label=id2label,
            metric=voter_metric,
            shrinkage=shrinkage,
            min_support_for_trust=min_support_for_trust,
        )
        voter_pred = _apply_class_voter(va_probs, w)
        soft_pred = _uniform_soft_vote(va_probs)

        voter_scores.append(_score_prob_matrix(voter_pred, va_y, id2label, selection_metric))
        soft_scores.append(_score_prob_matrix(soft_pred, va_y, id2label, selection_metric))

    voter_mean = float(np.mean(voter_scores))
    soft_mean = float(np.mean(soft_scores))
    return {
        "enabled": True,
        "selection_metric": selection_metric,
        "cv_folds_used": int(folds),
        "class_voter_scores": voter_scores,
        "soft_vote_scores": soft_scores,
        "class_voter_mean": voter_mean,
        "soft_vote_mean": soft_mean,
        "delta_class_minus_soft": voter_mean - soft_mean,
    }


def _meta_optuna_objective(
    trial,
    model_name: str,
    base_hp: dict,
    search_space: dict,
    oof_meta_X: np.ndarray,
    oof_y: np.ndarray,
    id2label: dict,
    metric: str,
    cv_folds: int,
    n_jobs: int,
    random_state: int,
) -> float:
    """Optuna trial objective for meta-learner HPO.

    Samples hyperparameters from ``search_space``, scores the resulting model
    via stratified k-fold CV on ``oof_meta_X`` / ``oof_y``.
    Test data is **never accessed** — the OOF matrix is the only data used here.

    Args:
        trial:        Optuna Trial object.
        model_name:   Meta-learner model alias (e.g. ``"logistic_regression"``).
        base_hp:      Fixed HPs from the spec (e.g. ``{"max_iter": 1000}``).
                      Searched keys override these.
        search_space: Dict of param → tuple spec or fixed value.
        oof_meta_X:   OOF meta-feature matrix (n_train × n_base_outputs).
        oof_y:        Integer label array (n_train,).
        id2label:     Integer → label string mapping.
        metric:       Scoring metric (``accuracy``, ``f1_macro``, …).
        cv_folds:     Stratified CV folds for scoring.
        n_jobs:       CPU parallelism.
        random_state: Seed.

    Returns:
        CV score (float, higher is better for all supported metrics).
    """
    hp = dict(base_hp)
    for key, spec in search_space.items():
        if isinstance(spec, tuple):
            kind = spec[0]
            if kind == "float":
                hp[key] = trial.suggest_float(key, spec[1], spec[2])
            elif kind == "log_float":
                hp[key] = trial.suggest_float(key, spec[1], spec[2], log=True)
            elif kind == "int":
                hp[key] = trial.suggest_int(key, spec[1], spec[2])
            elif kind == "categorical":
                hp[key] = trial.suggest_categorical(key, list(spec[1]))
            else:
                raise ValueError(
                    f"Unknown search space kind '{kind}' for param '{key}'. "
                    "Use 'float', 'log_float', 'int', or 'categorical'."
                )
        else:
            hp[key] = spec  # fixed value — no search

    candidate_spec = {"model_name": model_name, "hyperparams": hp}
    model = _resolve_meta_learner(candidate_spec, n_jobs=n_jobs, random_state=random_state)
    return _score_meta_candidate(
        model, oof_meta_X, oof_y, id2label, metric, cv_folds, random_state,
    )


# ---------------------------------------------------------------------------
# InSilicoVA integration helpers
# ---------------------------------------------------------------------------

def _insilicova_save_config(
    spec: dict,
    best_hp: dict | None,
    label2id: dict,
    id2label: dict,
    feature_cols: list[str] | None,
    output_dir: Path,
    label_col: str = "cause",
) -> None:
    """Persist InSilicoVA run configuration to *output_dir*.

    InSilicoVA uses ``data_type="customize"``: it learns symptom-cause
    probabilities from the training data at inference time.  The output
    ``indiv_prob`` columns are named directly by the user's cause labels
    from the training data — no cause mapping is needed.

    A dummy ``training_metadata.json`` is written so the standard
    resume-detection logic (which looks for that file) works unchanged.

    Args:
        spec:          Model spec dict (``model_name="insilicova"``).
        best_hp:       Hyperparams dict (``Nsim``, ``burnin``, ``thin``, …).
                       Falls back to ``spec["hyperparams"]`` when ``None``.
        label2id:      Shared label → integer map.
        id2label:      Shared integer → label map.
        feature_cols:  Fallback VA indicator columns when ``spec["va_cols"]``
                       is absent.
        output_dir:    Target directory (created if missing).
        label_col:     Column name in training DataFrame containing cause labels.
                       Passed to InSilicoVA as ``causes_train``. Default "cause".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hp = best_hp or spec.get("hyperparams") or {}

    config: dict = {
        # va_cols: raw VA indicator columns to pass to InSilicoVA (excludes label)
        "va_cols":     spec.get("va_cols", feature_cols),
        # label_col: cause column in training DataFrame (used as causes_train)
        "label_col":   spec.get("label_col", label_col),
        "hyperparams": hp,
        "id2label":    {str(k): v for k, v in id2label.items()},
        "label2id":    label2id,
    }
    with open(output_dir / "insilicova_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    # Dummy training_metadata.json — enables resume detection
    with open(output_dir / "training_metadata.json", "w") as fh:
        json.dump({"model_name": _INSILICOVA, "hyperparams": hp}, fh, indent=2)

    logger.info("InSilicoVA config saved to %s", output_dir)


def _insilicova_predict_df(
    config_dir: Path,
    df: pd.DataFrame,
    sorted_ids: list,
    train_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Run InSilicoVA (customize mode) on *df* and return a probability matrix.

    Uses ``data_type="customize"``: passes *train_df* (with the cause column)
    as the training reference so InSilicoVA learns symptom-cause probabilities
    from the actual training data rather than the fixed WHO database.  The
    ``indiv_prob`` columns returned by InSilicoVA are then named by the user's
    own cause labels — no cause mapping is required.

    Equivalent R call::

        codeVA(data = df, data.type = "customize", model = "InSilicoVA",
               data.train = train_df, causes.train = label_col,
               Nchain = 3, Nsim = 10000, auto.length = TRUE)

    Args:
        config_dir:  Directory containing ``insilicova_config.json``.
        df:          VA DataFrame of cases to classify (symptom columns only,
                     no label).  Pass ``reset_index(drop=True)`` so row
                     positions align with InSilicoVA's output.
        sorted_ids:  Sorted integer class IDs (from ``sorted(id2label.keys())``).
        train_df:    Training DataFrame including the cause label column.
                     Required for ``data_type="customize"``.  For OOF, pass
                     the fold's training rows; for test prediction, pass the
                     full training set.

    Returns:
        np.ndarray of shape ``(len(df), len(sorted_ids))``.

    Raises:
        ImportError:       ``pyinsilicova`` is not installed.
        FileNotFoundError: ``insilicova_config.json`` missing in *config_dir*.
        ValueError:        *train_df* is None (required for customize mode).
    """
    try:
        from pyinsilicova.insilicova import InSilicoVA
    except ImportError as exc:
        raise ImportError(
            "pyinsilicova is required for 'insilicova' base models.  "
            "Install with:  pip install pyinsilicova"
        ) from exc

    with open(config_dir / "insilicova_config.json") as fh:
        config = json.load(fh)

    id2label  = {int(k): v for k, v in config["id2label"].items()}
    va_cols   = config.get("va_cols")
    label_col = config.get("label_col", "cause")
    hp        = {k: v for k, v in (config.get("hyperparams") or {}).items()}

    if train_df is None:
        raise ValueError(
            "train_df is required for InSilicoVA (data_type='customize').  "
            "Pass the fold training rows for OOF, or the full training set "
            "for test prediction."
        )

    # Extract VA indicator columns (exclude label from train_df)
    val_va   = df[va_cols].reset_index(drop=True)   if va_cols else df.reset_index(drop=True)
    train_va = train_df[va_cols + [label_col]].reset_index(drop=True) if va_cols else train_df.reset_index(drop=True)

    logger.info(
        "Running InSilicoVA (customize) on %d cases with %d training samples ...",
        len(val_va), len(train_va),
    )
    isva = InSilicoVA(
        val_va,
        data_type="customize",
        train_data=train_va,
        causes_train=label_col,
        **hp,
    )
    isva.run()
    indiv_probs: pd.DataFrame = isva.indiv_prob   # cols = user's cause labels

    # Direct column lookup — output column names match user's cause labels
    n         = len(df)
    n_classes = len(sorted_ids)
    proba     = np.zeros((n, n_classes), dtype=float)

    missing = []
    for j, cid in enumerate(sorted_ids):
        label = id2label[cid]
        if label in indiv_probs.columns:
            proba[:, j] = indiv_probs[label].to_numpy(dtype=float)
        else:
            missing.append(label)

    if missing:
        logger.warning(
            "InSilicoVA: %d cause(s) not found in indiv_prob output: %s.  "
            "These columns receive probability 0 and rows are renormalised.",
            len(missing), missing,
        )
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        proba   /= row_sums

    return proba


# ---------------------------------------------------------------------------
# Standalone OOF function
# ---------------------------------------------------------------------------

def generate_oof_predictions(
    text_model_specs: list[BaseModelSpec],
    tabular_model_specs: list[BaseModelSpec],
    train_text_dataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    n_folds: int = 5,
    random_state: int = 42,
    output_dir: str | Path = "runs/ensemble/stacking/oof",
    # text training — val_size and early_stopping_patience are intentionally absent:
    # fold models train on the full fold training split with fixed hyperparams.
    # The fold boundary already defines train vs. held-out; no inner split needed.
    gradient_checkpointing: bool = False,
    batch_size: int = 32,
    # tabular training
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    # resume / disk
    resume: bool = True,
    save_fold_models: bool = False,
    cleanup_fold_files: bool = True,
    # optional pre-computed HP (from HPO stage; one dict per model in spec order)
    text_best_hp: list[dict] | None = None,
    tabular_best_hp: list[dict] | None = None,
    # InSilicoVA — raw DataFrame needed for pyinsilicova (not numpy arrays)
    train_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    label_col: str = "cause",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold probability predictions for all base models.

    For each fold and each base model, trains on the k-1 in-fold samples and
    predicts probabilities for the held-out fold.  OOF probabilities from all
    base models are concatenated into a meta-feature matrix covering the full
    training set.

    Meta-feature matrix column layout (one block of n_classes per model):
    ``[text_0_prob_0..C, text_1_prob_0..C, ..., tab_0_prob_0..C, ...]``

    Args:
        text_model_specs:     List of text base-model spec dicts.
        tabular_model_specs:  List of tabular base-model spec dicts.
        train_text_dataset:   Pre-tokenised ``ClassificationDataset`` for the
                              full training set, or a list of datasets aligned
                              with ``text_model_specs`` when the base models use
                              different tokenizers. Fold subsets are derived via
                              ``_SubsetDataset``.
        X_train:              Preprocessed tabular feature matrix (n_train×p).
        y_train:              Integer label array (n_train,).
        label2id:             Label → integer mapping.
        id2label:             Integer → label mapping.
        n_folds:              CV folds. Default 5.
        random_state:         Seed. Default 42.
        output_dir:           Root OOF directory.  Fold artifacts saved under
                              ``output_dir/fold_k/text_i/`` etc.
        gradient_checkpointing: Enable gradient checkpointing for text folds.
        batch_size:           Text inference batch size.
        n_jobs:               CPU parallelism for tabular models.
        use_gpu:              GPU flag for tabular models. None = auto-detect.
        resume:               Skip completed fold/model combos. Default True.
        save_fold_models:     Keep model weights after extracting OOF probs.
                              ``False`` (default) saves disk space.
        text_best_hp:         Pre-computed HP dicts for text models (from HPO).
                              ``None`` → use ``spec["hyperparams"]`` as-is.
        tabular_best_hp:      Pre-computed HP dicts for tabular models.
        train_df:             Full training DataFrame (required when any spec
                              has ``model_name="insilicova"``).
        feature_cols:         VA indicator columns for InSilicoVA.
        label_col:            Cause label column name in *train_df*.
                              Passed to InSilicoVA as ``causes_train``.
                              Default "cause".

    Returns:
        Tuple ``(oof_meta_X, oof_y)`` where:

        ``oof_meta_X``:  np.ndarray shape (n_train, n_models × n_classes).
                         Row i holds the OOF probability predictions for
                         training sample i from every base model.
        ``oof_y``:       np.ndarray shape (n_train,) — original integer labels
                         in train-set order (identical to input ``y_train``).

    Note:
        When any ``tabular_model_specs`` entry has ``model_name="insilicova"``,
        both ``train_df`` and ``feature_cols`` must be supplied.  InSilicoVA
        operates on the raw VA DataFrame (not the preprocessed numpy arrays)
        and has no separate training phase — the fold val rows are passed
        directly to :func:`_insilicova_predict_df`.
    """
    from sklearn.model_selection import StratifiedKFold

    # Lazy pipeline imports (avoid circular)
    from ..text.train   import train   as text_train
    from ..text.predict import predict as text_predict
    from ..tabular.train   import train   as tabular_train
    from ..tabular.predict import predict as tabular_predict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train     = len(y_train)
    sorted_ids  = sorted(id2label.keys())
    n_classes   = len(sorted_ids)
    n_text      = len(text_model_specs)
    n_tabular   = len(tabular_model_specs)
    n_total     = n_text + n_tabular

    if isinstance(train_text_dataset, (list, tuple)):
        if len(train_text_dataset) != n_text:
            raise ValueError(
                "When train_text_dataset is a list/tuple, it must contain one "
                f"dataset per text model ({n_text}); got {len(train_text_dataset)}."
            )
        train_text_datasets = list(train_text_dataset)
    else:
        # Backward compatibility for callers whose text models share a tokenizer.
        train_text_datasets = [train_text_dataset] * n_text

    oof_meta_X  = np.zeros((n_train, n_total * n_classes), dtype=float)
    oof_y       = y_train.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_splits = list(skf.split(np.arange(n_train), y_train))

    for fold_idx, (train_idx, val_idx) in enumerate(all_splits):
        fold_dir = output_dir / f"fold_{fold_idx}"
        logger.info("=== Fold %d/%d  (train=%d, val=%d) ===",
                    fold_idx + 1, n_folds, len(train_idx), len(val_idx))

        # ---------- Text base models ----------------------------------------
        for i, spec in enumerate(text_model_specs):
            model_dir      = fold_dir / f"text_{i}"
            oof_probs_file = model_dir / "oof_probs.npy"

            if resume and oof_probs_file.exists():
                logger.info("  [text_%d fold_%d] Resuming — loading %s", i, fold_idx, oof_probs_file)
                fold_probs = np.load(oof_probs_file)
            else:
                model_dir.mkdir(parents=True, exist_ok=True)
                model_name = spec["model_name"]
                max_length = spec.get("max_length", 512)
                use_lora   = spec.get("use_lora", False)
                gc         = spec.get("gradient_checkpointing", gradient_checkpointing)
                hp         = (text_best_hp[i] if text_best_hp else None) or spec.get("hyperparams") or {}

                logger.info("  [text_%d fold_%d] Training %s ...", i, fold_idx, model_name)
                model_text_dataset = train_text_datasets[i]
                fold_train_ds = _SubsetDataset(model_text_dataset, train_idx)
                fold_val_ds   = _SubsetDataset(model_text_dataset, val_idx)

                text_train(
                    train_dataset=fold_train_ds,
                    label2id=label2id, id2label=id2label,
                    model_name=model_name,
                    output_dir=model_dir / "weights",
                    hyperparams=hp,
                    val_size=None,                # fold boundary IS the val boundary;
                    use_lora=use_lora,            # no inner split — epochs fixed by HPO
                    gradient_checkpointing=gc,
                    early_stopping_patience=None, # no early stopping in fold training
                    resume=False,                 # never cross-contaminate fold checkpoints
                )

                result     = text_predict(
                    model_dir / "weights",
                    fold_val_ds,
                    batch_size=batch_size,
                    top_k=1,
                )
                fold_probs = _extract_probs(result, sorted_ids)  # (n_val, n_classes)
                np.save(oof_probs_file, fold_probs)
                logger.info("  [text_%d fold_%d] OOF probs saved (%d samples)", i, fold_idx, len(val_idx))

                if not save_fold_models:
                    shutil.rmtree(model_dir / "weights", ignore_errors=True)

            col_s = i * n_classes
            oof_meta_X[val_idx, col_s:col_s + n_classes] = fold_probs

        # ---------- Tabular base models -------------------------------------
        for i, spec in enumerate(tabular_model_specs):
            model_dir      = fold_dir / f"tabular_{i}"
            oof_probs_file = model_dir / "oof_probs.npy"

            if resume and oof_probs_file.exists():
                logger.info("  [tab_%d fold_%d] Resuming — loading %s", i, fold_idx, oof_probs_file)
                fold_probs = np.load(oof_probs_file)
            else:
                model_dir.mkdir(parents=True, exist_ok=True)
                model_name = spec["model_name"]
                logger.info("  [tab_%d fold_%d] Training %s ...", i, fold_idx, model_name)

                if model_name == _INSILICOVA:
                    # InSilicoVA (customize mode): learns symptom-cause probs
                    # from the fold training rows; predicts on fold val rows.
                    if train_df is None:
                        raise ValueError(
                            "train_df must be supplied to generate_oof_predictions() "
                            "when any tabular_model_spec has model_name='insilicova'."
                        )
                    hp = (tabular_best_hp[i] if tabular_best_hp else None) or spec.get("hyperparams") or {}
                    _insilicova_save_config(
                        spec, hp, label2id, id2label, feature_cols,
                        model_dir / "weights",
                        label_col=label_col,
                    )
                    fold_probs = _insilicova_predict_df(
                        model_dir / "weights",
                        train_df.iloc[val_idx].reset_index(drop=True),
                        sorted_ids,
                        train_df=train_df.iloc[train_idx].reset_index(drop=True),
                    )
                else:
                    hp = (tabular_best_hp[i] if tabular_best_hp else None) or spec.get("hyperparams")
                    tabular_train(
                        X_train=X_train[train_idx],
                        y_train=y_train[train_idx],
                        label2id=label2id, id2label=id2label,
                        model_name=model_name,
                        output_dir=model_dir / "weights",
                        hyperparams=hp,
                        random_state=random_state,
                        n_jobs=n_jobs, use_gpu=use_gpu,
                    )
                    result = tabular_predict(
                        model_dir / "weights",
                        X_test=X_train[val_idx],
                        y_test=y_train[val_idx],
                        top_k=1,
                    )
                    fold_probs = _extract_probs(result, sorted_ids)
                    if not save_fold_models:
                        shutil.rmtree(model_dir / "weights", ignore_errors=True)

                np.save(oof_probs_file, fold_probs)
                logger.info("  [tab_%d fold_%d] OOF probs saved (%d samples)", i, fold_idx, len(val_idx))

            col_s = (n_text + i) * n_classes
            oof_meta_X[val_idx, col_s:col_s + n_classes] = fold_probs

    # Persist assembled OOF matrix
    np.save(output_dir / "oof_meta_X.npy", oof_meta_X)
    np.save(output_dir / "oof_y.npy",      oof_y)

    meta_names = _build_meta_feature_names(text_model_specs, tabular_model_specs, sorted_ids)
    with open(output_dir / "meta_feature_names.json", "w") as fh:
        json.dump(meta_names, fh, indent=2)

    # Remove per-fold intermediate directories — only oof_meta_X.npy / oof_y.npy
    # are needed from this point on; fold dirs are resume checkpoints only.
    if cleanup_fold_files:
        for fold_dir in sorted(output_dir.glob("fold_*")):
            shutil.rmtree(fold_dir, ignore_errors=True)
        logger.info("Cleaned up fold directories from %s", output_dir)

    logger.info("OOF generation complete — meta_X shape: %s", oof_meta_X.shape)
    return oof_meta_X, oof_y


# ---------------------------------------------------------------------------
# StackingClassifier
# ---------------------------------------------------------------------------

class StackingClassifier:
    """End-to-end stacking (super learner) ensemble classifier.

    Implements a two-stage pipeline with explicit stage methods so that each
    stage can be run independently across separate Python sessions (resumable).

    Stage 1 — ``train_base_models(df, …)``
        Splits data, runs optional per-model HPO, executes the k-fold OOF loop,
        and retrains final base models on the full training set.

    Stage 2 — ``train_meta_learner_stage()``
        Loads OOF meta-features (from disk or in-memory), trains one or more
        meta-learner candidates, selects the best by CV metric, saves the winner.

    Stage 3 — ``predict_test()``
        Loads final base models, predicts on the hold-out test set, stacks
        probabilities, passes through the meta-learner, returns PredictionResult.

    ``run(df, …)`` calls all three stages in sequence.

    Attributes (populated after train_base_models())
    -------------------------------------------------
    train_df, test_df :   Split DataFrames.
    label2id, id2label :  Shared label maps.
    oof_meta_X :          OOF meta-feature matrix (n_train × n_models*n_classes).
    oof_y :               Integer label array for OOF rows.
    text_best_hp :        Best HP per text model (list of dicts).
    tabular_best_hp :     Best HP per tabular model (list of dicts).

    Attributes (populated after train_meta_learner_stage())
    --------------------------------------------------------
    meta_learner :        Fitted best meta-learner.
    meta_scores :         Dict {model_name: cv_score} for all candidates.

    Attributes (populated after predict_test() / run())
    ----------------------------------------------------
    predictions :         Final PredictionResult.
    """

    def __init__(
        self,
        text_models: list[BaseModelSpec],
        tabular_models: list[BaseModelSpec],
        output_dir: str | Path = "runs/ensemble/stacking",
        meta_learners: list[dict] | dict | None = None,
        meta_select_metric: str = "f1_macro",
        n_folds: int = 5,
        resume: bool = True,
        load_oof_from: str | Path | None = None,
    ):
        """Initialise StackingClassifier.

        Args:
            text_models:          List of text base-model spec dicts.
                                  Same format as SoftVotingClassifier.
                                  When ``load_oof_from`` is set, list **only
                                  the new models** to add — do not repeat
                                  models already in the source run.
            tabular_models:       List of tabular base-model spec dicts.
                                  Same rule as ``text_models``.
            output_dir:           Root output directory for this run.
            meta_learners:        One meta-learner spec dict, or a list of
                                  candidates (the best by CV score is chosen).
                                  Default: logistic regression (C=1.0).
            meta_select_metric:   Metric used to select among multiple
                                  meta-learner candidates.  Options:
                                  ``accuracy``, ``f1_macro`` (default),
                                  ``f1_weighted``, ``csmf_accuracy``.
            n_folds:              CV folds for the OOF loop. Default 5.
            resume:               Resume interrupted training. Default True.
            load_oof_from:        Path to an existing stacking ``output_dir``
                                  whose OOF predictions should be reused.
                                  When set:

                                  * The train/test split from that run is
                                    reused verbatim (row alignment is required).
                                  * OOF is only computed for the models in
                                    *this* ``text_models`` / ``tabular_models``
                                    (the newly added ones).
                                  * The combined OOF matrix
                                    ``[inherited | new]`` feeds Stage 2.
                                  * ``predict_test()`` draws final-model
                                    weights from the original run for inherited
                                    models and from ``output_dir/final/`` for
                                    new models.
                                  * Chaining is supported: the source run may
                                    itself have been created with
                                    ``load_oof_from``.

                                  Default ``None`` (standard full training).
        """
        if not text_models and not tabular_models:
            raise ValueError("At least one text or tabular model spec is required.")

        # Normalise meta_learners to a list
        if meta_learners is None:
            meta_learners = [dict(DEFAULT_META_LEARNER)]
        elif isinstance(meta_learners, dict):
            meta_learners = [meta_learners]

        self.text_models        = list(text_models)    if text_models    else []
        self.tabular_models     = list(tabular_models) if tabular_models else []
        self.output_dir         = Path(output_dir)
        self.meta_learner_specs = meta_learners
        self.meta_select_metric = meta_select_metric
        self.n_folds            = n_folds
        self.resume             = resume
        self.load_oof_from      = Path(load_oof_from) if load_oof_from else None

        # Populated after train_base_models()
        self.train_df:       pd.DataFrame | None = None
        self.test_df:        pd.DataFrame | None = None
        self.label2id:       dict | None          = None
        self.id2label:       dict | None          = None
        self.oof_meta_X:     np.ndarray | None    = None
        self.oof_y:          np.ndarray | None    = None
        self.text_best_hp:   list[dict]           = []
        self.tabular_best_hp: list[dict]          = []
        self._text_col:      str | None           = None
        self._feature_cols:  list[str] | None     = None
        self._label_col:     str | None           = None

        # Populated after train_meta_learner_stage()
        self.meta_learner: Any | None = None
        self.meta_scores:  dict       = {}
        self.class_voter_weights: np.ndarray | None = None
        self.class_voter_metadata: dict | None = None

        # Populated after predict_test()
        self.predictions: PredictionResult | None = None

    # ------------------------------------------------------------------
    # Internal: load OOF state from disk (for cross-session Stage 2/3)
    # ------------------------------------------------------------------

    def _oof_input_paths(self) -> dict:
        """The OOF files Stage 2 consumes — fingerprinted to detect staleness."""
        oof_dir = self.output_dir / "oof"
        return {
            "oof_meta_X": oof_dir / "oof_meta_X.npy",
            "oof_y": oof_dir / "oof_y.npy",
        }

    def _ensure_oof_loaded(self):
        """Load OOF artifacts from disk if not already in memory."""
        if self.oof_meta_X is not None:
            return

        oof_dir = self.output_dir / "oof"
        meta_X_file = oof_dir / "oof_meta_X.npy"
        if not meta_X_file.exists():
            raise RuntimeError(
                "OOF meta-features not found at %s.  "
                "Run train_base_models() first." % meta_X_file
            )

        self.oof_meta_X = np.load(meta_X_file)
        self.oof_y      = np.load(oof_dir / "oof_y.npy")

        lbl_path   = self.output_dir / "label2id.json"
        id2lbl_path = self.output_dir / "id2label.json"
        with open(lbl_path)   as f: self.label2id = json.load(f)
        with open(id2lbl_path) as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

        meta_path = oof_dir / "oof_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._text_col    = meta.get("text_col")
            self._feature_cols = meta.get("feature_cols")
            self._label_col   = meta.get("label_col")
            self.text_best_hp  = meta.get("text_best_hp",    [{}] * len(self.text_models))
            self.tabular_best_hp = meta.get("tabular_best_hp", [{}] * len(self.tabular_models))

        logger.info("OOF state loaded from disk — meta_X shape: %s", self.oof_meta_X.shape)

    def _resolve_final_dir(self, source: dict, inherited_from: str | None) -> Path:
        """Locate a base model's weights, tolerating a relocated run directory.

        ``model_sources`` records absolute paths resolved at stage-1 time.  A run
        opened later from a different machine or mount point — a synced folder, a
        different home, a symlinked root — still carries the original strings, so
        stage 2 fails to find weights sitting in its own ``final/``.  Re-anchor to
        this run when the stored path is gone and nothing was inherited from
        another run (where ``final/`` legitimately does not hold the weights).
        """
        stored = Path(source["final_dir"])
        if stored.exists() or inherited_from is not None:
            return stored

        local = self.output_dir / "final" / f"{source['type']}_{source['local_index']}"
        if not local.exists():
            return stored  # keep the original path in the error the caller raises

        logger.warning(
            "Recorded final_dir %s does not exist; using %s from this run directory.",
            stored, local,
        )
        return local

    # ------------------------------------------------------------------
    # Stage 1: train base models + generate OOF predictions
    # ------------------------------------------------------------------

    def train_base_models(
        self,
        df: pd.DataFrame,
        label_col: str,
        text_col: str | None = None,
        feature_cols: list[str] | None = None,
        # --- split ---
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        split_col: str | None = None,
        # --- text training ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 3,
        batch_size: int = 32,
        # --- tabular training ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        # --- fold model storage ---
        save_fold_models: bool = False,
        cleanup_fold_files: bool = True,
    ) -> dict:
        """Stage 1: split → (HPO) → k-fold OOF loop → final model training.

        After this stage all OOF artifacts and final base models are saved to
        disk.  The next stage (``train_meta_learner_stage``) can be run in a
        separate Python session.

        Args:
            df:                     Input DataFrame.
            label_col:              Cause-of-death label column.
            text_col:               Free-text narrative column.  Required when
                                    ``text_models`` is non-empty.
            feature_cols:           Tabular feature columns.  Required when
                                    ``tabular_models`` is non-empty.
            test_size:              Test fraction. Default 0.2.
            random_state:           Seed. Default 42.
            stratify:               Stratified split. Default True.
            val_size:               Internal val fraction for text fold models
                                    (early stopping).  Default 0.1.
            gradient_checkpointing: Enable gradient checkpointing. Default False.
            early_stopping_patience: Text early stopping patience. Default 3.
            batch_size:             Inference batch size for text. Default 32.
            n_jobs:                 CPU parallelism for tabular. Default -1.
            use_gpu:                GPU flag for tabular. None = auto-detect.
            encode_categoricals:    Categorical encoding. Default ``"ordinal"``.
            scale_numeric:          Apply StandardScaler. Default False.
            save_fold_models:       Keep fold model weights on disk. Default False.

        Returns:
            dict with ``oof_meta_X``, ``oof_y``, ``label2id``, ``id2label``,
            ``n_folds``, ``output_dir``.
        """
        if self.text_models and text_col is None:
            raise ValueError("text_col is required when text_models is non-empty.")
        if self.tabular_models and feature_cols is None:
            raise ValueError("feature_cols is required when tabular_models is non-empty.")

        # Lazy imports
        from ..utils.split            import split
        from ..text.dataset           import prepare_dataset as text_prepare
        from ..text.train             import train            as text_train
        from ..tabular.dataset        import prepare_dataset as tab_prepare
        from ..tabular.train          import train            as tab_train
        from ..text.hpo               import optimize         as text_optimize
        from ..tabular.hpo            import optimize         as tab_optimize

        self._text_col    = text_col
        self._feature_cols = feature_cols
        self._label_col   = label_col

        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # --- Inherited OOF loading (load_oof_from) --------------------------
        # When load_oof_from is set:
        #   • Reuse the exact train/test split from the source run (row alignment).
        #   • Only run OOF for the NEW models listed in text_models/tabular_models.
        #   • Concatenate [inherited_oof | new_oof] column-wise.
        #   • Store absolute final_dir paths in model_sources so predict_test()
        #     can locate weights for both old and new models across sessions.
        inherited_oof_X: np.ndarray | None = None
        inherited_model_sources: list[dict] = []

        if self.load_oof_from:
            _src = self.load_oof_from
            _ioof_file = _src / "oof" / "oof_meta_X.npy"
            if not _ioof_file.exists():
                raise FileNotFoundError(
                    f"oof_meta_X.npy not found in {_src / 'oof'}.  "
                    "Run train_base_models() on the source stacking run first."
                )
            inherited_oof_X = np.load(_ioof_file)
            logger.info(
                "load_oof_from: loaded inherited OOF matrix %s from %s",
                inherited_oof_X.shape, _src,
            )

            # Reuse the source split — OOF row i must mean the same training
            # sample in both inherited and new matrices.
            _src_data = _src / "data"
            if not (_src_data / "train_df.csv").exists():
                raise FileNotFoundError(
                    f"train_df.csv not found in {_src_data}.  "
                    "The source run must have been completed with train_base_models()."
                )
            logger.info(
                "load_oof_from: reusing train/test split from %s to guarantee "
                "OOF row alignment.  The df argument is still used to prepare "
                "features for the new models.", _src,
            )
            self.train_df = pd.read_csv(_src_data / "train_df.csv")
            self.test_df  = pd.read_csv(_src_data / "test_df.csv")

            if inherited_oof_X.shape[0] != len(self.train_df):
                raise ValueError(
                    f"Inherited OOF has {inherited_oof_X.shape[0]} rows but "
                    f"train_df.csv has {len(self.train_df)} rows — shape mismatch.  "
                    "Ensure load_oof_from points to the correct source run."
                )

            # Load model_sources from source metadata (supports chaining)
            _src_meta_path = _src / "oof" / "oof_metadata.json"
            if _src_meta_path.exists():
                with open(_src_meta_path) as _f:
                    _src_meta = json.load(_f)
                if "model_sources" in _src_meta:
                    inherited_model_sources = _src_meta["model_sources"]
                else:
                    # Backward compat: construct from old flat metadata fields
                    _old_text    = _src_meta.get("text_specs", [])
                    _old_tabular = _src_meta.get("tabular_specs", [])
                    _old_nc      = _src_meta.get("n_classes", 1)
                    _col = 0
                    for _j, _spec in enumerate(_old_text):
                        inherited_model_sources.append({
                            "type": "text", "local_index": _j, "spec": _spec,
                            "final_dir": str((_src / "final" / f"text_{_j}").resolve()),
                            "col_start": _col, "n_cols": _old_nc,
                        })
                        _col += _old_nc
                    for _j, _spec in enumerate(_old_tabular):
                        inherited_model_sources.append({
                            "type": "tabular", "local_index": _j, "spec": _spec,
                            "final_dir": str((_src / "final" / f"tabular_{_j}").resolve()),
                            "col_start": _col, "n_cols": _old_nc,
                        })
                        _col += _old_nc
            else:
                logger.warning(
                    "No oof_metadata.json found in %s — cannot reconstruct "
                    "inherited model_sources.  predict_test() may fail for "
                    "inherited models.", _src / "oof",
                )
        else:
            # --- Step 1: split ----------------------------------------------
            self.train_df, self.test_df = split(
                df, label_col=label_col, text_col=text_col,
                test_size=test_size, random_state=random_state, stratify=stratify,
                split_col=split_col,
            )
            logger.info("Split: %d train / %d test", len(self.train_df), len(self.test_df))

        # Save (or re-save) splits for predict_test() in future sessions
        self.train_df.to_csv(data_dir / "train_df.csv", index=False)
        self.test_df.to_csv(data_dir  / "test_df.csv",  index=False)

        # --- Step 2: prepare one tokenized dataset per tokenizer ------------
        label2id = id2label = None
        train_text_datasets = []

        if self.text_models:
            dataset_cache = {}
            for i, spec in enumerate(self.text_models):
                cache_key = (spec["model_name"], spec.get("max_length", 512))
                if cache_key not in dataset_cache:
                    dataset_cache[cache_key] = text_prepare(
                        self.train_df, self.test_df,
                        text_col=text_col, label_col=label_col,
                        model_name=spec["model_name"],
                        max_length=spec.get("max_length", 512),
                    )
                train_text_ds, _test_text_ds, model_label2id, model_id2label = (
                    dataset_cache[cache_key]
                )
                if label2id is None:
                    label2id, id2label = model_label2id, model_id2label
                elif model_label2id != label2id or model_id2label != id2label:
                    raise ValueError(
                        "Text base models produced inconsistent label maps while "
                        f"preparing dataset {i} ({spec['model_name']})."
                    )
                train_text_datasets.append(train_text_ds)

        # Tabular preprocessing
        X_train = X_test = y_train = y_test = None
        preprocessor = feature_names = None

        if self.tabular_models:
            (X_train, X_test, y_train, y_test,
             preprocessor, lbl2id, id2lbl, feature_names) = tab_prepare(
                self.train_df, self.test_df,
                feature_cols=feature_cols, label_col=label_col,
                encode_categoricals=encode_categoricals,
                scale_numeric=scale_numeric,
            )
            if label2id is None:
                label2id, id2label = lbl2id, id2lbl

        self.label2id = label2id
        self.id2label = id2label
        sorted_ids    = sorted(id2label.keys())
        n_classes     = len(sorted_ids)

        # Persist label maps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "label2id.json", "w") as fh:
            json.dump(label2id, fh, indent=2)
        with open(self.output_dir / "id2label.json", "w") as fh:
            json.dump({str(k): v for k, v in id2label.items()}, fh, indent=2)

        # Save tabular test arrays for predict_test()
        if X_test is not None:
            np.save(data_dir / "X_test.npy", X_test)
            np.save(data_dir / "y_test.npy", y_test)

        # --- Step 3: optional HPO (once, before OOF loop) -------------------
        hpo_dir = self.output_dir / "hpo"
        self.text_best_hp    = []
        self.tabular_best_hp = []

        for i, spec in enumerate(self.text_models):
            if spec.get("use_optimize", False):
                logger.info("Text model %d (%s): running HPO ...", i, spec["model_name"])
                best_hp, _ = text_optimize(
                    train_dataset=train_text_datasets[i],
                    label2id=label2id, id2label=id2label,
                    model_name=spec["model_name"],
                    output_dir=hpo_dir / f"text_{i}",
                    n_trials=spec.get("n_trials", 20),
                    metric=spec.get("optimize_metric", "f1_macro"),
                    search_space=spec.get("search_space"),
                    random_state=random_state,
                    use_lora=spec.get("use_lora", False),
                    gradient_checkpointing=gradient_checkpointing,
                    early_stopping_patience=early_stopping_patience,
                )
                self.text_best_hp.append(best_hp)
            else:
                self.text_best_hp.append(spec.get("hyperparams") or {})

        for i, spec in enumerate(self.tabular_models):
            if spec.get("use_optimize", False):
                if spec["model_name"] == _INSILICOVA:
                    logger.info(
                        "Tabular model %d (insilicova): HPO not applicable — "
                        "InSilicoVA uses a fixed Bayesian cause database.  "
                        "Using spec hyperparams.", i
                    )
                    self.tabular_best_hp.append(spec.get("hyperparams") or {})
                else:
                    logger.info("Tabular model %d (%s): running HPO ...", i, spec["model_name"])
                    best_hp, _ = tab_optimize(
                        X_train=X_train, y_train=y_train,
                        label2id=label2id, id2label=id2label,
                        model_name=spec["model_name"],
                        output_dir=hpo_dir / f"tabular_{i}",
                        n_trials=spec.get("n_trials", 20),
                        metric=spec.get("optimize_metric", "f1_macro"),
                        search_space=spec.get("search_space"),
                        use_cv=spec.get("use_cv", True),
                        n_cv_folds=spec.get("n_cv_folds", 3),
                        random_state=random_state,
                        n_jobs=n_jobs, use_gpu=use_gpu,
                    )
                    self.tabular_best_hp.append(best_hp)
            else:
                self.tabular_best_hp.append(spec.get("hyperparams"))

        # --- Step 4: k-fold OOF loop ----------------------------------------
        oof_dir = self.output_dir / "oof"
        n_inherited_cols = inherited_oof_X.shape[1] if inherited_oof_X is not None else 0
        n_new_models     = len(self.text_models) + len(self.tabular_models)
        expected_cols    = n_inherited_cols + n_new_models * n_classes

        # Resume check: skip loop only when assembled matrix has the right shape.
        # If load_oof_from was used and the file exists but has only new-model
        # columns (incomplete previous run), re-generate and re-combine.
        _oof_file = oof_dir / "oof_meta_X.npy"
        _skip_oof = (
            self.resume
            and _oof_file.exists()
            and np.load(_oof_file, mmap_mode="r").shape[1] == expected_cols
        )

        if _skip_oof:
            logger.info("OOF meta-features already assembled — skipping OOF loop.")
            self.oof_meta_X = np.load(_oof_file)
            self.oof_y      = np.load(oof_dir / "oof_y.npy")
        else:
            new_oof_X, self.oof_y = generate_oof_predictions(
                text_model_specs=self.text_models,
                tabular_model_specs=self.tabular_models,
                train_text_dataset=train_text_datasets if self.text_models else None,
                X_train=X_train,
                y_train=y_train if y_train is not None else np.array([]),
                label2id=label2id,
                id2label=id2label,
                n_folds=self.n_folds,
                random_state=random_state,
                output_dir=oof_dir,
                gradient_checkpointing=gradient_checkpointing,
                batch_size=batch_size,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
                resume=self.resume,
                save_fold_models=save_fold_models,
                cleanup_fold_files=cleanup_fold_files,
                text_best_hp=self.text_best_hp    or None,
                tabular_best_hp=self.tabular_best_hp or None,
                train_df=self.train_df,
                feature_cols=feature_cols,
            )

            # Prepend inherited columns (if any) and overwrite the file.
            if inherited_oof_X is not None:
                self.oof_meta_X = np.concatenate([inherited_oof_X, new_oof_X], axis=1)
                logger.info(
                    "Combined inherited OOF %s + new OOF %s → %s",
                    inherited_oof_X.shape, new_oof_X.shape, self.oof_meta_X.shape,
                )
            else:
                self.oof_meta_X = new_oof_X

            # generate_oof_predictions() already saved new_oof_X; overwrite with
            # the combined matrix so future resume picks up the full matrix.
            oof_dir.mkdir(parents=True, exist_ok=True)
            np.save(_oof_file,            self.oof_meta_X)
            np.save(oof_dir / "oof_y.npy", self.oof_y)

        # Build model_sources — flat list of all models (inherited + new) in
        # column order.  Absolute final_dir paths survive across sessions and
        # across directories, enabling predict_test() to find weights for both
        # inherited models (in the source run) and new models (in output_dir).
        final_dir = self.output_dir / "final"
        col_offset = n_inherited_cols
        new_model_sources: list[dict] = []
        for i, spec in enumerate(self.text_models):
            new_model_sources.append({
                "type":        "text",
                "local_index": i,
                "spec":        spec,
                "final_dir":   str((final_dir / f"text_{i}").resolve()),
                "col_start":   col_offset + i * n_classes,
                "n_cols":      n_classes,
            })
        for i, spec in enumerate(self.tabular_models):
            new_model_sources.append({
                "type":        "tabular",
                "local_index": i,
                "spec":        spec,
                "final_dir":   str((final_dir / f"tabular_{i}").resolve()),
                "col_start":   col_offset + (len(self.text_models) + i) * n_classes,
                "n_cols":      n_classes,
            })
        all_model_sources = inherited_model_sources + new_model_sources

        # Save OOF metadata for cross-session handoff
        oof_meta_names = []
        for src in all_model_sources:
            mn = src["spec"]["model_name"].replace("/", "_").replace("-", "_")
            prefix = f"text_{src['local_index']}_{mn}" if src["type"] == "text" \
                     else f"tab_{src['local_index']}_{mn}"
            for cid in sorted_ids:
                oof_meta_names.append(f"{prefix}_prob_{cid}")

        oof_metadata = {
            "n_folds":          self.n_folds,
            "n_text_models":    len(self.text_models),
            "n_tabular_models": len(self.tabular_models),
            "n_classes":        n_classes,
            "n_train":          len(self.train_df),
            "n_test":           len(self.test_df),
            "text_col":         text_col,
            "feature_cols":     feature_cols,
            "label_col":        label_col,
            "label2id":         label2id,
            "id2label":         {str(k): v for k, v in id2label.items()},
            "text_specs":       self.text_models,
            "tabular_specs":    self.tabular_models,
            "model_sources":    all_model_sources,
            "meta_feature_names": oof_meta_names,
            "text_best_hp":     self.text_best_hp,
            "tabular_best_hp":  self.tabular_best_hp,
            "inherited_from":   str(self.load_oof_from.resolve()) if self.load_oof_from else None,
        }
        with open(oof_dir / "oof_metadata.json", "w") as fh:
            json.dump(oof_metadata, fh, indent=2, default=str)

        # --- Step 5: train final base models on full training set -----------
        final_dir = self.output_dir / "final"

        for i, spec in enumerate(self.text_models):
            model_dir  = final_dir / f"text_{i}"
            done_marker = model_dir / "training_metadata.json"

            if self.resume and done_marker.exists():
                logger.info("Final text model %d already trained — skipping.", i)
                continue

            logger.info("Training final text model %d/%d: %s",
                        i + 1, len(self.text_models), spec["model_name"])
            final_val = None if spec.get("use_optimize") else val_size

            text_train(
                train_dataset=train_text_datasets[i],
                label2id=label2id, id2label=id2label,
                model_name=spec["model_name"],
                output_dir=model_dir,
                hyperparams=self.text_best_hp[i] or {},
                val_size=final_val,
                use_lora=spec.get("use_lora", False),
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                resume=self.resume,
            )

        for i, spec in enumerate(self.tabular_models):
            model_dir   = final_dir / f"tabular_{i}"
            done_marker = model_dir / "training_metadata.json"

            if self.resume and done_marker.exists():
                logger.info("Final tabular model %d already trained — skipping.", i)
                continue

            logger.info("Training final tabular model %d/%d: %s",
                        i + 1, len(self.tabular_models), spec["model_name"])

            if spec["model_name"] == _INSILICOVA:
                _insilicova_save_config(
                    spec, self.tabular_best_hp[i], label2id, id2label,
                    feature_cols, model_dir, label_col=label_col,
                )
            else:
                tab_train(
                    X_train=X_train, y_train=y_train,
                    label2id=label2id, id2label=id2label,
                    model_name=spec["model_name"],
                    output_dir=model_dir,
                    hyperparams=self.tabular_best_hp[i],
                    preprocessor=preprocessor, feature_names=feature_names,
                    random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu,
                )

        logger.info("Stage 1 complete. OOF shape: %s", self.oof_meta_X.shape)
        return {
            "oof_meta_X": self.oof_meta_X,
            "oof_y":      self.oof_y,
            "label2id":   label2id,
            "id2label":   id2label,
            "n_folds":    self.n_folds,
            "output_dir": self.output_dir,
        }

    # ------------------------------------------------------------------
    # Stage 2: train meta-learner(s) + select best
    # ------------------------------------------------------------------

    def train_meta_learner_stage(
        self,
        meta_learners: list[dict] | dict | None = None,
        metric: str | None = None,
        meta_cv_folds: int = 3,
        random_state: int = 42,
        n_jobs: int = -1,
        use_optimize: bool = False,
        n_trials: int = 30,
        search_space: dict | None = None,
    ) -> dict:
        """Stage 2: train meta-learner candidates and select the best.

        Can be called after ``train_base_models()`` in the same session, or in
        a fresh session (OOF data loaded automatically from disk).

        **No data leakage guarantee:** this stage uses only the OOF meta-feature
        matrix (``oof_meta_X``) and OOF labels (``oof_y``).  Test data is never
        accessed here, whether Optuna search is enabled or not.

        **Fixed-spec selection (default, ``use_optimize=False``):**
        When multiple meta-learner specs are provided, each is scored via
        stratified ``meta_cv_folds``-fold CV on the OOF meta-features.  The
        highest-scoring model is fitted on all OOF data and saved.

        **Optuna HP search (``use_optimize=True``):**
        Exactly one meta-learner spec must be provided (the model type to
        search).  Optuna runs ``n_trials`` trials, each scored via stratified
        ``meta_cv_folds``-fold CV on the OOF meta-features.  The best
        hyperparameters are used to fit the final meta-learner on all OOF data.
        This is a single round of selection — no two-stage leakage risk.

        Args:
            meta_learners:   Override ``self.meta_learner_specs``.  One dict or
                             a list of dicts.  ``None`` uses the specs from
                             ``__init__``.
            metric:          Selection metric.  Default: ``self.meta_select_metric``
                             (``f1_macro`` unless overridden at init).
            meta_cv_folds:   CV folds for meta-learner scoring. Default 3.
            random_state:    Seed. Default 42.
            n_jobs:          CPU parallelism for meta-learner instantiation.
            use_optimize:    Run Optuna HP search instead of fixed-spec
                             selection.  Requires exactly one spec.  Default
                             ``False``.
            n_trials:        Number of Optuna trials when ``use_optimize=True``.
                             Default 30.
            search_space:    HP search space dict for Optuna.  Keys are param
                             names; values are tuple specs
                             ``("log_float", lo, hi)``, ``("float", lo, hi)``,
                             ``("int", lo, hi)``, ``("categorical", [...])``,
                             or a fixed scalar.  ``None`` uses
                             ``DEFAULT_META_SEARCH_SPACES[model_name]``.
                             Only used when ``use_optimize=True``.

        Returns:
            dict with ``meta_learner``, ``meta_scores``, ``best_meta_name``,
            ``output_dir``.
        """
        self._ensure_oof_loaded()

        if meta_learners is not None:
            specs = [meta_learners] if isinstance(meta_learners, dict) else list(meta_learners)
        else:
            specs = self.meta_learner_specs

        metric = metric or self.meta_select_metric

        meta_dir = self.output_dir / "meta_learner"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Branch A: Optuna HP search (use_optimize=True)
        # ------------------------------------------------------------------
        if use_optimize:
            if len(specs) != 1:
                raise ValueError(
                    "use_optimize=True requires exactly one meta-learner spec "
                    "(the model type whose hyperparameters will be searched). "
                    f"Got {len(specs)} specs.  Pass a single dict or remove "
                    "extra entries from meta_learners."
                )
            spec       = specs[0]
            model_name = spec["model_name"]
            base_hp    = dict(spec.get("hyperparams") or {})

            # Resolve search space: user override > DEFAULT_META_SEARCH_SPACES > {}
            if search_space is not None:
                active_space = search_space
            elif model_name in DEFAULT_META_SEARCH_SPACES:
                active_space = DEFAULT_META_SEARCH_SPACES[model_name]
            else:
                logger.warning(
                    "No default search space for meta-learner '%s'. "
                    "Pass search_space= explicitly or choose a supported model.",
                    model_name,
                )
                active_space = {}

            logger.info(
                "Stage 2: Optuna HP search for meta-learner '%s' "
                "(%d trials, metric=%s) ...",
                model_name, n_trials, metric,
            )

            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            study.optimize(
                lambda trial: _meta_optuna_objective(
                    trial,
                    model_name=model_name,
                    base_hp=base_hp,
                    search_space=active_space,
                    oof_meta_X=self.oof_meta_X,
                    oof_y=self.oof_y,
                    id2label=self.id2label,
                    metric=metric,
                    cv_folds=meta_cv_folds,
                    n_jobs=n_jobs,
                    random_state=random_state,
                ),
                n_trials=n_trials,
                catch=(Exception,),
            )

            n_completed = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
            if n_completed == 0:
                raise RuntimeError(
                    "All Optuna meta-learner trials failed. "
                    "Check that the model dependencies are installed and the "
                    "search space bounds are valid."
                )

            best_hp    = {**base_hp, **study.best_trial.params}
            best_name  = model_name
            best_score = study.best_value
            scores     = {model_name: round(best_score, 6)}

            logger.info(
                "Meta-learner Optuna complete: best %s = %.4f, params = %s",
                metric, best_score, best_hp,
            )

            # Save trial CSV
            try:
                import pandas as _pd
                trials_df = study.trials_dataframe()
                trials_df.to_csv(meta_dir / f"hpo_meta_{model_name}.csv", index=False)
            except Exception:
                pass

            # Build best spec with found HPs for final fit
            best_spec = {"model_name": model_name, "hyperparams": best_hp}

        # ------------------------------------------------------------------
        # Branch B: fixed-spec candidate selection (default)
        # ------------------------------------------------------------------
        else:
            logger.info("Stage 2: training %d meta-learner candidate(s) ...", len(specs))

            scores = {}
            for spec in specs:
                name  = spec["model_name"]
                model = _resolve_meta_learner(spec, n_jobs=n_jobs, random_state=random_state)

                if len(specs) > 1:
                    score = _score_meta_candidate(
                        model, self.oof_meta_X, self.oof_y,
                        self.id2label, metric, meta_cv_folds, random_state,
                    )
                    scores[name] = round(score, 6)
                    logger.info("  %s  %s = %.4f", name, metric, score)
                else:
                    scores[name] = None

            if len(specs) > 1:
                best_name = max(scores, key=lambda k: scores[k])
                logger.info(
                    "Best meta-learner: %s (%.4f)", best_name, scores[best_name],
                )
            else:
                best_name = specs[0]["model_name"]

            best_spec = next(s for s in specs if s["model_name"] == best_name)

        # ------------------------------------------------------------------
        # Fit winner on all OOF data and save
        # ------------------------------------------------------------------
        best_model = _resolve_meta_learner(best_spec, n_jobs=n_jobs, random_state=random_state)
        best_model.fit(self.oof_meta_X, self.oof_y)

        self.meta_learner = best_model
        self.meta_scores  = scores

        # Sanitize and strip all RNG objects before pickling to prevent
        # cross-NumPy-version joblib failures (MT19937 path changed in NumPy 2.x).
        # Three-layer defence: sanitize attrs → strip via __dict__ walk →
        # copyreg patch catches anything missed (C-extension slots, etc.).
        prepare_estimator_for_joblib(best_model)
        with null_rng_pickler():
            joblib.dump(best_model, meta_dir / "meta_learner.joblib")
        with open(meta_dir / "meta_scores.json", "w") as fh:
            json.dump(scores, fh, indent=2)

        meta_metadata = {
            "best_meta_name":     best_name,
            "meta_select_metric": metric,
            "meta_cv_folds":      meta_cv_folds,
            "use_optimize":       use_optimize,
            "n_trials":           n_trials if use_optimize else None,
            "meta_scores":        scores,
            "best_spec":          best_spec,
            "oof_meta_X_shape":   list(self.oof_meta_X.shape),
        }
        # Fingerprint the OOF matrix this meta-learner was fitted on, so a later
        # predict_test() can tell whether the base models have been re-run since.
        record_inputs(meta_metadata, self._oof_input_paths())
        with open(meta_dir / "meta_learner_metadata.json", "w") as fh:
            json.dump(meta_metadata, fh, indent=2, default=str)

        logger.info("Stage 2 complete. Meta-learner saved to %s", meta_dir)
        return {
            "meta_learner":    best_model,
            "meta_scores":     scores,
            "best_meta_name":  best_name,
            "output_dir":      self.output_dir,
        }

    def train_class_voter_stage(
        self,
        metric: str = "f1",
        shrinkage: float = 0.1,
        min_support_for_trust: float = 20.0,
        fallback_to_soft: bool = True,
        fallback_metric: str = "f1_macro",
        fallback_cv_folds: int = 3,
        fallback_random_state: int = 42,
        fallback_tolerance: float = 0.0,
    ) -> dict:
        """Stage 2 alternative: learn class-aware voting weights from OOF data.

        Uses only the OOF meta-feature matrix and OOF labels. This is a linear,
        interpretable combiner that learns a different model-weight profile for
        each class, without fitting a meta-learner over the concatenated OOF
        features.

        The learned weights are applied via predicted-class soft-gating
        (implemented in :func:`_apply_class_voter`): for each sample, each
        model gets the weight associated with the class that model predicts as
        most likely, then gates are normalized across models.

        Args:
            metric: Per-class OOF scoring metric used to rank models within
                    each class.  ``"brier"`` (default) computes
                    1 − mean((p − y_bin)²) from soft probabilities — reliable
                    on small class samples and degrades gracefully when a class
                    is rare or absent in OOF data.  ``"recall"``,
                    ``"precision"``, ``"f1"`` use hard argmax decisions and are
                    provided for diagnostic comparison.
            shrinkage:
                    Base blend factor toward uniform per-class weights for
                    well-supported classes.  Classes with fewer OOF examples
                    than ``min_support_for_trust`` receive additional shrinkage
                    on top of this base (support-adaptive).
                    ``0.0`` = trust learned OOF weights for all classes.
                    ``1.0`` = uniform voting for all classes.  Default ``0.5``.
            min_support_for_trust:
                    OOF training-set count below which a class is considered
                    insufficiently supported and its weights are pulled
                    more strongly toward uniform.  Default ``20``.
            fallback_to_soft:
                    If True (default), run a stratified CV safeguard on OOF
                    data comparing learned class-voter vs uniform soft voting.
                    If class-voter underperforms by more than
                    ``fallback_tolerance``, final saved weights are replaced by
                    exact uniform weights (safe fallback).
            fallback_metric:
                    Metric for the fallback comparison CV.  Uses
                    ``score_predictions`` supported metrics. Default
                    ``"f1_macro"``.
            fallback_cv_folds:
                    Requested stratified folds for fallback CV. Default ``3``.
                    If class support is too low, folds are reduced
                    automatically; if fewer than 2 folds are possible, fallback
                    CV is skipped.
            fallback_random_state:
                    Random seed for fallback CV split. Default ``42``.
            fallback_tolerance:
                    Non-negative margin (in score units) that class-voter must
                    beat soft-vote by on fallback CV in order to be kept.
                    Final fallback decision keeps class-voter only when:
                    ``class_cv > soft_cv + fallback_tolerance``.
                    Default ``0.0`` (ties prefer uniform soft vote).
        """
        self._ensure_oof_loaded()
        if fallback_tolerance < 0:
            raise ValueError("fallback_tolerance must be >= 0.")

        oof_meta_path = self.output_dir / "oof" / "oof_metadata.json"
        if not oof_meta_path.exists():
            raise RuntimeError(
                "OOF metadata not found at %s. Run train_base_models() first." % oof_meta_path
            )
        with open(oof_meta_path) as fh:
            oof_meta = json.load(fh)
        model_sources = oof_meta.get("model_sources") or []
        if not model_sources:
            raise RuntimeError("OOF metadata is missing model_sources; cannot train class voter.")

        sorted_ids = sorted(self.id2label.keys())
        prob_matrices = [
            self.oof_meta_X[:, int(src["col_start"]): int(src["col_start"]) + int(src["n_cols"])]
            for src in model_sources
        ]
        class_weights = _learn_class_voter_weights(
            prob_matrices=prob_matrices,
            y_true=self.oof_y,
            id2label=self.id2label,
            metric=metric,
            shrinkage=shrinkage,
            min_support_for_trust=min_support_for_trust,
        )
        combined_oof = _apply_class_voter(prob_matrices, class_weights)
        soft_oof = _uniform_soft_vote(prob_matrices)

        from ..utils.metrics import score_predictions, CV_METRICS

        metric_names = list(CV_METRICS)
        true_labels = [self.id2label[int(y)] for y in self.oof_y]
        learned_result = _assemble_prediction_result(combined_oof, true_labels, self.id2label, top_k=3)
        soft_result = _assemble_prediction_result(soft_oof, true_labels, self.id2label, top_k=3)
        learned_scores = {
            name: float(score_predictions(learned_result.top1, metric=name))
            for name in metric_names
        }
        soft_scores = {
            name: float(score_predictions(soft_result.top1, metric=name))
            for name in metric_names
        }

        fallback_report = None
        use_uniform_fallback = False
        if fallback_to_soft:
            fallback_report = _compare_class_voter_vs_soft_cv(
                prob_matrices=prob_matrices,
                y_true=self.oof_y,
                id2label=self.id2label,
                voter_metric=metric,
                shrinkage=shrinkage,
                min_support_for_trust=min_support_for_trust,
                selection_metric=fallback_metric,
                cv_folds=fallback_cv_folds,
                random_state=fallback_random_state,
            )
            if fallback_report.get("enabled"):
                voter_cv = float(fallback_report["class_voter_mean"])
                soft_cv = float(fallback_report["soft_vote_mean"])
                # Conservative safeguard: keep class-voter only when it
                # demonstrably beats soft-vote on CV by the configured margin.
                use_uniform_fallback = voter_cv <= (soft_cv + fallback_tolerance)
            else:
                logger.warning(
                    "Class-voter fallback CV skipped: %s",
                    fallback_report.get("reason", "unknown reason"),
                )

        if use_uniform_fallback:
            logger.info(
                "Class-voter fallback activated: learned weights underperform "
                "soft vote on CV (%s %.4f vs %.4f). Saving uniform weights.",
                fallback_metric,
                float(fallback_report["class_voter_mean"]),
                float(fallback_report["soft_vote_mean"]),
            )
            n_models = len(model_sources)
            n_classes = len(sorted_ids)
            class_weights = np.full((n_models, n_classes), 1.0 / n_models)

        final_oof = _apply_class_voter(prob_matrices, class_weights)
        final_result = _assemble_prediction_result(final_oof, true_labels, self.id2label, top_k=3)
        scores = {
            name: float(score_predictions(final_result.top1, metric=name))
            for name in metric_names
        }

        class_voter_dir = self.output_dir / "class_voter"
        class_voter_dir.mkdir(parents=True, exist_ok=True)

        np.save(class_voter_dir / "class_weights.npy", class_weights)
        weights_payload = {
            "model_names": [src["spec"]["model_name"] for src in model_sources],
            "class_ids": sorted_ids,
            "class_labels": [self.id2label[cid] for cid in sorted_ids],
            "class_weights": class_weights.tolist(),
        }
        with open(class_voter_dir / "class_weights.json", "w") as fh:
            json.dump(weights_payload, fh, indent=2)

        metadata = {
            "metric": metric,
            "shrinkage": shrinkage,
            "min_support_for_trust": min_support_for_trust,
            "fallback_to_soft": fallback_to_soft,
            "fallback_metric": fallback_metric,
            "fallback_cv_folds": fallback_cv_folds,
            "fallback_random_state": fallback_random_state,
            "fallback_tolerance": fallback_tolerance,
            "used_uniform_fallback": use_uniform_fallback,
            "oof_meta_X_shape": list(self.oof_meta_X.shape),
            "n_models": len(model_sources),
            "n_classes": len(sorted_ids),
            "model_names": [src["spec"]["model_name"] for src in model_sources],
            "learned_scores": learned_scores,
            "soft_vote_scores": soft_scores,
            "fallback_cv_report": fallback_report,
            "scores": scores,
        }
        # Same fingerprint as the meta-learner: these weights are learned from
        # the OOF matrix and go stale when it changes.
        record_inputs(metadata, self._oof_input_paths())
        with open(class_voter_dir / "class_voter_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

        self.class_voter_weights = class_weights
        self.class_voter_metadata = metadata

        logger.info("Stage 2 complete. Class-aware voter saved to %s", class_voter_dir)
        return {
            "class_weights": class_weights,
            "scores": scores,
            "learned_scores": learned_scores,
            "soft_vote_scores": soft_scores,
            "used_uniform_fallback": use_uniform_fallback,
            "fallback_cv_report": fallback_report,
            "output_dir": self.output_dir,
        }

    # ------------------------------------------------------------------
    # Stage 3: test prediction via meta-learner
    # ------------------------------------------------------------------

    def predict_test(
        self,
        top_k: int = 3,
        batch_size: int = 32,
        on_stale: str = "auto",
    ) -> PredictionResult:
        """Stage 3: predict on the held-out test set using the meta-learner.

        Loads final base models and meta-learner from disk if not already in
        memory.  Stacks test probability matrices into a meta-feature vector
        for each test sample, then passes through the meta-learner.

        Can be called after ``train_meta_learner_stage()`` in the same session,
        or in a fresh session (all artifacts reloaded from disk).

        Args:
            top_k:      Number of top classes in the topk output. Default 3.
            batch_size: Text inference batch size. Default 32.

        Returns:
            :class:`~multimodalva.utils.types.PredictionResult`.
        """
        # Lazy imports
        from ..text.dataset    import prepare_dataset as text_prepare
        from ..text.predict    import predict          as text_predict
        from ..tabular.dataset import prepare_dataset as tab_prepare
        from ..tabular.predict import predict          as tab_predict

        # Ensure meta-learner is loaded
        if self.meta_learner is None:
            meta_model_path = self.output_dir / "meta_learner" / "meta_learner.joblib"
            if not meta_model_path.exists():
                raise RuntimeError(
                    "Meta-learner not found at %s.  "
                    "Run train_meta_learner_stage() first." % meta_model_path
                )
            self.meta_learner = load_joblib_compat(meta_model_path)
            logger.info("Meta-learner loaded from %s", meta_model_path)

            # The meta-learner was fitted on the OOF matrix. If the base models
            # have been re-run since, that matrix has changed and this
            # meta-learner no longer matches it.
            meta_meta_path = meta_model_path.parent / "meta_learner_metadata.json"
            if meta_meta_path.exists():
                with open(meta_meta_path) as fh:
                    saved_meta = json.load(fh)
                guard_inputs(
                    check_inputs(saved_meta, self._oof_input_paths()),
                    "The saved meta-learner",
                    "re-run train_meta_learner_stage()",
                    on_stale=on_stale,
                )

        # Ensure label maps are loaded
        self._ensure_oof_loaded()

        # Load test data
        data_dir = self.output_dir / "data"
        train_df = pd.read_csv(data_dir / "train_df.csv")
        test_df  = pd.read_csv(data_dir / "test_df.csv")

        sorted_ids = sorted(self.id2label.keys())
        n_classes  = len(sorted_ids)
        n_test     = len(test_df)

        # Load model_sources — flat ordered list of all base models
        # (inherited + new) with absolute final_dir paths and column ranges.
        # Falls back to constructing from self.text_models / self.tabular_models
        # for runs created before model_sources was introduced.
        _oof_meta_path = self.output_dir / "oof" / "oof_metadata.json"
        model_sources: list[dict] | None = None
        _inherited_from: str | None = None
        if _oof_meta_path.exists():
            with open(_oof_meta_path) as _f:
                _oof_meta = json.load(_f)
            model_sources = _oof_meta.get("model_sources")
            _inherited_from = _oof_meta.get("inherited_from")

        if not model_sources:
            # Backward-compat fallback: only current-run models, no inheritance
            final_dir_fb = self.output_dir / "final"
            model_sources = []
            for i, spec in enumerate(self.text_models):
                model_sources.append({
                    "type": "text", "local_index": i, "spec": spec,
                    "final_dir": str((final_dir_fb / f"text_{i}").resolve()),
                    "col_start": i * n_classes, "n_cols": n_classes,
                })
            for i, spec in enumerate(self.tabular_models):
                model_sources.append({
                    "type": "tabular", "local_index": i, "spec": spec,
                    "final_dir": str((final_dir_fb / f"tabular_{i}").resolve()),
                    "col_start": (len(self.text_models) + i) * n_classes,
                    "n_cols": n_classes,
                })

        n_total_cols = sum(s["n_cols"] for s in model_sources)
        meta_X_test  = np.zeros((n_test, n_total_cols), dtype=float)

        # Pre-load tabular test arrays once (shared by all sklearn tabular models).
        # All tabular models in model_sources share the same feature space, so
        # one preprocessed X_test works for all of them.
        X_test_npy = data_dir / "X_test.npy"
        y_test_npy = data_dir / "y_test.npy"
        _has_sklearn_tab = any(
            s["type"] == "tabular" and s["spec"].get("model_name") != _INSILICOVA
            for s in model_sources
        )
        X_test = y_test = None
        if _has_sklearn_tab and X_test_npy.exists():
            X_test = np.load(X_test_npy)
            y_test = np.load(y_test_npy)
        elif _has_sklearn_tab:
            (_, X_test, _, y_test, _, _, _, _) = tab_prepare(
                train_df, test_df,
                feature_cols=self._feature_cols,
                label_col=self._label_col,
                encode_categoricals="ordinal",
            )

        # --- Iterate over all model sources (inherited + new) ---------------
        for source in model_sources:
            model_dir  = self._resolve_final_dir(source, _inherited_from)
            spec       = source["spec"]
            col_s      = source["col_start"]
            n_cols     = source["n_cols"]
            model_name = spec["model_name"]

            if source["type"] == "text":
                max_length = spec.get("max_length", 512)
                logger.info(
                    "Predicting test — text model %s (weights: %s) ...",
                    model_name, model_dir,
                )
                _, test_text_ds, _, _ = text_prepare(
                    train_df, test_df,
                    text_col=self._text_col, label_col=self._label_col,
                    model_name=model_name, max_length=max_length,
                )
                result     = text_predict(model_dir, test_text_ds, batch_size=batch_size, top_k=1)
                fold_probs = _extract_probs(result, sorted_ids)

            else:  # tabular
                logger.info(
                    "Predicting test — tabular model %s (weights: %s) ...",
                    model_name, model_dir,
                )
                if model_name == _INSILICOVA:
                    fold_probs = _insilicova_predict_df(
                        model_dir, test_df, sorted_ids, train_df=train_df,
                    )
                else:
                    result     = tab_predict(model_dir, X_test, y_test, top_k=1)
                    fold_probs = _extract_probs(result, sorted_ids)

            meta_X_test[:, col_s:col_s + n_cols] = fold_probs

        # --- Meta-learner final prediction ----------------------------------
        meta_proba = self.meta_learner.predict_proba(meta_X_test)  # (n_test, n_classes)

        # Reorder columns to canonical sorted_ids order
        classes   = list(self.meta_learner.classes_)
        col_order = [classes.index(cid) for cid in sorted_ids]
        meta_proba = meta_proba[:, col_order]

        true_labels = test_df[self._label_col].tolist()
        self.predictions = _assemble_prediction_result(
            meta_proba, true_labels, self.id2label, top_k
        )

        # Save predictions
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        self.predictions.top1.to_csv(pred_dir / "top1.csv", index=False)
        self.predictions.topk.to_csv(pred_dir / "topk.csv", index=False)
        self.predictions.full.to_csv(pred_dir / "full.csv", index=False)

        voted_acc = (
            self.predictions.top1["true_label"] == self.predictions.top1["predicted_label"]
        ).mean()
        logger.info("Stage 3 complete. Test accuracy: %.4f", voted_acc)

        return self.predictions

    def predict_test_class_voter(
        self,
        top_k: int = 3,
        batch_size: int = 32,
        on_stale: str = "auto",
    ) -> PredictionResult:
        """Stage 3 alternative: predict test data using saved class-aware weights.

        Args:
            top_k: Number of ranked classes to return.
            batch_size: Inference batch size for text base models.
        on_stale: What to do when the saved Stage 2 result was built from an
            out-of-fold matrix that has since changed (for example because a base
            model was re-run). ``"auto"``/``"error"`` stop with an explanation,
            ``"warn"`` continues with a warning, ``"ignore"`` continues silently.
            Results saved before input tracking existed are not checked.
        """
        if self.class_voter_weights is None:
            class_voter_dir = self.output_dir / "class_voter"
            weights_path = class_voter_dir / "class_weights.npy"
            meta_path = class_voter_dir / "class_voter_metadata.json"
            if not weights_path.exists():
                raise RuntimeError(
                    "Class-aware voter weights not found at %s. Run train_class_voter_stage() first."
                    % weights_path
                )
            self.class_voter_weights = np.load(weights_path)
            if meta_path.exists():
                with open(meta_path) as fh:
                    self.class_voter_metadata = json.load(fh)
                # These weights were learned from the OOF matrix; re-running a
                # base model changes it and leaves the weights stale.
                guard_inputs(
                    check_inputs(self.class_voter_metadata, self._oof_input_paths()),
                    "The saved class-aware voting weights",
                    "re-run train_class_voter_stage()",
                    on_stale=on_stale,
                )

        # Lazy imports
        from ..text.dataset    import prepare_dataset as text_prepare
        from ..text.predict    import predict          as text_predict
        from ..tabular.dataset import prepare_dataset as tab_prepare
        from ..tabular.predict import predict          as tab_predict

        self._ensure_oof_loaded()

        data_dir = self.output_dir / "data"
        train_df = pd.read_csv(data_dir / "train_df.csv")
        test_df = pd.read_csv(data_dir / "test_df.csv")

        sorted_ids = sorted(self.id2label.keys())
        n_classes = len(sorted_ids)

        oof_meta_path = self.output_dir / "oof" / "oof_metadata.json"
        with open(oof_meta_path) as fh:
            oof_meta = json.load(fh)
        model_sources = oof_meta.get("model_sources") or []
        if not model_sources:
            raise RuntimeError("OOF metadata is missing model_sources; cannot run class-aware voter.")

        if self.class_voter_weights.shape != (len(model_sources), n_classes):
            raise ValueError(
                "Loaded class-aware weights shape %s does not match current model_sources/classes (%d, %d)."
                % (self.class_voter_weights.shape, len(model_sources), n_classes)
            )

        X_test_npy = data_dir / "X_test.npy"
        y_test_npy = data_dir / "y_test.npy"
        _has_sklearn_tab = any(
            s["type"] == "tabular" and s["spec"].get("model_name") != _INSILICOVA
            for s in model_sources
        )
        X_test = y_test = None
        if _has_sklearn_tab and X_test_npy.exists():
            X_test = np.load(X_test_npy)
            y_test = np.load(y_test_npy)
        elif _has_sklearn_tab:
            (_, X_test, _, y_test, _, _, _, _) = tab_prepare(
                train_df, test_df,
                feature_cols=self._feature_cols,
                label_col=self._label_col,
                encode_categoricals="ordinal",
            )

        prob_matrices = []
        for source in model_sources:
            model_dir = self._resolve_final_dir(source, oof_meta.get("inherited_from"))
            spec = source["spec"]
            model_name = spec["model_name"]

            if source["type"] == "text":
                max_length = spec.get("max_length", 512)
                logger.info(
                    "Predicting test — text model %s (weights: %s) ...",
                    model_name, model_dir,
                )
                _, test_text_ds, _, _ = text_prepare(
                    train_df, test_df,
                    text_col=self._text_col, label_col=self._label_col,
                    model_name=model_name, max_length=max_length,
                )
                result = text_predict(model_dir, test_text_ds, batch_size=batch_size, top_k=1)
                fold_probs = _extract_probs(result, sorted_ids)
            else:
                logger.info(
                    "Predicting test — tabular model %s (weights: %s) ...",
                    model_name, model_dir,
                )
                if model_name == _INSILICOVA:
                    fold_probs = _insilicova_predict_df(
                        model_dir, test_df, sorted_ids, train_df=train_df,
                    )
                else:
                    result = tab_predict(model_dir, X_test, y_test, top_k=1)
                    fold_probs = _extract_probs(result, sorted_ids)
            prob_matrices.append(fold_probs)

        combined = _apply_class_voter(prob_matrices, self.class_voter_weights)
        true_labels = test_df[self._label_col].tolist()
        predictions = _assemble_prediction_result(combined, true_labels, self.id2label, top_k)

        pred_dir = self.output_dir / "class_voter" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        predictions.top1.to_csv(pred_dir / "top1.csv", index=False)
        predictions.topk.to_csv(pred_dir / "topk.csv", index=False)
        predictions.full.to_csv(pred_dir / "full.csv", index=False)

        voted_acc = (
            predictions.top1["true_label"] == predictions.top1["predicted_label"]
        ).mean()
        logger.info("Class-aware voter Stage 3 complete. Test accuracy: %.4f", voted_acc)
        return predictions

    # ------------------------------------------------------------------
    # Convenience: all stages in sequence
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
        split_col: str | None = None,
        # --- text ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 3,
        batch_size: int = 32,
        # --- tabular ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        # --- fold storage ---
        save_fold_models: bool = False,
        cleanup_fold_files: bool = True,
        # --- meta-learner ---
        meta_learners: list[dict] | dict | None = None,
        meta_select_metric: str | None = None,
        meta_cv_folds: int = 3,
        meta_use_optimize: bool = False,
        meta_n_trials: int = 30,
        meta_search_space: dict | None = None,
        # --- inference ---
        top_k: int = 3,
    ) -> dict:
        """Convenience wrapper: Stage 1 → Stage 2 → Stage 3 in sequence.

        Equivalent to calling::

            clf.train_base_models(df, ...)
            clf.train_meta_learner_stage(...)
            predictions = clf.predict_test(...)

        Returns:
            dict with ``predictions``, ``oof_meta_X``, ``meta_learner``,
            ``meta_scores``, ``label2id``, ``id2label``, ``output_dir``.
        """
        self.train_base_models(
            df=df, label_col=label_col,
            text_col=text_col, feature_cols=feature_cols,
            test_size=test_size, random_state=random_state, stratify=stratify,
            split_col=split_col,
            val_size=val_size,
            gradient_checkpointing=gradient_checkpointing,
            early_stopping_patience=early_stopping_patience,
            batch_size=batch_size,
            n_jobs=n_jobs, use_gpu=use_gpu,
            encode_categoricals=encode_categoricals,
            scale_numeric=scale_numeric,
            save_fold_models=save_fold_models,
            cleanup_fold_files=cleanup_fold_files,
        )
        self.train_meta_learner_stage(
            meta_learners=meta_learners,
            metric=meta_select_metric,
            meta_cv_folds=meta_cv_folds,
            random_state=random_state,
            n_jobs=n_jobs,
            use_optimize=meta_use_optimize,
            n_trials=meta_n_trials,
            search_space=meta_search_space,
        )
        predictions = self.predict_test(top_k=top_k, batch_size=batch_size)

        # Save top-level metadata
        metadata = {
            "output_dir":       str(self.output_dir),
            "n_text_models":    len(self.text_models),
            "n_tabular_models": len(self.tabular_models),
            "n_folds":          self.n_folds,
            "meta_scores":      self.meta_scores,
            "label2id":         self.label2id,
            "id2label":         {str(k): v for k, v in self.id2label.items()},
            "n_train":          len(self.train_df),
            "n_test":           len(self.test_df),
            "n_classes":        len(self.label2id),
        }
        with open(self.output_dir / "training_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        return {
            "predictions":  predictions,
            "oof_meta_X":   self.oof_meta_X,
            "meta_learner": self.meta_learner,
            "meta_scores":  self.meta_scores,
            "label2id":     self.label2id,
            "id2label":     self.id2label,
            "output_dir":   self.output_dir,
        }
