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

    Other combiners (same OOF input, same PredictionResult output):
        train_class_voter_stage()        / predict_test_class_voter()
            per-model × per-class weights (class-aware voting).
        train_ensemble_selection_stage() / predict_test_ensemble_selection()
            one weight per model by greedy ensemble selection
            (Caruana et al. 2004; see ensemble_selection.py).
        predict_test_simple_average()
            equal-weight average; nothing to train.
    Every Stage 3 gets the base models' test probabilities from one shared
    helper, computed once per run and reused.

    compare_combiners_stage():
        Nested CV over the OOF rows: simple average, class-aware voting,
        ensemble selection and each meta-learner scored on identical folds,
        using training data only. combiner="best" uses it to choose.

    Convenience: run() calls the stages in sequence. run(combiner=...) picks
    one combiner, a list of them (sharing one stage 1), or "best"; and
    run(oof_from=...) reuses a finished run's stage 1 without retraining.

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
    │   ├── text_0/               — Optuna HPO artifacts (if the spec searched)
    │   └── tabular_0/
    ├── final/
    │   ├── text_0/               — base model retrained on full training set
    │   ├── text_1/
    │   └── tabular_0/
    ├── meta_learner/
    │   ├── meta_learner.joblib   — fitted best meta-learner
    │   ├── meta_scores.json      — CV scores for all candidate meta-learners
    │   └── meta_learner_metadata.json
    ├── predictions/              — the main result (the first combiner, or the one "best" chose)
    ├── meta_learner/predictions/ — also written by predict_test()
    ├── meta_learner_<name>/      — one meta-learner named in combiner=; same layout as meta_learner/
    ├── simple_average/predictions/
    ├── class_voter/              — class_weights.{npy,json}, metadata, predictions/
    ├── ensemble_selection/
    │   ├── ensemble_weights.npy  — one weight per base model (sums to 1)
    │   ├── ensemble_weights.json — weights by "<position>:<model_name>"
    │   ├── trajectory.csv        — OOF score after each selection step, per bag
    │   ├── ensemble_selection_metadata.json
    │   └── predictions/
    ├── combiner_comparison/
    │   ├── combiner_comparison.csv  — one row per combiner: cv_mean, cv_std, fold scores
    │   └── combiner_comparison.json — settings, per-fold results, best_combiner
    ├── label2id.json
    ├── id2label.json
    └── training_metadata.json

HPO per base model:
    Set ``"hyperparams": Optimize(...)`` on any model spec.  The search runs ONCE on the full
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

InSilicoVA base model — NOT SUPPORTED YET (v1):
    The ``model_name="insilicova"`` hook below is unfinished scaffolding. Do not
    describe it, in documentation or in a paper, as a working feature. Two
    things block it:

    * ``pyinsilicova`` does not install on Python >= 3.12, this package's floor;
    * more fundamentally, it cannot train a probbase against a user's own cause
      list. It works with InSilicoVA's native causes only, which is not the
      setting this package is built for (a study's own cause grouping).

    What to do today: run InSilicoVA where it works — ``pyinsilicova`` if your
    causes are its native ones, otherwise the R implementation — and bring its
    assignments into the comparison as predictions. ``results`` takes predicted
    labels (and probabilities, if you have them) from any source, so an
    externally produced model lands on the same leaderboard, with the same
    metrics and bootstrap intervals, as one trained here.

    The rest of this section describes the hook as designed, for whoever
    finishes it:

    * No training phase — InSilicoVA applies a fixed Bayesian symptom-cause
      probability database (WHO 2016 or PHMRC), so a search is ignored.
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

    Would require:  ``pip install pyinsilicova`` — see the two blockers above.

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
from ..utils.predictions import (
    assemble_predictions, resolve_test_ids, save_predictions,
)
from ..utils.runtime import track_run
from ..utils.seeds import seed_everything, set_determinism
from ..utils.numpy_compat import (
    load_joblib_compat,
    null_rng_pickler,
    prepare_estimator_for_joblib,
)

from ..utils.hpo_defaults import TABULAR_SPEC_DEFAULTS, TEXT_SPEC_DEFAULTS
from ..utils.optimize_config import (
    resolve_search_resume,
    _log_hp_source,
    resolve_spec_hyperparams,
)
from ..utils.provenance import check_inputs, guard_inputs, record_inputs

logger = logging.getLogger(__name__)


BaseModelSpec = dict

#: Named combiners StackingClassifier.run(combiner=...) accepts. Several can be
#: given; all share one stage 1. combiner= also accepts a meta-learner model
#: name (``"logistic_regression"``, ``"lightgbm"``, …) to use that one
#: meta-learner alone, and ``"best"`` to let compare_combiners_stage() choose.
STACKING_COMBINERS: tuple[str, ...] = (
    "meta_learner", "simple_average", "class_aware_voting", "ensemble_selection",
)


#: The alias for the one meta-learner that is not a tabular model.
_LR_ALIAS = "logistic_regression"

#: The candidate list a stacking run uses when the caller names none: one
#: multinomial logistic regression.
#:
#: A list of one, not a bare dict, because ``meta_learners=`` is a list of
#: candidates and the default is simply a short one — adding a candidate is then
#: an edit to this list rather than a change of type.
#:
#: LR is the default on purpose. The meta-learner is fitted on probabilities that
#: already came from strong models, so the second stage is a re-weighting job, and
#: a linear model on ~(n_models x n_causes) features is the conservative choice at
#: the row counts verbal-autopsy studies have. Whether to compare several
#: candidates depends on the data, so it is the caller's decision: a run that asks
#: for nothing should not silently do something wider than it was asked for.
#:
#: The ten names are otherwise on the same footing — this one is the default, not
#: a recommendation over the rest.
#:
#: Any of the ten names in :func:`_meta_learner_names` may be passed. For a model
#: that is not one of the ten, run stage 1 alone (``oof_only=True``), take
#: ``oof_meta_X`` / ``oof_y`` from the result, and fit it yourself — the error for
#: an unknown name says so.
DEFAULT_META_LEARNERS: list[dict] = [
    {
        "model_name":  _LR_ALIAS,
        "hyperparams": {"max_iter": 1000, "C": 1.0},
    },
]


# Alias used to identify InSilicoVA specs throughout this module
_INSILICOVA = "insilicova"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SubsetDataset:
    """Lightweight wrapper exposing a contiguous-index subset of a Dataset.

    Compatible with any ``torch.utils.data.Dataset`` via ``__len__`` /
    ``__getitem__``.  Used to feed fold-specific training / validation slices
    to ``train_text()`` and ``predict_text()`` without re-tokenising.
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
    All tabular model aliases from ``tabular.train.TABULAR_MODELS`` are also
    accepted (lightgbm, catboost, xgboost, random_forest, mlp, …), and are built
    by :func:`~multimodalva.tabular.train._build_model`, so a model behaves the
    same as a meta-learner as it does as a base model.

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

    if model_name == _LR_ALIAS:
        from sklearn.linear_model import LogisticRegression
        hp.setdefault("max_iter",      1000)
        hp.setdefault("C",             1.0)
        hp.setdefault("solver",        "lbfgs")   # lbfgs supports multinomial multiclass natively
        hp.setdefault("n_jobs",        n_jobs)
        hp.setdefault("random_state",  random_state)
        return LogisticRegression(**hp)

    # Every other name is a tabular model alias, so build it with the tabular
    # pipeline's own constructor rather than a second copy of the same logic.
    #
    # This used to re-implement the registry lookup, the n_jobs / forced / CUDA
    # parameter handling and the seed injection, and the seed half was wrong:
    # it set random_state on everything except catboost, so `naive_bayes` and
    # `knn` — both advertised by _meta_learner_names(), neither accepting a
    # random_state — raised TypeError the moment anyone passed them, while
    # catboost (which does accept random_state) was left unseeded. Nobody hit
    # either, because the two were never passed until someone tried them.
    # _build_model checks the constructor signature, treats a **kwargs
    # constructor as accepting the seed (xgboost's does), and leaves the two
    # deterministic models alone.
    from ..tabular.train import TABULAR_MODELS, _build_model

    if model_name not in TABULAR_MODELS:
        # Normally unreachable: _check_meta_learner_specs rejects this before
        # stage 1. Kept because this function is also called directly.
        raise ValueError(
            f"Unknown meta-learner {model_name!r}. Accepted: "
            f"{', '.join(_meta_learner_names())}.\n" + _UNKNOWN_META_LEARNER_HINT
        )

    # Silence per-fold training output: the meta-learner is fitted once per
    # candidate per CV fold, which is otherwise thousands of lines. setdefault,
    # so a caller asking for verbose output still gets it. Meta-learner specific,
    # which is why it stays here rather than moving into _build_model.
    if model_name == "catboost":
        hp.setdefault("verbose", 0)
    elif model_name == "lightgbm":
        hp.setdefault("verbose", -1)
    elif model_name == "xgboost":
        hp.setdefault("verbosity", 0)

    return _build_model(
        model_name, hp, random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu
    )


def _meta_learner_names() -> tuple[str, ...]:
    """Every name :func:`_resolve_meta_learner` accepts.

    Imported lazily, like ``_resolve_meta_learner`` itself: the tabular
    registry pulls in the gradient-boosting libraries.
    """
    from ..tabular.train import TABULAR_MODELS
    return (_LR_ALIAS, *TABULAR_MODELS)


#: What to do about a meta-learner this package cannot build. Kept as one string
#: because three places raise it and they must say the same thing.
_UNKNOWN_META_LEARNER_HINT = (
    "Stacking can only fit a meta-learner it knows how to build. For any other "
    "model, compute stage 1 and fit the second stage yourself:\n"
    "    out = run(task='stacking', ..., oof_only=True)\n"
    "    X, y = out['oof_meta_X'], out['oof_y']\n"
    "    my_model.fit(X, y)\n"
    "oof_meta_X is the out-of-fold probability matrix "
    "(n_train x n_models*n_causes); oof/oof_metadata.json names every column in "
    "meta_feature_names. Nothing about the model has to come from this package."
)


def _check_meta_learner_specs(specs: list[dict]) -> None:
    """Reject meta-learner candidates this package cannot build, and duplicates.

    Called **before stage 1**, on purpose. Every candidate is instantiated in
    stage 2, so an unbuildable name would otherwise surface only after every base
    model had been trained over every fold — hours of GPU time on a submitted job
    that nobody is watching, ending in a crash that was knowable at the start.
    """
    names = [s["model_name"] for s in specs]

    accepted = set(_meta_learner_names())
    unknown = [n for n in names if n not in accepted]
    if unknown:
        raise ValueError(
            f"meta_learners names {', '.join(repr(n) for n in unknown)}, which "
            f"stacking cannot build. Accepted: {', '.join(_meta_learner_names())}.\n"
            + _UNKNOWN_META_LEARNER_HINT
        )

    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(
            f"meta_learners lists {', '.join(repeated)} more than once. Results "
            "are reported by model_name, so give each candidate a different model."
        )


def _score_meta_candidate(
    meta_model,
    meta_X: np.ndarray,
    y_oof: np.ndarray,
    id2label: dict,
    metric: str,
    cv_folds: int,
    split_seed: int,
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
        split_seed:   Seed for the CV folds. The model's own seed is already set
                      on ``meta_model``; this only decides which rows form each
                      fold, so every candidate is scored on the same folds.

    Returns:
        Mean CV score (float).
    """
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from ..utils.metrics import score_predictions

    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=split_seed)
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
    result = assemble_predictions(
        probs, id2label,
        true_labels=true_labels, top_k=1,
    )
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
    split_seed: int,
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

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
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


def _fit_class_voter(
    prob_matrices: list[np.ndarray],
    y_true: np.ndarray,
    id2label: dict,
    metric: str,
    shrinkage: float,
    min_support_for_trust: float,
    fallback_to_soft: bool,
    fallback_metric: str,
    fallback_cv_folds: int,
    split_seed: int,
    fallback_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, bool, dict | None]:
    """The class-aware voter as ``train_class_voter_stage()`` fits it.

    Learns the model × class weights, then — when ``fallback_to_soft`` — runs
    the CV check against uniform soft voting and replaces the weights with
    uniform ones unless the learned voter wins by more than
    ``fallback_tolerance``. The one implementation behind both the stage and
    the combiner comparison, so the comparison scores the voter as it is
    actually trained.

    Returns:
        ``(learned_weights, final_weights, used_uniform_fallback,
        fallback_report)``. ``fallback_report`` is ``None`` when
        ``fallback_to_soft`` is False.
    """
    learned = _learn_class_voter_weights(
        prob_matrices=prob_matrices,
        y_true=y_true,
        id2label=id2label,
        metric=metric,
        shrinkage=shrinkage,
        min_support_for_trust=min_support_for_trust,
    )
    report = None
    use_uniform = False
    if fallback_to_soft:
        report = _compare_class_voter_vs_soft_cv(
            prob_matrices=prob_matrices,
            y_true=y_true,
            id2label=id2label,
            voter_metric=metric,
            shrinkage=shrinkage,
            min_support_for_trust=min_support_for_trust,
            selection_metric=fallback_metric,
            cv_folds=fallback_cv_folds,
            split_seed=split_seed,
        )
        if report.get("enabled"):
            # Conservative safeguard: keep class-voter only when it
            # demonstrably beats soft-vote on CV by the configured margin.
            use_uniform = (float(report["class_voter_mean"])
                           <= float(report["soft_vote_mean"]) + fallback_tolerance)
    if use_uniform:
        n_models, n_classes = learned.shape
        return learned, np.full((n_models, n_classes), 1.0 / n_models), True, report
    return learned, learned, False, report


def _model_source_keys(model_sources: list[dict]) -> list[str]:
    """Unique report key per base model: ``"<position>:<model_name>"``.

    ``model_name`` alone is not unique — two runs of the same checkpoint, or an
    inherited ``text_0`` next to a new ``text_0``, share it.
    """
    return [f"{pos}:{src['spec']['model_name']}" for pos, src in enumerate(model_sources)]


def _stack_oof_probs(oof_meta_X: np.ndarray, model_sources: list[dict]) -> np.ndarray:
    """Slice the OOF meta-feature matrix into (n_models, n_samples, n_classes)."""
    return np.stack([
        oof_meta_X[:, int(src["col_start"]): int(src["col_start"]) + int(src["n_cols"])]
        for src in model_sources
    ], axis=0)


# Adapters giving the Stage 2 combiners the fit(preds, y) / predict(preds)
# interface that cross_validate_combiner() expects.  preds is
# (n_models, n_samples, n_classes); y holds class positions 0..n_classes-1.
# Each adapter calls the implementation its training stage uses, so the
# comparison scores a combiner exactly as it would be trained.

class _SimpleAverageCombiner:
    def fit(self, preds, y):
        return self

    def predict(self, preds):
        return _uniform_soft_vote(list(preds))


class _ClassVoterCombiner:
    """The class-aware voter as train_class_voter_stage() fits it, fallback included.

    ``fit_kwargs`` are ``_fit_class_voter()``'s settings; the caller takes them
    from ``train_class_voter_stage()``'s own defaults, so there is one set.
    """

    def __init__(self, **fit_kwargs):
        self.kwargs = fit_kwargs

    def fit(self, preds, y):
        # y already holds positions, so an identity id2label keeps
        # _learn_class_voter_weights' position ↔ id mapping trivial.
        id2label = {i: str(i) for i in range(preds.shape[2])}
        _, self.weights, self.used_uniform_fallback, _ = _fit_class_voter(
            list(preds), y, id2label, **self.kwargs,
        )
        return self

    def predict(self, preds):
        return _apply_class_voter(list(preds), self.weights)


class _MetaLearnerCombiner:
    """One meta-learner spec fitted on the concatenated OOF probabilities."""

    def __init__(self, spec: dict, n_jobs: int, random_state: int):
        self.spec, self.n_jobs, self.random_state = spec, n_jobs, random_state

    @staticmethod
    def _features(preds):
        # (n_models, n, c) → (n, n_models*c), the column order of oof_meta_X.
        return np.concatenate(list(preds), axis=1)

    def fit(self, preds, y):
        self.n_classes = preds.shape[2]
        self.model = _resolve_meta_learner(
            self.spec, n_jobs=self.n_jobs, random_state=self.random_state,
        )
        self.model.fit(self._features(preds), y)
        return self

    def predict(self, preds):
        proba = self.model.predict_proba(self._features(preds))
        out = np.zeros((proba.shape[0], self.n_classes))
        out[:, np.asarray(self.model.classes_, dtype=int)] = proba
        return out


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
    split_seed: int | None = None,
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
        random_state:         Seed for each fold model's training. Default 42.
        split_seed:           Seed for which rows land in which OOF fold.
                              ``None`` (the default) reuses ``random_state``,
                              so existing callers are unaffected.
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
    from ..text.train   import train_text
    from ..text.predict import predict_text
    from ..tabular.train   import train_tabular
    from ..tabular.predict import predict_tabular

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

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True,
        random_state=split_seed if split_seed is not None else random_state,
    )
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
                # max_length is applied where the text is tokenised, not here.
                use_lora   = spec.get("use_lora", False)
                gc         = spec.get("gradient_checkpointing", gradient_checkpointing)
                hp         = (text_best_hp[i] if text_best_hp else None) or spec.get("hyperparams") or {}

                logger.info("  [text_%d fold_%d] Training %s ...", i, fold_idx, model_name)
                model_text_dataset = train_text_datasets[i]
                fold_train_ds = _SubsetDataset(model_text_dataset, train_idx)
                fold_val_ds   = _SubsetDataset(model_text_dataset, val_idx)

                train_text(
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
                    random_state=random_state,    # fold split varies; init must not
                )

                result     = predict_text(
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
                    train_tabular(
                        X_train=X_train[train_idx],
                        y_train=y_train[train_idx],
                        label2id=label2id, id2label=id2label,
                        model_name=model_name,
                        output_dir=model_dir / "weights",
                        hyperparams=hp,
                        random_state=random_state,
                        n_jobs=n_jobs, use_gpu=use_gpu,
                    )
                    result = predict_tabular(
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

    Other combiners — ``train_class_voter_stage()`` /
    ``predict_test_class_voter()`` (class-aware voting),
    ``train_ensemble_selection_stage()`` / ``predict_test_ensemble_selection()``
    (greedy ensemble selection) and ``predict_test_simple_average()``
    (equal weights). ``compare_combiners_stage()`` scores all of them by nested
    CV on the OOF rows.

    ``run(df, …)`` calls the stages in sequence. ``combiner`` picks one
    combiner, a list of them (sharing one stage 1) or ``"best"``;
    ``oof_from`` reuses a finished run's stage 1.

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
    best_meta_name :      The candidate chosen.

    Attributes (populated after the alternative Stage 2 methods)
    ------------------------------------------------------------
    class_voter_weights, class_voter_metadata :
                          After train_class_voter_stage().
    ensemble_selection_weights, ensemble_selection_metadata :
                          After train_ensemble_selection_stage().
    combiner_comparison, best_combiner :
                          Table and winner from compare_combiners_stage().

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
                                  candidates — each is scored by CV on the OOF
                                  matrix and the best is refit on all of it.
                                  Default: :data:`DEFAULT_META_LEARNERS`, one
                                  multinomial logistic regression. Which models
                                  are worth comparing depends on the data, so
                                  comparing several is done by passing them. Any
                                  of the ten names :func:`_meta_learner_names`
                                  returns may be used, all on the same footing;
                                  one this package cannot build is refused before
                                  stage 1 with a pointer to ``oof_only=True``,
                                  which hands back ``oof_meta_X`` / ``oof_y`` to
                                  fit whatever you like.
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
        # Specs may be empty only when the base models come from an existing
        # run via run(oof_from=...); train_base_models() enforces it otherwise.
        text_models = list(text_models or [])
        tabular_models = list(tabular_models or [])

        # Normalise meta_learners to a list
        if meta_learners is None:
            # Deep enough to cover the nested hyperparams dict. dict(spec) alone
            # is shallow, so a caller editing clf.meta_learner_specs[i]
            # ["hyperparams"] would edit DEFAULT_META_LEARNERS itself and change
            # the default for every later run in the process — the same trap
            # Optimize.__post_init__ guards against for its own dict fields.
            meta_learners = [
                {**spec, "hyperparams": dict(spec.get("hyperparams") or {})}
                for spec in DEFAULT_META_LEARNERS
            ]
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
        self._id_col:        str | None           = None
        # The seeds stage 1 ran with, so stages 2 and 3 can inherit them even
        # when they run in a fresh session.
        # Where stage-1 artifacts (oof/, final/, data/, label maps) are read
        # from. None means this run's own output_dir; run(oof_from=...) points
        # it at an earlier stacking run so its out-of-fold predictions and base
        # models are reused instead of recomputed.
        self._oof_from:      Path | None          = None
        # Base models' test probabilities, shared by every Stage 3 (see
        # _predict_test_base_probs). Cleared whenever stage 1 is re-run.
        self._test_probs_cache = None
        self._split_seed:    int | None           = None
        self._train_seed:    int | None           = None
        self._feature_cols:  list[str] | None     = None
        self._label_col:     str | None           = None

        # Populated after train_meta_learner_stage()
        self.meta_learner: Any | None = None
        self.meta_scores:  dict       = {}
        self.class_voter_weights: np.ndarray | None = None
        self.ensemble_selection_weights: np.ndarray | None = None
        self.ensemble_selection_metadata: dict | None = None
        self.combiner_comparison: pd.DataFrame | None = None
        self.best_combiner: str | None = None
        self.class_voter_metadata: dict | None = None

        # Populated after predict_test()
        self.predictions: PredictionResult | None = None

    # ------------------------------------------------------------------
    # Internal: load OOF state from disk (for cross-session Stage 2/3)
    # ------------------------------------------------------------------

    def _oof_input_paths(self) -> dict:
        """The OOF files Stage 2 consumes — fingerprinted to detect staleness."""
        oof_dir = self._stage1_dir / "oof"
        return {
            "oof_meta_X": oof_dir / "oof_meta_X.npy",
            "oof_y": oof_dir / "oof_y.npy",
        }

    def _ensure_oof_loaded(self):
        """Load OOF artifacts from disk if not already in memory."""
        if self.oof_meta_X is not None:
            return

        oof_dir = self._stage1_dir / "oof"
        meta_X_file = oof_dir / "oof_meta_X.npy"
        if not meta_X_file.exists():
            raise RuntimeError(
                "OOF meta-features not found at %s.  "
                "Run train_base_models() first." % meta_X_file
            )

        self.oof_meta_X = np.load(meta_X_file)
        self.oof_y      = np.load(oof_dir / "oof_y.npy")

        lbl_path   = self._stage1_dir / "label2id.json"
        id2lbl_path = self._stage1_dir / "id2label.json"
        with open(lbl_path)   as f: self.label2id = json.load(f)
        with open(id2lbl_path) as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

        meta_path = oof_dir / "oof_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._text_col    = meta.get("text_col")
            self._id_col      = meta.get("id_col")
            self._split_seed  = meta.get("split_seed")
            self._train_seed  = meta.get("train_seed")
            self._feature_cols = meta.get("feature_cols")
            self._label_col   = meta.get("label_col")
            self.text_best_hp  = meta.get("text_best_hp",    [{}] * len(self.text_models))
            self.tabular_best_hp = meta.get("tabular_best_hp", [{}] * len(self.tabular_models))

        logger.info("OOF state loaded from disk — meta_X shape: %s", self.oof_meta_X.shape)

    @property
    def _stage1_dir(self) -> Path:
        """The directory stage-1 artifacts are read from (see ``_oof_from``)."""
        return self._oof_from if self._oof_from is not None else self.output_dir

    def _inherit_seeds(
        self, split_seed: int | None, train_seed: int | None,
    ) -> tuple[int, int]:
        """Fill unset stage-2/3 seeds from the ones stage 1 ran with.

        Stages 2 and 3 can run in a fresh session, and they used to default
        their seed to 42 regardless of what stage 1 used — so re-running stage 2
        alone after a stage 1 with seed 7 silently mixed two seeds in one
        result. An explicit argument still wins; ``None`` means "whatever stage
        1 used", falling back to 42 only for runs made before the seeds were
        recorded.
        """
        resolved_split = split_seed if split_seed is not None else self._split_seed
        resolved_train = train_seed if train_seed is not None else self._train_seed
        if resolved_split is None or resolved_train is None:
            logger.warning(
                "No seeds recorded from stage 1 (run made before seeds were "
                "saved to oof_metadata.json); using 42 for any unset seed."
            )
        return (
            42 if resolved_split is None else int(resolved_split),
            42 if resolved_train is None else int(resolved_train),
        )

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

        local = self._stage1_dir / "final" / f"{source['type']}_{source['local_index']}"
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
        split_seed: int = 42,
        train_seed: int = 42,
        deterministic: bool = False,
        stratify: bool = True,
        split_col: str | None = None,
        # --- text training ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 4,
        batch_size: int = 32,
        # --- tabular training ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        # --- fold model storage ---
        save_fold_models: bool = False,
        cleanup_fold_files: bool = True,
        # --- row identity ---
        id_col: str | None = None,
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
            split_seed:             Seed for every row-partitioning decision —
                                    the train/test split, the OOF folds, each
                                    base model's search folds and early-stopping
                                    slice. Default 42.
            train_seed:             Seed for every base model's training and
                                    search sampler. Default 42. Both seeds are
                                    saved to ``oof/oof_metadata.json`` so later
                                    stages inherit them.
            deterministic:          Demand bit-for-bit repeatable kernels, at a
                                    cost in speed and robustness. Default False.
            stratify:               Stratified split. Default True.
            val_size:               Internal val fraction for text fold models
                                    (early stopping).  Default 0.1.
            gradient_checkpointing: Enable gradient checkpointing. Default False.
            early_stopping_patience: Text early stopping patience. Default 4, the
                                    same as TextClassifier, so a text base model
                                    stops the way it would on its own.
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
        self._test_probs_cache = None   # new base models → new test probabilities
        if not self.text_models and not self.tabular_models:
            raise ValueError(
                "At least one text or tabular model spec is required to train "
                "base models. To reuse the base models of an earlier stacking "
                "run instead, call run(oof_from=<that run's output_dir>)."
            )
        if self.text_models and text_col is None:
            raise ValueError("text_col is required when text_models is non-empty.")
        if self.tabular_models and feature_cols is None:
            raise ValueError("feature_cols is required when tabular_models is non-empty.")

        # Lazy imports
        from ..utils.split            import split
        from ..text.dataset           import prepare_text_dataset
        from ..text.train             import train_text
        from ..tabular.dataset        import prepare_tabular_dataset
        from ..tabular.train          import train_tabular
        from ..text.hpo               import search_text
        from ..tabular.hpo            import search_tabular

        self._text_col    = text_col
        self._id_col      = id_col
        self._split_seed  = split_seed
        self._train_seed  = train_seed
        set_determinism(deterministic)
        seed_everything(train_seed)
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
                test_size=test_size, random_state=split_seed, stratify=stratify,
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
                    dataset_cache[cache_key] = prepare_text_dataset(
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
            # Every tabular base model reads one shared feature matrix (the OOF
            # loop and the saved X_test assume it), so a per-model preprocessing
            # setting cannot be honoured. It used to be ignored silently — the
            # same spec behaves differently in voting, which does honour it.
            for i, spec in enumerate(self.tabular_models):
                for key, run_value in (("encode_categoricals", encode_categoricals),
                                       ("scale_numeric", scale_numeric)):
                    if key in spec and spec[key] != run_value:
                        raise ValueError(
                            f"tabular_models[{i}] ({spec.get('model_name')}) sets "
                            f"{key}={spec[key]!r}, but stacking prepares one "
                            f"feature matrix for all tabular base models "
                            f"({key}={run_value!r}). Set {key} on run() instead, "
                            "or use voting, which prepares each base model "
                            "separately."
                        )
            (X_train, X_test, y_train, y_test,
             preprocessor, lbl2id, id2lbl, feature_names) = prepare_tabular_dataset(
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
            tag = f"text_{i}"
            kind, fixed, search = resolve_spec_hyperparams(spec, TEXT_SPEC_DEFAULTS)
            if kind == "search":
                logger.info(
                    "Base model %s (%s): hyperparameters FROM SEARCH — %s.",
                    tag, spec["model_name"], search.describe(),
                )
                best_hp, _ = search_text(
                    search,
                    resume=resolve_search_resume(search, self.resume),
                    train_dataset=train_text_datasets[i],
                    label2id=label2id, id2label=id2label,
                    model_name=spec["model_name"],
                    output_dir=hpo_dir / tag,
                    random_state=train_seed,
                    split_seed=split_seed,
                    use_lora=spec.get("use_lora", TEXT_SPEC_DEFAULTS["use_lora"]),
                    use_focal=spec.get("use_focal", TEXT_SPEC_DEFAULTS["use_focal"]),
                    gradient_checkpointing=gradient_checkpointing,
                    early_stopping_patience=early_stopping_patience,
                    use_fast=spec.get("use_fast", TEXT_SPEC_DEFAULTS["use_fast"]),
                )
                logger.info(
                    "Base model %s (%s): search finished — best %s, records in %s.",
                    tag, spec["model_name"], best_hp, hpo_dir / tag,
                )
                self.text_best_hp.append(best_hp)
            else:
                hp = fixed or {}
                _log_hp_source(tag, spec["model_name"], hp)
                self.text_best_hp.append(hp)

        for i, spec in enumerate(self.tabular_models):
            tag = f"tabular_{i}"
            kind, fixed, search = resolve_spec_hyperparams(spec, TABULAR_SPEC_DEFAULTS)
            if kind == "search":
                if spec["model_name"] == _INSILICOVA:
                    logger.info(
                        "Tabular model %d (insilicova): HPO not applicable — "
                        "InSilicoVA uses a fixed Bayesian cause database.  "
                        "Using spec hyperparams.", i
                    )
                    self.tabular_best_hp.append(fixed or {})
                else:
                    logger.info(
                        "Base model %s (%s): hyperparameters FROM SEARCH — %s.",
                        tag, spec["model_name"], search.describe(),
                    )
                    best_hp, _ = search_tabular(
                        search,
                        resume=resolve_search_resume(search, self.resume),
                        X_train=X_train, y_train=y_train,
                        label2id=label2id, id2label=id2label,
                        model_name=spec["model_name"],
                        output_dir=hpo_dir / tag,
                        random_state=train_seed,
                        split_seed=split_seed,
                        n_jobs=n_jobs, use_gpu=use_gpu,
                    )
                    logger.info(
                        "Base model %s (%s): search finished — best %s, records in %s.",
                        tag, spec["model_name"], best_hp, hpo_dir / tag,
                    )
                    self.tabular_best_hp.append(best_hp)
            else:
                hp = fixed
                _log_hp_source(tag, spec["model_name"], hp)
                self.tabular_best_hp.append(hp)

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
                random_state=train_seed,
                split_seed=split_seed,
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
            "id_col":           id_col,
            "split_seed":       split_seed,
            "train_seed":       train_seed,
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
            # A search already settled the epoch count, so the final fit uses
            # the whole training split; otherwise train_text() carves an
            # internal validation slice for early stopping.
            _searched = resolve_spec_hyperparams(spec, TEXT_SPEC_DEFAULTS)[0] == "search"
            final_val = None if _searched else val_size

            train_text(
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
                random_state=train_seed,
                split_seed=split_seed,
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
                train_tabular(
                    X_train=X_train, y_train=y_train,
                    label2id=label2id, id2label=id2label,
                    model_name=spec["model_name"],
                    output_dir=model_dir,
                    hyperparams=self.tabular_best_hp[i],
                    preprocessor=preprocessor, feature_names=feature_names,
                    random_state=train_seed, n_jobs=n_jobs, use_gpu=use_gpu,
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
        split_seed: int | None = None,
        train_seed: int | None = None,
        n_jobs: int = -1,
    ) -> dict:
        """Stage 2: score the meta-learner candidates and fit the best one.

        Can be called after ``train_base_models()`` in the same session, or in
        a fresh session (OOF data loaded automatically from disk).

        **No data leakage:** this stage uses only the OOF meta-feature matrix
        (``oof_meta_X``) and OOF labels (``oof_y``). Test data is never touched.

        Every candidate — including a lone one — is scored by stratified
        ``meta_cv_folds``-fold CV on the OOF meta-features, so the run always
        records how well its meta-learner does. The best is refitted on all OOF
        data and saved.

        There is deliberately no hyperparameter search here. The meta-learner
        sees a small, low-dimensional input (``n_models × n_classes`` columns),
        where a few cheap fixed candidates — the default multinomial logistic
        regression with ``C=1``, optionally alongside a tree model — are
        sufficient, and a search on the same OOF rows would add a second layer
        of selection for little gain.

        Args:
            meta_learners:   Override ``self.meta_learner_specs``.  One dict or
                             a list of dicts, each ``{"model_name": ...,
                             "hyperparams": {...}}``. ``None`` uses the specs
                             from ``__init__``.
            metric:          Selection metric.  Default: ``self.meta_select_metric``
                             (``f1_macro`` unless overridden at init).
            meta_cv_folds:   CV folds for meta-learner scoring. Default 3.
            split_seed:      Seed for the CV folds every candidate is scored on.
                             ``None`` (default) inherits the ``split_seed`` stage
                             1 ran with, read from ``oof/oof_metadata.json``.
            train_seed:      Seed for each meta-learner. ``None`` (default)
                             inherits stage 1's ``train_seed``.
            n_jobs:          CPU parallelism for meta-learner instantiation.

        Returns:
            dict with ``meta_learner``, ``meta_scores``, ``best_meta_name``,
            ``meta_select_metric``, ``output_dir``.
        """
        self._ensure_oof_loaded()
        split_seed, train_seed = self._inherit_seeds(split_seed, train_seed)

        if meta_learners is not None:
            specs = [meta_learners] if isinstance(meta_learners, dict) else list(meta_learners)
        else:
            specs = self.meta_learner_specs

        metric = metric or self.meta_select_metric

        meta_dir = self.output_dir / "meta_learner"
        meta_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Stage 2: scoring %d meta-learner candidate(s) by %d-fold CV on "
            "%s ...", len(specs), meta_cv_folds, metric,
        )
        scores: dict = {}
        for spec in specs:
            name  = spec["model_name"]
            model = _resolve_meta_learner(spec, n_jobs=n_jobs, random_state=train_seed)
            score = _score_meta_candidate(
                model, self.oof_meta_X, self.oof_y,
                self.id2label, metric, meta_cv_folds, split_seed,
            )
            scores[name] = round(score, 6)
            logger.info("  %-20s CV %s = %.4f", name, metric, score)

        best_name = max(scores, key=lambda k: scores[k])
        best_spec = next(s for s in specs if s["model_name"] == best_name)
        logger.info(
            "Meta-learner chosen: %s (CV %s = %.4f)%s",
            best_name, metric, scores[best_name],
            "" if len(specs) > 1 else " — the only candidate",
        )

        # ------------------------------------------------------------------
        # Fit winner on all OOF data and save
        # ------------------------------------------------------------------
        best_model = self._fit_save_meta_learner(
            best_spec, meta_dir, scores, best_name, metric, meta_cv_folds,
            split_seed, train_seed, n_jobs,
        )

        self.meta_learner   = best_model
        self.meta_scores    = scores
        self.best_meta_name = best_name
        self.meta_select_metric_used = metric

        logger.info("Stage 2 complete. Meta-learner saved to %s", meta_dir)
        return {
            "meta_learner":       best_model,
            "meta_scores":        scores,
            "best_meta_name":     best_name,
            "meta_select_metric": metric,
            "output_dir":         self.output_dir,
        }

    def _fit_save_meta_learner(
        self,
        spec: dict,
        meta_dir: Path,
        scores: dict,
        best_name: str,
        metric: str,
        meta_cv_folds: int,
        split_seed: int,
        train_seed: int,
        n_jobs: int,
    ):
        """Fit one meta-learner spec on all OOF rows and save it to ``meta_dir``.

        Writes ``meta_learner.joblib``, ``meta_scores.json`` and
        ``meta_learner_metadata.json`` — the layout ``predict_test()`` and
        ``results.validation`` read. Returns the fitted model.
        """
        meta_dir.mkdir(parents=True, exist_ok=True)
        model = _resolve_meta_learner(spec, n_jobs=n_jobs, random_state=train_seed)
        model.fit(self.oof_meta_X, self.oof_y)

        # Sanitize and strip all RNG objects before pickling to prevent
        # cross-NumPy-version joblib failures (MT19937 path changed in NumPy 2.x).
        # Three-layer defence: sanitize attrs → strip via __dict__ walk →
        # copyreg patch catches anything missed (C-extension slots, etc.).
        prepare_estimator_for_joblib(model)
        with null_rng_pickler():
            joblib.dump(model, meta_dir / "meta_learner.joblib")
        with open(meta_dir / "meta_scores.json", "w") as fh:
            json.dump(scores, fh, indent=2)

        meta_metadata = {
            "best_meta_name":     best_name,
            "meta_select_metric": metric,
            "meta_cv_folds":      meta_cv_folds,
            "meta_scores":        scores,
            "best_spec":          spec,
            "oof_meta_X_shape":   list(self.oof_meta_X.shape),
            "split_seed":         split_seed,
            "train_seed":         train_seed,
        }
        # Fingerprint the OOF matrix this meta-learner was fitted on, so a later
        # predict_test() can tell whether the base models have been re-run since.
        record_inputs(meta_metadata, self._oof_input_paths())
        with open(meta_dir / "meta_learner_metadata.json", "w") as fh:
            json.dump(meta_metadata, fh, indent=2, default=str)
        return model

    def _meta_learner_spec(self, name: str, meta_learners: list[dict] | None) -> dict:
        """The spec a meta-learner model name listed in ``combiner`` stands for.

        The matching entry of ``meta_learners`` (or of the specs given at
        construction) supplies its hyperparameters; a name with no entry falls
        back to :data:`DEFAULT_META_LEARNERS` if it is one of those, and to the
        model library's own defaults otherwise.
        """
        specs = meta_learners if meta_learners is not None else self.meta_learner_specs
        for spec in specs:
            if spec["model_name"] == name:
                return spec
        for spec in DEFAULT_META_LEARNERS:
            if spec["model_name"] == name:
                return {**spec, "hyperparams": dict(spec.get("hyperparams") or {})}
        return {"model_name": name}

    def _train_single_meta_learner(
        self,
        name: str,
        meta_learners: list[dict] | None,
        metric: str | None,
        meta_cv_folds: int,
        split_seed: int | None,
        train_seed: int | None,
        n_jobs: int,
    ):
        """Stage 2 for one meta-learner named in ``combiner`` (no choosing).

        Saved to ``meta_learner_<name>/`` so several can sit side by side; its
        CV score is recorded the same way ``train_meta_learner_stage()``
        records a candidate's.
        """
        self._ensure_oof_loaded()
        split_seed, train_seed = self._inherit_seeds(split_seed, train_seed)
        metric = metric or self.meta_select_metric
        spec = self._meta_learner_spec(name, meta_learners)
        score = _score_meta_candidate(
            _resolve_meta_learner(spec, n_jobs=n_jobs, random_state=train_seed),
            self.oof_meta_X, self.oof_y, self.id2label, metric, meta_cv_folds, split_seed,
        )
        logger.info("Stage 2: meta-learner %s — CV %s = %.4f", name, metric, score)
        return self._fit_save_meta_learner(
            spec, self.output_dir / f"meta_learner_{name}", {name: round(score, 6)},
            name, metric, meta_cv_folds, split_seed, train_seed, n_jobs,
        )

    def _meta_learner_test_proba(self, model, batch_size: int) -> tuple[np.ndarray, pd.DataFrame]:
        """A fitted meta-learner's test probabilities, columns in class-id order."""
        test_probs, test_df, model_sources = self._predict_test_base_probs(batch_size)
        sorted_ids = sorted(self.id2label.keys())

        # The meta-learner reads base models side by side, in the OOF column
        # layout it was fitted on.
        n_total_cols = sum(src["n_cols"] for src in model_sources)
        meta_X_test = np.zeros((len(test_df), n_total_cols), dtype=float)
        for probs, src in zip(test_probs, model_sources):
            col_s = src["col_start"]
            meta_X_test[:, col_s:col_s + src["n_cols"]] = probs

        meta_proba = model.predict_proba(meta_X_test)  # (n_test, n_classes)

        # Reorder columns to canonical sorted_ids order
        classes   = list(model.classes_)
        col_order = [classes.index(cid) for cid in sorted_ids]
        return meta_proba[:, col_order], test_df

    def train_class_voter_stage(
        self,
        metric: str = "f1",
        shrinkage: float = 0.1,
        min_support_for_trust: float = 20.0,
        fallback_to_soft: bool = True,
        fallback_metric: str = "f1_macro",
        fallback_cv_folds: int = 3,
        split_seed: int | None = None,
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
                    each class.  Default ``"f1"``.  ``"f1"``, ``"recall"`` and
                    ``"precision"`` use hard argmax decisions.  ``"brier"``
                    computes 1 − mean((p − y_bin)²) from soft probabilities,
                    which degrades more gracefully when a class is rare or
                    absent in OOF data.
            shrinkage:
                    Base blend factor toward uniform per-class weights for
                    well-supported classes.  Classes with fewer OOF examples
                    than ``min_support_for_trust`` receive additional shrinkage
                    on top of this base (support-adaptive).
                    ``0.0`` = trust learned OOF weights for all classes.
                    ``1.0`` = uniform voting for all classes.  Default ``0.1``.
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
            split_seed:
                    Seed for the fallback CV folds. ``None`` (default) inherits
                    the ``split_seed`` stage 1 ran with. Replaces
                    ``fallback_random_state``: the folds are a partitioning
                    decision, so they follow the same seed as every other one.
            fallback_tolerance:
                    Non-negative margin (in score units) that class-voter must
                    beat soft-vote by on fallback CV in order to be kept.
                    Final fallback decision keeps class-voter only when:
                    ``class_cv > soft_cv + fallback_tolerance``.
                    Default ``0.0`` (ties prefer uniform soft vote).
        """
        self._ensure_oof_loaded()
        split_seed, _ = self._inherit_seeds(split_seed, None)
        if fallback_tolerance < 0:
            raise ValueError("fallback_tolerance must be >= 0.")

        oof_meta_path = self._stage1_dir / "oof" / "oof_metadata.json"
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
        learned_weights, class_weights, use_uniform_fallback, fallback_report = _fit_class_voter(
            prob_matrices=prob_matrices,
            y_true=self.oof_y,
            id2label=self.id2label,
            metric=metric,
            shrinkage=shrinkage,
            min_support_for_trust=min_support_for_trust,
            fallback_to_soft=fallback_to_soft,
            fallback_metric=fallback_metric,
            fallback_cv_folds=fallback_cv_folds,
            split_seed=split_seed,
            fallback_tolerance=fallback_tolerance,
        )
        combined_oof = _apply_class_voter(prob_matrices, learned_weights)
        soft_oof = _uniform_soft_vote(prob_matrices)

        from ..utils.metrics import score_predictions, CV_METRICS

        metric_names = list(CV_METRICS)
        true_labels = [self.id2label[int(y)] for y in self.oof_y]
        learned_result = assemble_predictions(
            combined_oof, self.id2label,
            true_labels=true_labels, top_k=3,
        )
        soft_result = assemble_predictions(
            soft_oof, self.id2label,
            true_labels=true_labels, top_k=3,
        )
        learned_scores = {
            name: float(score_predictions(learned_result.top1, metric=name))
            for name in metric_names
        }
        soft_scores = {
            name: float(score_predictions(soft_result.top1, metric=name))
            for name in metric_names
        }

        if fallback_report is not None and not fallback_report.get("enabled"):
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

        final_oof = _apply_class_voter(prob_matrices, class_weights)
        final_result = assemble_predictions(
            final_oof, self.id2label,
            true_labels=true_labels, top_k=3,
        )
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
            "split_seed": split_seed,
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

    # ------------------------------------------------------------------
    # Shared by every Stage 3: base models' test probabilities
    # ------------------------------------------------------------------

    def _load_model_sources(self) -> tuple[list[dict], str | None]:
        """Every base model this run combines, in OOF column order.

        Returns ``(model_sources, inherited_from)``. ``model_sources`` records,
        per base model, its spec, where its final weights are and which OOF
        columns it owns. Runs made before that list was saved have it rebuilt
        from ``text_models`` / ``tabular_models``.
        """
        oof_meta_path = self._stage1_dir / "oof" / "oof_metadata.json"
        model_sources: list[dict] | None = None
        inherited_from: str | None = None
        if oof_meta_path.exists():
            with open(oof_meta_path) as fh:
                oof_meta = json.load(fh)
            model_sources = oof_meta.get("model_sources")
            inherited_from = oof_meta.get("inherited_from")

        if not model_sources:
            n_classes = len(self.id2label)
            final_dir = self._stage1_dir / "final"
            model_sources = []
            for i, spec in enumerate(self.text_models):
                model_sources.append({
                    "type": "text", "local_index": i, "spec": spec,
                    "final_dir": str((final_dir / f"text_{i}").resolve()),
                    "col_start": i * n_classes, "n_cols": n_classes,
                })
            for i, spec in enumerate(self.tabular_models):
                model_sources.append({
                    "type": "tabular", "local_index": i, "spec": spec,
                    "final_dir": str((final_dir / f"tabular_{i}").resolve()),
                    "col_start": (len(self.text_models) + i) * n_classes,
                    "n_cols": n_classes,
                })
        if not model_sources:
            raise RuntimeError(
                "No base models found for %s. Run train_base_models() first."
                % self._stage1_dir
            )
        return model_sources, inherited_from

    def _oof_positions(self) -> np.ndarray:
        """OOF labels as positions along the class axis (sorted class ids)."""
        sorted_ids = np.array(sorted(self.id2label.keys()))
        return np.searchsorted(sorted_ids, np.asarray(self.oof_y, dtype=int))

    def _predict_test_base_probs(
        self, batch_size: int,
    ) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
        """Each final base model's test probabilities, ``(n_models, n_test, n_classes)``.

        The one implementation every Stage 3 uses — meta-learner, class-aware
        voter and ensemble selection used to carry a copy each. The result is
        cached on the instance, so combining the same stage 1 several ways
        (``run(combiner=[...])``) runs each base model's inference once;
        text inference is the expensive part.

        Returns:
            ``(probs, test_df, model_sources)``.
        """
        from ..text.dataset    import prepare_text_dataset
        from ..text.predict    import predict_text
        from ..tabular.dataset import prepare_tabular_dataset
        from ..tabular.predict import predict_tabular

        key = (str(self._stage1_dir), int(batch_size))
        cached = getattr(self, "_test_probs_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        self._ensure_oof_loaded()
        data_dir = self._stage1_dir / "data"
        train_df = pd.read_csv(data_dir / "train_df.csv")
        test_df  = pd.read_csv(data_dir / "test_df.csv")
        sorted_ids = sorted(self.id2label.keys())
        model_sources, inherited_from = self._load_model_sources()

        # One preprocessed X_test serves every sklearn tabular model: they all
        # share the feature space prepared in stage 1.
        X_test = y_test = None
        if any(s["type"] == "tabular" and s["spec"].get("model_name") != _INSILICOVA
               for s in model_sources):
            if (data_dir / "X_test.npy").exists():
                X_test = np.load(data_dir / "X_test.npy")
                y_test = np.load(data_dir / "y_test.npy")
            else:
                (_, X_test, _, y_test, _, _, _, _) = prepare_tabular_dataset(
                    train_df, test_df,
                    feature_cols=self._feature_cols,
                    label_col=self._label_col,
                    encode_categoricals="ordinal",
                )

        prob_matrices = []
        for source in model_sources:
            model_dir  = self._resolve_final_dir(source, inherited_from)
            spec       = source["spec"]
            model_name = spec["model_name"]
            logger.info(
                "Predicting test — %s model %s (weights: %s) ...",
                source["type"], model_name, model_dir,
            )
            if source["type"] == "text":
                _, test_text_ds, _, _ = prepare_text_dataset(
                    train_df, test_df,
                    text_col=self._text_col, label_col=self._label_col,
                    model_name=model_name, max_length=spec.get("max_length", 512),
                )
                result = predict_text(model_dir, test_text_ds, batch_size=batch_size, top_k=1)
                probs = _extract_probs(result, sorted_ids)
            elif model_name == _INSILICOVA:
                probs = _insilicova_predict_df(model_dir, test_df, sorted_ids, train_df=train_df)
            else:
                result = predict_tabular(model_dir, X_test, y_test, top_k=1)
                probs = _extract_probs(result, sorted_ids)
            prob_matrices.append(probs)

        out = (np.stack(prob_matrices, axis=0), test_df, model_sources)
        self._test_probs_cache = (key, out)
        return out

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

        meta_proba, test_df = self._meta_learner_test_proba(self.meta_learner, batch_size)

        true_labels = test_df[self._label_col].tolist()
        self.predictions = assemble_predictions(
            meta_proba, self.id2label,
            true_labels=true_labels, top_k=top_k,
            ids=resolve_test_ids(test_df, self._id_col),
        )

        # Save predictions in the layout every pipeline uses:
        # <output_dir>/predictions/predictions_{top1,full,topk}.csv
        save_predictions(self.predictions, self.output_dir / "predictions")
        save_predictions(self.predictions, self.output_dir / "meta_learner" / "predictions")

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

        self._ensure_oof_loaded()
        n_classes = len(self.id2label)
        model_sources, _ = self._load_model_sources()
        if self.class_voter_weights.shape != (len(model_sources), n_classes):
            raise ValueError(
                "Loaded class-aware weights shape %s does not match current model_sources/classes (%d, %d)."
                % (self.class_voter_weights.shape, len(model_sources), n_classes)
            )
        test_probs, test_df, _ = self._predict_test_base_probs(batch_size)
        prob_matrices = list(test_probs)

        combined = _apply_class_voter(prob_matrices, self.class_voter_weights)
        true_labels = test_df[self._label_col].tolist()
        predictions = assemble_predictions(
            combined, self.id2label,
            true_labels=true_labels, top_k=top_k,
            ids=resolve_test_ids(test_df, self._id_col),
        )

        save_predictions(predictions, self.output_dir / "class_voter" / "predictions")

        voted_acc = (
            predictions.top1["true_label"] == predictions.top1["predicted_label"]
        ).mean()
        logger.info("Class-aware voter Stage 3 complete. Test accuracy: %.4f", voted_acc)
        return predictions

    def _predict_test_single_meta_learner(
        self, name: str, model, top_k: int, batch_size: int,
    ) -> PredictionResult:
        """Stage 3 for one meta-learner named in ``combiner``."""
        meta_proba, test_df = self._meta_learner_test_proba(model, batch_size)
        predictions = assemble_predictions(
            meta_proba, self.id2label,
            true_labels=test_df[self._label_col].tolist(), top_k=top_k,
            ids=resolve_test_ids(test_df, self._id_col),
        )
        save_predictions(predictions, self.output_dir / f"meta_learner_{name}" / "predictions")
        acc = (predictions.top1["true_label"] == predictions.top1["predicted_label"]).mean()
        logger.info("Meta-learner %s Stage 3 complete. Test accuracy: %.4f", name, acc)
        return predictions

    def predict_test_simple_average(
        self,
        top_k: int = 3,
        batch_size: int = 32,
    ) -> PredictionResult:
        """Stage 3 for ``combiner="simple_average"``: equal-weight average.

        Averages the final base models' test probabilities with equal weights.
        Nothing is learned, so there is no Stage 2. These are the same base
        models every other combiner uses, which is what separates this from
        ``task="voting"`` — that pipeline trains its own base models.
        Predictions are saved under ``output_dir/simple_average/predictions/``.

        Args:
            top_k:      Number of ranked classes to return.
            batch_size: Inference batch size for text base models.
        """
        self._ensure_oof_loaded()
        test_probs, test_df, _ = self._predict_test_base_probs(batch_size)
        predictions = assemble_predictions(
            _uniform_soft_vote(list(test_probs)), self.id2label,
            true_labels=test_df[self._label_col].tolist(), top_k=top_k,
            ids=resolve_test_ids(test_df, self._id_col),
        )
        save_predictions(predictions, self.output_dir / "simple_average" / "predictions")
        acc = (predictions.top1["true_label"] == predictions.top1["predicted_label"]).mean()
        logger.info("Simple average Stage 3 complete. Test accuracy: %.4f", acc)
        return predictions

    # ------------------------------------------------------------------
    # Convenience: all stages in sequence
    # ------------------------------------------------------------------

    def train_ensemble_selection_stage(
        self,
        metric: str | None = None,
        ensemble_size: int = 100,
        use_best_in_trajectory: bool = True,
        sorted_init: int = 0,
        n_bags: int = 1,
        bag_fraction: float = 0.5,
        train_seed: int | None = None,
    ) -> dict:
        """Stage 2 alternative: greedy ensemble selection (Caruana et al. 2004).

        Uses only the OOF meta-feature matrix and OOF labels.  Learns one
        non-negative weight per base model (summing to 1) by repeatedly adding,
        with replacement, the model that most improves ``metric`` on the
        averaged OOF probabilities.  See
        :class:`~multimodalva.ensemble.ensemble_selection.EnsembleSelection`.

        Args:
            metric:                 Selection metric.  Default
                                    ``self.meta_select_metric``
                                    (``f1_macro``).  Any
                                    ``score_predictions`` metric, or
                                    ``"log_loss"``.
            ensemble_size:          Greedy iterations.  Default 100.
            use_best_in_trajectory: Keep the best iteration's weights.  Default
                                    ``True``.
            sorted_init:            Seed with the top-k single models.  Default 0.
            n_bags:                 Bagged selection over random model subsets;
                                    1 (default) = no bagging.
            bag_fraction:           Share of models per bag.  Default 0.5.
            train_seed:             Seed for the bags. ``None`` (default)
                                    inherits the ``train_seed`` stage 1 ran with.

        Returns:
            dict with ``weights`` (``{model_key: weight}``), ``scores`` (in-sample
            OOF scores of the weighted average on ``CV_METRICS``), ``oof_score``,
            ``best_single_score``, ``simple_average_score``, ``output_dir``.
            Model keys are ``"<position>:<model_name>"``.
        """
        from .ensemble_selection import EnsembleSelection
        from ..utils.metrics import score_predictions, CV_METRICS

        self._ensure_oof_loaded()
        _, train_seed = self._inherit_seeds(None, train_seed)
        metric = metric or self.meta_select_metric
        model_sources, _ = self._load_model_sources()
        keys = _model_source_keys(model_sources)
        preds = _stack_oof_probs(self.oof_meta_X, model_sources)

        es = EnsembleSelection(
            metric=metric,
            ensemble_size=ensemble_size,
            use_best_in_trajectory=use_best_in_trajectory,
            sorted_init=sorted_init,
            n_bags=n_bags,
            bag_fraction=bag_fraction,
            random_state=train_seed,
        ).fit(preds, self._oof_positions(), model_names=keys)

        true_labels = [self.id2label[int(y)] for y in self.oof_y]
        combined_result = assemble_predictions(
            es.predict(preds), self.id2label, true_labels=true_labels, top_k=3,
        )
        scores = {
            name: float(score_predictions(combined_result.top1, metric=name))
            for name in CV_METRICS
        }

        es_dir = self.output_dir / "ensemble_selection"
        es_dir.mkdir(parents=True, exist_ok=True)
        np.save(es_dir / "ensemble_weights.npy", es.weights_)
        weights_payload = {
            "model_keys": keys,
            "model_names": [src["spec"]["model_name"] for src in model_sources],
            "weights": es.weights_.tolist(),
            "weights_by_model": es.weights_by_model_,
            "selected_models": es.selected_models_,
        }
        with open(es_dir / "ensemble_weights.json", "w") as fh:
            json.dump(weights_payload, fh, indent=2)
        pd.DataFrame(
            es.trajectory_.T, columns=[f"bag_{b}" for b in range(es.trajectory_.shape[0])],
        ).rename_axis("step").to_csv(es_dir / "trajectory.csv")

        metadata = {
            "metric": metric,
            "greater_is_better": es.greater_is_better_,
            "ensemble_size": ensemble_size,
            "use_best_in_trajectory": use_best_in_trajectory,
            "sorted_init": sorted_init,
            "n_bags": n_bags,
            "bag_fraction": bag_fraction,
            "train_seed": train_seed,
            "bags": es.bags_,
            "oof_meta_X_shape": list(self.oof_meta_X.shape),
            "n_models": len(model_sources),
            "n_classes": int(preds.shape[2]),
            "model_keys": keys,
            "weights_by_model": es.weights_by_model_,
            "selected_models": es.selected_models_,
            # In-sample: fitted and scored on the same OOF rows.  Use
            # compare_combiners_stage() for the out-of-sample comparison.
            "oof_score": es.oof_score_,
            "best_single_model": es.best_single_model_,
            "best_single_score": es.best_single_score_,
            "simple_average_score": es.simple_average_score_,
            "scores": scores,
        }
        # Learned from the OOF matrix; stale once a base model is re-run.
        record_inputs(metadata, self._oof_input_paths())
        with open(es_dir / "ensemble_selection_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

        self.ensemble_selection_weights = es.weights_
        self.ensemble_selection_metadata = metadata

        logger.info(
            "Ensemble selection (%s, in-sample OOF): weighted %.4f | best single %.4f (%s) "
            "| simple average %.4f",
            metric, es.oof_score_, es.best_single_score_, es.best_single_model_,
            es.simple_average_score_,
        )
        for key in es.selected_models_:
            logger.info("  weight %.3f  %s", es.weights_by_model_[key], key)
        logger.info("Stage 2 complete. Ensemble selection saved to %s", es_dir)
        return {
            "weights": es.weights_by_model_,
            "scores": scores,
            "oof_score": es.oof_score_,
            "best_single_score": es.best_single_score_,
            "simple_average_score": es.simple_average_score_,
            "output_dir": self.output_dir,
        }

    def predict_test_ensemble_selection(
        self,
        top_k: int = 3,
        batch_size: int = 32,
        on_stale: str = "auto",
    ) -> PredictionResult:
        """Stage 3 alternative: combine test probabilities with the saved weights.

        Each base model's test probabilities come from its final model
        (retrained on the full training set in Stage 1), then are combined as
        ``Σ_m weight_m × probs_m``.  Predictions are saved under
        ``output_dir/ensemble_selection/predictions/``.

        Args:
            top_k:      Number of ranked classes to return.
            batch_size: Inference batch size for text base models.
            on_stale:   As in :meth:`predict_test_class_voter`.
        """
        es_dir = self.output_dir / "ensemble_selection"
        if self.ensemble_selection_weights is None:
            weights_path = es_dir / "ensemble_weights.npy"
            if not weights_path.exists():
                raise RuntimeError(
                    "Ensemble-selection weights not found at %s. "
                    "Run train_ensemble_selection_stage() first." % weights_path
                )
            self.ensemble_selection_weights = np.load(weights_path)
            meta_path = es_dir / "ensemble_selection_metadata.json"
            if meta_path.exists():
                with open(meta_path) as fh:
                    self.ensemble_selection_metadata = json.load(fh)
                guard_inputs(
                    check_inputs(self.ensemble_selection_metadata, self._oof_input_paths()),
                    "The saved ensemble-selection weights",
                    "re-run train_ensemble_selection_stage()",
                    on_stale=on_stale,
                )

        self._ensure_oof_loaded()
        model_sources, _ = self._load_model_sources()
        if len(self.ensemble_selection_weights) != len(model_sources):
            raise ValueError(
                "Loaded ensemble-selection weights have %d entries but there are %d base models."
                % (len(self.ensemble_selection_weights), len(model_sources))
            )

        test_probs, test_df, _ = self._predict_test_base_probs(batch_size)
        combined = np.tensordot(self.ensemble_selection_weights, test_probs, axes=(0, 0))
        predictions = assemble_predictions(
            combined, self.id2label,
            true_labels=test_df[self._label_col].tolist(), top_k=top_k,
            ids=resolve_test_ids(test_df, self._id_col),
        )
        save_predictions(predictions, es_dir / "predictions")

        acc = (predictions.top1["true_label"] == predictions.top1["predicted_label"]).mean()
        logger.info("Ensemble-selection Stage 3 complete. Test accuracy: %.4f", acc)
        return predictions

    # ------------------------------------------------------------------
    # Out-of-sample comparison of Stage 2 combiners
    # ------------------------------------------------------------------

    def compare_combiners_stage(
        self,
        metric: str | None = None,
        n_combiner_folds: int = 5,
        split_seed: int | None = None,
        train_seed: int | None = None,
        meta_learners: list[dict] | dict | None = None,
        class_voter_kwargs: dict | None = None,
        ensemble_selection_kwargs: dict | None = None,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Compare the Stage 2 combiners by nested CV over the OOF rows.

        This is how ``combiner="best"`` chooses, and it can also be called on
        its own. It uses training data only — the test set is not touched — so
        a combiner can be chosen before any test result is seen.

        Fitting a combiner on all OOF rows and scoring it on the same rows is
        in-sample, and favours the more flexible combiner. Here every combiner
        is fitted on k-1 folds of the OOF rows and scored on the held-out fold,
        with the **same** stratified folds for all of them
        (:func:`~multimodalva.ensemble.ensemble_selection.cross_validate_combiner`).
        Each is scored as its training stage fits it.

        Rows, simplest first (ties go to the earlier row):
            ``simple_average``      equal-weight average (nothing learned)
            ``class_aware_voting``  the class-aware voter, including its
                                    fallback to equal weights
            ``ensemble_selection``  greedy ensemble selection
            one row per meta-learner, named by its ``model_name`` — the same
                                    names ``combiner=`` accepts

        Args:
            metric:                    Scoring metric for every row, and
                                       ensemble selection's selection metric.
                                       Default ``self.meta_select_metric``.
            n_combiner_folds:          Folds. Default 5 (reduced if a class
                                       is smaller).
            split_seed:                Seed for the folds every combiner is
                                       scored on (and for the class voter's
                                       fallback CV). ``None`` (default)
                                       inherits stage 1's.
            train_seed:                Seed for the meta-learners and the
                                       ensemble-selection bags. ``None``
                                       (default) inherits stage 1's.
            meta_learners:             Meta-learner specs, one row each.
                                       Default: the specs given at
                                       construction.
            class_voter_kwargs:        Settings for the class voter, as
                                       :meth:`train_class_voter_stage` takes
                                       them; defaults are that method's.
            ensemble_selection_kwargs: Settings for ensemble selection, as
                                       :meth:`train_ensemble_selection_stage`
                                       takes them.
            n_jobs:                    Meta-learner parallelism.

        Returns:
            DataFrame, one row per combiner, also written to
            ``output_dir/combiner_comparison/combiner_comparison.csv``. The
            ``.json`` beside it holds the settings, per-fold results and
            ``best_combiner``, the row with the best ``cv_mean``.
        """
        import inspect

        from .ensemble_selection import EnsembleSelection, cross_validate_combiner

        self._ensure_oof_loaded()
        split_seed, train_seed = self._inherit_seeds(split_seed, train_seed)
        metric = metric or self.meta_select_metric
        model_sources, _ = self._load_model_sources()
        preds = _stack_oof_probs(self.oof_meta_X, model_sources)
        y = self._oof_positions()

        # The class voter's settings default to train_class_voter_stage()'s own,
        # read from its signature so there is one set of defaults.
        voter_defaults = {
            name: p.default
            for name, p in inspect.signature(self.train_class_voter_stage).parameters.items()
            if name != "split_seed"
        }
        cv_kwargs = {**voter_defaults, **(class_voter_kwargs or {}), "split_seed": split_seed}

        es_kwargs = dict(ensemble_selection_kwargs or {})
        es_seed = es_kwargs.pop("train_seed", None)
        es_seed = train_seed if es_seed is None else es_seed
        es_kwargs["metric"] = es_kwargs.get("metric") or metric

        if meta_learners is None:
            specs = self.meta_learner_specs
        else:
            specs = [meta_learners] if isinstance(meta_learners, dict) else list(meta_learners)
        _check_meta_learner_specs(specs)

        factories = {
            "simple_average":     _SimpleAverageCombiner,
            "class_aware_voting": lambda: _ClassVoterCombiner(**cv_kwargs),
            "ensemble_selection": lambda: EnsembleSelection(random_state=es_seed, **es_kwargs),
        }
        for spec in specs:
            factories[spec["model_name"]] = (
                lambda spec=spec: _MetaLearnerCombiner(spec, n_jobs, train_seed)
            )

        rows, reports = [], {}
        for name, factory in factories.items():
            logger.info("Combiner comparison: %s ...", name)
            rep = cross_validate_combiner(
                preds, y, factory,
                n_splits=n_combiner_folds, random_state=split_seed, metric=metric,
            )
            reports[name] = rep
            rows.append({
                "combiner": name,
                "metric": metric,
                "cv_mean": rep["mean"],
                "cv_std": rep["std"],
                "n_folds": rep["n_splits_used"],
                **{f"fold_{i}": s for i, s in enumerate(rep["fold_scores"])},
            })

        table = pd.DataFrame(rows)
        greater_is_better = next(iter(reports.values()))["greater_is_better"]
        # Ties go to the earlier (simpler) row: argmax/argmin return the first.
        means = table["cv_mean"].to_numpy()
        best = str(table["combiner"].iloc[int(np.argmax(means) if greater_is_better
                                              else np.argmin(means))])

        out_dir = self.output_dir / "combiner_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "combiner_comparison.csv", index=False)
        payload = {
            "metric": metric,
            "greater_is_better": greater_is_better,
            "best_combiner": best,
            "n_combiner_folds": n_combiner_folds,
            "split_seed": split_seed,
            "train_seed": train_seed,
            "class_voter_kwargs": cv_kwargs,
            "ensemble_selection_kwargs": {**es_kwargs, "train_seed": es_seed},
            "meta_learners": specs,
            "model_keys": _model_source_keys(model_sources),
            "results": reports,
        }
        record_inputs(payload, self._oof_input_paths())
        with open(out_dir / "combiner_comparison.json", "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        logger.info("Combiner comparison (%s, %d-fold nested CV on OOF) — best: %s\n%s",
                    metric, int(table["n_folds"].iloc[0]), best,
                    table[["combiner", "cv_mean", "cv_std"]].to_string(index=False))
        self.combiner_comparison = table
        self.best_combiner = best
        return table

    @track_run("stacking", monitor_gpu=True)
    def run(
        self,
        df: "pd.DataFrame | None" = None,
        label_col: str | None = None,
        text_col: str | None = None,
        feature_cols: list[str] | None = None,
        # --- split ---
        test_size: float = 0.2,
        split_seed: int = 42,
        train_seed: int = 42,
        deterministic: bool = False,
        stratify: bool = True,
        split_col: str | None = None,
        # --- text ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 4,
        batch_size: int = 32,
        # --- tabular ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        # --- fold storage ---
        save_fold_models: bool = False,
        cleanup_fold_files: bool = True,
        # --- stage 2: how the base models are combined ---
        combiner: "str | list[str]" = "meta_learner",
        oof_from: "str | Path | None" = None,
        oof_only: bool = False,
        meta_learners: list[dict] | dict | None = None,
        meta_select_metric: str | None = None,
        meta_cv_folds: int = 3,
        class_voter_kwargs: dict | None = None,
        ensemble_selection_kwargs: dict | None = None,
        n_combiner_folds: int = 5,
        # --- inference ---
        top_k: int = 3,
        id_col: str | None = None,
    ) -> dict:
        """Stage 1 (out-of-fold predictions) → stage 2 (combine) → stage 3 (test).

        ``combiner`` picks how the base models are combined:

        ``"meta_learner"`` (default)
            A meta-learner trained on the out-of-fold probabilities (stacking
            proper). With several ``meta_learners`` candidates, the best by
            ``meta_cv_folds``-fold CV is used.
        ``"simple_average"``
            Equal-weight average of the base models' probabilities; nothing is
            learned. Uses the same base models as every other combiner.
        ``"class_aware_voting"``
            Per-cause weights over the base models, learned from the
            out-of-fold predictions. ``class_voter_kwargs`` go to
            :meth:`train_class_voter_stage`.
        ``"ensemble_selection"``
            Greedy ensemble selection (Caruana et al. 2004): one non-negative
            weight per base model. ``ensemble_selection_kwargs`` go to
            :meth:`train_ensemble_selection_stage`.
        a meta-learner model name — ``"logistic_regression"``, ``"lightgbm"``, …
            That one meta-learner, with no choosing between candidates. Its
            hyperparameters come from the entry of ``meta_learners`` with the
            same ``model_name``, or the package defaults if there is none.
            Saved under ``meta_learner_<name>/``.
        ``"best"``
            Chooses one of the above using training data only:
            :meth:`compare_combiners_stage` scores simple average, class-aware
            voting, ensemble selection and each ``meta_learners`` candidate by
            ``n_combiner_folds``-fold nested CV on the out-of-fold rows, and
            only the winner is trained and applied to the test set. The test
            set plays no part in the choice. Cannot be listed with others.

        Pass a **list** to get several from one set of out-of-fold predictions.
        Stage 1 is the expensive part and runs once; so does each base model's
        inference on the test set. The first combiner listed is the main result.

        ``meta_select_metric`` is the metric for every choice made in stage 2:
        between meta-learner candidates, by ensemble selection, and by
        ``"best"``.

        ``oof_only=True`` does the opposite: it computes stage 1, writes it, and
returns before any combiner is fitted — for running the expensive half on a
GPU box and the combiners later (or repeatedly) somewhere cheap. The result
dict then carries the OOF matrix, ``oof_dir``, ``oof_metadata`` (which records
the base models, the split and the seeds), and ``predictions`` is ``None``. Continue with ``run(oof_from=<that output_dir>)``.
Passing both raises.

``oof_from`` skips stage 1 entirely. Point it at the ``output_dir`` of an
        earlier stacking run and its out-of-fold predictions, trained base
        models, train/test split, label maps and seeds are reused; nothing is
        retrained. ``df``, the model specs and the seed arguments are then not
        used — stage 2 must see the split and seeds its out-of-fold
        predictions were made with. Leave it ``None`` to compute stage 1.

        Outputs:
            Each combiner writes its predictions to its own folder —
            ``meta_learner/predictions/``, ``simple_average/predictions/``,
            ``class_voter/predictions/``, ``ensemble_selection/predictions/``,
            ``meta_learner_<name>/predictions/`` — and ``predictions/`` holds
            the main result, as for every other pipeline. ``"best"`` also
            writes ``combiner_comparison/``.

        Returns:
            dict with ``predictions`` (the main result), ``combiner`` (the
            combiners run, main first), ``combiner_predictions``
            (``{combiner: PredictionResult}``), ``combiner_chosen`` and
            ``combiner_comparison`` (set by ``"best"``, else ``None``),
            ``best_meta_name``, ``meta_select_metric``, ``meta_scores``,
            ``class_voter`` (the voter's report), ``ensemble_selection_weights``,
            ``oof_meta_X``, ``oof_from``, ``meta_learner``, ``label2id``,
            ``id2label`` and ``output_dir``.
        """
        requested = [combiner] if isinstance(combiner, str) else list(combiner)
        if meta_learners is not None and isinstance(meta_learners, dict):
            meta_learners = [meta_learners]
        # Checked before stage 1 so a typo does not cost a full OOF loop.
        choose_best = requested == ["best"]
        if "best" in requested and not choose_best:
            raise ValueError(
                "combiner='best' chooses one combiner by itself; do not list it "
                "with others."
            )
        meta_names = _meta_learner_names()
        unknown = [c for c in requested
                   if c != "best" and c not in STACKING_COMBINERS and c not in meta_names]
        if unknown or not requested:
            raise ValueError(
                f"Unknown combiner {unknown or combiner!r}. Use one of "
                f"{', '.join(STACKING_COMBINERS)}, a meta-learner model name "
                f"({', '.join(meta_names)}), a list of those, or 'best'.\n"
                + _UNKNOWN_META_LEARNER_HINT
            )
        _check_meta_learner_specs(
            meta_learners if meta_learners is not None else self.meta_learner_specs
        )
        methods = list(dict.fromkeys(requested))    # drop repeats, keep order

        # --- stage 1: compute it, or reuse an earlier run's -----------------
        if oof_from is not None:
            source = Path(oof_from)
            if not (source / "oof" / "oof_meta_X.npy").is_file():
                raise FileNotFoundError(
                    f"oof_from={str(source)!r} has no oof/oof_meta_X.npy. Point it "
                    "at the output_dir of a finished stacking run (its stage 1)."
                )
            self._oof_from = source
            self._test_probs_cache = None
            self._ensure_oof_loaded()
            with open(source / "oof" / "oof_metadata.json") as fh:
                source_meta = json.load(fh)
            if not self.text_models and not self.tabular_models:
                self.text_models = list(source_meta.get("text_specs") or [])
                self.tabular_models = list(source_meta.get("tabular_specs") or [])
            logger.info(
                "Reusing stage 1 from %s: %d out-of-fold rows × %d columns, "
                "split_seed=%s, train_seed=%s. No base model is retrained; the "
                "seed arguments of this call are not used.",
                source, self.oof_meta_X.shape[0], self.oof_meta_X.shape[1],
                self._split_seed, self._train_seed,
            )
            if df is not None:
                logger.info("oof_from is set, so df is not used.")
            stage_split = stage_train = None      # inherit the source run's
            n_train = source_meta.get("n_train")
            n_test = source_meta.get("n_test")
        else:
            if df is None or label_col is None:
                raise ValueError("df and label_col are required unless oof_from is given.")
            self.train_base_models(
                df=df, label_col=label_col,
                text_col=text_col, feature_cols=feature_cols,
                test_size=test_size, split_seed=split_seed, train_seed=train_seed,
                deterministic=deterministic, stratify=stratify,
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
                id_col=id_col,
            )
            stage_split, stage_train = split_seed, train_seed
            n_train, n_test = len(self.train_df), len(self.test_df)

        # --- stop here when only stage 1 was asked for -----------------------
        # Stage 1 is the expensive half and often belongs on different hardware
        # from the combiners. What it wrote is what run(oof_from=...) reads, so
        # a later call picks up exactly here.
        if oof_only:
            if oof_from is not None:
                raise ValueError(
                    "oof_only=True with oof_from= has nothing to do: oof_from "
                    "reuses a finished stage 1, and oof_only stops after "
                    "computing one. Pass one or the other."
                )
            logger.info(
                "oof_only=True — stage 1 done, stopping before the combiners. "
                "Continue later with run(oof_from=%r, combiner=...).",
                str(self.output_dir),
            )
            return {
                "oof_meta_X":    self.oof_meta_X,
                "oof_y":         self.oof_y,
                "oof_dir":       self._stage1_dir / "oof",
                "oof_metadata":  self._stage1_dir / "oof" / "oof_metadata.json",
                "output_dir":    self.output_dir,
                "label2id":      self.label2id,
                "id2label":      self.id2label,
                "n_train":       n_train,
                "n_test":        n_test,
                "predictions":   None,
                "oof_only":      True,
            }

        # --- "best": choose one combiner on the OOF rows alone ---------------
        comparison = None
        if choose_best:
            comparison = self.compare_combiners_stage(
                metric=meta_select_metric,
                n_combiner_folds=n_combiner_folds,
                split_seed=stage_split,
                train_seed=stage_train,
                meta_learners=meta_learners,
                class_voter_kwargs=class_voter_kwargs,
                ensemble_selection_kwargs=ensemble_selection_kwargs,
                n_jobs=n_jobs,
            )
            methods = [self.best_combiner]
            logger.info("combiner='best' chose %s; training and applying only it.",
                        self.best_combiner)

        # --- stage 2 + 3 for each combiner ----------------------------------
        results: dict[str, PredictionResult] = {}
        voter_report = None
        for method in methods:
            if method == "meta_learner":
                self.train_meta_learner_stage(
                    meta_learners=meta_learners,
                    metric=meta_select_metric,
                    meta_cv_folds=meta_cv_folds,
                    split_seed=stage_split,
                    train_seed=stage_train,
                    n_jobs=n_jobs,
                )
                results[method] = self.predict_test(top_k=top_k, batch_size=batch_size)
            elif method == "simple_average":
                results[method] = self.predict_test_simple_average(
                    top_k=top_k, batch_size=batch_size,
                )
            elif method == "class_aware_voting":
                voter_report = self.train_class_voter_stage(
                    split_seed=stage_split, **(class_voter_kwargs or {}),
                )
                results[method] = self.predict_test_class_voter(
                    top_k=top_k, batch_size=batch_size,
                )
            elif method == "ensemble_selection":
                es_kwargs = dict(ensemble_selection_kwargs or {})
                es_kwargs.setdefault("metric", meta_select_metric)
                es_kwargs.setdefault("train_seed", stage_train)
                self.train_ensemble_selection_stage(**es_kwargs)
                results[method] = self.predict_test_ensemble_selection(
                    top_k=top_k, batch_size=batch_size,
                )
            else:  # one meta-learner, by model name (validated above)
                model = self._train_single_meta_learner(
                    method, meta_learners, meta_select_metric, meta_cv_folds,
                    stage_split, stage_train, n_jobs,
                )
                results[method] = self._predict_test_single_meta_learner(
                    method, model, top_k=top_k, batch_size=batch_size,
                )

        # The main result goes where every pipeline's predictions live. Written
        # last, so another combiner's stage 3 cannot overwrite it.
        main = results[methods[0]]
        save_predictions(main, self.output_dir / "predictions")
        self.predictions = main

        use_meta = "meta_learner" in results
        es_weights = (
            (self.ensemble_selection_metadata or {}).get("weights_by_model")
            if "ensemble_selection" in results else None
        )
        metadata = {
            "output_dir":         str(self.output_dir),
            "combiner_requested": requested,
            "combiner":           methods,
            "combiner_chosen":    self.best_combiner if choose_best else None,
            "combiner_comparison": ("combiner_comparison/combiner_comparison.csv"
                                    if choose_best else None),
            "oof_from":           str(self._oof_from) if self._oof_from else None,
            "n_text_models":      len(self.text_models),
            "n_tabular_models":   len(self.tabular_models),
            "n_folds":            self.n_folds,
            "split_seed":         self._split_seed,
            "train_seed":         self._train_seed,
            "best_meta_name":     getattr(self, "best_meta_name", None) if use_meta else None,
            "meta_select_metric": getattr(self, "meta_select_metric_used", None) if use_meta else None,
            "meta_scores":        self.meta_scores if use_meta else None,
            "class_voter":        _jsonable_report(voter_report),
            "ensemble_selection_weights": es_weights,
            "label2id":           self.label2id,
            "id2label":           {str(k): v for k, v in self.id2label.items()},
            "n_train":            n_train,
            "n_test":             n_test,
            "n_classes":          len(self.label2id),
        }
        with open(self.output_dir / "training_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        return {
            "predictions":                main,
            "combiner":                   methods,
            "combiner_predictions":       results,
            "combiner_chosen":            metadata["combiner_chosen"],
            "combiner_comparison":        comparison,
            "best_meta_name":             metadata["best_meta_name"],
            "meta_select_metric":         metadata["meta_select_metric"],
            "meta_scores":                metadata["meta_scores"],
            "class_voter":                voter_report,
            "ensemble_selection_weights": es_weights,
            "oof_meta_X":                 self.oof_meta_X,
            "oof_from":                   self._oof_from,
            "meta_learner":               self.meta_learner if use_meta else None,
            "label2id":                   self.label2id,
            "id2label":                   self.id2label,
            "output_dir":                 self.output_dir,
        }


def _jsonable_report(report: dict | None) -> dict | None:
    """The class voter's report, minus arrays, for training_metadata.json."""
    if report is None:
        return None
    out = {}
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            continue
        out[key] = value
    return out
