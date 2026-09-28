"""
Greedy ensemble selection — a combiner for stacking Stage 2.

Caruana, Niculescu-Mizil, Crew & Ksikes (2004), "Ensemble Selection from
Libraries of Models" (ICML).  The same algorithm AutoGluon uses for its
WeightedEnsemble.

Starting from an empty ensemble, each iteration adds the base model whose
inclusion most improves the metric of the averaged out-of-fold predictions.
Models are added *with replacement*, so a model picked k times out of N gets
weight k / N.  The weights are non-negative and sum to 1, so a weighted
average of probability matrices is itself a probability matrix.

This module has no dependency on the rest of the ensemble pipeline: it works on
plain arrays, so it can combine OOF predictions produced anywhere.
:class:`~multimodalva.ensemble.stacking.StackingClassifier` wires it into the
stage API (``train_ensemble_selection_stage()`` /
``predict_test_ensemble_selection()``).

Public API:
    EnsembleSelection
        Fit weights on OOF predictions, apply them to test predictions.
    cross_validate_combiner(oof_preds, y_true, combiner_factory, ...)
        Nested CV over OOF rows: fit a combiner on k-1 folds, score the held-out
        fold.  Fitting a combiner on the full OOF and scoring on the same OOF
        is in-sample; this gives the out-of-sample estimate.

Layouts:
    classification  (n_models, n_samples, n_classes) probabilities, or
                    (n_models, n_samples) positive-class probabilities (binary)
    regression      (n_models, n_samples)

    ``y_true`` for classification holds class *positions* 0..n_classes-1 along
    the last axis (for binary input: 0 or 1).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..utils.metrics import METRIC_DIRECTION, score_predictions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric handling
# ---------------------------------------------------------------------------

def _resolve_scorer(
    metric: str | Callable,
    greater_is_better: bool | None,
    problem_type: str,
) -> tuple[Callable[[np.ndarray, np.ndarray], float], bool]:
    """Return ``(score_fn, greater_is_better)`` with ``score_fn(y, preds)``.

    ``score_fn`` returns the metric in its natural units (log loss stays a
    positive loss).  Callers that maximize multiply by ``+1`` / ``-1``.

    * A package metric name (``accuracy``, ``f1_macro``, ``csmf_accuracy``, …)
      is label-based: combined probabilities are reduced to their argmax class
      and scored with :func:`~multimodalva.utils.metrics.score_predictions`.
    * ``"log_loss"`` scores the probabilities directly; lower is better.
    * A callable receives ``(y_true, combined_preds)`` unchanged — probabilities
      for classification, values for regression — and ``greater_is_better``
      (default ``True``) states its direction.
    """
    if problem_type not in {"classification", "regression"}:
        raise ValueError("problem_type must be 'classification' or 'regression'.")

    if callable(metric):
        gib = True if greater_is_better is None else bool(greater_is_better)
        return (lambda y, p: float(metric(y, p))), gib

    if problem_type == "regression":
        raise ValueError(
            "Regression needs a callable metric (e.g. an RMSE function) with "
            "greater_is_better=False; the package's named metrics are for "
            "classification."
        )

    natural_gib = METRIC_DIRECTION.get(metric, "maximize") == "maximize"
    if greater_is_better is not None and bool(greater_is_better) != natural_gib:
        raise ValueError(
            f"greater_is_better={greater_is_better} contradicts the direction of "
            f"metric '{metric}' ({METRIC_DIRECTION.get(metric, 'maximize')})."
        )

    if metric == "log_loss":
        from sklearn.metrics import log_loss

        def score_fn(y, p):
            p = _as_class_matrix(p)
            return float(log_loss(y, p, labels=np.arange(p.shape[1])))

        return score_fn, False

    # Label-based metric: validated once here rather than on every call.
    score_predictions(pd.DataFrame({"true_label": [0], "predicted_label": [0]}), metric)

    def score_fn(y, p):
        pred = np.argmax(_as_class_matrix(p), axis=1)
        top1 = pd.DataFrame({"true_label": y, "predicted_label": pred})
        return float(score_predictions(top1, metric=metric))

    return score_fn, True


def _as_class_matrix(p: np.ndarray) -> np.ndarray:
    """(n,) positive-class probabilities → (n, 2); (n, c) returned unchanged."""
    if p.ndim == 1:
        return np.column_stack([1.0 - p, p])
    return p


def _check_layout(preds: np.ndarray, problem_type: str) -> np.ndarray:
    preds = np.asarray(preds, dtype=float)
    if problem_type == "regression" and preds.ndim != 2:
        raise ValueError(
            f"Regression predictions must be (n_models, n_samples); got {preds.shape}."
        )
    if problem_type == "classification" and preds.ndim not in (2, 3):
        raise ValueError(
            "Classification predictions must be (n_models, n_samples, n_classes) "
            f"or (n_models, n_samples) for binary; got {preds.shape}."
        )
    if preds.shape[0] < 1:
        raise ValueError("At least one model is required.")
    return preds


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

class EnsembleSelection:
    """Greedy forward ensemble selection with replacement (Caruana et al. 2004).

    Args:
        metric:                 Package metric name (``f1_macro`` default,
                                ``accuracy``, ``balanced_accuracy``,
                                ``f1_weighted``, ``csmf_accuracy``,
                                ``log_loss``) or a callable
                                ``metric(y_true, combined_preds)``.
        ensemble_size:          Greedy iterations.  Default 100.
        use_best_in_trajectory: Keep the counts from the iteration with the best
                                OOF score instead of those after the last
                                iteration.  Default ``True``.
        sorted_init:            Seed the ensemble with the top-k models by
                                individual OOF score before the greedy loop.
                                Default 0 (empty start).
        n_bags:                 Bagged ensemble selection: run the selection on
                                ``n_bags`` random model subsets and average the
                                weight vectors.  Default 1 — one run over all
                                models; ``bag_fraction`` is then not used.
        bag_fraction:           Share of models drawn (without replacement) for
                                each bag, rounded up.  Default 0.5.
        random_state:           Seed for the bag draws.  Default 42.
        greater_is_better:      Direction of a callable ``metric``.  For a named
                                metric the direction is known and this may be
                                left ``None``.
        problem_type:           ``"classification"`` (default) or
                                ``"regression"``.

    Attributes (after :meth:`fit`):
        weights_:               (n_models,) non-negative, sums to 1.
        weights_by_model_:      ``{model_name: weight}`` for every model.
        selected_models_:       Names with non-zero weight, highest weight first.
        trajectory_:            (n_bags, n_steps) OOF score after each step, in
                                the metric's natural units.  With
                                ``sorted_init > 0`` the first step is the seeded
                                ensemble.
        oof_score_:             Score of the weighted average on the OOF rows it
                                was fitted on (in-sample — use
                                :func:`cross_validate_combiner` to compare
                                combiners).
        best_single_score_:     Best individual model's OOF score.
        best_single_model_:     Its name.
        simple_average_score_:  OOF score of the equal-weight average.
        greater_is_better_:     Direction used for the scores above.
    """

    def __init__(
        self,
        metric: str | Callable = "f1_macro",
        ensemble_size: int = 100,
        use_best_in_trajectory: bool = True,
        sorted_init: int = 0,
        n_bags: int = 1,
        bag_fraction: float = 0.5,
        random_state: int | None = 42,
        greater_is_better: bool | None = None,
        problem_type: str = "classification",
    ):
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")
        if sorted_init < 0:
            raise ValueError("sorted_init must be >= 0.")
        if n_bags < 1:
            raise ValueError("n_bags must be >= 1.")
        if not (0.0 < bag_fraction <= 1.0):
            raise ValueError("bag_fraction must be in (0, 1].")
        self.metric = metric
        self.ensemble_size = int(ensemble_size)
        self.use_best_in_trajectory = bool(use_best_in_trajectory)
        self.sorted_init = int(sorted_init)
        self.n_bags = int(n_bags)
        self.bag_fraction = float(bag_fraction)
        self.random_state = random_state
        self.greater_is_better = greater_is_better
        self.problem_type = problem_type

    # ------------------------------------------------------------------
    def fit(
        self,
        oof_preds: np.ndarray,
        y_true: np.ndarray,
        model_names: list[str] | None = None,
    ) -> "EnsembleSelection":
        """Learn the model weights from OOF predictions; returns ``self``."""
        preds = _check_layout(oof_preds, self.problem_type)
        y = np.asarray(y_true)
        if preds.shape[1] != len(y):
            raise ValueError(
                f"oof_preds has {preds.shape[1]} samples but y_true has {len(y)}."
            )
        n_models = preds.shape[0]
        if model_names is None:
            model_names = [f"model_{i}" for i in range(n_models)]
        model_names = [str(n) for n in model_names]
        if len(model_names) != n_models:
            raise ValueError(
                f"{len(model_names)} model_names for {n_models} models."
            )
        if len(set(model_names)) != n_models:
            raise ValueError("model_names must be unique (weights are reported by name).")

        score_fn, gib = _resolve_scorer(self.metric, self.greater_is_better, self.problem_type)
        sign = 1.0 if gib else -1.0

        def maximize(p):
            s = sign * score_fn(y, p)
            return s if np.isfinite(s) else -np.inf

        if self.n_bags == 1:
            bags = [np.arange(n_models)]
        else:
            rng = np.random.default_rng(self.random_state)
            bag_size = max(1, math.ceil(self.bag_fraction * n_models))
            # Sorted so ties inside a bag still go to the lowest original index.
            bags = [
                np.sort(rng.choice(n_models, size=bag_size, replace=False))
                for _ in range(self.n_bags)
            ]

        weights = np.zeros(n_models, dtype=float)
        trajectories = []
        for idx in bags:
            bag_weights, traj = self._select(preds[idx], maximize)
            weights[idx] += bag_weights
            trajectories.append(sign * np.asarray(traj))
        weights /= len(bags)
        weights /= weights.sum()  # exact 1 after float averaging

        self.n_models_ = n_models
        self.model_names_ = model_names
        self.weights_ = weights
        self.weights_by_model_ = dict(zip(model_names, weights.tolist()))
        order = np.argsort(-weights, kind="stable")
        self.selected_models_ = [model_names[i] for i in order if weights[i] > 0]
        self.trajectory_ = np.vstack(trajectories)
        self.bags_ = [b.tolist() for b in bags]
        self.greater_is_better_ = gib

        singles = [score_fn(y, preds[m]) for m in range(n_models)]
        best = int(np.argmax(sign * np.asarray(singles)))
        self.best_single_model_ = model_names[best]
        self.best_single_score_ = float(singles[best])
        self.simple_average_score_ = float(score_fn(y, preds.mean(axis=0)))
        self.oof_score_ = float(score_fn(y, self.predict(preds)))
        return self

    def _select(self, preds: np.ndarray, maximize: Callable) -> tuple[np.ndarray, list]:
        """One greedy run over ``preds`` (a bag). Returns weights and trajectory."""
        n_models = preds.shape[0]
        running = np.zeros_like(preds[0])
        counts = np.zeros(n_models, dtype=int)
        history: list[np.ndarray] = []
        trajectory: list[float] = []

        if self.sorted_init > 0:
            singles = [maximize(preds[m]) for m in range(n_models)]
            # Stable sort on the negated score: ties keep the lower index first.
            seed = np.argsort(-np.asarray(singles), kind="stable")[: self.sorted_init]
            for m in seed:
                running += preds[m]
                counts[m] += 1
            trajectory.append(maximize(running / counts.sum()))
            history.append(counts.copy())

        for _ in range(self.ensemble_size):
            n_in = counts.sum()
            best_m, best_s = 0, -np.inf
            for m in range(n_models):
                s = maximize((running + preds[m]) / (n_in + 1))
                if s > best_s:  # strict: ties go to the lowest index
                    best_m, best_s = m, s
            running += preds[best_m]
            counts[best_m] += 1
            trajectory.append(best_s)
            history.append(counts.copy())

        if self.use_best_in_trajectory:
            # argmax returns the first maximum: the smallest ensemble reaching it.
            final = history[int(np.argmax(trajectory))]
        else:
            final = history[-1]
        return final / final.sum(), trajectory

    # ------------------------------------------------------------------
    def predict(self, test_preds: np.ndarray) -> np.ndarray:
        """Weighted sum over the model axis.

        ``test_preds`` has the same layout as the OOF predictions passed to
        :meth:`fit`.  Returns (n_samples, n_classes) for multiclass input and
        (n_samples,) for binary positive-class or regression input.
        """
        if not hasattr(self, "weights_"):
            raise RuntimeError("EnsembleSelection is not fitted; call fit() first.")
        preds = _check_layout(test_preds, self.problem_type)
        if preds.shape[0] != self.n_models_:
            raise ValueError(
                f"test_preds has {preds.shape[0]} models; fitted on {self.n_models_}."
            )
        return np.tensordot(self.weights_, preds, axes=(0, 0))

    def __repr__(self) -> str:
        return (
            f"EnsembleSelection(metric={self.metric!r}, "
            f"ensemble_size={self.ensemble_size}, n_bags={self.n_bags})"
        )


# ---------------------------------------------------------------------------
# Nested CV for combiners
# ---------------------------------------------------------------------------

def cross_validate_combiner(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    combiner_factory: Callable[[], Any],
    n_splits: int = 5,
    random_state: int | None = 42,
    metric: str | Callable = "f1_macro",
    greater_is_better: bool | None = None,
    problem_type: str = "classification",
) -> dict:
    """Out-of-sample score of a combiner, by cross-validating over OOF rows.

    Each fold fits a fresh combiner from ``combiner_factory()`` on the other
    folds' rows and scores its ``predict()`` on the held-out rows.  Folds are
    stratified for classification and depend only on ``y_true``,
    ``n_splits`` and ``random_state`` — so combiners compared with the same
    arguments are scored on identical folds.

    If a class has fewer rows than ``n_splits``, the fold count is reduced to
    the smallest class count (as the class-voter fallback CV does); fewer than 2
    possible folds raises ``ValueError``.

    Args:
        oof_preds:        Same layouts as :class:`EnsembleSelection`.
        y_true:           Labels aligned with the sample axis.
        combiner_factory: Zero-argument callable returning an unfitted object
                          with ``fit(oof_preds, y_true)`` and
                          ``predict(preds)``.
        n_splits:         Requested folds.  Default 5.
        random_state:     Fold shuffle seed.  Default 42.
        metric / greater_is_better / problem_type:
                          How held-out folds are scored; same meaning as in
                          :class:`EnsembleSelection`.

    Returns:
        dict with ``mean``, ``std`` (population std over folds),
        ``fold_scores``, ``n_splits_used``, ``metric``, ``greater_is_better``.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    preds = _check_layout(oof_preds, problem_type)
    y = np.asarray(y_true)
    score_fn, gib = _resolve_scorer(metric, greater_is_better, problem_type)

    if problem_type == "classification":
        _, counts = np.unique(y, return_counts=True)
        folds = min(int(n_splits), int(counts.min()))
        if folds < 2:
            raise ValueError(
                f"Stratified CV not possible: smallest class has {counts.min()} row(s)."
            )
        if folds < n_splits:
            logger.warning(
                "cross_validate_combiner: n_splits reduced %d → %d (smallest class count).",
                n_splits, folds,
            )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        folds = int(n_splits)
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    fold_scores = []
    for tr, va in cv.split(np.zeros(len(y)), y):
        combiner = combiner_factory()
        combiner.fit(preds[:, tr], y[tr])
        fold_scores.append(float(score_fn(y[va], combiner.predict(preds[:, va]))))

    return {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "n_splits_used": int(folds),
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        "greater_is_better": gib,
    }
