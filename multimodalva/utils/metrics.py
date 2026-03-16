"""
Shared evaluation metrics and Optuna sampling utility used by both
text/hpo.py and tabular/hpo.py.

Public API:
    csmf_accuracy(y_true, y_pred)           — WHO/InsilicoVA population-level metric
    cccsmf_accuracy(y_true, y_pred)         — chance-corrected CSMF accuracy
    score_predictions(top1_df, metric)      — scalar score from a top1 DataFrame
    log_loss_from_full(full_df, id2label)   — log loss from a full probability DataFrame
    sample_hyperparams(trial, search_space) — Optuna trial → hyperparameter dict

Constants:
    METRIC_DIRECTION  — maps metric name → "minimize" | "maximize" for study creation
"""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Metrics where lower is better.  All others default to "maximize".
# Used by text/hpo.py and tabular/hpo.py when creating Optuna studies and
# configuring Ray Tune's search direction.
METRIC_DIRECTION: dict[str, str] = {
    "log_loss": "minimize",
}


def csmf_accuracy(y_true, y_pred) -> float:
    """Compute CSMF (Cause-Specific Mortality Fraction) accuracy.

    The standard evaluation metric in verbal autopsy research, used by
    InsilicoVA, InterVA, and WHO benchmarks. Measures how well the model
    recovers population-level cause-of-death distributions rather than
    individual-level assignments.

    Formula:
        CSMF_accuracy = 1 - Σ|CSMF_pred_c - CSMF_true_c| /
                            (2 * (1 - min(CSMF_true_c)))

    Where CSMF_c is the fraction of all deaths assigned to cause c.
    The denominator normalises for the minimum true CSMF so that the score
    remains in [0, 1] regardless of class imbalance.

    Args:
        y_true: Array-like of true cause-of-death labels (strings or integers).
        y_pred: Array-like of predicted cause-of-death labels (same type as y_true).

    Returns:
        CSMF accuracy in [0, 1]; higher is better.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    csmf_true = np.array([(y_true == c).sum() / n for c in classes])
    csmf_pred = np.array([(y_pred == c).sum() / n for c in classes])

    min_true = csmf_true.min()
    # Degenerate case: all deaths from a single cause → denominator = 0.
    # Return 1.0 if predictions also assign all deaths to that cause, else 0.0.
    if min_true == 1.0:
        return 1.0 if np.allclose(csmf_true, csmf_pred) else 0.0

    return 1.0 - np.abs(csmf_pred - csmf_true).sum() / (2 * (1 - min_true))


# Expected CSMF accuracy under random (chance) assignment.
# Derived analytically for uniform random classifiers; standard constant
# used in the InsilicoVA / PHMRC verbal autopsy literature.
_CSMF_CHANCE = 0.632


def cccsmf_accuracy(y_true, y_pred) -> float:
    """Compute chance-corrected CSMF accuracy (CCCSMF).

    Adjusts the raw CSMF accuracy for the expected performance of a random
    classifier, analogous to Cohen's kappa for individual-level accuracy.
    Values above 0 indicate performance better than chance; 1.0 is perfect.

    Formula:
        CCCSMF_accuracy = (CSMF_accuracy - 0.632) / (1 - 0.632)

    The constant 0.632 is the expected CSMF accuracy under random assignment,
    as established in the PHMRC verbal autopsy benchmarking literature
    (Murray et al., 2011).

    Args:
        y_true: Array-like of true cause-of-death labels.
        y_pred: Array-like of predicted cause-of-death labels.

    Returns:
        Chance-corrected CSMF accuracy.  Values in (-∞, 1]; higher is better.
        Negative values indicate performance worse than random.
    """
    raw = csmf_accuracy(y_true, y_pred)
    return (raw - _CSMF_CHANCE) / (1.0 - _CSMF_CHANCE)


def score_predictions(top1_df, metric: str) -> float:
    """Compute a scalar score from a top1 predictions DataFrame.

    Works with the `top1` attribute of PredictionResult from either
    text.predict() or tabular.predict().

    Args:
        top1_df: DataFrame with columns ``true_label`` and ``predicted_label``.
                 Pass ``result.top1`` from a PredictionResult.
        metric:  One of "accuracy", "balanced_accuracy", "f1_macro",
                 "f1_weighted", "csmf_accuracy".

    Returns:
        Scalar score (higher is better for all supported metrics).

    Raises:
        ValueError: If metric is not recognised.
    """
    y_true = top1_df["true_label"]
    y_pred = top1_df["predicted_label"]
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric == "f1_weighted":
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if metric == "csmf_accuracy":
        return csmf_accuracy(y_true, y_pred)
    raise ValueError(
        f"Unknown metric '{metric}'. "
        "Choose from: accuracy, balanced_accuracy, f1_macro, f1_weighted, csmf_accuracy, log_loss."
        "  Note: log_loss requires the full probability DataFrame — use log_loss_from_full()."
    )


def log_loss_from_full(full_df: "pd.DataFrame", id2label: dict) -> float:
    """Compute log loss (cross-entropy) from a full probability DataFrame.

    Measures calibration quality: how well the predicted probability
    distributions match the true labels.  **Lower is better**; 0.0 is perfect.

    Unlike accuracy / F1 / CSMF, which only look at the argmax prediction,
    log loss penalises overconfident wrong predictions heavily.  This makes it
    especially important for ensemble methods (soft voting, stacking) that
    consume the raw probability matrix — poorly calibrated probabilities
    degrade ensemble performance even when individual top-1 accuracy is high.

    Args:
        full_df:   The ``full`` DataFrame from a ``PredictionResult``.
                   Expected columns: ``true_label``, ``prob_0``, ``prob_1``, …
                   Pass ``result.full`` directly.
        id2label:  Integer class ID → label string mapping.
                   Keys may be ``int`` or ``str`` (both handled).

    Returns:
        Log loss (cross-entropy) as a float ≥ 0.  Lower is better.

    Raises:
        ValueError: If ``full_df`` has no ``prob_*`` columns or lacks
                    ``true_label``, or if ``id2label`` is missing an entry.
    """
    import pandas as pd  # noqa: PLC0415
    from sklearn.metrics import log_loss  # noqa: PLC0415

    if "true_label" not in full_df.columns:
        raise ValueError("full_df must contain a 'true_label' column.")

    prob_cols = sorted(
        [c for c in full_df.columns if str(c).startswith("prob_")],
        key=lambda x: int(str(x).split("_")[1]),
    )
    if not prob_cols:
        raise ValueError(
            "full_df has no prob_* columns.  Pass result.full directly."
        )

    y_true  = full_df["true_label"]
    y_proba = full_df[prob_cols].values

    # Build ordered label list matching the prob column order
    labels = []
    for i in range(len(prob_cols)):
        label = id2label.get(i, id2label.get(str(i)))
        if label is None:
            raise ValueError(
                f"id2label has no entry for class index {i}.  "
                f"Available keys: {list(id2label.keys())[:10]}."
            )
        labels.append(label)

    return float(log_loss(y_true, y_proba, labels=labels))


def sample_hyperparams(trial: optuna.Trial, search_space: dict) -> dict:
    """Sample one set of hyperparameters from the search space for a given trial.

    Search space spec format (same convention used in text/hpo.py and tabular/hpo.py):
        ("float_log", low, high)   — log-uniform float; best for learning rates
        ("float",     low, high)   — uniform float
        ("int",       low, high)   — uniform int
        ("categorical", [values])  — discrete choices

    Args:
        trial:        Optuna Trial object.
        search_space: Dict mapping parameter name → spec tuple.

    Returns:
        Dict mapping parameter name → sampled value.

    Raises:
        ValueError: If a spec tuple has an unrecognised type string.
    """
    hp = {}
    for name, spec in search_space.items():
        kind = spec[0]
        if kind == "float_log":
            hp[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            hp[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            hp[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            hp[name] = trial.suggest_categorical(name, list(spec[1]))
        else:
            raise ValueError(f"Unknown search space type '{kind}' for parameter '{name}'.")
    return hp
