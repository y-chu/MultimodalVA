"""Calibration metrics: how well predicted probabilities match observed outcomes.

Accuracy and F1 only look at the top-ranked cause. These look at the probability
attached to it, which is what matters when a downstream user acts on the number
rather than the label.

    ECE   Expected Calibration Error — average gap between predicted probability
          and observed frequency, computed per cause and then averaged so that
          every cause counts equally regardless of how common it is.
    MCE   Maximum Calibration Error — the single worst cause-bin gap. Sensitive
          to sparse bins by design: it is the worst case, not the typical one.
    Brier Mean squared error of the probabilities, averaged over causes. Lower
          is better for all three.

Typical use, straight from a pipeline result::

    from multimodalva.results import calibration_summary

    res = run(task="text", ...)
    calibration_summary(res["predictions"].full, res["predictions"].id2label)
    # {'ECE': 0.0412, 'MCE': 0.3810, 'Brier': 0.0234, 'n': 1000, 'n_causes': 19}

Brier accepts a bootstrap confidence interval::

    calibration_summary(full_df, id2label, n_boot=2000)
    # {'Brier': 0.0224, 'Brier_lo': 0.0209, 'Brier_hi': 0.0239, ...}

**ECE and MCE are deliberately reported without a confidence interval.** Both are
biased upward at small sample sizes, and a bootstrap resample holds only about
63% distinct rows, so it behaves like a smaller dataset and scores higher than
the full sample. Measured on 1000 held-out cases: ECE point 0.0121 against a
bootstrap mean of 0.0152 (+25%), MCE 0.635 against 0.722 (+14%), while accuracy,
Brier and CSMF accuracy come back unbiased under the very same resamples. The
shift is large enough that the interval can sit entirely above the point
estimate, so quoting one would misstate the uncertainty rather than describe it.

Why the two behave differently: ECE averages
``|observed frequency - predicted probability|``, and an absolute value cannot
cancel noise, so a noisier observed frequency can only push the score up. MCE
takes the single worst bin, which is the sparsest and therefore the noisiest.
Brier is a plain mean, so its noise cancels. Sampling *without* replacement at
half the rows reproduces the same upward shift, which confirms the cause is the
sample size rather than the duplicated rows.

If you need an interval for ECE anyway, the FAQ ("Why is there no confidence
interval for ECE and MCE?") shows what to compute it from and gives runnable
code, so the choice of method stays yours and is stated explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

__all__ = [
    "classwise_bin_data",
    "ece_score",
    "mce_score",
    "brier_multiclass",
    "calibration_summary",
]

DEFAULT_N_BINS = 10


def classwise_bin_data(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str],
    n_bins: int = DEFAULT_N_BINS,
) -> pd.DataFrame:
    """Return one row per non-empty cause-specific probability bin.

    Each cause is binned separately on its own predicted probability, which is
    what makes the resulting ECE a cause-macro quantity rather than a
    top-label one.

    Args:
        y_true_bin: One-hot true labels, shape ``(n_samples, n_classes)``.
        y_prob:     Predicted probabilities, same shape.
        classes:    Cause names, ordered to match the columns.
        n_bins:     Number of equal-width probability bins. Default 10.

    Returns:
        DataFrame with ``cause``, ``bin``, ``bin_lower``, ``bin_upper``, ``n``,
        ``mean_probability``, ``observed_frequency`` and ``absolute_gap``.
        Causes that never occur in ``y_true_bin`` are omitted.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    for c, cause in enumerate(classes):
        if y_true_bin[:, c].sum() == 0:
            continue
        for bin_index in range(n_bins):
            lo, hi = edges[bin_index], edges[bin_index + 1]
            # The first bin includes its lower edge so probabilities of exactly
            # 0 are counted; every later bin is left-open to avoid double count.
            if bin_index == 0:
                mask = (y_prob[:, c] >= lo) & (y_prob[:, c] <= hi)
            else:
                mask = (y_prob[:, c] > lo) & (y_prob[:, c] <= hi)
            if not mask.any():
                continue
            mean_probability = float(y_prob[mask, c].mean())
            observed_frequency = float(y_true_bin[mask, c].mean())
            rows.append(
                {
                    "cause": cause,
                    "bin": bin_index + 1,
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "n": int(mask.sum()),
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                    "absolute_gap": abs(observed_frequency - mean_probability),
                }
            )
    return pd.DataFrame(rows)


def ece_score(bin_data: pd.DataFrame, n_test: int) -> float:
    """Cause-macro Expected Calibration Error; each cause contributes equally.

    Within a cause, bins are weighted by how many samples fall in them; across
    causes the per-cause values are averaged unweighted, so a rare cause with
    badly calibrated probabilities is not hidden by a common well-calibrated one.

    Args:
        bin_data: Output of :func:`classwise_bin_data`.
        n_test:   Number of samples the bins were built from.

    Returns:
        ECE in [0, 1]; lower is better. NaN when ``bin_data`` is empty.
    """
    if bin_data.empty:
        return float("nan")
    per_cause = bin_data.assign(
        weighted_gap=lambda d: d["n"] / n_test * d["absolute_gap"]
    ).groupby("cause")["weighted_gap"].sum()
    return float(per_cause.mean())


def mce_score(bin_data: pd.DataFrame) -> float:
    """Largest cause-bin calibration gap; sparse-bin sensitive by design.

    Args:
        bin_data: Output of :func:`classwise_bin_data`.

    Returns:
        MCE in [0, 1]; lower is better. NaN when ``bin_data`` is empty.
    """
    return float(bin_data["absolute_gap"].max()) if len(bin_data) else float("nan")


def brier_multiclass(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    """Multi-class Brier score: mean over all classes of the per-class score.

    Args:
        y_true_bin: One-hot true labels, shape ``(n_samples, n_classes)``.
        y_prob:     Predicted probabilities, same shape.

    Returns:
        Brier score; lower is better, 0.0 is perfect.
    """
    scores = [
        brier_score_loss(y_true_bin[:, c], y_prob[:, c])
        for c in range(y_prob.shape[1])
    ]
    return float(np.mean(scores))


def _unpack_full(full_df: pd.DataFrame, id2label: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Turn a ``PredictionResult.full`` table into one-hot labels + probabilities."""
    ids = sorted(int(k) for k in id2label)
    classes = [id2label[k] if k in id2label else id2label[str(k)] for k in ids]

    prob_cols = [f"prob_{i}" for i in ids]
    missing = [c for c in prob_cols if c not in full_df.columns]
    if missing:
        raise ValueError(
            f"full_df is missing probability columns {missing[:5]}. "
            "Pass the `full` table from a PredictionResult together with its id2label."
        )
    if "true_label" not in full_df.columns:
        raise ValueError("full_df must contain a 'true_label' column.")

    y_prob = full_df[prob_cols].to_numpy(dtype=float)
    label_to_col = {name: j for j, name in enumerate(classes)}
    y_true = full_df["true_label"].map(label_to_col)
    if y_true.isna().any():
        unknown = sorted(set(full_df.loc[y_true.isna(), "true_label"].astype(str)))[:5]
        raise ValueError(
            f"true_label values {unknown} are not in id2label. "
            "The predictions and the label map must come from the same run."
        )
    y_true_bin = np.zeros_like(y_prob)
    y_true_bin[np.arange(len(y_true)), y_true.to_numpy(dtype=int)] = 1.0
    return y_true_bin, y_prob, classes


def calibration_summary(
    full_df: pd.DataFrame,
    id2label: dict,
    *,
    n_bins: int = DEFAULT_N_BINS,
    n_boot: int = 0,
    ci: float = 95.0,
    random_state: int = 42,
) -> dict[str, float]:
    """ECE, MCE and Brier for one set of predictions, optionally with a CI.

    Args:
        full_df:      The ``full`` table from a ``PredictionResult`` — a
                      ``true_label`` column plus ``prob_0``, ``prob_1``, ….
        id2label:     Integer class ID → cause name, from the same run.
        n_bins:       Probability bins per cause. Default 10.
        n_boot:       Bootstrap resamples used for the Brier interval. ``0``
                      (default) skips them. ECE and MCE never receive an
                      interval — see the module docstring for why, and the FAQ
                      for how to build one yourself if you need it.
        ci:           Confidence level in percent. Default 95.
        random_state: Seed for the bootstrap resampling.

    Returns:
        ``{"ECE": …, "MCE": …, "Brier": …, "n": …, "n_causes": …}``, plus
        ``Brier_lo`` and ``Brier_hi`` when ``n_boot > 0``.

    Raises:
        ValueError: If the frame and the label map do not line up.
    """
    y_true_bin, y_prob, classes = _unpack_full(full_df, id2label)
    n = len(full_df)

    def _point(idx: np.ndarray) -> tuple[float, float, float]:
        yt, yp = y_true_bin[idx], y_prob[idx]
        bins = classwise_bin_data(yt, yp, classes, n_bins)
        return ece_score(bins, len(idx)), mce_score(bins), brier_multiclass(yt, yp)

    all_idx = np.arange(n)
    ece, mce, brier = _point(all_idx)
    out: dict[str, float] = {
        "ECE": ece, "MCE": mce, "Brier": brier,
        "n": float(n), "n_causes": float(len(classes)),
    }
    if n_boot <= 0:
        return out

    # Only Brier is resampled. ECE and MCE are biased upward at smaller
    # effective sample sizes, which is exactly what a bootstrap resample is, so
    # an interval built this way would describe a different quantity from the
    # point estimate it is printed next to.
    rng = np.random.default_rng(random_state)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = brier_multiclass(y_true_bin[idx], y_prob[idx])

    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    out["Brier_lo"] = float(np.percentile(draws, lo_q))
    out["Brier_hi"] = float(np.percentile(draws, hi_q))
    return out
