"""Tests for the calibration metrics (ECE, MCE, Brier).

These definitions are the ones the manuscript reports, so the arithmetic is
pinned here rather than left to inspection: a perfectly calibrated set of
probabilities must score ~0, a confidently wrong one must score near the
worst case, and the cause-macro weighting must not let a common cause mask a
rare one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multimodalva.results.calibration import (
    brier_multiclass,
    calibration_summary,
    classwise_bin_data,
    ece_score,
    mce_score,
)


def _full_df(y_true_idx, y_prob, id2label):
    """Build a PredictionResult.full-shaped frame."""
    cols = {f"prob_{i}": y_prob[:, i] for i in range(y_prob.shape[1])}
    cols["true_label"] = [id2label[i] for i in y_true_idx]
    return pd.DataFrame(cols)


def test_perfect_confident_predictions_are_well_calibrated():
    """One-hot probabilities that are always right: every metric at its floor."""
    n, k = 60, 3
    rng = np.random.default_rng(0)
    y_idx = rng.integers(0, k, n)
    y_prob = np.zeros((n, k))
    y_prob[np.arange(n), y_idx] = 1.0

    y_bin = np.zeros_like(y_prob)
    y_bin[np.arange(n), y_idx] = 1.0

    bins = classwise_bin_data(y_bin, y_prob, ["a", "b", "c"])
    assert ece_score(bins, n) == pytest.approx(0.0, abs=1e-12)
    assert mce_score(bins) == pytest.approx(0.0, abs=1e-12)
    assert brier_multiclass(y_bin, y_prob) == pytest.approx(0.0, abs=1e-12)


def test_confidently_wrong_predictions_are_badly_calibrated():
    """Always certain and always wrong: maximum possible gap."""
    n, k = 40, 2
    y_idx = np.zeros(n, dtype=int)
    y_prob = np.tile([0.0, 1.0], (n, 1)).astype(float)  # certain of class 1

    y_bin = np.zeros_like(y_prob)
    y_bin[np.arange(n), y_idx] = 1.0

    bins = classwise_bin_data(y_bin, y_prob, ["a", "b"])
    assert mce_score(bins) == pytest.approx(1.0)
    assert brier_multiclass(y_bin, y_prob) == pytest.approx(1.0)


def test_ece_weights_every_cause_equally():
    """A rare, badly calibrated cause must not be hidden by a common good one.

    Cause 'a' is 90 of 100 rows and predicted perfectly. Cause 'b' is the other
    10, and the model always says its probability is 0, so it is missed
    entirely. Worked by hand:

    * cause a — the 90 'a' rows sit at p=1 with observed frequency 1 (gap 0),
      the 10 'b' rows sit at p=0 with observed frequency 0 (gap 0) → 0.0
    * cause b — all 100 rows sit at p=0, observed frequency 10/100 = 0.1
      (gap 0.1, weight 100/100) → 0.1

    Cause-macro ECE averages those two: 0.05. Weighting causes by prevalence
    instead would give 0.9·0 + 0.1·0.1 = 0.01, five times smaller — which is the
    reason the manuscript uses the macro form.
    """
    n_a, n_b = 90, 10
    y_bin = np.zeros((n_a + n_b, 2))
    y_bin[:n_a, 0] = 1.0
    y_bin[n_a:, 1] = 1.0

    y_prob = np.zeros((n_a + n_b, 2))
    y_prob[:n_a, 0] = 1.0            # cause a: right and certain
    y_prob[n_a:, 0] = 0.0
    y_prob[n_a:, 1] = 0.0            # cause b: certain it is absent, but present

    bins = classwise_bin_data(y_bin, y_prob, ["a", "b"])
    ece = ece_score(bins, n_a + n_b)
    assert ece == pytest.approx(0.05)

    per_cause = bins.assign(w=lambda d: d["n"] / (n_a + n_b) * d["absolute_gap"])
    prevalence_weighted = (
        per_cause.groupby("cause")["w"].sum() * np.array([n_a, n_b]) / (n_a + n_b)
    ).sum()
    assert ece > prevalence_weighted


def test_calibration_summary_from_full_dataframe():
    """The public entry point works on a PredictionResult.full-shaped frame."""
    id2label = {0: "cardiac", 1: "hiv", 2: "injury"}
    rng = np.random.default_rng(7)
    n = 120
    y_idx = rng.integers(0, 3, n)
    y_prob = rng.dirichlet(np.ones(3), size=n)

    out = calibration_summary(_full_df(y_idx, y_prob, id2label), id2label)
    assert set(out) >= {"ECE", "MCE", "Brier", "n", "n_causes"}
    assert out["n"] == n and out["n_causes"] == 3
    for key in ("ECE", "MCE", "Brier"):
        assert 0.0 <= out[key] <= 1.0


def test_only_brier_gets_a_confidence_interval():
    """ECE and MCE must never be given an interval.

    Both are biased upward on small samples, and a bootstrap resample is a small
    sample, so an interval would describe a different sample size from the point
    estimate printed beside it. See FAQ.md for the measurements and for how to
    build one deliberately if a study calls for it.
    """
    id2label = {0: "cardiac", 1: "hiv"}
    rng = np.random.default_rng(3)
    n = 200
    y_idx = rng.integers(0, 2, n)
    y_prob = rng.dirichlet(np.ones(2), size=n)

    out = calibration_summary(
        _full_df(y_idx, y_prob, id2label), id2label, n_boot=200, random_state=1
    )
    assert out["Brier_lo"] <= out["Brier"] <= out["Brier_hi"]
    for absent in ("ECE_lo", "ECE_hi", "MCE_lo", "MCE_hi"):
        assert absent not in out, f"{absent} must not be reported"




def test_no_ci_keys_without_bootstrap():
    id2label = {0: "a", 1: "b"}
    rng = np.random.default_rng(11)
    y_prob = rng.dirichlet(np.ones(2), size=30)
    out = calibration_summary(_full_df(rng.integers(0, 2, 30), y_prob, id2label), id2label)
    assert not any(k.endswith(("_lo", "_hi")) for k in out)


def test_mismatched_label_map_is_reported_clearly():
    id2label = {0: "a", 1: "b"}
    rng = np.random.default_rng(5)
    df = _full_df(rng.integers(0, 2, 20), rng.dirichlet(np.ones(2), size=20), id2label)
    df.loc[0, "true_label"] = "not_a_cause"
    with pytest.raises(ValueError, match="not in id2label"):
        calibration_summary(df, id2label)


def test_missing_probability_columns_are_reported_clearly():
    id2label = {0: "a", 1: "b"}
    df = pd.DataFrame({"true_label": ["a", "b"], "prob_0": [0.5, 0.5]})
    with pytest.raises(ValueError, match="missing probability columns"):
        calibration_summary(df, id2label)
