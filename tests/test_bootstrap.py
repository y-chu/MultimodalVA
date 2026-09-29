"""
Tests for the bootstrap confidence-interval helpers.

The important guarantee is that the fast confusion-matrix metrics agree exactly
with scikit-learn, since that is what makes 1000 resamples affordable. The rest
cover the input handling users will actually hit.

Run: pytest tests/test_bootstrap.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from multimodalva.results import bootstrap_ci, paired_bootstrap_ci, predictions_frame
from multimodalva.results.bootstrap import _confusion_metrics, _encode
from multimodalva.utils.metrics import cccsmf_accuracy, csmf_accuracy

CAUSES = ["HIV/AIDS", "Malaria", "Pneumonia", "Stroke", "Injury", "Other"]


def _demo_frame(n=240, seed=0, n_models=2):
    """Predictions with realistic class imbalance and unequal model skill."""
    rng = np.random.default_rng(seed)
    weights = np.array([0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
    truth = rng.choice(CAUSES, size=n, p=weights)
    frame = {"true_label": truth}
    for i, skill in enumerate(np.linspace(0.75, 0.45, n_models)):
        keep = rng.random(n) < skill
        noise = rng.choice(CAUSES, size=n)
        frame[f"model_{i}"] = np.where(keep, truth, noise)
    return pd.DataFrame(frame)


# ---------------------------------------------------------------------------
# Metric parity with scikit-learn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_confusion_metrics_match_sklearn(seed):
    frame = _demo_frame(seed=seed)
    metrics = [
        "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "precision_weighted", "recall_macro",
        "recall_weighted", "csmf_accuracy", "cccsmf_accuracy",
    ]
    y_true, y_pred, n_classes = _encode(frame, "true_label", ["model_0"])
    yp = y_pred["model_0"]

    got = _confusion_metrics(y_true, yp, n_classes, metrics)
    expected = {
        "accuracy": accuracy_score(y_true, yp),
        "balanced_accuracy": balanced_accuracy_score(y_true, yp),
        "f1_macro": f1_score(y_true, yp, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, yp, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, yp, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, yp, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, yp, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, yp, average="weighted", zero_division=0),
        "csmf_accuracy": csmf_accuracy(y_true, yp),
        "cccsmf_accuracy": cccsmf_accuracy(y_true, yp),
    }
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, abs=1e-12), name


def test_metric_parity_on_resamples():
    """Parity must hold on resamples too, where classes can go missing."""
    frame = _demo_frame(n=120, seed=7)
    y_true, y_pred, n_classes = _encode(frame, "true_label", ["model_0"])
    yp = y_pred["model_0"]
    rng = np.random.default_rng(11)

    for _ in range(25):
        idx = rng.integers(0, y_true.size, y_true.size)
        got = _confusion_metrics(y_true[idx], yp[idx], n_classes,
                                 ["f1_macro", "balanced_accuracy", "csmf_accuracy"])
        assert got["f1_macro"] == pytest.approx(
            f1_score(y_true[idx], yp[idx], average="macro", zero_division=0), abs=1e-12)
        assert got["balanced_accuracy"] == pytest.approx(
            balanced_accuracy_score(y_true[idx], yp[idx]), abs=1e-12)
        assert got["csmf_accuracy"] == pytest.approx(
            csmf_accuracy(y_true[idx], yp[idx]), abs=1e-12)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_interval_brackets_estimate_and_is_reproducible():
    frame = _demo_frame()
    first = bootstrap_ci(frame, n_boot=200, random_state=42)
    again = bootstrap_ci(frame, n_boot=200, random_state=42)
    pd.testing.assert_frame_equal(first, again)

    for model in first.index:
        for metric in ("accuracy", "f1_macro", "csmf_accuracy"):
            lo = first.loc[model, f"{metric}_ci_lower"]
            hi = first.loc[model, f"{metric}_ci_upper"]
            assert lo <= first.loc[model, metric] <= hi
            assert lo < hi


def test_point_estimate_matches_direct_computation():
    frame = _demo_frame()
    ci = bootstrap_ci(frame, n_boot=50, metrics=["accuracy"])
    direct = accuracy_score(frame["true_label"], frame["model_0"]) * 100
    assert ci.loc["model_0", "accuracy"] == pytest.approx(direct)


def test_wider_interval_with_fewer_cases():
    narrow = bootstrap_ci(_demo_frame(n=600, seed=3), n_boot=300, metrics=["accuracy"])
    wide = bootstrap_ci(_demo_frame(n=80, seed=3), n_boot=300, metrics=["accuracy"])
    span = lambda t: t.loc["model_0", "accuracy_ci_upper"] - t.loc["model_0", "accuracy_ci_lower"]
    assert span(wide) > span(narrow)


def test_long_and_formatted_layouts():
    frame = _demo_frame()
    long_df = bootstrap_ci(frame, n_boot=50, metrics=["accuracy", "f1_macro"],
                           long_format=True, formatted=True)
    assert set(long_df.columns) >= {"model", "metric", "estimate", "ci_lower",
                                    "ci_upper", "bootstrap_se", "n", "n_boot",
                                    "formatted"}
    assert len(long_df) == 4          # 2 models x 2 metrics
    assert "(" in long_df["formatted"].iloc[0]


def test_rejects_unknown_metric_and_bad_n_boot():
    frame = _demo_frame()
    with pytest.raises(ValueError, match="Unknown metric"):
        bootstrap_ci(frame, metrics=["auroc"])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(frame, n_boot=0)


# ---------------------------------------------------------------------------
# predictions_frame
# ---------------------------------------------------------------------------

def test_accepts_prediction_results_and_top1_frames():
    frame = _demo_frame()
    top1 = pd.DataFrame({"true_label": frame["true_label"],
                         "predicted_label": frame["model_0"]})
    built = predictions_frame({"a": top1, "b": top1})
    assert list(built.columns) == ["true_label", "a", "b"]
    assert (built["a"] == built["b"]).all()


def test_rejects_models_from_different_splits():
    frame = _demo_frame()
    a = pd.DataFrame({"true_label": frame["true_label"],
                      "predicted_label": frame["model_0"]})
    b = a.iloc[::-1].reset_index(drop=True)          # same rows, different order
    with pytest.raises(ValueError, match="different order"):
        predictions_frame({"a": a, "b": b})

    with pytest.raises(ValueError, match="test cases"):
        predictions_frame({"a": a, "b": a.iloc[:50]})


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        predictions_frame({})


# ---------------------------------------------------------------------------
# paired_bootstrap_ci
# ---------------------------------------------------------------------------

def test_paired_difference_matches_point_estimates():
    frame = _demo_frame(n_models=3)
    out = paired_bootstrap_ci(frame, model_a="model_0", model_b="model_2",
                              n_boot=200, metrics=["accuracy"])
    assert len(out) == 1
    row = out.iloc[0]
    direct = (accuracy_score(frame["true_label"], frame["model_0"])
              - accuracy_score(frame["true_label"], frame["model_2"])) * 100
    assert row["difference"] == pytest.approx(direct)
    assert row["ci_lower"] <= row["difference"] <= row["ci_upper"]


def test_paired_detects_a_real_difference():
    """model_0 is built to be clearly better than model_2."""
    frame = _demo_frame(n=500, n_models=3)
    out = paired_bootstrap_ci(frame, model_a="model_0", model_b="model_2",
                              n_boot=400, metrics=["accuracy"])
    row = out.iloc[0]
    assert row["ci_lower"] > 0            # interval excludes zero
    assert row["prob_reversed"] < 0.05


def test_identical_models_give_zero_difference():
    frame = _demo_frame()
    frame["copy"] = frame["model_0"]
    out = paired_bootstrap_ci(frame, model_a="model_0", model_b="copy",
                              n_boot=100, metrics=["accuracy", "f1_macro"])
    assert (out["difference"].abs() < 1e-12).all()
    assert (out["ci_lower"].abs() < 1e-12).all()
    assert (out["ci_upper"].abs() < 1e-12).all()


def test_reference_mode_compares_every_model():
    frame = _demo_frame(n_models=3)
    out = paired_bootstrap_ci(frame, reference="model_2", n_boot=100,
                              metrics=["accuracy"])
    assert set(out["model"]) == {"model_0", "model_1"}
    assert (out["compared_with"] == "model_2").all()


def test_requires_a_comparison_target():
    frame = _demo_frame()
    with pytest.raises(ValueError, match="reference"):
        paired_bootstrap_ci(frame, n_boot=10)
    with pytest.raises(ValueError, match="both model_a and model_b"):
        paired_bootstrap_ci(frame, model_a="model_0", n_boot=10)
    with pytest.raises(ValueError, match="not found"):
        paired_bootstrap_ci(frame, model_a="model_0", model_b="ghost", n_boot=10)


def test_shared_resamples_across_models():
    """Same seed must give the same draws, whichever models are present."""
    frame = _demo_frame(n_models=3)
    two = bootstrap_ci(frame[["true_label", "model_0", "model_1"]],
                       n_boot=100, metrics=["accuracy"])
    three = bootstrap_ci(frame, n_boot=100, metrics=["accuracy"])
    assert two.loc["model_0", "accuracy_ci_lower"] == pytest.approx(
        three.loc["model_0", "accuracy_ci_lower"])
def test_bootstrap_indices_are_batched_instead_of_allocating_b_by_n():
    from multimodalva.results.bootstrap import _index_batches

    n, n_boot = 10_000, 10_000
    start, first = next(_index_batches(n, n_boot, 42, target_bytes=80_000))
    assert start == 0
    assert first.shape[1] == n
    assert first.shape[0] < n_boot
    assert first.nbytes <= 80_000
