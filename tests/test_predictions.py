"""Tests for the shared prediction-table builder.

Five pipelines used to render their probability matrix into top1/full/topk
independently. These tests pin the behaviour of the single implementation that
replaced them, including the parts that differed between the old copies:
tie-breaking, the optional ``id`` column, and unlabelled input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multimodalva.utils.predictions import assemble_predictions, save_predictions


def _id2label(n):
    return {i: f"cause_{i}" for i in range(n)}


def test_columns_and_ordering():
    """top1/full/topk carry the documented columns in the documented order."""
    proba = np.array([[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]])
    id2label = _id2label(3)
    res = assemble_predictions(proba, id2label, true_labels=["cause_1", "cause_0"])

    assert list(res.top1.columns) == ["true_label", "predicted_label", "predicted_prob"]
    assert list(res.full.columns) == ["true_label", "prob_0", "prob_1", "prob_2"]
    assert res.top1["predicted_label"].tolist() == ["cause_1", "cause_0"]
    assert res.top1["predicted_prob"].tolist() == pytest.approx([0.7, 0.6])
    assert res.topk["top1_label"].tolist() == ["cause_1", "cause_0"]
    assert res.topk["top2_label"].tolist() == ["cause_2", "cause_2"]


def test_unlabelled_data_has_no_true_label_column():
    """The point of the shared builder: scoring records with no known cause."""
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    res = assemble_predictions(proba, _id2label(2))

    for table in (res.top1, res.full, res.topk):
        assert "true_label" not in table.columns
    assert res.top1["predicted_label"].tolist() == ["cause_1", "cause_0"]


def test_ids_become_a_leading_column():
    """Every pipeline gets id support; voting/stacking/feature_fusion lacked it."""
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    res = assemble_predictions(proba, _id2label(2), ids=["rec_a", "rec_b"])

    for table in (res.top1, res.full, res.topk):
        assert list(table.columns)[0] == "id"
        assert table["id"].tolist() == ["rec_a", "rec_b"]


def test_ties_break_toward_the_lower_class_id():
    """Pinned because the old copies disagreed with each other.

    The ensemble copies sorted ascending and reversed, which puts the *highest*
    class ID first among equal probabilities; the text pipeline used
    ``torch.topk``, which puts the lowest first. A stable sort of the negated
    matrix matches the text pipeline and is deterministic by contract, rather
    than relying on the ordering NumPy's unstable quicksort happens to produce.
    """
    proba = np.full((1, 4), 0.25)
    res = assemble_predictions(proba, _id2label(4), top_k=4)

    assert res.topk.iloc[0]["top1_label"] == "cause_0"
    assert res.topk.iloc[0]["top4_label"] == "cause_3"
    assert res.top1.iloc[0]["predicted_label"] == "cause_0"


def test_top_k_capped_at_class_count():
    proba = np.array([[0.3, 0.7]])
    res = assemble_predictions(proba, _id2label(2), top_k=10)
    assert "top2_label" in res.topk.columns
    assert "top3_label" not in res.topk.columns


def test_full_probabilities_are_untouched():
    """`full` must reproduce the input matrix exactly — it feeds calibration."""
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(5), size=20)
    res = assemble_predictions(proba, _id2label(5))
    np.testing.assert_allclose(res.full[[f"prob_{i}" for i in range(5)]].to_numpy(), proba)


def test_shape_mismatches_are_reported_clearly():
    proba = np.zeros((3, 4))
    with pytest.raises(ValueError, match="id2label has 2 classes"):
        assemble_predictions(proba, _id2label(2))
    with pytest.raises(ValueError, match="true_labels has 2 entries"):
        assemble_predictions(proba, _id2label(4), true_labels=["a", "b"])
    with pytest.raises(ValueError, match="must be 2-D"):
        assemble_predictions(np.zeros(4), _id2label(4))


def test_save_predictions_writes_three_csvs(tmp_path):
    proba = np.array([[0.2, 0.8]])
    res = assemble_predictions(proba, _id2label(2), true_labels=["cause_1"])
    save_predictions(res, tmp_path / "out", prefix="preds")

    for name in ("top1", "full", "topk"):
        path = tmp_path / "out" / f"preds_{name}.csv"
        assert path.exists()
        assert len(pd.read_csv(path)) == 1
