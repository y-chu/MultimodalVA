"""
Tests for row identifiers travelling from the input data to the predictions.

Passing ``id_col`` should put an ``id`` column in every prediction table, still
aligned with the right row after the pipeline drops rows with a missing label.

Run: pytest tests/test_ids.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multimodalva.results import bootstrap_ci, paired_bootstrap_ci, predictions_frame
from multimodalva.tabular import TabularClassifier
from multimodalva.tabular.dataset import valid_label_mask

CAUSES = ["HIV/AIDS", "Malaria", "Pneumonia", "Stroke"]
FEATURES = ["fever", "cough", "age"]


def _demo_df(n=160, seed=0, missing_labels=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(CAUSES, n)
    df = pd.DataFrame({
        "record_id": [f"VA{i:05d}" for i in range(n)],
        "fever": rng.integers(0, 2, n),
        "cough": rng.integers(0, 2, n),
        "age": rng.integers(1, 90, n),
        "cause": labels,
    })
    if missing_labels:
        # Empty strings, not NaN: split() rejects NaN labels up front, so the
        # rows that actually reach prepare_text_dataset()'s drop look like this.
        df.loc[df.index[:missing_labels], "cause"] = ""
    return df


# ---------------------------------------------------------------------------
# Identifiers through the tabular pipeline
# ---------------------------------------------------------------------------

def test_id_column_appears_in_every_prediction_table(tmp_path):
    df = _demo_df()
    clf = TabularClassifier(model_name="random_forest", output_dir=tmp_path / "run")
    out = clf.run(df=df, feature_cols=FEATURES, label_col="cause",
                  id_col="record_id", hyperparams={"n_estimators": 10})

    preds = out["predictions"]
    for table in (preds.top1, preds.full, preds.topk):
        assert table.columns[0] == "id"
        assert table["id"].is_unique
        assert set(table["id"]) <= set(df["record_id"])

    # and the ids reach the saved CSVs analysts actually read
    saved = pd.read_csv(tmp_path / "run" / "predictions" / "predictions_top1.csv")
    assert saved.columns[0] == "id"


def test_ids_identify_the_right_rows(tmp_path):
    """The id must name the record whose true label is in the same row."""
    df = _demo_df()
    clf = TabularClassifier(model_name="random_forest", output_dir=tmp_path / "run")
    out = clf.run(df=df, feature_cols=FEATURES, label_col="cause",
                  id_col="record_id", hyperparams={"n_estimators": 10})

    top1 = out["predictions"].top1
    truth = df.set_index("record_id")["cause"]
    for record_id, true_label in zip(top1["id"], top1["true_label"]):
        assert truth[record_id] == true_label


def test_ids_stay_aligned_when_rows_are_dropped(tmp_path):
    """Rows with a missing label are dropped; ids must follow that drop."""
    df = _demo_df(missing_labels=12)
    clf = TabularClassifier(model_name="random_forest", output_dir=tmp_path / "run")
    out = clf.run(df=df, feature_cols=FEATURES, label_col="cause",
                  id_col="record_id", hyperparams={"n_estimators": 10})

    top1 = out["predictions"].top1
    dropped = set(df.loc[~valid_label_mask(df, "cause"), "record_id"])
    assert not (set(top1["id"]) & dropped)

    truth = df.set_index("record_id")["cause"]
    for record_id, true_label in zip(top1["id"], top1["true_label"]):
        assert truth[record_id] == true_label


def test_no_id_column_when_id_col_is_not_given(tmp_path):
    df = _demo_df()
    clf = TabularClassifier(model_name="random_forest", output_dir=tmp_path / "run")
    out = clf.run(df=df, feature_cols=FEATURES, label_col="cause",
                  hyperparams={"n_estimators": 10})
    assert "id" not in out["predictions"].top1.columns


def test_unknown_id_col_is_reported(tmp_path):
    df = _demo_df()
    clf = TabularClassifier(model_name="random_forest", output_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="id_col"):
        clf.run(df=df, feature_cols=FEATURES, label_col="cause",
                id_col="not_a_column", hyperparams={"n_estimators": 10})


# ---------------------------------------------------------------------------
# Identifiers make the downstream comparison guards stronger
# ---------------------------------------------------------------------------

def _labelled_top1(seed, n=60, shuffle=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "id": [f"VA{i:05d}" for i in range(n)],
        "true_label": rng.choice(CAUSES, n),
    })
    frame["predicted_label"] = np.where(
        rng.random(n) < 0.6, frame["true_label"], rng.choice(CAUSES, n))
    if shuffle:
        frame = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    return frame


def test_models_are_matched_on_id_not_row_order():
    a = _labelled_top1(0)
    b = a.sample(frac=1, random_state=7).reset_index(drop=True)   # same rows, shuffled
    built = predictions_frame({"a": a, "b": b})

    assert built.columns[0] == "id"
    # b was reordered onto a's ids, so both true-label views agree row by row
    lookup = b.set_index("id")["predicted_label"]
    assert (built["b"].to_numpy() == lookup.reindex(built["id"]).to_numpy()).all()


def test_different_records_are_rejected_even_with_matching_labels():
    a = _labelled_top1(0, n=60)
    b = _labelled_top1(0, n=60)
    b["id"] = [f"OTHER{i:05d}" for i in range(60)]     # same labels, other records
    with pytest.raises(ValueError, match="different records"):
        predictions_frame({"a": a, "b": b})


def test_mixing_models_with_and_without_ids_is_rejected():
    a = _labelled_top1(0)
    b = a.drop(columns=["id"])
    with pytest.raises(ValueError, match="no 'id' column"):
        predictions_frame({"a": a, "b": b})


def test_id_column_is_not_scored_as_a_model():
    a = _labelled_top1(0)
    b = a.copy()
    b["predicted_label"] = b["predicted_label"].sample(frac=1, random_state=3).to_numpy()
    frame = predictions_frame({"a": a, "b": b})
    ci = bootstrap_ci(frame, n_boot=50, metrics=["accuracy"])
    assert set(ci.index) == {"a", "b"}

    paired = paired_bootstrap_ci(frame, model_a="a", model_b="b",
                                 n_boot=50, metrics=["accuracy"])
    assert len(paired) == 1
