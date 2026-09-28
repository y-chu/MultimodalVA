"""Every task's output carries an ``id`` column when ``id_col`` is given.

That is what lets a user merge predictions back onto their own records, or onto
an external dataset. It used to be silently dropped by four of the six tasks.
"""

from __future__ import annotations

import inspect

import pytest

import multimodalva as mv


def _df(n=15):
    df = mv.data("va_sample", n_per_class=n)
    return df.assign(rowid=[f"R{i:04d}" for i in range(len(df))])


def _features(df):
    return [c for c in df.columns if c not in ("cause_of_death", "narrative", "rowid", "id")]


@pytest.mark.parametrize("task,extra", [
    ("tabular", {"model": "random_forest", "hyperparams": {"n_estimators": 10}}),
    ("voting", {"tabular_models": [{"model_name": "random_forest"}, {"model_name": "naive_bayes"}]}),
    ("stacking", {"tabular_models": [{"model_name": "random_forest"}, {"model_name": "naive_bayes"}],
                  "init_kwargs": {"n_folds": 3}}),
])
def test_output_tables_start_with_id(task, extra, tmp_path):
    df = _df()
    r = mv.run(task=task, data=df, label_col="cause_of_death", features=_features(df),
               id_col="rowid", output_dir=tmp_path, **extra)
    for name in ("top1", "full", "topk"):
        table = getattr(r["predictions"], name)
        assert table.columns[0] == "id", f"{task}.{name} does not start with id"
        assert set(table["id"]) <= set(df["rowid"]), f"{task}.{name} ids are not the source ids"
    # and the saved files carry it too, since that is what a later merge reads
    saved = next(tmp_path.rglob("predictions/predictions_top1.csv"))
    assert saved.read_text().split(",", 1)[0] == "id"


@pytest.mark.parametrize("path,cls", [
    ("multimodalva.text.text_classifier", "TextClassifier"),
    ("multimodalva.ensemble.data_fusion_classifier", "DataFusionClassifier"),
    ("multimodalva.ensemble.feature_fusion", "FeatureFusionClassifier"),
])
def test_slow_tasks_accept_id_col(path, cls):
    """Text-model tasks are too slow to train here; check they take the argument."""
    import importlib

    run = getattr(importlib.import_module(path), cls).run
    assert "id_col" in inspect.signature(run).parameters


def test_id_col_is_never_used_as_a_feature():
    from multimodalva.runner import _resolve_features

    df = _df(5)
    auto = _resolve_features(df, None, "cause_of_death", "narrative", None, None, id_col="rowid")
    assert "rowid" not in auto
    with pytest.raises(ValueError, match="also listed in features"):
        _resolve_features(df, ["rowid", "i019a"], "cause_of_death", None, None, None, id_col="rowid")


@pytest.mark.parametrize("prefused", [False, True])
def test_data_fusion_carries_id_col_into_the_split(prefused, monkeypatch, tmp_path):
    """Both data-fusion branches must keep id_col (and a split column).

    The fused table used to keep only the label and the fused text, so
    run(task="data_fusion", id_col=...) crashed once run() began passing id_col
    on — and the pre-fused branch also dropped a fixed split column.
    """
    import numpy as np
    import multimodalva.ensemble.data_fusion_classifier as D

    seen = {}

    class Stop(Exception):
        pass

    def spy(train_df, test_df, *a, **kw):
        seen["cols"] = set(train_df.columns) | set(test_df.columns)
        seen["test_ids"] = set(test_df[kw["id_col"]])
        raise Stop

    import multimodalva.text.dataset as TD   # data fusion imports it lazily from here
    monkeypatch.setattr(TD, "prepare_text_dataset", spy)
    df = _df(8)
    feats = [c for c in df.columns if c.startswith("i") and c[1:4].isdigit()][:20]
    if prefused:
        df = df.assign(fused_text=df["narrative"])
    df = df.assign(fold=np.where(np.arange(len(df)) % 4 == 0, "test", "train"))
    with pytest.raises(Stop):
        D.DataFusionClassifier(model_name="bert-base-uncased", output_dir=tmp_path).run(
            df, text_col="narrative", feature_cols=feats, label_col="cause_of_death",
            id_col="rowid", split_col="fold",
        )
    assert "rowid" in seen["cols"]
    assert seen["test_ids"] == set(df.loc[df["fold"] == "test", "rowid"])
