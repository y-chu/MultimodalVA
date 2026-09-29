"""One name, ``resume``, for continuing an interrupted run — on every task."""

from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest

import multimodalva as mv
from multimodalva import Optimize
from multimodalva.utils.optimize_config import resolve_search_resume

RUNS = [
    ("multimodalva.text.text_classifier", "TextClassifier", "run"),
    ("multimodalva.tabular.tabular_classifier", "TabularClassifier", "run"),
    ("multimodalva.ensemble.data_fusion_classifier", "DataFusionClassifier", "run"),
    ("multimodalva.ensemble.feature_fusion", "FeatureFusionClassifier", "run"),
    ("multimodalva.ensemble.voting", "SoftVotingClassifier", "run"),
    # stacking keeps it on the object: one object spans several stage calls
    ("multimodalva.ensemble.stacking", "StackingClassifier", "__init__"),
]


@pytest.mark.parametrize("mod,cls,meth", RUNS)
def test_every_task_calls_it_resume(mod, cls, meth):
    params = inspect.signature(getattr(getattr(importlib.import_module(mod), cls), meth)).parameters
    assert "resume" in params and params["resume"].default is True
    assert "resume_training" not in params


def test_search_follows_the_run_unless_told_otherwise():
    assert resolve_search_resume(Optimize(), run_resume=True) is True
    assert resolve_search_resume(Optimize(), run_resume=False) is False
    assert resolve_search_resume(Optimize(resume=False), run_resume=True) is False
    assert resolve_search_resume(Optimize(resume=True), run_resume=False) is True


def test_old_name_is_rejected_with_the_new_one(tmp_path):
    df = mv.data("va_sample", n_per_class=4)
    with pytest.raises(TypeError, match="resume="):
        mv.run(task="tabular", data=df, label_col="cause_of_death",
               output_dir=tmp_path, resume_training=False)


def test_run_routes_resume_to_the_stacking_object(tmp_path, monkeypatch):
    seen = {}
    from multimodalva.ensemble import stacking

    real_init = stacking.StackingClassifier.__init__

    def spy(self, *a, **kw):
        seen["resume"] = kw.get("resume")
        real_init(self, *a, **kw)
        raise RuntimeError("stop after construction")

    monkeypatch.setattr(stacking.StackingClassifier, "__init__", spy)
    df = mv.data("va_sample", n_per_class=4)
    with pytest.raises(RuntimeError, match="stop after construction"):
        mv.run(task="stacking", data=df, label_col="cause_of_death",
               tabular_models=[{"model_name": "naive_bayes"}], output_dir=tmp_path,
               resume=False)
    assert seen["resume"] is False


def test_resume_manifest_accepts_only_the_same_data_and_configuration(tmp_path):
    from multimodalva.utils.provenance import (
        enforce_resume_manifest,
        make_resume_manifest,
    )

    data = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    first = make_resume_manifest(dataframes={"train": data}, config={"seed": 7})
    path = tmp_path / "resume_manifest.json"
    enforce_resume_manifest(
        path, first, resume=True, artifacts_exist=False, what="test run"
    )
    # recorded_at differs, but the content signature is identical.
    same = make_resume_manifest(dataframes={"train": data.copy()}, config={"seed": 7})
    enforce_resume_manifest(
        path, same, resume=True, artifacts_exist=True, what="test run"
    )

    changed_data = data.copy()
    changed_data.loc[0, "x"] = 99
    changed = make_resume_manifest(
        dataframes={"train": changed_data}, config={"seed": 7}
    )
    with pytest.raises(RuntimeError, match="data/split"):
        enforce_resume_manifest(
            path, changed, resume=True, artifacts_exist=True, what="test run"
        )

    changed_config = make_resume_manifest(
        dataframes={"train": data}, config={"seed": 8}
    )
    with pytest.raises(RuntimeError, match="configuration"):
        enforce_resume_manifest(
            path, changed_config, resume=True, artifacts_exist=True, what="test run"
        )


def test_resume_refuses_legacy_artifacts_with_no_manifest(tmp_path):
    from multimodalva.utils.provenance import (
        enforce_resume_manifest,
        make_resume_manifest,
    )

    expected = make_resume_manifest(
        dataframes={"train": pd.DataFrame({"x": [1]})}, config={}
    )
    with pytest.raises(RuntimeError, match="does not"):
        enforce_resume_manifest(
            tmp_path / "resume_manifest.json", expected,
            resume=True, artifacts_exist=True, what="legacy run",
        )


def test_default_hpo_resume_names_change_with_the_content():
    import numpy as np

    from multimodalva.tabular.hpo import _tabular_search_signature

    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    config = {"metric": "f1_macro", "space": {"depth": ("int", 2, 4)}}
    same = _tabular_search_signature(X.copy(), y.copy(), dict(config))
    assert same == _tabular_search_signature(X, y, config)
    changed = X.copy()
    changed[0, 0] = -1
    assert same != _tabular_search_signature(changed, y, config)
    assert same != _tabular_search_signature(X, y, {**config, "metric": "accuracy"})
