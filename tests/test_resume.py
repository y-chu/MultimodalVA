"""One name, ``resume``, for continuing an interrupted run — on every task."""

from __future__ import annotations

import importlib
import inspect

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
