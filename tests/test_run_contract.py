"""What every run leaves behind and returns, whichever pipeline made it.

A downstream script — a leaderboard, a report, a LOPO loop — should not need a
branch per pipeline. Every run() returns the same core keys and writes the same
core files; pipeline-specific extras come on top.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import multimodalva as mv
from multimodalva import Optimize
from multimodalva.utils.runtime import RUN_RESULT_KEYS

SPECS = [{"model_name": "random_forest"}, {"model_name": "naive_bayes"}]


@pytest.fixture(scope="module")
def data():
    df = mv.data("va_sample", n_per_class=12)
    df = df.assign(rowid=[f"R{i:04d}" for i in range(len(df))])
    feats = [c for c in df.columns if c not in ("cause_of_death", "narrative", "rowid", "id")]
    return df, feats


def _run(data, tmp, task, **kw):
    df, feats = data
    return mv.run(task=task, data=df, label_col="cause_of_death", features=feats,
                  id_col="rowid", output_dir=tmp, **kw)


@pytest.fixture(scope="module")
def runs(data, tmp_path_factory):
    t = tmp_path_factory.mktemp
    return {
        "tabular_fixed": _run(data, t("tf"), "tabular", model="random_forest",
                              hyperparams={"n_estimators": 10}),
        "tabular_search": _run(data, t("ts"), "tabular", model="random_forest",
                               hyperparams=Optimize(n_trials=2, cv_folds=2)),
        "voting": _run(data, t("vo"), "voting", tabular_models=SPECS),
        "stacking": _run(data, t("st"), "stacking", tabular_models=SPECS,
                         init_kwargs={"n_folds": 3}),
    }


@pytest.mark.parametrize("name", ["tabular_fixed", "tabular_search", "voting", "stacking"])
def test_every_run_returns_the_guaranteed_keys(runs, name):
    r = runs[name]
    missing = [k for k in RUN_RESULT_KEYS if k not in r]
    assert not missing, f"{name} run() result lacks {missing}"
    assert r["train_metadata"], f"{name} returned no train_metadata"


@pytest.mark.parametrize("name", ["tabular_fixed", "tabular_search", "voting", "stacking"])
def test_every_run_writes_validation_and_runtime(runs, name):
    out = runs[name]["output_dir"]
    assert (out / "validation.json").is_file()
    assert (out / "runtime" / "pipeline_runtime.json").is_file()
    report = json.loads((out / "runtime" / "pipeline_runtime.json").read_text())
    assert report["stages"] and report["stages"][0]["status"] == "ok"


def test_validation_source_matches_how_the_model_was_fitted(runs):
    src = {n: r["validation"]["source"] for n, r in runs.items()}
    assert src["tabular_search"] == "hpo_cv"
    assert src["tabular_fixed"] == "none", "a fixed-HP tabular fit has no validation data"
    assert src["stacking"] == "meta_cv"
    assert src["voting"] == "none"
    assert runs["tabular_search"]["validation"]["scores"]["f1_macro"] > 0
    assert runs["tabular_search"]["best_hyperparams"], "a search returns its best hyperparameters"


def test_validation_leaderboard_ranks_and_never_uses_test(runs):
    from multimodalva.results import validation_leaderboard

    board = validation_leaderboard({n: r for n, r in runs.items()})
    assert list(board.columns[:1]) == ["val_f1_macro"]
    assert np.isnan(board.loc["tabular_fixed", "val_f1_macro"]), "must stay missing, not use test"
    assert board.index[-1] in ("tabular_fixed", "voting"), "missing scores sort last"


def test_validation_leaderboard_reads_runs_without_validation_json(runs, tmp_path):
    """Runs made before validation.json existed are derived from their artifacts."""
    import shutil
    from multimodalva.results import validation_leaderboard

    old = tmp_path / "old_run"
    shutil.copytree(runs["tabular_search"]["output_dir"], old)
    (old / "validation.json").unlink()
    board = validation_leaderboard({"old": old})
    assert board.loc["old", "source"] == "hpo_cv"
    assert board.loc["old", "val_f1_macro"] == pytest.approx(
        runs["tabular_search"]["validation"]["scores"]["f1_macro"])


# --- preprocessing: one setting, honoured the same way everywhere --------------

def test_voting_takes_a_run_level_encoding_and_specs_override_it(data, tmp_path):
    seen = []
    import multimodalva.tabular.dataset as D
    orig = D.prepare_tabular_dataset

    def spy(*a, **kw):
        seen.append(kw.get("encode_categoricals"))
        return orig(*a, **kw)

    import unittest.mock as mock
    with mock.patch.object(D, "prepare_tabular_dataset", spy):
        _run(data, tmp_path, "voting", encode_categoricals="onehot",
             tabular_models=[{"model_name": "random_forest"},
                             {"model_name": "naive_bayes", "encode_categoricals": "ordinal"}])
    assert seen == ["onehot", "ordinal"], seen


def test_stacking_refuses_a_spec_encoding_it_cannot_honour(data, tmp_path):
    with pytest.raises(ValueError, match="one\\s+feature matrix"):
        _run(data, tmp_path, "stacking", encode_categoricals="ordinal",
             tabular_models=[{"model_name": "random_forest", "encode_categoricals": "onehot"},
                             {"model_name": "naive_bayes"}])


# --- text knobs -----------------------------------------------------------------

@pytest.mark.parametrize("path,cls", [
    ("multimodalva.text.text_classifier", "TextClassifier"),
    ("multimodalva.ensemble.data_fusion_classifier", "DataFusionClassifier"),
])
def test_text_pipelines_expose_val_size_and_n_jobs(path, cls):
    import importlib

    params = inspect.signature(getattr(importlib.import_module(path), cls).run).parameters
    assert params["val_size"].default == 0.1
    assert params["n_jobs"].default is None


def test_feature_fusion_exposes_val_size():
    from multimodalva.ensemble.feature_fusion import FeatureFusionClassifier

    assert inspect.signature(FeatureFusionClassifier.run).parameters["val_size"].default is None


@pytest.mark.parametrize("name,pipeline", [
    ("tabular_fixed", "tabular"), ("voting", "voting"), ("stacking", "stacking"),
])
def test_training_metadata_names_the_pipeline(runs, name, pipeline):
    """So a loader can read the task instead of guessing it from the files."""
    meta = json.loads((runs[name]["output_dir"] / "training_metadata.json").read_text())
    assert meta["pipeline"] == pipeline
    assert meta["multimodalva_version"]


def test_text_worker_count_is_decided_at_training_time(monkeypatch):
    """n_jobs sets MULTIMODALVA_DATALOADER_WORKERS for a run, after import.

    The default used to be computed at import, so it ignored that variable; on
    Apple Silicon n_jobs=0 then disabled the "no workers" override while keeping
    the import-time count, and started a dozen worker processes.
    """
    pytest.importorskip("torch")
    from multimodalva.text import train as T

    assert T.TEXT_DEFAULT_HYPERPARAMS["dataloader_num_workers"] is None
    monkeypatch.setenv("MULTIMODALVA_DATALOADER_WORKERS", "3")
    assert T._resolve_dataloader_workers(None) == 3
    monkeypatch.setenv("MULTIMODALVA_DATALOADER_WORKERS", "0")
    assert T._resolve_dataloader_workers(None) == 0
    assert T._resolve_dataloader_workers(5) == 5, "an explicit value wins"


def test_text_n_jobs_is_scoped_to_the_run():
    import os
    from multimodalva.utils.runtime import set_dataloader_workers

    key = "MULTIMODALVA_DATALOADER_WORKERS"
    before = os.environ.get(key)
    with set_dataloader_workers(2):
        assert os.environ[key] == "2"
    assert os.environ.get(key) == before


def test_early_stopping_slice_survives_small_data():
    """A search trial on small data used to crash in the stratified split.

    sklearn refuses a stratified validation slice with fewer rows than classes;
    inside a search every trial failed and the summary blamed GPU memory. The
    slice now grows to one row per class, or goes unstratified where even that
    is impossible. Data that split fine before takes the unchanged path.
    """
    pytest.importorskip("torch")
    from multimodalva.text.train import _val_split

    class Tiny:
        def __init__(self, labels):
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, i):
            return self.labels[i]

    # 11 classes, 84 rows: 10% is 9 rows < 11 classes -> used to raise
    ds = Tiny([i % 11 for i in range(84)])
    tr, va = _val_split(ds, 0.1, random_state=0)
    assert len(va) == 11 and len(tr) + len(va) == 84
    # a singleton class cannot be stratified at all -> random split, no crash
    ds = Tiny([0] * 20 + [1] * 20 + [2])
    tr, va = _val_split(ds, 0.2, random_state=0)
    assert len(tr) + len(va) == 41
    # the ordinary case is unchanged
    ds = Tiny([i % 3 for i in range(300)])
    tr, va = _val_split(ds, 0.1, random_state=0)
    assert len(va) == 30


def test_log_loss_does_not_depend_on_how_the_ids_are_ordered():
    """sklearn aligns probability columns to *sorted* labels, not to the order
    given in ``labels=`` — it only warns. Our columns are in class-id order, so
    an id order that is not alphabetical silently produced a wrong number
    (2.455 where the answer was 0.031). Package runs were safe because their
    label encoders sort, but predictions brought in from another tool need not.
    """
    import numpy as np
    import pandas as pd

    from multimodalva.utils.metrics import log_loss_from_full

    def log_loss_for(causes):
        id2label = dict(enumerate(causes))
        lab2id = {v: k for k, v in id2label.items()}
        rng = np.random.default_rng(0)
        truth = rng.choice(causes, 200)
        p = np.full((200, len(causes)), 0.01)
        for row, cause in enumerate(truth):
            p[row, lab2id[cause]] = 1 - 0.01 * (len(causes) - 1)
        full = pd.DataFrame(
            p, columns=[f"prob_{i}" for i in range(len(causes))]
        ).assign(true_label=truth)
        return log_loss_from_full(full, id2label)

    causes = ["HIV/AIDS", "TB", "Pneumonia", "Injury"]      # ids not alphabetical
    assert causes != sorted(causes), "this test needs a non-alphabetical id order"
    expected = -np.log(0.97)
    assert log_loss_for(causes) == pytest.approx(expected, abs=1e-4)
    # and the alphabetical case, which was always right, is unchanged
    assert log_loss_for(sorted(causes)) == pytest.approx(expected, abs=1e-4)
