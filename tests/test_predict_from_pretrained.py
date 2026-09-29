"""predict_from_pretrained(): load a trained model, score new data, labelled or not."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import multimodalva as mv
from multimodalva.inference.api import _resolve_pretrained_source
from multimodalva.inference.backends import PRETRAINED_BACKENDS, load_pretrained_backend
from multimodalva.inference.checks import (
    PretrainedChecks,
    _is_placeholder_feature_cols,
    _is_placeholder_id2label,
)
from multimodalva.utils.types import PredictionResult

LABEL = "cause_of_death"


@pytest.fixture(scope="module")
def data():
    df = mv.data("va_sample", n_per_class=12)
    df = df.assign(rowid=[f"R{i:04d}" for i in range(len(df))])
    feats = [c for c in df.columns if c not in (LABEL, "narrative", "rowid", "id")]
    return df, feats


@pytest.fixture(scope="module")
def tabular_run(data, tmp_path_factory):
    df, feats = data
    out = tmp_path_factory.mktemp("tab")
    r = mv.run(task="tabular", data=df, label_col=LABEL, features=feats,
               id_col="rowid", model="random_forest",
               hyperparams={"n_estimators": 10}, output_dir=out)
    return out, r


# --- tabular ----------------------------------------------------------------

def test_labelled_reproduces_the_saved_run(data, tabular_run):
    df, _ = data
    out, r = tabular_run
    test_ids = r["predictions"].top1["id"].tolist()
    new = df.set_index("rowid").loc[test_ids].reset_index()
    res = mv.predict_from_pretrained(out, new, label_col=LABEL, id_col="rowid")
    assert isinstance(res, PredictionResult)
    assert isinstance(res.checks, PretrainedChecks)
    assert res.checks.task == "tabular" and res.checks.task_source == "training_metadata"
    assert res.checks.feature_cols_source == "preprocessor"
    assert res.checks.labelled
    saved = r["predictions"].full
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    np.testing.assert_allclose(res.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy())
    assert res.top1["true_label"].tolist() == r["predictions"].top1["true_label"].tolist()


def test_unlabelled_omits_true_label(data, tabular_run):
    df, _ = data
    out, _ = tabular_run
    res = mv.predict_from_pretrained(out, df.drop(columns=[LABEL]))
    for table in (res.top1, res.full, res.topk):
        assert "true_label" not in table.columns
    assert not res.checks.labelled
    assert res.checks.n_rows == len(df)


def test_final_dir_is_accepted_too(data, tabular_run):
    df, _ = data
    out, _ = tabular_run
    res = mv.predict_from_pretrained(out / "final", df.head(5))
    assert len(res.top1) == 5


def test_columns_matched_by_name_not_position(data, tabular_run):
    df, feats = data
    out, _ = tabular_run
    base = mv.predict_from_pretrained(out, df)
    shuffled = df[list(reversed(df.columns))].assign(extra_col=1)
    res = mv.predict_from_pretrained(out, shuffled)
    pd.testing.assert_frame_equal(base.full, res.full)
    assert "extra_col" in res.checks.dropped_input_cols


def test_missing_column_errors_by_default_and_names_it(data, tabular_run):
    df, feats = data
    out, _ = tabular_run
    with pytest.raises(ValueError, match=feats[0]):
        mv.predict_from_pretrained(out, df.drop(columns=[feats[0]]))


def test_missing_column_fill_na(data, tabular_run):
    df, feats = data
    out, _ = tabular_run
    with pytest.warns(UserWarning, match="Filled 1 missing"):
        res = mv.predict_from_pretrained(out, df.drop(columns=[feats[0]]),
                                         missing_feature_method="fill_na")
    assert res.checks.missing_feature_cols == [feats[0]]


def test_na_values_in_present_column_do_not_warn(data, tabular_run):
    df, feats = data
    out, _ = tabular_run
    df = df.copy()
    df.loc[0, feats[0]] = np.nan
    res = mv.predict_from_pretrained(out, df)
    assert not any(feats[0] in w for w in res.checks.warnings)


def test_bad_missing_feature_method(data, tabular_run):
    df, _ = data
    out, _ = tabular_run
    with pytest.raises(ValueError, match="missing_feature_method"):
        mv.predict_from_pretrained(out, df, missing_feature_method="drop")


def test_unknown_labels_warn(data, tabular_run):
    df, _ = data
    out, _ = tabular_run
    df = df.copy()
    df.loc[0, LABEL] = "Not A Cause"
    res = mv.predict_from_pretrained(out, df, label_col=LABEL)
    assert any("Not A Cause" in w for w in res.checks.warnings)


def test_save_dir(data, tabular_run, tmp_path):
    df, _ = data
    out, _ = tabular_run
    mv.predict_from_pretrained(out, df, save_dir=tmp_path)
    assert (tmp_path / "predictions_full.csv").is_file()


# --- foreign tabular models -------------------------------------------------

def _foreign(tmp_path, X, y, id2label=None):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    if id2label is not None:
        (tmp_path / "id2label.json").write_text(json.dumps(id2label))
    return model


def test_foreign_dataframe_model_with_string_classes(tmp_path):
    X = pd.DataFrame({"a": [0, 1, 0, 1, 2, 2], "b": [1, 1, 0, 0, 1, 0]})
    y = ["x", "y", "x", "y", "z", "z"]
    _foreign(tmp_path, X, y)
    with pytest.warns(UserWarning, match="does not record which MultimodalVA"):
        res = mv.predict_from_pretrained(tmp_path, X[["b", "a"]])
    assert res.checks.task_source == "structure"
    assert res.checks.feature_cols_source == "native"
    assert res.checks.id2label_source == "native"
    assert set(res.id2label.values()) == {"x", "y", "z"}


def test_foreign_array_model_needs_feature_cols(tmp_path):
    X = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    _foreign(tmp_path, X, [0, 1, 0, 1], id2label={"0": "p", "1": "q"})
    df = pd.DataFrame(X, columns=["a", "b"])
    with pytest.raises(ValueError, match="feature_cols"):
        mv.predict_from_pretrained(tmp_path, df)
    res = mv.predict_from_pretrained(tmp_path, df, feature_cols=["a", "b"])
    assert res.checks.feature_cols_source == "caller"
    assert res.checks.id2label_source == "sidecar"


def test_missing_id2label_gives_integer_ids(tmp_path):
    X = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    _foreign(tmp_path, X, [0, 1, 0, 1])
    df = pd.DataFrame(X, columns=["a", "b"])
    with pytest.warns(UserWarning, match="integer class ids"):
        res = mv.predict_from_pretrained(tmp_path, df, feature_cols=["a", "b"])
    assert res.id2label == {0: "class_0", 1: "class_1"}
    assert res.checks.id2label_source == "missing"


# --- placeholders, dispatch, source resolution ------------------------------

def test_placeholders_count_as_missing():
    assert _is_placeholder_id2label({0: "LABEL_0", 1: "LABEL_1"})
    assert not _is_placeholder_id2label({0: "Injury", 1: "LABEL_1"})
    assert _is_placeholder_feature_cols(["0", "1", "2"])
    assert not _is_placeholder_feature_cols(["age", "sex"])


def test_unsupported_task_names_what_is_supported():
    with pytest.raises(ValueError, match="tabular"):
        load_pretrained_backend("stacking")


def test_registry_is_lazy():
    assert all(":" in target for target in PRETRAINED_BACKENDS.values())


def test_hub_url_and_bad_source(monkeypatch, tmp_path):
    calls = []
    fake = type(sys)("huggingface_hub")
    fake.snapshot_download = lambda repo_id, **kw: calls.append((repo_id, kw)) or str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    _resolve_pretrained_source("https://huggingface.co/org/name/tree/main", {"revision": "v1"})
    _resolve_pretrained_source("org/name.v2", {})
    assert calls == [("org/name", {"revision": "v1"}), ("org/name.v2", {})]
    with pytest.raises(FileNotFoundError):
        _resolve_pretrained_source("no/such/dir", {})


def test_import_does_not_pull_torch():
    code = ("import sys, multimodalva as mv; mv.predict_from_pretrained; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_existing_prediction_results_default_checks_to_none(tabular_run):
    _, r = tabular_run
    assert r["predictions"].checks is None


# --- text (tiny model; skipped when it cannot be downloaded) ----------------

@pytest.fixture(scope="module")
def text_run(data, tmp_path_factory):
    pytest.importorskip("transformers")
    df, _ = data
    out = tmp_path_factory.mktemp("txt")
    try:
        r = mv.run(task="text", data=df, label_col=LABEL, text_col="narrative",
                   id_col="rowid", model="prajjwal1/bert-tiny",
                   hyperparams={"epochs": 1, "batch_size": 8}, output_dir=out)
    except OSError as exc:  # offline
        pytest.skip(f"tiny model unavailable: {exc}")
    return out, r


def test_text_reproduces_saved_probabilities(data, text_run):
    df, _ = data
    out, r = text_run
    test_ids = r["predictions"].top1["id"].tolist()
    new = df.set_index("rowid").loc[test_ids].reset_index()
    res = mv.predict_from_pretrained(out, new, text_col="narrative",
                                     label_col=LABEL, id_col="rowid")
    assert res.checks.task == "text"
    assert res.checks.id2label_source in ("native", "sidecar")
    saved = r["predictions"].full
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    np.testing.assert_allclose(res.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy(), atol=1e-5)
    assert res.top1["predicted_label"].tolist() == r["predictions"].top1["predicted_label"].tolist()


def test_text_unlabelled_and_empty_text(data, text_run):
    df, _ = data
    out, _ = text_run
    new = df.drop(columns=[LABEL]).head(6).copy()
    new.loc[0, "narrative"] = None
    res = mv.predict_from_pretrained(out, new, text_col="narrative")
    assert "true_label" not in res.top1.columns
    assert len(res.top1) == 6
    assert res.checks.empty_text_rows == 1


def test_text_needs_text_col(data, text_run):
    df, _ = data
    out, _ = text_run
    with pytest.raises(ValueError, match="text_col"):
        mv.predict_from_pretrained(out, df)


@pytest.mark.parametrize("fixture", ["tabular_run", "text_run"])
def test_every_backend_honours_the_shared_contract(request, data, fixture):
    """Same method names must mean the same inputs and outputs on every backend."""
    df, _ = data
    out, _ = request.getfixturevalue(fixture)
    task = "tabular" if fixture == "tabular_run" else "text"
    backend_cls = load_pretrained_backend(task)
    checks = PretrainedChecks(source=str(out), artifact_dir=str(out / "final"),
                              task=task, task_source="caller")
    kwargs = {"text_col": "narrative"} if task == "text" else {}
    backend = backend_cls.from_pretrained(out / "final", checks, **kwargs)
    assert set(backend.id2label) == set(range(len(backend.id2label)))
    proba = backend.predict_proba(backend.prepare_inputs(df.head(7)))
    assert isinstance(proba, np.ndarray) and proba.shape == (7, len(backend.id2label))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert backend.check_artifact() is None


# --- feature fusion (AutoMM stubbed: training it needs a GPU box) -----------

@pytest.fixture
def automm_artifact(tmp_path, monkeypatch):
    """A feature-fusion run directory plus a stand-in AutoMM predictor."""
    (tmp_path / "automm_model").mkdir()
    id2label = {"0": "Injury", "1": "Neoplasms"}
    (tmp_path / "training_metadata.json").write_text(json.dumps({
        "pipeline": "feature_fusion", "multimodalva_version": "0.1.0",
        "feature_cols": ["narrative", "age_group"], "id2label": id2label,
    }))
    (tmp_path / "id2label.json").write_text(json.dumps(id2label))

    class _Predictor:
        class_labels = ["Neoplasms", "Injury"]  # AutoMM's own order differs

        @classmethod
        def load(cls, path, **kw):
            _Predictor.loaded_from = path
            return cls()

        def predict_proba(self, df):
            _Predictor.seen_cols = list(df.columns)
            return pd.DataFrame({"Neoplasms": [0.8] * len(df),
                                 "Injury": [0.2] * len(df)})

    autogluon = type(sys)("autogluon")
    multimodal = type(sys)("autogluon.multimodal")
    multimodal.MultiModalPredictor = _Predictor
    autogluon.multimodal = multimodal
    monkeypatch.setitem(sys.modules, "autogluon", autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", multimodal)
    return tmp_path, _Predictor


def test_feature_fusion_columns_and_class_order(automm_artifact):
    tmp_path, predictor = automm_artifact
    df = pd.DataFrame({"age_group": ["45-59", "60+"], "narrative": ["a", "b"],
                       "extra": [1, 2]})
    res = mv.predict_from_pretrained(tmp_path, df)
    assert res.checks.task == "feature_fusion"
    assert res.checks.task_source == "training_metadata"
    assert predictor.seen_cols == ["narrative", "age_group"]  # trained order
    assert res.checks.dropped_input_cols == ["extra"]
    # AutoMM's column order is by label name; ours is by class id.
    assert res.id2label == {0: "Injury", 1: "Neoplasms"}
    np.testing.assert_allclose(res.full[["prob_0", "prob_1"]].to_numpy(),
                               [[0.2, 0.8], [0.2, 0.8]])
    assert res.top1["predicted_label"].tolist() == ["Neoplasms", "Neoplasms"]


def test_feature_fusion_missing_column(automm_artifact):
    tmp_path, _ = automm_artifact
    df = pd.DataFrame({"narrative": ["a"]})
    with pytest.raises(ValueError, match="age_group"):
        mv.predict_from_pretrained(tmp_path, df)
    with pytest.warns(UserWarning, match="Filled 1 missing"):
        mv.predict_from_pretrained(tmp_path, df, missing_feature_method="fill_na")


def test_feature_fusion_needs_the_predictor_dir(automm_artifact, tmp_path):
    bare = tmp_path / "no_automm"
    bare.mkdir()
    (bare / "training_metadata.json").write_text(json.dumps({"pipeline": "feature_fusion"}))
    with pytest.raises(FileNotFoundError, match="automm_model"):
        mv.predict_from_pretrained(bare, pd.DataFrame({"a": [1]}))


# --- ensembles --------------------------------------------------------------

SPECS = [{"model_name": "random_forest", "hyperparams": {"n_estimators": 10}},
         {"model_name": "naive_bayes"}]


@pytest.fixture(scope="module")
def voting_run(data, tmp_path_factory):
    df, feats = data
    out = tmp_path_factory.mktemp("vote")
    r = mv.run(task="voting", data=df, label_col=LABEL, features=feats,
               id_col="rowid", tabular_models=SPECS, output_dir=out)
    return out, r


@pytest.fixture(scope="module")
def stacking_run(data, tmp_path_factory):
    df, feats = data
    out = tmp_path_factory.mktemp("stack")
    r = mv.run(task="stacking", data=df, label_col=LABEL, features=feats,
               id_col="rowid", tabular_models=SPECS, output_dir=out,
               init_kwargs={"n_folds": 3},
               combiner=["meta_learner", "class_aware_voting",
                             "ensemble_selection"])
    return out, r


def _test_rows(df, run_result):
    ids = run_result["predictions"].top1["id"].tolist()
    return df.set_index("rowid").loc[ids].reset_index()


def test_voting_reproduces_its_own_run(data, voting_run):
    df, _ = data
    out, r = voting_run
    res = mv.predict_ensemble_from_pretrained(
        out, _test_rows(df, r), label_col=LABEL, id_col="rowid")
    assert res.checks.task == "voting" and res.checks.combiner == "soft_voting"
    assert [b["task"] for b in res.checks.base_models] == ["tabular", "tabular"]
    saved = r["predictions"].full
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    np.testing.assert_allclose(res.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy())


def test_stacking_reproduces_its_own_run(data, stacking_run):
    df, _ = data
    out, r = stacking_run
    res = mv.predict_ensemble_from_pretrained(
        out, _test_rows(df, r), label_col=LABEL, id_col="rowid")
    assert res.checks.task == "stacking"
    assert res.checks.combiner == "meta_learner"
    assert len(res.checks.base_models) == 2
    saved = r["predictions"].full
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    np.testing.assert_allclose(res.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy())


def test_stacking_unlabelled_and_combiner_choice(data, stacking_run):
    df, _ = data
    out, _ = stacking_run
    res = mv.predict_ensemble_from_pretrained(out, df.drop(columns=[LABEL]).head(5))
    assert "true_label" not in res.top1.columns and len(res.top1) == 5
    with pytest.raises(ValueError, match="combiner must be one this run fitted"):
        mv.predict_ensemble_from_pretrained(out, df.head(2), combiner="mean")



@pytest.mark.parametrize("combiner", ["class_aware_voting", "ensemble_selection"])
def test_stacking_other_combiners_reproduce_their_own_predictions(
        data, stacking_run, combiner):
    df, _ = data
    out, r = stacking_run
    saved = pd.read_csv(Path(r["output_dir"]) /
                        ("class_voter" if combiner == "class_aware_voting"
                         else "ensemble_selection") /
                        "predictions" / "predictions_full.csv")
    res = mv.predict_ensemble_from_pretrained(
        out, _test_rows(df, r), label_col=LABEL, id_col="rowid",
        combiner=combiner)
    assert res.checks.combiner == combiner
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    np.testing.assert_allclose(res.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy())


def test_stacking_relocated_base_models(data, stacking_run, tmp_path):
    """model_sources holds absolute paths; a moved run must still work."""
    df = data[0]
    run_dir = Path(stacking_run[1]["output_dir"])   # output_dir/stacking/
    moved = tmp_path / "moved_run"
    shutil.copytree(run_dir, moved)
    oof = moved / "oof" / "oof_metadata.json"
    meta = json.loads(oof.read_text())
    for src in meta["model_sources"]:
        src["final_dir"] = str(tmp_path / "gone" / Path(src["final_dir"]).name)
    oof.write_text(json.dumps(meta))
    res = mv.predict_ensemble_from_pretrained(moved, df.head(4))
    assert all(str(moved) in b["dir"] for b in res.checks.base_models)


def test_single_model_entry_rejects_an_ensemble(data, voting_run):
    df, _ = data
    out, _ = voting_run
    with pytest.raises(ValueError, match="does not support task 'voting'"):
        mv.predict_from_pretrained(out, df.head(2))


def test_ensemble_entry_rejects_a_single_model(data, tabular_run):
    df, _ = data
    out, _ = tabular_run
    with pytest.raises(ValueError, match="not an ensemble"):
        mv.predict_ensemble_from_pretrained(out, df.head(2))


@pytest.mark.skipif(not os.environ.get("MULTIMODALVA_SLOW_TESTS"),
                    reason="trains a real AutoMM model (~4 min); "
                           "set MULTIMODALVA_SLOW_TESTS=1")
def test_feature_fusion_round_trip_on_a_real_automm_model(data, tmp_path_factory):
    """The stub above cannot check the AutoGluon boundary itself; this does.

    Trains a tiny AutoMM model and asks predict_from_pretrained() to reproduce
    that run's own test probabilities. It caught a real bug the stub missed:
    ``predictor.class_labels`` is a NumPy array, so ``or []`` raised.
    """
    pytest.importorskip("autogluon.multimodal")
    df, feats = data
    out = tmp_path_factory.mktemp("ff")
    r = mv.run(task="feature_fusion", data=df, label_col=LABEL,
               text_col="narrative", features=feats, id_col="rowid",
               output_dir=out, model="prajjwal1/bert-tiny",
               init_kwargs={"preset": "medium_quality"}, time_limit=120)
    saved = r["predictions"].full
    rows = df.set_index("rowid").loc[saved["id"]].reset_index()
    res = mv.predict_from_pretrained(r["output_dir"], rows, text_col="narrative",
                                     label_col=LABEL, id_col="rowid")
    assert res.checks.task == "feature_fusion"
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    np.testing.assert_allclose(res.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy())
    assert (res.top1["predicted_label"].tolist()
            == r["predictions"].top1["predicted_label"].tolist())


# --- combiners the renamed stage 2 added ------------------------------------
# simple_average fits nothing, a meta-learner named in combiner= lives in
# meta_learner_<name>/, and combiner="best" records its winner in
# combiner_chosen. All three postdate the first version of this module.

@pytest.fixture(scope="module")
def stacking_run_extra_combiners(data, tmp_path_factory):
    df, feats = data
    out = tmp_path_factory.mktemp("stack_extra")
    r = mv.run(task="stacking", data=df, label_col=LABEL, features=feats,
               id_col="rowid", tabular_models=SPECS, output_dir=out,
               init_kwargs={"n_folds": 3},
               # the named meta-learner first, so that the default resolution
               # cannot pass by picking the first *fitted* combiner instead of
               # the one the run delivered.
               combiner=["logistic_regression", "simple_average"])
    return out, r


@pytest.mark.parametrize("combiner", ["simple_average", "logistic_regression"])
def test_extra_combiners_reproduce_their_own_predictions(
        data, stacking_run_extra_combiners, combiner):
    df, _ = data
    out, r = stacking_run_extra_combiners
    root = Path(r["output_dir"])
    sub = ("simple_average" if combiner == "simple_average"
           else f"meta_learner_{combiner}")
    saved = pd.read_csv(root / sub / "predictions" / "predictions_full.csv")
    rows = df.set_index("rowid").loc[saved["id"]].reset_index()
    res = mv.predict_ensemble_from_pretrained(
        out, rows, label_col=LABEL, id_col="rowid", combiner=combiner)
    assert res.checks.combiner == combiner
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    np.testing.assert_allclose(res.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy(), atol=1e-12)


def test_default_combiner_follows_what_the_run_delivered(
        data, stacking_run_extra_combiners):
    """No combiner= given: reuse the run's own first/chosen combiner."""
    df, _ = data
    out, r = stacking_run_extra_combiners
    meta = json.loads((Path(r["output_dir"]) / "training_metadata.json").read_text())
    expected = meta.get("combiner_chosen") or meta["combiner"][0]
    res = mv.predict_ensemble_from_pretrained(out, df.head(4), id_col="rowid")
    assert res.checks.combiner == expected
    saved = r["predictions"].full
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    rows = df.set_index("rowid").loc[saved["id"]].reset_index()
    again = mv.predict_ensemble_from_pretrained(out, rows, id_col="rowid")
    np.testing.assert_allclose(again.full[prob_cols].to_numpy(),
                               saved[prob_cols].to_numpy(), atol=1e-12)


# --- the CLI ----------------------------------------------------------------
#
# The point of `multimodalva predict` is that an external user does not have to
# know which of the two Python entry points their artifact needs, and does not
# discover a mistake after a job has burned an hour of GPU time. So these tests
# are about the routing and the guard rails, not about the predictions again.

def _cli(*argv):
    from multimodalva.cli import main
    return main(["predict", *argv])


@pytest.fixture(scope="module")
def csv_data(data, tmp_path_factory):
    df, _ = data
    path = tmp_path_factory.mktemp("csv") / "new.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_cli_routes_a_single_model_and_reports_metrics(
        tabular_run, csv_data, tmp_path, capsys):
    out, _ = tabular_run
    assert _cli("--source", str(out), "--data", csv_data,
                "--label-col", LABEL, "--id-col", "rowid",
                "--output-dir", str(tmp_path)) == 0
    printed = capsys.readouterr().out
    assert "Detected: tabular (from the run's own metadata)" in printed
    assert "f1_macro" in printed and "csmf_accuracy" in printed
    assert (tmp_path / "predictions_full.csv").is_file()


def test_cli_routes_an_ensemble_without_being_told(voting_run, csv_data, capsys):
    out, _ = voting_run
    assert _cli("--source", str(out), "--data", csv_data) == 0
    printed = capsys.readouterr().out
    assert "Detected: voting" in printed
    assert "Combined 2 base models with soft_voting" in printed


def test_cli_unlabelled_prints_no_metrics(tabular_run, csv_data, capsys):
    out, _ = tabular_run
    assert _cli("--source", str(out), "--data", csv_data) == 0
    printed = capsys.readouterr().out
    assert "f1_macro" not in printed
    assert "Scored" in printed


def test_cli_task_conflict_with_recorded_metadata_exits(tabular_run, csv_data, capsys):
    """A wrong --task must fail now, not after an inference pass."""
    out, _ = tabular_run
    assert _cli("--source", str(out), "--data", csv_data, "--task", "text") == 2
    err = capsys.readouterr().err
    assert "contradicts this run" in err and "--force" in err


def test_cli_force_overrides_a_recorded_conflict(tabular_run, csv_data, capsys):
    out, _ = tabular_run
    # Forced through, so it fails further on (a text model needs a text column)
    # rather than on the conflict itself.
    assert _cli("--source", str(out), "--data", csv_data,
                "--task", "text", "--force") == 2
    captured = capsys.readouterr()
    assert "contradicts this run" not in captured.err
    assert "text_col" in captured.err


def test_cli_task_override_of_a_guess_only_warns(text_run, csv_data, tmp_path,
                                                capsys):
    """Detection from the files present is a guess; --task wins over it.

    A data-fusion artifact *is* a Hugging Face sequence classifier, so structure
    alone reads it as text. Saying so must not be an error — this is the case
    --task exists for.
    """
    out, _ = text_run
    artifact = tmp_path / "foreign_hf"
    shutil.copytree(out / "final", artifact)
    (artifact / "training_metadata.json").unlink()   # a plain Hub download
    assert _cli("--source", str(artifact), "--data", csv_data,
                "--text-col", "narrative", "--task", "data_fusion") == 0
    captured = capsys.readouterr()
    assert "Detected: data_fusion (from --task)" in captured.out
    assert "fused long text" in captured.out          # the extra hint
    assert "suggests 'text'" in captured.err          # warned, not fatal


def test_cli_dry_run_scores_nothing(tabular_run, csv_data, tmp_path, capsys):
    out, _ = tabular_run
    assert _cli("--source", str(out), "--data", csv_data, "--label-col", LABEL,
                "--output-dir", str(tmp_path), "--dry-run") == 0
    printed = capsys.readouterr().out
    assert "Feature columns:" in printed and "0 missing" in printed
    assert "performance will be reported" in printed
    assert "nothing was scored or written" in printed
    assert not list(tmp_path.glob("*.csv"))


def test_cli_dry_run_names_missing_columns(tabular_run, data, tmp_path, capsys):
    df, feats = data
    thin = tmp_path / "thin.csv"
    df.drop(columns=[feats[0]]).to_csv(thin, index=False)
    out, _ = tabular_run
    assert _cli("--source", str(out), "--data", str(thin), "--dry-run") == 0
    printed = capsys.readouterr().out
    assert "1 missing" in printed and feats[0] in printed


def test_cli_dry_run_on_a_stacking_run_names_the_combiner(
        stacking_run, csv_data, capsys):
    out, _ = stacking_run
    assert _cli("--source", str(out), "--data", csv_data, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert "Base models: 2" in printed
    assert "Combiner:" in printed


def test_cli_rejects_flags_that_do_not_apply(tabular_run, voting_run, csv_data,
                                             capsys):
    out, _ = tabular_run
    assert _cli("--source", str(out), "--data", csv_data,
                "--combiner", "meta_learner") == 2
    assert "--combiner applies to a stacking run" in capsys.readouterr().err
    vote, _ = voting_run
    assert _cli("--source", str(vote), "--data", csv_data,
                "--feature-cols", "a,b") == 2
    assert "does not apply to an ensemble" in capsys.readouterr().err


def test_cli_unreadable_source_exits_two(csv_data, tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert _cli("--source", str(tmp_path / "empty"), "--data", csv_data) == 2
    assert "Cannot tell what kind of model" in capsys.readouterr().err


def test_voting_run_without_base_models_says_what_to_do(tmp_path):
    """Voting over already-computed predictions keeps no models to re-score with."""
    (tmp_path / "training_metadata.json").write_text(json.dumps({"pipeline": "voting"}))
    with pytest.raises(ValueError, match="vote_from_results"):
        mv.predict_ensemble_from_pretrained(tmp_path, pd.DataFrame({"a": [1]}))


def test_tabular_backend_rejects_unknown_backend_kwargs_before_loading(tmp_path):
    from multimodalva.inference.checks import PretrainedChecks
    from multimodalva.inference.tabular_backend import TabularPretrainedBackend

    checks = PretrainedChecks(
        source=str(tmp_path), artifact_dir=str(tmp_path), task="tabular",
        task_source="caller", multimodalva_version=None,
    )
    with pytest.raises(TypeError, match="silently ignored"):
        TabularPretrainedBackend.from_pretrained(
            tmp_path, checks, invented_option=True,
        )


def test_malformed_training_metadata_is_not_silently_ignored(tmp_path):
    from multimodalva.inference.api import _read_training_metadata

    (tmp_path / "training_metadata.json").write_text("{not valid json")
    with pytest.raises(ValueError, match="Cannot read training metadata"):
        _read_training_metadata(tmp_path, tmp_path)
