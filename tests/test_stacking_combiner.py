"""Stacking's second stage: which combiners run, and reusing stage 1.

The meta-learner, simple average, class-aware voting and ensemble selection all
sit on the same out-of-fold predictions. Stage 1 is the expensive part, so run()
must be able to apply several of them to one set of OOF predictions, to choose
one on the OOF rows alone (combiner="best"), and to apply any of them to a
finished run's OOF predictions without retraining anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import multimodalva as mv

SPECS = [{"model_name": "random_forest"}, {"model_name": "naive_bayes"}]
ALL = ["meta_learner", "class_aware_voting", "ensemble_selection"]


@pytest.fixture(scope="module")
def data():
    df = mv.data("va_sample", n_per_class=12)
    df = df.assign(rowid=[f"R{i:04d}" for i in range(len(df))])
    feats = [c for c in df.columns if c not in ("cause_of_death", "narrative", "rowid", "id")]
    return df, feats


@pytest.fixture(scope="module")
def all_run(data, tmp_path_factory):
    df, feats = data
    out = tmp_path_factory.mktemp("all")
    r = mv.run(task="stacking", data=df, label_col="cause_of_death", features=feats,
               id_col="rowid", tabular_models=SPECS, init_kwargs={"n_folds": 3},
               combiner=ALL, ensemble_selection_kwargs={"ensemble_size": 10},
               output_dir=out)
    return r, out / "stacking"


def test_a_list_runs_every_method_from_one_stage_one(all_run):
    r, d = all_run
    assert r["combiner"] == ALL
    assert set(r["combiner_predictions"]) == set(ALL)
    assert r["predictions"] is r["combiner_predictions"]["meta_learner"], "first listed is main"
    assert r["combiner_chosen"] is None and r["combiner_comparison"] is None
    for folder in ("meta_learner", "class_voter", "ensemble_selection"):
        assert (d / folder / "predictions" / "predictions_top1.csv").is_file(), folder
    assert (d / "predictions" / "predictions_top1.csv").is_file()
    for name, pr in r["combiner_predictions"].items():
        assert pr.top1.columns[0] == "id", f"{name} lost the id column"


def test_main_predictions_file_is_the_first_method(data, tmp_path):
    """predictions/ must hold the first method even if another ran after it."""
    df, feats = data
    r = mv.run(task="stacking", data=df, label_col="cause_of_death", features=feats,
               tabular_models=SPECS, init_kwargs={"n_folds": 3},
               combiner=["class_aware_voting", "meta_learner"], output_dir=tmp_path)
    import pandas as pd
    saved = pd.read_csv(tmp_path / "stacking" / "predictions" / "predictions_full.csv")
    voter = r["combiner_predictions"]["class_aware_voting"].full
    assert np.allclose(saved.filter(like="prob_").to_numpy(), voter.filter(like="prob_").to_numpy())


def test_chosen_meta_learner_is_reported(all_run):
    """The winner is whichever candidate scored best, and every candidate is scored.

    This used to assert ``best_meta_name == "logistic_regression"``, which held
    only because the default was a single LR candidate — with one candidate the
    winner is a foregone conclusion. Since 2026-09-27 the default compares the
    four in ``DEFAULT_META_LEARNERS``, so the winner depends on the data. Pinning
    a different name would just re-pin an accident of this 28-row fixture; what
    matters is that the reported winner really is the argmax and that the run
    records all four scores.
    """
    from multimodalva.ensemble.stacking import DEFAULT_META_LEARNERS

    r, d = all_run
    expected = [spec["model_name"] for spec in DEFAULT_META_LEARNERS]

    assert r["meta_select_metric"] == "f1_macro"
    # Every candidate is scored, a lone one included, so the run always records
    # how well the chosen combiner did on the OOF matrix.
    assert set(r["meta_scores"]) == set(expected)
    assert all(v is not None for v in r["meta_scores"].values())

    best = max(r["meta_scores"], key=lambda k: r["meta_scores"][k])
    assert r["best_meta_name"] == best, r["meta_scores"]
    assert r["best_meta_name"] in expected

    meta = json.loads((d / "training_metadata.json").read_text())
    assert meta["best_meta_name"] == r["best_meta_name"]
    assert meta["combiner"] == ALL
    assert meta["ensemble_selection_weights"]


@pytest.mark.parametrize("method,key", [
    ("class_aware_voting", "class_aware_voting"),
    ("ensemble_selection", "ensemble_selection"),
])
def test_oof_from_reuses_stage_one_without_retraining(all_run, data, tmp_path, method, key):
    r, d = all_run
    df, _ = data
    reused = mv.run(task="stacking", data=df, label_col="cause_of_death",
                    combiner=method, oof_from=str(d),
                    ensemble_selection_kwargs={"ensemble_size": 10}, output_dir=tmp_path)
    out = tmp_path / "stacking"
    assert not (out / "final").exists() and not (out / "oof").exists(), "stage 1 was recomputed"
    a = r["combiner_predictions"][key].full.filter(like="prob_").to_numpy()
    b = reused["predictions"].full.filter(like="prob_").to_numpy()
    assert np.allclose(a, b), f"reused OOF gave a different {method} result"
    assert (out / "predictions" / "predictions_top1.csv").is_file(), "main result not written"


def test_unknown_combiner_is_rejected_before_stage_one(data, tmp_path):
    df, feats = data
    with pytest.raises(ValueError, match="Unknown combiner"):
        mv.run(task="stacking", data=df, label_col="cause_of_death", features=feats,
               tabular_models=SPECS, combiner=["meta_learner", "voting"],
               output_dir=tmp_path)
    assert not (tmp_path / "stacking" / "oof").exists()


META = [{"model_name": "logistic_regression", "hyperparams": {"max_iter": 1000, "C": 1.0}},
        {"model_name": "random_forest", "hyperparams": {"n_estimators": 25}}]


def test_meta_learner_names_and_simple_average_each_get_a_folder(all_run, data, tmp_path):
    """Listed meta-learners must not overwrite each other, or meta_learner/."""
    import pandas as pd

    r, d = all_run
    df, _ = data
    listed = ["simple_average", "ensemble_selection", "random_forest", "logistic_regression"]
    out = mv.run(task="stacking", data=df, label_col="cause_of_death",
                 combiner=listed, oof_from=str(d),
                 init_kwargs={"meta_learners": META},
                 ensemble_selection_kwargs={"ensemble_size": 10}, output_dir=tmp_path)
    s = tmp_path / "stacking"
    assert out["combiner"] == listed
    for folder in ("simple_average", "ensemble_selection",
                   "meta_learner_random_forest", "meta_learner_logistic_regression"):
        assert (s / folder / "predictions" / "predictions_full.csv").is_file(), folder
    assert not (s / "meta_learner").exists(), "a listed meta-learner wrote meta_learner/"
    main = pd.read_csv(s / "predictions" / "predictions_full.csv").filter(like="prob_")
    first = out["combiner_predictions"]["simple_average"].full.filter(like="prob_")
    assert np.allclose(main.to_numpy(), first.to_numpy()), "main is not the first listed"

    # The spec with the same model_name supplies the hyperparameters.
    rf = json.loads((s / "meta_learner_random_forest" / "meta_learner_metadata.json").read_text())
    assert rf["best_spec"]["hyperparams"] == {"n_estimators": 25}
    assert rf["best_meta_name"] == "random_forest" and rf["meta_scores"]["random_forest"]


def test_best_chooses_on_oof_and_matches_naming_the_winner(all_run, data, tmp_path):
    """"best" = the comparison's winner, trained exactly as if it were named."""
    from multimodalva.results.validation import summarize_validation

    r, d = all_run
    df, _ = data
    best_dir = tmp_path / "best"
    out = mv.run(task="stacking", data=df, label_col="cause_of_death",
                 combiner="best", oof_from=str(d),
                 init_kwargs={"meta_learners": META}, n_combiner_folds=3,
                 ensemble_selection_kwargs={"ensemble_size": 10}, output_dir=best_dir)
    table = out["combiner_comparison"]
    assert table["combiner"].tolist() == [
        "simple_average", "class_aware_voting", "ensemble_selection",
        "logistic_regression", "random_forest",
    ]
    winner = table["combiner"].iloc[int(table["cv_mean"].argmax())]
    assert out["combiner_chosen"] == winner and out["combiner"] == [winner]
    assert set(out["combiner_predictions"]) == {winner}, "only the winner is applied"

    s = best_dir / "stacking"
    meta = json.loads((s / "training_metadata.json").read_text())
    assert meta["combiner_requested"] == ["best"] and meta["combiner_chosen"] == winner
    assert (s / "combiner_comparison" / "combiner_comparison.csv").is_file()

    named = mv.run(task="stacking", data=df, label_col="cause_of_death",
                   combiner=winner, oof_from=str(d),
                   init_kwargs={"meta_learners": META},
                   ensemble_selection_kwargs={"ensemble_size": 10},
                   output_dir=tmp_path / "named")
    a = out["predictions"].full.filter(like="prob_").to_numpy()
    b = named["predictions"].full.filter(like="prob_").to_numpy()
    assert np.allclose(a, b), "'best' trained its winner differently from naming it"

    v = summarize_validation(s)
    assert v["source"] == "combiner_cv" and v["chosen"] == winner
    assert v["scores"]["f1_macro"] == pytest.approx(float(table["cv_mean"].max()))


def test_meta_learner_search_is_gone():
    import inspect

    from multimodalva.ensemble.stacking import StackingClassifier

    for meth in ("run", "train_meta_learner_stage"):
        params = inspect.signature(getattr(StackingClassifier, meth)).parameters
        assert "meta_hyperparams" not in params
        assert "meta_use_optimize" not in params and "use_optimize" not in params


# --- stage 1 and stage 2 as separate calls ----------------------------------
# The two stages want different hardware: stage 1 trains every base model over
# every fold, stage 2 fits a combiner on the saved out-of-fold matrix in
# seconds. oof_only= stops after the first; oof_from= starts at the second.

def _tiny_specs():
    return [{"model_name": "random_forest", "hyperparams": {"n_estimators": 5}},
            {"model_name": "naive_bayes", "hyperparams": {}}]


@pytest.fixture(scope="module")
def stage1_only(tmp_path_factory):
    import multimodalva as mv

    df = mv.data("va_sample", n_per_class=10)
    df = df.assign(rowid=[f"R{i}" for i in range(len(df))])
    feats = [c for c in df.columns if c not in ("cause_of_death", "narrative", "rowid")]
    out = tmp_path_factory.mktemp("stage1")
    r = mv.run(task="stacking", data=df, label_col="cause_of_death", features=feats,
               id_col="rowid", tabular_models=_tiny_specs(), output_dir=out,
               init_kwargs={"n_folds": 3}, oof_only=True)
    return r


def test_oof_only_stops_before_any_combiner(stage1_only):
    r = stage1_only
    root = Path(r["output_dir"])
    assert r["predictions"] is None
    assert r["oof_meta_X"].shape[0] == r["oof_y"].shape[0]
    assert Path(r["oof_metadata"]).is_file()
    for combiner_dir in ("meta_learner", "simple_average", "class_voter",
                         "ensemble_selection", "predictions"):
        assert not (root / combiner_dir).exists(), (
            f"{combiner_dir} was written although only stage 1 was asked for"
        )


def test_stage_two_runs_later_from_the_saved_oof(stage1_only, tmp_path):
    """And more than once, with different combiners, retraining nothing."""
    import multimodalva as mv

    first = mv.run(task="stacking", output_dir=tmp_path / "s2",
                   oof_from=stage1_only["output_dir"],
                   combiner=["meta_learner", "simple_average"])
    again = mv.run(task="stacking", output_dir=tmp_path / "s3",
                   oof_from=stage1_only["output_dir"],
                   combiner="class_aware_voting")
    assert len(first["predictions"].top1) == len(again["predictions"].top1)
    # stage 1 was not recomputed: no fold directories under either stage-2 run
    for out in (tmp_path / "s2", tmp_path / "s3"):
        assert not list(out.rglob("fold_*"))


def test_oof_only_with_oof_from_is_refused(tmp_path, stage1_only):
    import multimodalva as mv

    with pytest.raises(ValueError, match="nothing to do"):
        mv.run(task="stacking", output_dir=tmp_path / "both",
               oof_from=stage1_only["output_dir"], oof_only=True)


def test_data_is_still_required_without_oof_from(tmp_path):
    import multimodalva as mv

    with pytest.raises(TypeError, match="needs data= and label_col="):
        mv.run(task="tabular", output_dir=tmp_path / "nodata")
