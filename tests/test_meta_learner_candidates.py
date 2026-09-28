"""Every meta-learner name the package advertises actually works.

``_meta_learner_names()`` returns ten: ``logistic_regression`` plus all nine
tabular aliases, because ``_resolve_meta_learner`` delegates to the tabular
registry. Two of them used to raise ``TypeError`` the moment anyone passed them —
``naive_bayes`` and ``knn`` accept no ``random_state``, and the resolver injected
one into everything except catboost — while catboost, which does accept it, was
left unseeded. Nobody hit either, because the analysis only ever used four
candidates (LR, LightGBM, XGBoost, random forest).

The fix was to stop re-implementing what ``tabular.train._build_model`` already
does. These tests pin the outcome: an advertised name is a usable name, and the
seed reaches every model that can take one.
"""

from __future__ import annotations

import numpy as np
import pytest

from multimodalva.ensemble.stacking import (
    DEFAULT_META_LEARNERS,
    _meta_learner_names,
    _resolve_meta_learner,
)


@pytest.fixture(scope="module")
def oof():
    """An OOF meta-feature matrix: 3 base models × 4 causes, row-normalised."""
    rng = np.random.default_rng(0)
    X = rng.random((120, 12))
    X /= X.sum(axis=1, keepdims=True)
    y = rng.integers(0, 4, 120)
    return X, y


@pytest.mark.parametrize("name", _meta_learner_names())
def test_every_advertised_meta_learner_fits_and_predicts_proba(name, oof):
    X, y = oof
    model = _resolve_meta_learner({"model_name": name}, n_jobs=1, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    # Stacking needs probabilities, not labels: the combiner scatters these into
    # the full id2label width.
    assert proba.shape == (len(y), len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", ["naive_bayes", "knn"])
def test_seedless_models_are_not_given_a_random_state(name):
    # The specific regression: these two constructors accept no random_state and
    # no **kwargs, so injecting one raised TypeError. They are deterministic and
    # need no seed.
    model = _resolve_meta_learner({"model_name": name}, n_jobs=1, random_state=42)
    assert "random_state" not in model.get_params()


@pytest.mark.parametrize("name", ["logistic_regression", "lightgbm", "xgboost",
                                  "random_forest", "mlp", "gbdt", "svm", "catboost"])
def test_the_seed_reaches_every_model_that_takes_one(name):
    # catboost is in this list on purpose: it was excluded by the old resolver
    # and so ran unseeded, even though CatBoostClassifier accepts random_state.
    model = _resolve_meta_learner({"model_name": name}, n_jobs=1, random_state=7)
    assert model.get_params().get("random_state") == 7


def test_caller_hyperparams_win_over_the_injected_defaults():
    model = _resolve_meta_learner(
        {"model_name": "random_forest", "hyperparams": {"random_state": 99}},
        n_jobs=1, random_state=7,
    )
    assert model.get_params()["random_state"] == 99


def test_svm_still_gets_probability_true():
    # _FORCED_PARAMS, which the resolver would have lost had it kept its own copy
    # of the parameter handling: SVC has no predict_proba without it.
    model = _resolve_meta_learner({"model_name": "svm"}, n_jobs=1, random_state=42)
    assert model.get_params()["probability"] is True


def test_unknown_name_raises_naming_what_is_accepted():
    with pytest.raises(ValueError, match="Unknown meta-learner"):
        _resolve_meta_learner({"model_name": "not_a_model"})


def test_ten_names_accepted_one_used_by_default():
    # Two numbers that are easy to conflate: ten names are *accepted*, and the
    # default candidate list holds exactly one — a multinomial logistic
    # regression. Comparing several is a decision about a particular study, so it
    # is the caller's: Analysis/ picks four of the ten and passes them
    # explicitly. A run that says nothing must not silently do something wider.
    assert len(_meta_learner_names()) == 10
    assert [s["model_name"] for s in DEFAULT_META_LEARNERS] == ["logistic_regression"]


def test_the_default_is_a_list_so_adding_a_candidate_is_not_a_type_change():
    assert isinstance(DEFAULT_META_LEARNERS, list)


def test_every_default_candidate_is_an_accepted_name():
    accepted = set(_meta_learner_names())
    for spec in DEFAULT_META_LEARNERS:
        assert spec["model_name"] in accepted


def test_every_default_candidate_carries_explicit_hyperparams():
    # The meta-learner is never hyperparameter-searched, so a candidate with no
    # settings would silently run on its library defaults.
    for spec in DEFAULT_META_LEARNERS:
        assert spec.get("hyperparams"), spec["model_name"]


def test_default_candidate_names_are_unique():
    # Candidates are reported by model_name in meta_scores.json.
    names = [s["model_name"] for s in DEFAULT_META_LEARNERS]
    assert len(names) == len(set(names))


def test_the_defaults_are_copied_not_shared(oof):
    # A caller mutating a returned spec must not poison the module constant, and
    # the same spec object is reused across candidates in an ensemble.
    from multimodalva.ensemble.stacking import StackingClassifier

    clf = StackingClassifier(text_models=[], tabular_models=[], output_dir="/tmp/x")
    clf.meta_learner_specs[0]["hyperparams"]["C"] = 999
    assert DEFAULT_META_LEARNERS[0]["hyperparams"]["C"] == 1.0



# --- a model outside the ten: refused early, pointed at oof_only -------------

def test_an_unbuildable_candidate_is_refused_before_stage_one():
    """The point of the check: fail on the way in, not after the GPU hours.

    Every candidate is instantiated in stage 2, so without this an unknown name
    surfaces only once every base model has been trained over every fold — on a
    submitted job, hours of compute spent on a crash that was knowable at the
    start.
    """
    import pytest

    from multimodalva.ensemble.stacking import _check_meta_learner_specs

    with pytest.raises(ValueError) as exc:
        _check_meta_learner_specs([{"model_name": "ridge_classifier"}])
    msg = str(exc.value)
    assert "ridge_classifier" in msg
    assert "oof_only=True" in msg, "the error must name the escape hatch"
    assert "oof_meta_X" in msg
    # And it lists what would have worked.
    for name in ("logistic_regression", "mlp", "svm"):
        assert name in msg


def test_duplicate_candidates_are_still_refused():
    import pytest

    from multimodalva.ensemble.stacking import _check_meta_learner_specs

    with pytest.raises(ValueError, match="more than once"):
        _check_meta_learner_specs(
            [{"model_name": "lightgbm"}, {"model_name": "lightgbm"}]
        )


def test_the_default_list_passes_its_own_check():
    from multimodalva.ensemble.stacking import _check_meta_learner_specs

    _check_meta_learner_specs(DEFAULT_META_LEARNERS)


def test_all_ten_accepted_names_pass_the_check():
    from multimodalva.ensemble.stacking import _check_meta_learner_specs

    for name in _meta_learner_names():
        _check_meta_learner_specs([{"model_name": name}])


def test_the_hint_is_one_string_so_every_raiser_says_the_same_thing():
    # Three places raise it: the pre-stage-1 check, _resolve_meta_learner's
    # fallback, and the unknown-combiner error.
    import multimodalva.ensemble.stacking as st

    assert "oof_only=True" in st._UNKNOWN_META_LEARNER_HINT
