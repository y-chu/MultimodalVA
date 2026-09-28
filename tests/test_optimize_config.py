"""``hyperparams=`` says where a model's hyperparameters come from.

Three cases, one argument: a dict of fixed values, ``"default"`` for the model
library's own, or ``Optimize(...)`` to search. ``Optimize`` carries the search
settings with it, so an ensemble base-model spec has one key instead of six.

The flat ``use_optimize=True`` spelling is gone. A spec still carrying it must
fail loudly rather than silently not searching — that is what several of these
tests check.
"""

from __future__ import annotations

import pytest

from multimodalva import Optimize
from multimodalva.utils.hpo_defaults import (
    TABULAR_SPEC_DEFAULTS,
    TEXT_SPEC_DEFAULTS,
)
from multimodalva.utils.optimize_config import (
    resolve_hyperparams,
    resolve_spec_hyperparams,
    optimize_from_flags,
)


# --- which of the three cases is it? ---------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ({"learning_rate": 3e-5}, "fixed"),
        ("default", "default"),
        ("DEFAULT", "default"),
        (None, "default"),
        (Optimize(), "search"),
    ],
)
def test_resolve_hyperparams_kinds(value, expected):
    assert resolve_hyperparams(value)[0] == expected


def test_search_string_is_rejected_with_a_usable_message():
    """A bare string cannot carry search settings, so it must not be accepted."""
    with pytest.raises(ValueError) as exc:
        resolve_hyperparams("search")
    assert "Optimize()" in str(exc.value)


def test_unknown_keyword_names_the_alternatives():
    with pytest.raises(ValueError) as exc:
        resolve_hyperparams("auto")
    msg = str(exc.value)
    assert "default" in msg and "Optimize(" in msg


def test_wrong_type_is_a_type_error():
    with pytest.raises(TypeError):
        resolve_hyperparams(5)


# --- the removed spelling must fail loudly, not silently --------------------

@pytest.mark.parametrize("stale_key, value", [
    ("use_optimize", True),
    ("n_trials", 50),
    ("optimize_metric", "csmf_accuracy"),
    ("search_space", {"n_estimators": ("int", 10, 20)}),
    ("use_cv", False),
    ("n_cv_folds", 5),
    ("resume_hpo", False),
])
def test_removed_spec_keys_raise_and_say_what_to_use(stale_key, value):
    """Silently ignoring these would mean quietly not searching at all."""
    with pytest.raises(ValueError) as exc:
        resolve_spec_hyperparams(
            {"model_name": "lightgbm", stale_key: value}, TABULAR_SPEC_DEFAULTS
        )
    msg = str(exc.value)
    assert stale_key in msg
    assert "Optimize(" in msg
    assert "lightgbm" in msg


def test_error_lists_every_stale_key_at_once():
    with pytest.raises(ValueError) as exc:
        resolve_spec_hyperparams(
            {"model_name": "x", "use_optimize": True, "n_trials": 9},
            TABULAR_SPEC_DEFAULTS,
        )
    assert "n_trials" in str(exc.value) and "use_optimize" in str(exc.value)


def test_optimize_from_flags_is_for_flat_surfaces_only():
    """The CLI and config files are flat, so they still build an Optimize."""
    assert optimize_from_flags(False, n_trials=20) is None
    built = optimize_from_flags(True, n_trials=20, optimize_metric="csmf_accuracy")
    assert built == Optimize(n_trials=20, metric="csmf_accuracy")


# --- family defaults fill only what the caller left alone -------------------

def test_family_defaults_fill_unset_fields():
    _, _, s = resolve_spec_hyperparams({"model_name": "x", "hyperparams": Optimize()},
                                       TABULAR_SPEC_DEFAULTS)
    assert s.n_trials == TABULAR_SPEC_DEFAULTS["n_trials"]
    assert s.metric == TABULAR_SPEC_DEFAULTS["optimize_metric"]


def test_explicit_search_fields_survive_family_defaults():
    """An explicitly chosen metric must not be replaced by the family default."""
    _, _, s = resolve_spec_hyperparams(
        {"model_name": "x", "hyperparams": Optimize(metric="accuracy", n_trials=7)},
        TEXT_SPEC_DEFAULTS,
    )
    assert s.metric == "accuracy"
    assert s.n_trials == 7


def test_training_knobs_are_not_optimize_fields():
    """use_lora / use_focal change training whether or not a search runs.

    They belong beside ``model`` in the spec, not inside an object that only
    exists when searching — otherwise fixed-hyperparameter runs would have
    nowhere to set them.
    """
    import dataclasses
    names = {f.name for f in dataclasses.fields(Optimize)}
    assert "use_lora" not in names
    assert "use_focal" not in names


def test_text_and_tabular_get_their_own_trial_budget():
    _, _, t = resolve_spec_hyperparams({"model_name": "x", "hyperparams": Optimize()},
                                       TEXT_SPEC_DEFAULTS)
    _, _, b = resolve_spec_hyperparams({"model_name": "x", "hyperparams": Optimize()},
                                       TABULAR_SPEC_DEFAULTS)
    assert t.n_trials == 30 and b.n_trials == 50


# --- the other two cases ----------------------------------------------------

def test_fixed_values_pass_through_untouched():
    kind, fixed, search = resolve_spec_hyperparams(
        {"model_name": "knn", "hyperparams": {"n_neighbors": 7}}, TABULAR_SPEC_DEFAULTS
    )
    assert (kind, fixed, search) == ("fixed", {"n_neighbors": 7}, None)


def test_silent_spec_means_library_defaults():
    kind, fixed, search = resolve_spec_hyperparams({"model_name": "knn"},
                                                   TABULAR_SPEC_DEFAULTS)
    assert kind == "default" and fixed is None and search is None


def test_fixed_values_are_copied_not_aliased():
    original = {"n_neighbors": 7}
    _, fixed, _ = resolve_spec_hyperparams(
        {"model_name": "knn", "hyperparams": original}, TABULAR_SPEC_DEFAULTS
    )
    fixed["n_neighbors"] = 99
    assert original["n_neighbors"] == 7


# --- the log line -----------------------------------------------------------

def test_describe_names_what_was_configured():
    line = Optimize(metric="csmf_accuracy", n_trials=50,
                  space={"learning_rate": ("float_log", 1e-5, 1e-4)}).describe()
    assert "csmf_accuracy" in line
    assert "50" in line
    assert "learning_rate" in line


def test_search_objects_with_the_same_settings_are_equal():
    assert Optimize(n_trials=5) == Optimize(n_trials=5)
    assert Optimize(n_trials=5) != Optimize(n_trials=6)


def test_search_copies_its_dicts():
    """Freezing stops rebinding, not in-place edits — so the dicts are copied.

    An ensemble often hands the same Optimize to several base models; editing the
    caller's dict afterwards must not reach into them.
    """
    space = {"n_estimators": ("int", 10, 20)}
    s = Optimize(space=space)
    space["n_estimators"] = ("int", 999, 1000)
    assert s.space == {"n_estimators": ("int", 10, 20)}
