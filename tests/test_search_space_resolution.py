"""A caller's search_space keys are used exactly as passed; the rest stay adaptive.

The contract has two halves and both matter:

* keys the caller names are searched over **exactly** the ranges they wrote —
  nothing narrows, rescales or re-profiles them;
* keys the caller does not name keep their adaptive value and **are still
  searched**, so a partial dict never silently shrinks the search.

The run log has to state both, because a run whose real search space differs from
what the caller wrote — in either direction — is the thing this guards against.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from multimodalva.utils.hpo_defaults import merge_search_space

ADAPTIVE = {
    "learning_rate": ("float_log", 1e-5, 1e-3),
    "batch_size": ("categorical", [8, 16]),
    "epochs": ("int", 2, 6),
    "weight_decay": ("float", 0.0, 0.1),
}


def test_no_override_leaves_the_adaptive_space_alone():
    active = dict(ADAPTIVE)
    assert merge_search_space(active, None, logging.getLogger("t")) == ADAPTIVE


def test_named_keys_are_used_exactly_as_passed():
    """Deliberately absurd ranges: nothing may adjust them."""
    active = dict(ADAPTIVE)
    user = {"learning_rate": ("float", 0.5, 0.6), "epochs": ("int", 1, 1)}
    out = merge_search_space(active, user, logging.getLogger("t"))
    assert out["learning_rate"] == ("float", 0.5, 0.6)
    assert out["epochs"] == ("int", 1, 1)


def test_unnamed_keys_keep_their_adaptive_value_and_are_still_searched():
    active = dict(ADAPTIVE)
    out = merge_search_space(active, {"learning_rate": ("float", 0.5, 0.6)},
                             logging.getLogger("t"))
    assert out["batch_size"] == ADAPTIVE["batch_size"]
    assert out["weight_decay"] == ADAPTIVE["weight_decay"]
    assert set(out) == set(ADAPTIVE), "a partial dict must not shrink the search"


def test_a_key_the_adaptive_space_lacks_is_added():
    active = dict(ADAPTIVE)
    out = merge_search_space(active, {"label_smoothing": ("float", 0.0, 0.2)},
                             logging.getLogger("t"))
    assert out["label_smoothing"] == ("float", 0.0, 0.2)


def test_the_log_states_both_halves(caplog):
    """Which keys are the caller's, and which came from the adaptive space."""
    with caplog.at_level(logging.INFO):
        merge_search_space(dict(ADAPTIVE), {"learning_rate": ("float", 0.5, 0.6)},
                           logging.getLogger("t"))
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "used exactly as passed" in msg
    assert "learning_rate" in msg
    assert "come from the adaptive space" in msg
    assert "batch_size" in msg and "epochs" in msg and "weight_decay" in msg


def test_a_full_override_says_the_adaptive_space_contributed_nothing(caplog):
    with caplog.at_level(logging.INFO):
        merge_search_space(dict(ADAPTIVE), dict(ADAPTIVE), logging.getLogger("t"))
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "contributes nothing" in msg


# --- end to end, through the real adaptive machinery ------------------------

def test_the_adaptive_mechanism_never_touches_a_callers_range():
    """The profile/class-tier adjustments run on the defaults, before the merge."""
    from multimodalva.tabular.search_spaces import get_tabular_default_search_space

    adaptive = get_tabular_default_search_space(
        "lightgbm", np.random.default_rng(0).random((500, 40)), "auto", 11)
    user = {"learning_rate": ("float", 0.5, 0.6), "n_estimators": ("int", 7, 8)}
    out = merge_search_space(dict(adaptive), dict(user), logging.getLogger("t"))
    for key, value in user.items():
        assert out[key] == value, f"{key} was modified after the caller set it"
    # and the rest of the adaptive space survives
    assert set(out) == set(adaptive)


def test_data_fusion_keeps_the_memory_critical_keys_when_a_space_is_partial():
    """A partial space used to drop these and exhaust GPU memory.

    Merging is what protects them: the caller names learning_rate, and batch_size
    and gradient_accumulation_steps keep their calibrated values instead of
    vanishing.
    """
    from multimodalva.ensemble.data_fusion_classifier import (
        DATA_FUSION_DEFAULT_SEARCH_SPACE,
    )

    out = merge_search_space(dict(DATA_FUSION_DEFAULT_SEARCH_SPACE),
                             {"learning_rate": ("float_log", 1e-6, 1e-5)},
                             logging.getLogger("t"))
    for key in ("batch_size", "gradient_accumulation_steps"):
        assert out[key] == DATA_FUSION_DEFAULT_SEARCH_SPACE[key]


# --- a malformed space is refused at construction, not at sampling ----------
#
# The searches run study.optimize(..., catch=(Exception,)) so one bad
# configuration cannot kill a long search. A malformed *space* therefore used to
# fail every trial silently and reach the end of the budget with nothing
# completed — after the queue wait on a cluster — and the errors that escaped the
# sampler were unreadable (a NaN bound arrived as OverflowError out of NumPy).

@pytest.mark.parametrize("space,expected", [
    ({"lr": None},                          "expected a tuple"),
    ({"lr": ()},                            "expected a tuple"),
    ({"lr": ("flot", 1e-5, 1e-3)},          "is not one of"),
    ({"lr": ("float", None, 1e-3)},         "is not a number"),
    ({"lr": ("float", float("nan"), 1e-3)}, "is not finite"),
    ({"lr": ("float", float("inf"), 1.0)},  "is not finite"),
    ({"lr": ("float", 1e-3, 1e-5)},         "greater than"),
    ({"lr": ("float", 1e-5)},               "exactly a low and a high"),
    ({"lr": ("categorical", None)},         "must be a list of choices"),
    ({"lr": ("categorical", [])},           "at least one choice"),
])
def test_a_malformed_space_is_refused_when_the_optimize_is_built(space, expected):
    from multimodalva import Optimize

    with pytest.raises(ValueError, match="malformed"):
        Optimize(space=space)
    with pytest.raises(ValueError, match=expected):
        Optimize(space=space)


def test_every_problem_is_named_at_once():
    """Not one error per run — fixing them one submission at a time is the cost."""
    from multimodalva import Optimize

    with pytest.raises(ValueError) as exc:
        Optimize(space={"a": ("float", float("nan"), 1.0),
                        "b": ("categorical", []),
                        "c": None})
    msg = str(exc.value)
    assert "a:" in msg and "b:" in msg and "c:" in msg


def test_well_formed_spaces_are_accepted():
    from multimodalva import Optimize

    ok = Optimize(space={
        "learning_rate": ("float_log", 1e-5, 1e-3),
        "weight_decay": ("float", 0.0, 0.1),
        "epochs": ("int", 2, 6),
        "batch_size": ("categorical", [8, 16]),
        "pinned": ("categorical", [8]),          # a single choice pins the value
        "equal_bounds": ("float", 0.5, 0.5),     # degenerate but meaningful
    })
    assert len(ok.space) == 6
    assert Optimize(space=None).space is None


def test_a_single_choice_pins_the_value_every_trial():
    """The documented way to hold one hyperparameter fixed while searching others."""
    import optuna

    from multimodalva.utils.metrics import sample_hyperparams

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = {"learning_rate": ("float_log", 1e-5, 1e-3),
             "batch_size": ("categorical", [8])}
    seen = set()

    def objective(trial):
        seen.add(sample_hyperparams(trial, space)["batch_size"])
        return 0.0

    optuna.create_study().optimize(objective, n_trials=5)
    assert seen == {8}
