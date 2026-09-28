"""A base model inside an ensemble must be trained the way it would be alone.

Stacking cannot call ``TextClassifier.run`` / ``TabularClassifier.run``: each of
those owns its own train/test split, and stacking needs ONE split, then k folds,
then a full-data refit. So it drives the low-level functions itself, and every
setting has to reach it through the base-model spec.

That hand-copying is what drifted. Before these tests a text base model searched
with ``metric="f1_macro"`` and LoRA off while the same model run on its own
searched with ``metric="accuracy"`` and LoRA on — a different objective and a
different space, for the same model on the same data.

Since then the search settings moved into :class:`Optimize`, which is passed
through as one object and cannot drift. What is left in the spec defaults is the
training knobs, which still have to match the classifier, plus the per-family
trial budget. These tests pin both.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from multimodalva import Optimize
from multimodalva.tabular import TabularClassifier
from multimodalva.tabular.hpo import optimize_tabular
from multimodalva.text import TextClassifier
from multimodalva.text.hpo import optimize_text
from multimodalva.utils.hpo_defaults import (
    TABULAR_SPEC_DEFAULTS,
    TEXT_SPEC_DEFAULTS,
)


def _default(func, name):
    param = inspect.signature(func).parameters.get(name)
    assert param is not None, f"{func.__qualname__} has no parameter {name!r}"
    return param.default


def _optimize_default(name):
    for f in dataclasses.fields(Optimize):
        if f.name == name:
            return f.default
    raise AssertionError(f"Optimize has no field {name!r}")


# --- training knobs: spec must agree with the classifier -------------------

TEXT_TRAINING_KNOBS = [("use_lora", "use_lora"),
                       ("use_focal", "use_focal"),
                       ("use_fast", "use_fast")]


@pytest.mark.parametrize("spec_key, run_param", TEXT_TRAINING_KNOBS)
def test_text_training_knobs_match_the_classifier(spec_key, run_param):
    assert TEXT_SPEC_DEFAULTS[spec_key] == _default(TextClassifier.run, run_param), (
        f"stacking text spec default {spec_key!r} has drifted from "
        f"TextClassifier.run({run_param}=). A text base model would be trained "
        f"differently inside an ensemble than on its own."
    )


# --- search settings: one source of truth, and it is Optimize ---------------

def test_search_settings_are_not_duplicated_in_spec_defaults():
    """Anything Optimize owns must not also sit in the spec defaults.

    Two copies of a default is exactly how the last drift happened.
    """
    owned_by_optimize = {f.name for f in dataclasses.fields(Optimize)}
    aliases = {"use_cv": "cv", "n_cv_folds": "cv_folds", "resume_hpo": "resume",
               "search_space": "space"}
    for defaults, name in [(TEXT_SPEC_DEFAULTS, "TEXT"),
                           (TABULAR_SPEC_DEFAULTS, "TABULAR")]:
        for key in defaults:
            field = aliases.get(key, key)
            assert field not in owned_by_optimize or key in {
                "n_trials", "optimize_metric", "search_space_profile"
            }, (
                f"{name}_SPEC_DEFAULTS[{key!r}] duplicates an Optimize field. "
                "Search settings belong on Optimize only."
            )


def test_classifiers_no_longer_carry_search_arguments():
    """The old flat arguments are gone; hyperparams= is the only way in."""
    for cls in (TextClassifier, TabularClassifier):
        params = set(inspect.signature(cls.run).parameters)
        for gone in ("use_optimize", "n_trials", "optimize_metric",
                     "search_space", "use_cv", "n_cv_folds", "resume_hpo"):
            assert gone not in params, (
                f"{cls.__name__}.run still takes {gone}=. It belongs on "
                "Optimize now."
            )
        assert "hyperparams" in params


def test_optimize_defaults_are_the_sensible_ones():
    assert _optimize_default("metric") == "f1_macro"
    assert _optimize_default("cv") is True
    assert _optimize_default("cv_folds") == 3
    assert _optimize_default("resume") is None    # follows the run's resume=
    assert _optimize_default("n_trials") is None   # filled per family


def test_trial_budget_differs_by_family():
    """One text trial costs far more than one tabular trial."""
    assert TEXT_SPEC_DEFAULTS["n_trials"] == 30
    assert TABULAR_SPEC_DEFAULTS["n_trials"] == 50


# --- the objective ----------------------------------------------------------

def test_search_metric_is_f1_macro_everywhere():
    """One default objective across all six pipelines.

    ``accuracy`` rewards getting the common causes right, so it is the wrong
    default for verbal autopsy, where the rare causes are the point. The
    analysis this package was built for uses ``f1_macro`` throughout.
    """
    from multimodalva.ensemble.data_fusion_classifier import DataFusionClassifier
    from multimodalva.ensemble.feature_fusion import FeatureFusionClassifier

    assert _optimize_default("metric") == "f1_macro"
    assert _default(optimize_text, "metric") == "f1_macro"
    assert _default(optimize_tabular, "metric") == "f1_macro"
    assert _default(FeatureFusionClassifier.__init__, "eval_metric") == "f1_macro"
    assert TEXT_SPEC_DEFAULTS["optimize_metric"] == "f1_macro"
    assert TABULAR_SPEC_DEFAULTS["optimize_metric"] == "f1_macro"
    # data fusion delegates to the text pipeline, so it has no metric of its own
    assert "optimize_metric" not in inspect.signature(DataFusionClassifier.run).parameters


def test_tabular_spec_still_exposes_search_space_profile():
    """Tabular-only, so it stays a spec key rather than an Optimize field."""
    assert TABULAR_SPEC_DEFAULTS["search_space_profile"] == _default(
        optimize_tabular, "search_space_profile"
    )


# --- run() must not hand a stage an argument it does not take ---------------

def test_stacking_run_delegates_only_arguments_the_stages_accept():
    """``run()`` is a thin wrapper over three stage methods.

    Renaming an argument on a stage without updating ``run()`` produces a
    TypeError only once a real stacking run reaches that stage, which unit
    tests never do. This reads the call sites instead.
    """
    import ast
    import pathlib

    import multimodalva.ensemble.stacking as mod

    source = pathlib.Path(mod.__file__).read_text()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "StackingClassifier")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    run = methods["run"]

    problems = []
    for node in ast.walk(run):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name) and func.value.id == "self"):
            continue
        target = methods.get(func.attr)
        if target is None:
            continue
        accepted = {a.arg for a in target.args.args} | {
            a.arg for a in target.args.kwonlyargs
        }
        has_kwargs = target.args.kwarg is not None
        for kw in node.keywords:
            if kw.arg and kw.arg not in accepted and not has_kwargs:
                problems.append(f"run() passes {kw.arg}= to {func.attr}(), "
                                f"which does not accept it")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path,cls,meth", [
    ("multimodalva.ensemble.voting", "SoftVotingClassifier", "run"),
    ("multimodalva.ensemble.stacking", "StackingClassifier", "run"),
    ("multimodalva.ensemble.stacking", "StackingClassifier", "train_base_models"),
])
def test_ensemble_text_training_defaults_match_the_classifier(path, cls, meth):
    """A text base model must stop training the way it would on its own.

    early_stopping_patience was 3 in the ensembles and 4 in TextClassifier, so
    the same model trained differently inside an ensemble than alone.
    """
    import importlib

    fn = getattr(getattr(importlib.import_module(path), cls), meth)
    for name in ("early_stopping_patience", "val_size"):
        assert _default(fn, name) == _default(TextClassifier.run, name), (
            f"{cls}.{meth}({name}=) default differs from TextClassifier.run"
        )
