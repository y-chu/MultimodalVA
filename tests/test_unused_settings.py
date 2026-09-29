"""Settings a caller passed that the pipeline will not act on must be named.

One ``Optimize`` object is accepted by every task, but not every task honours
every field. Feature fusion hands its search to AutoGluon AutoMM, which brings
its own scheduler, searcher, holdout and metric — so ``backend=``, ``metric=``,
``cv=`` and the rest reach nothing. Dropping those silently is the hardest kind
of problem for a user to notice, especially in a queued cluster job.
"""

from __future__ import annotations

import logging

import pytest

from multimodalva.utils.optimize_config import Optimize, warn_unused_settings

HONOURED = frozenset({"n_trials", "space"})


def test_defaults_are_not_reported():
    """A default the caller never set is not a request."""
    assert warn_unused_settings(Optimize(), HONOURED, pipeline="demo") == []


def test_honoured_fields_are_not_reported():
    search = Optimize(n_trials=20, space={"lr": ("float", 1e-5, 1e-3)})
    assert warn_unused_settings(search, HONOURED, pipeline="demo") == []


def test_unhonoured_fields_are_named_in_declaration_order():
    search = Optimize(backend="ray", metric="csmf_accuracy", cv=False, n_trials=20)
    assert warn_unused_settings(search, HONOURED, pipeline="demo") == [
        "metric", "cv", "backend",
    ]


def test_extra_counts_as_set_only_when_non_empty():
    assert warn_unused_settings(Optimize(extra={}), HONOURED, pipeline="demo") == []
    assert warn_unused_settings(
        Optimize(extra={"ray_address": "auto"}), HONOURED, pipeline="demo"
    ) == ["extra"]


def test_the_warning_names_the_pipeline_and_the_reason(caplog):
    search = Optimize(backend="ray")
    with caplog.at_level(logging.WARNING):
        warn_unused_settings(search, HONOURED, pipeline="feature_fusion",
                             note="AutoMM runs the search itself.")
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "feature_fusion" in msg
    assert "backend" in msg
    assert "AutoMM runs the search itself." in msg


def test_feature_fusion_declares_what_it_honours():
    """The set is what the pipeline actually reads off the Optimize object."""
    import inspect

    from multimodalva.ensemble import feature_fusion as ff

    assert ff.FEATURE_FUSION_HONOURED_SETTINGS == {"n_trials", "space"}
    import ast

    # Every attribute read off the Optimize object, found properly rather than by
    # slicing text: reading a new one without declaring it would make the
    # honoured set — and therefore the warning — a lie.
    tree = ast.parse(inspect.getsource(ff))
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "hp_search"
    }
    used -= {"with_defaults", "describe"}          # helpers, not settings
    undeclared = sorted(used - ff.FEATURE_FUSION_HONOURED_SETTINGS)
    assert not undeclared, (
        f"feature_fusion reads {undeclared} off Optimize but does not declare "
        "them in FEATURE_FUSION_HONOURED_SETTINGS, so warn_unused_settings() "
        "would wrongly report them as ignored"
    )


def test_feature_fusion_calls_the_warning():
    import inspect

    from multimodalva.ensemble import feature_fusion as ff

    assert "warn_unused_settings(" in inspect.getsource(ff)


# --- the text search: space_profile is a tabular concept --------------------

def test_text_search_warns_about_space_profile(caplog):
    """`Optimize(space_profile=...)` reaches nothing on the text path.

    The text search spaces adapt to sample size and class count and have no
    profiles; only the tabular spaces do. The docstring said "Tabular only", but
    behaviour was silent, which is no help to someone who did not read it.
    """
    from unittest.mock import patch

    from multimodalva.text.hpo import search_text

    kw = dict(resume=False, train_dataset=None, label2id={}, id2label={},
              model_name="bert", output_dir="/tmp/unused_settings_probe")
    with caplog.at_level(logging.WARNING):
        with patch("multimodalva.text.hpo.optimize_text", return_value=({}, None)):
            search_text(Optimize(space_profile="wide", n_trials=3), **kw)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "space_profile" in msg and "text search" in msg


def test_text_search_is_quiet_when_every_setting_applies(caplog):
    from unittest.mock import patch

    from multimodalva.text.hpo import search_text

    kw = dict(resume=False, train_dataset=None, label2id={}, id2label={},
              model_name="bert", output_dir="/tmp/unused_settings_probe")
    with caplog.at_level(logging.WARNING):
        with patch("multimodalva.text.hpo.optimize_text", return_value=({}, None)):
            search_text(Optimize(n_trials=3, metric="csmf_accuracy", cv=True), **kw)
    assert not [r for r in caplog.records if "ignored by" in r.getMessage()]


def test_tabular_search_honours_space_profile():
    """The counterpart: tabular does read it, so it must not be warned about."""
    import ast
    import inspect

    from multimodalva.tabular import hpo as tab

    tree = ast.parse(inspect.getsource(tab))
    reads = {
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id == "search"
    }
    assert "space_profile" in reads


# --- drift guard: a new Optimize field must be handled or declared ----------
#
# warn_unused_settings() derives its honoured set from the dataclass, so a field
# added to Optimize is treated as honoured by default. That is the right default
# for noise — it never invents a false warning — but it means a genuinely
# unhandled new field would be silent again. This closes that: every field has to
# be read by the search module, or listed here as deliberately not applicable.

#: Fields no search module reads directly, with why.
_HANDLED_ELSEWHERE = {
    "resume": "resolved by optimize_config.resolve_search_resume()",
}
_NOT_APPLICABLE = {
    "multimodalva/text/hpo.py": {
        "space_profile": "tabular-only; the text spaces have no profiles "
                         "(warned about at the search_text() call)",
    },
    "multimodalva/tabular/hpo.py": {},
}


@pytest.mark.parametrize("module_path", sorted(_NOT_APPLICABLE))
def test_every_optimize_field_is_read_or_declared(module_path):
    import ast
    import dataclasses
    import pathlib

    import multimodalva

    root = pathlib.Path(multimodalva.__file__).parent.parent
    src = (root / module_path).read_text()
    reads = {
        n.attr for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id == "search"
    }
    declared = set(_HANDLED_ELSEWHERE) | set(_NOT_APPLICABLE[module_path])
    unhandled = sorted(
        f.name for f in dataclasses.fields(Optimize)
        if f.name not in reads and f.name not in declared
    )
    assert not unhandled, (
        f"{module_path} neither reads {unhandled} off Optimize nor declares them. "
        "Either act on the field, or add it to _NOT_APPLICABLE here and to the "
        "honoured set at the call to warn_unused_settings(), so the user is told "
        "it was dropped instead of it vanishing."
    )
