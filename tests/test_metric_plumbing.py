"""The metric a caller asks for must be the metric the search optimises.

``Optimize(metric=...)` travels: hyperparams= or a base-model spec →
resolve_hyperparams() → one Optimize → search_text()/search_tabular() → the
backend's objective. Seven call sites feed those two functions (text, tabular,
data fusion via TextClassifier, and the text/tabular halves of voting and
stacking), and a single one that rebuilt the object or passed its own metric=
would silently optimise something the caller did not ask for.
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import patch

import pytest

from multimodalva.utils.optimize_config import Optimize

_FUNNELS = ("search_text", "search_tabular")


def _call_sites():
    """Every call to search_text/search_tabular in the package, as AST nodes."""
    import pathlib

    import multimodalva

    root = pathlib.Path(multimodalva.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in _FUNNELS):
                yield path.relative_to(root), node


def test_every_call_site_passes_the_callers_optimize_through():
    """First positional argument, unmodified — not a rebuilt Optimize."""
    sites = list(_call_sites())
    assert len(sites) >= 7, f"expected the known call sites, found {len(sites)}"
    for rel, node in sites:
        assert node.args, f"{rel}:{node.lineno} calls {node.func.id}() with no search object"
        first = node.args[0]
        assert isinstance(first, ast.Name), (
            f"{rel}:{node.lineno} passes a constructed object to {node.func.id}(); "
            "the caller's Optimize must be forwarded as-is or its metric can differ "
            f"from what was asked for (got {ast.dump(first)[:60]})"
        )


def test_no_call_site_overrides_the_metric():
    for rel, node in _call_sites():
        overridden = [kw.arg for kw in node.keywords if kw.arg == "metric"]
        assert not overridden, (
            f"{rel}:{node.lineno} passes metric= to {node.func.id}(), which would "
            "override Optimize(metric=...) that the caller set"
        )


@pytest.mark.parametrize("metric", ["f1_macro", "csmf_accuracy", "accuracy"])
@pytest.mark.parametrize("backend", ["optuna", "ray"])
def test_search_text_forwards_the_metric_to_either_backend(metric, backend):
    from multimodalva.text import hpo

    target = "optimize_text" if backend == "optuna" else "optimize_text_ray"
    with patch.object(hpo, target, return_value=({}, None)) as fake:
        with patch("multimodalva.utils.optimize_config.resolve_backend",
                   return_value=backend):
            hpo.search_text(
                Optimize(metric=metric, n_trials=2), resume=False,
                train_dataset=None, label2id={}, id2label={},
                model_name="bert", output_dir="/tmp/metric_probe",
            )
    assert fake.call_args.kwargs["metric"] == metric


@pytest.mark.parametrize("metric", ["f1_macro", "csmf_accuracy"])
@pytest.mark.parametrize("backend", ["optuna", "ray"])
def test_search_tabular_forwards_the_metric_to_either_backend(metric, backend):
    import numpy as np

    from multimodalva.tabular import hpo

    target = "optimize_tabular" if backend == "optuna" else "optimize_tabular_ray"
    with patch.object(hpo, target, return_value=({}, None)) as fake:
        with patch("multimodalva.utils.optimize_config.resolve_backend",
                   return_value=backend):
            hpo.search_tabular(
                Optimize(metric=metric, n_trials=2), resume=False,
                X_train=np.zeros((4, 2)), y_train=np.array([0, 1, 0, 1]),
                label2id={"a": 0, "b": 1}, id2label={0: "a", 1: "b"},
                model_name="lightgbm", output_dir="/tmp/metric_probe",
            )
    assert fake.call_args.kwargs["metric"] == metric


def test_feature_fusion_selects_on_eval_metric_not_optimize_metric():
    """The one pipeline where Optimize(metric=) does not apply — and it says so."""
    from multimodalva.ensemble import feature_fusion as ff

    assert "metric" not in ff.FEATURE_FUSION_HONOURED_SETTINGS
    src = inspect.getsource(ff)
    assert "warn_unused_settings(" in src
