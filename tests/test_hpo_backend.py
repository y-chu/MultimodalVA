"""``Optimize(backend=)`` picks the search engine, and ``pruning=`` reaches it.

Two things are checked here that used to be impossible or silently broken:

* The Ray backend is reachable from one call. Before 2026-09-27 the only way to
  it was the low-level ``optimize_text_ray()`` / ``optimize_tabular_ray()``, so
  an HPC user had to drop out of the one-call API — and ensembles could not use
  it at all, because their base-model searches are internal.
* ``Optimize(pruning=...)`` is forwarded as ``enable_pruning=``. It used to be
  declared and then dropped on the floor at every call site, which
  ``pruning=None`` hid because it happens to equal each family's own default.

Neither backend actually runs here: both dispatchers are called with the real
``Optimize`` and a stubbed backend function, so the tests assert the *mapping*,
which is where the seven call sites used to disagree.
"""

from __future__ import annotations

import pytest

from multimodalva import Optimize
from multimodalva.utils.optimize_config import (
    BACKENDS,
    resolve_backend,
    resolve_pruning,
)


# --- the field itself -------------------------------------------------------

def test_backend_defaults_to_auto():
    assert Optimize().backend == "auto"


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_documented_backend_is_accepted(backend):
    assert Optimize(backend=backend).backend == backend


def test_unknown_backend_raises_naming_the_valid_ones():
    with pytest.raises(ValueError, match="not recognised"):
        Optimize(backend="rey")


def test_backend_is_part_of_equality():
    assert Optimize(backend="ray") != Optimize(backend="optuna")


def test_with_defaults_does_not_override_an_explicit_backend():
    # A run-level default must never overwrite what a base-model spec said.
    assert Optimize(backend="optuna").with_defaults(backend="ray").backend == "optuna"
    assert Optimize().with_defaults(backend="ray").backend == "ray"


# --- resolve_backend --------------------------------------------------------

@pytest.mark.parametrize("explicit", ["optuna", "ray"])
def test_explicit_backend_is_returned_unchanged(explicit):
    assert resolve_backend(explicit) == explicit


def test_auto_picks_ray_under_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert resolve_backend("auto") == "ray"


def test_auto_picks_optuna_without_cuda_or_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_backend("auto") == "optuna"


def test_auto_picks_ray_with_cuda(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_backend("auto") == "ray"


# --- ray lives in an extra: auto must not resolve to a backend that is absent --

def _pretend_ray_missing(monkeypatch):
    import multimodalva.utils.optimize_config as oc

    monkeypatch.setattr(oc, "ray_is_available", lambda: False)


def test_auto_falls_back_to_optuna_when_ray_is_not_installed(monkeypatch):
    # The whole point of the [ray] extra: a CUDA machine without ray still runs.
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    _pretend_ray_missing(monkeypatch)
    assert resolve_backend("auto") == "optuna"


def test_that_fallback_warns_rather_than_passing_silently(monkeypatch, caplog):
    # Silence would be worse than an error here: the job would run its trials one
    # at a time, produce correct results, and nobody would notice until the wall
    # clock did.
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    _pretend_ray_missing(monkeypatch)
    with caplog.at_level("WARNING"):
        resolve_backend("auto")
    assert "multimodalva[ray]" in caplog.text
    assert "SLURM" in caplog.text


def test_no_warning_when_ray_would_not_have_been_used_anyway(monkeypatch, caplog):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    _pretend_ray_missing(monkeypatch)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        assert resolve_backend("auto") == "optuna"
    assert "multimodalva[ray]" not in caplog.text


def test_explicit_ray_is_never_downgraded(monkeypatch):
    # An explicit request deserves an explicit ImportError from the backend,
    # not a silent switch to a different search engine.
    _pretend_ray_missing(monkeypatch)
    assert resolve_backend("ray") == "ray"


def test_ray_availability_is_a_spec_lookup_not_an_import():
    # Importing ray costs seconds and starts background machinery; this is called
    # just to resolve a string.
    import sys

    from multimodalva.utils.optimize_config import ray_is_available

    sys.modules.pop("ray", None)
    assert isinstance(ray_is_available(), bool)
    assert "ray" not in sys.modules


def test_auto_survives_a_torch_probe_that_raises(monkeypatch):
    # A driver mismatch makes torch.cuda.is_available() raise rather than
    # return False; resolving a backend must not take the whole run down.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    import torch

    def boom():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert resolve_backend("auto") == "optuna"


def test_describe_reports_the_resolved_backend(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    line = Optimize().describe()
    assert "backend=ray (from auto)" in line
    # An explicitly named backend is not annotated as coming from auto.
    assert "from auto" not in Optimize(backend="ray").describe()


# --- resolve_pruning --------------------------------------------------------

def test_pruning_none_follows_the_family_default():
    assert resolve_pruning(Optimize(), family_default=True) is True
    assert resolve_pruning(Optimize(), family_default=False) is False


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_pruning_beats_the_family_default(explicit):
    assert resolve_pruning(Optimize(pruning=explicit), family_default=not explicit) is explicit


def test_describe_mentions_pruning():
    assert "pruning=family default" in Optimize().describe()
    assert "pruning=False" in Optimize(pruning=False).describe()


# --- the dispatchers --------------------------------------------------------

@pytest.fixture
def text_calls(monkeypatch):
    """Record what search_text() passes, without running a search."""
    import multimodalva.text.hpo as hpo

    seen: dict = {}

    def fake_optuna(**kwargs):
        seen["backend"] = "optuna"
        seen["kwargs"] = kwargs
        return {"epochs": 3}, "study"

    def fake_ray(**kwargs):
        seen["backend"] = "ray"
        seen["kwargs"] = kwargs
        return {"epochs": 3}, "result_grid"

    monkeypatch.setattr(hpo, "optimize_text", fake_optuna)
    monkeypatch.setattr(hpo, "optimize_text_ray", fake_ray)
    return seen


COMMON = dict(
    train_dataset=object(),
    label2id={"a": 0},
    id2label={0: "a"},
    model_name="prajjwal1/bert-tiny",
    output_dir="/tmp/nowhere",
    random_state=42,
    split_seed=7,
    use_lora=False,
    use_focal=False,
    gradient_checkpointing=False,
    early_stopping_patience=4,
    use_fast=True,
)


def test_search_text_routes_to_optuna(text_calls):
    from multimodalva.text.hpo import search_text

    search_text(Optimize(backend="optuna"), resume=True, **COMMON)
    assert text_calls["backend"] == "optuna"


def test_search_text_routes_to_ray(text_calls):
    from multimodalva.text.hpo import search_text

    search_text(Optimize(backend="ray"), resume=True, **COMMON)
    assert text_calls["backend"] == "ray"


def test_optuna_path_gets_load_if_exists_and_enable_pruning(text_calls):
    from multimodalva.text.hpo import search_text
    from multimodalva.text.hpo import TEXT_PRUNING_DEFAULT

    search_text(Optimize(backend="optuna"), resume=False, **COMMON)
    kw = text_calls["kwargs"]
    assert kw["load_if_exists"] is False
    assert kw["enable_pruning"] is TEXT_PRUNING_DEFAULT
    assert "resume" not in kw


def test_pruning_false_actually_reaches_the_optuna_backend(text_calls):
    # The regression this file exists for: Optimize(pruning=False) used to be
    # accepted, logged, and then ignored.
    from multimodalva.text.hpo import search_text

    search_text(Optimize(backend="optuna", pruning=False), resume=True, **COMMON)
    assert text_calls["kwargs"]["enable_pruning"] is False


def test_ray_path_gets_resume_and_no_pruning_argument(text_calls):
    from multimodalva.text.hpo import search_text

    search_text(Optimize(backend="ray"), resume=False, **COMMON)
    kw = text_calls["kwargs"]
    assert kw["resume"] is False
    assert "load_if_exists" not in kw
    assert "enable_pruning" not in kw


def test_ray_path_warns_that_pruning_is_ignored(text_calls, caplog):
    from multimodalva.text.hpo import search_text

    with caplog.at_level("WARNING"):
        search_text(Optimize(backend="ray", pruning=False), resume=True, **COMMON)
    assert "ignored by the Ray backend" in caplog.text


def test_cv_settings_reach_both_backends(text_calls):
    from multimodalva.text.hpo import search_text

    for backend in ("optuna", "ray"):
        search_text(
            Optimize(backend=backend, cv=True, cv_folds=5), resume=True, **COMMON
        )
        kw = text_calls["kwargs"]
        assert kw["use_cv"] is True, backend
        assert kw["n_cv_folds"] == 5, backend


def test_n_trials_none_is_left_to_the_backend_default(text_calls):
    from multimodalva.text.hpo import search_text

    search_text(Optimize(backend="ray"), resume=True, **COMMON)
    # Ray resolves None to 30 or 60 depending on ASHA; passing it through would
    # freeze one of those choices at the call site.
    assert "n_trials" not in text_calls["kwargs"]

    search_text(Optimize(backend="ray", n_trials=12), resume=True, **COMMON)
    assert text_calls["kwargs"]["n_trials"] == 12


def test_extra_reaches_the_ray_backend_verbatim(text_calls):
    from multimodalva.text.hpo import search_text

    search_text(
        Optimize(backend="ray", extra={"ray_address": "auto", "use_asha": True}),
        resume=True,
        **COMMON,
    )
    kw = text_calls["kwargs"]
    assert kw["ray_address"] == "auto"
    assert kw["use_asha"] is True


def test_split_seed_reaches_both_text_backends(text_calls):
    from multimodalva.text.hpo import search_text

    for backend in ("optuna", "ray"):
        search_text(Optimize(backend=backend), resume=True, **COMMON)
        assert text_calls["kwargs"]["split_seed"] == 7, backend


# --- tabular dispatcher -----------------------------------------------------

@pytest.fixture
def tabular_calls(monkeypatch):
    import multimodalva.tabular.hpo as hpo

    seen: dict = {}

    def fake(**kwargs):
        seen["kwargs"] = kwargs
        return {"n_estimators": 100}, "study"

    def fake_ray(**kwargs):
        seen["kwargs"] = kwargs
        seen["ray"] = True
        return {"n_estimators": 100}, "grid"

    monkeypatch.setattr(hpo, "optimize_tabular", fake)
    monkeypatch.setattr(hpo, "optimize_tabular_ray", fake_ray)
    return seen


TABULAR_COMMON = dict(
    X_train=None,
    y_train=None,
    label2id={"a": 0},
    id2label={0: "a"},
    model_name="lightgbm",
    output_dir="/tmp/nowhere",
    random_state=42,
    split_seed=7,
    n_jobs=1,
    use_gpu=False,
)


def test_search_tabular_maps_space_profile_and_pruning(tabular_calls):
    from multimodalva.tabular.hpo import search_tabular, TABULAR_PRUNING_DEFAULT

    search_tabular(
        Optimize(backend="optuna", space_profile="wide"), resume=True, **TABULAR_COMMON
    )
    kw = tabular_calls["kwargs"]
    assert kw["search_space_profile"] == "wide"
    assert kw["enable_pruning"] is TABULAR_PRUNING_DEFAULT
    assert kw["load_if_exists"] is True


def test_search_tabular_routes_to_ray_with_split_seed(tabular_calls):
    from multimodalva.tabular.hpo import search_tabular

    search_tabular(Optimize(backend="ray"), resume=True, **TABULAR_COMMON)
    assert tabular_calls.get("ray") is True
    # split_seed had no counterpart on the Ray path until 2026-09-27; without it
    # switching backend silently redrew the folds.
    assert tabular_calls["kwargs"]["split_seed"] == 7


# --- the Ray signatures the dispatchers rely on -----------------------------

def test_ray_backends_accept_the_arguments_the_dispatcher_sends():
    import inspect

    from multimodalva.text.hpo import optimize_text_ray
    from multimodalva.tabular.hpo import optimize_tabular_ray

    text = inspect.signature(optimize_text_ray).parameters
    for name in ("use_cv", "n_cv_folds", "split_seed", "resume", "metric", "search_space"):
        assert name in text, name

    tab = inspect.signature(optimize_tabular_ray).parameters
    for name in ("use_cv", "n_cv_folds", "split_seed", "resume", "search_space_profile"):
        assert name in tab, name


def test_text_ray_trial_fn_supports_both_scoring_modes():
    import inspect

    from multimodalva.text.hpo import _ray_trial_fn

    p = inspect.signature(_ray_trial_fn).parameters
    for name in ("opt_train", "opt_val", "train_dataset", "cv_splits", "metric_key"):
        assert name in p, name


# --- CLI --------------------------------------------------------------------

def test_cli_hpo_backend_builds_an_optimize():
    from multimodalva.utils.optimize_config import optimize_from_flags

    search = optimize_from_flags(True, backend="ray")
    assert isinstance(search, Optimize)
    assert search.backend == "ray"


def test_cli_hpo_backend_without_optimize_is_refused():
    from multimodalva.cli import main

    with pytest.raises(SystemExit, match="only mean something"):
        main(["run", "--task", "text", "--data", "demo", "--label-col", "y",
              "--output-dir", "/tmp/nowhere", "--hpo-backend", "ray"])



# --- packaging: ray is an extra, not a core dependency ----------------------

def _pyproject() -> dict:
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


RAY_ONLY_DEPS = ("ray", "grpcio", "protobuf", "optuna-integration")


def test_core_dependencies_contain_nothing_ray_only():
    # 363 MB of them, measured 2026-09-27 — more than torch. And it keeps the
    # protobuf<5 ceiling, this project's most conflict-prone pin, out of the
    # dependency set of users who never run a Ray search.
    core = _pyproject()["project"]["dependencies"]
    leaked = [d for d in core if any(d.startswith(n) for n in RAY_ONLY_DEPS)]
    assert leaked == [], f"ray-only dependencies leaked back into core: {leaked}"


def test_optuna_itself_stays_in_core():
    # The Optuna backend is the default and produced every published result.
    core = _pyproject()["project"]["dependencies"]
    assert any(d.startswith("optuna>") or d.startswith("optuna=") for d in core)


def test_ray_extra_exists_and_all_includes_it():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "ray" in extras, "the ImportError messages promise multimodalva[ray]"
    assert any(d.startswith("ray[tune]") for d in extras["ray"])
    assert any(d.startswith("ray[tune]") for d in extras["all"]), "[all] must be complete"


def test_ray_pin_stays_inside_autogluons_window():
    # autogluon.core[raytune] asks for ray[default,tune]>=2.43,<2.53, and
    # [feature_fusion] pulls it in. A wider ceiling here makes pip backtrack or,
    # worse, lets a bare install land a ray that adding [feature_fusion] later
    # has to downgrade.
    extras = _pyproject()["project"]["optional-dependencies"]
    ray_pin = next(d for d in extras["ray"] if d.startswith("ray[tune]"))
    assert "<2.53" in ray_pin, ray_pin
