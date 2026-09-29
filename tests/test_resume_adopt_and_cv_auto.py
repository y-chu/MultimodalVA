"""Two escape hatches, and the limits that make them safe.

``resume="adopt"`` exists because the content manifest refuses artifacts made
before manifests existed, which would otherwise strand every run directory
created before 2026-09-28. ``Optimize(cv_folds="auto")`` exists because an
integer fold count raises on data whose rarest cause cannot fill every fold, and
the rarest cause is not always known before the job starts.

Both are overrides, so each test below pins what they must *not* do as tightly
as what they do: adopt must not paper over a manifest that disagrees, and "auto"
must not invent a fold count the data cannot support.
"""

from __future__ import annotations

import numpy as np
import pytest

import multimodalva as mv
from multimodalva import Optimize
from multimodalva.runner import (
    _RESUME_ARTIFACTS, SUPPORTED_TASKS, _normalize_task, _resume_dir,
)
from multimodalva.utils.optimize_config import (
    CV_FOLDS_AUTO, CV_FOLDS_DEFAULT, resolve_cv_folds,
)

LABEL = "cause_of_death"


@pytest.fixture(scope="module")
def df():
    return mv.data("va_sample", n_per_class=10)


@pytest.fixture(scope="module")
def feats(df):
    return [c for c in df.columns if c not in (LABEL, "narrative", "id")]


def _cfg(df, feats, **kw):
    return dict(task="tabular", data=df, label_col=LABEL, features=feats,
                model="naive_bayes", **kw)


def _thin(df, cause, keep):
    """``df`` with all but ``keep`` rows of ``cause`` removed."""
    drop = df[df[LABEL] == cause].index[:(df[LABEL] == cause).sum() - keep]
    return df.drop(index=drop).reset_index(drop=True)


# ---------------------------------------------------------------------------
# resume="adopt"
# ---------------------------------------------------------------------------
def test_a_directory_with_no_manifest_is_refused_by_default(df, feats, tmp_path):
    out = tmp_path / "legacy"
    mv.run(output_dir=out, **_cfg(df, feats))
    (out / "resume_manifest.json").unlink()          # a pre-guard run
    with pytest.raises(RuntimeError, match="resume_manifest.json does not"):
        mv.run(output_dir=out, **_cfg(df, feats))


def test_adopt_reuses_it_and_records_the_current_signature(df, feats, tmp_path, caplog):
    out = tmp_path / "legacy"
    mv.run(output_dir=out, **_cfg(df, feats))
    manifest = out / "resume_manifest.json"
    manifest.unlink()

    with caplog.at_level("WARNING"):
        result = mv.run(output_dir=out, resume="adopt", **_cfg(df, feats))
    assert result["predictions"] is not None
    assert manifest.is_file(), "adopting must record the signature of this call"
    said = "\n".join(r.getMessage() for r in caplog.records)
    assert "adopt" in said and "rather than on a check" in said, (
        "adopting is an override and has to say so out loud")

    # and from here on the directory is checked normally, with no adopt needed
    assert mv.run(output_dir=out, **_cfg(df, feats))["predictions"] is not None


def test_adopt_does_not_override_a_manifest_that_disagrees(df, feats, tmp_path):
    """The case adopt must refuse: a signature exists and contradicts this call.

    A missing manifest is a missing check; a manifest that disagrees is positive
    evidence of a different run, and adopting that is the silent mixing the
    guard exists to prevent.
    """
    out = tmp_path / "run"
    mv.run(output_dir=out, **_cfg(df, feats))
    with pytest.raises(RuntimeError, match="differs from this call"):
        mv.run(output_dir=out, resume="adopt", split_seed=7, **_cfg(df, feats))


def test_a_fresh_run_needs_no_adopt(df, feats, tmp_path):
    """Artifacts the current pipeline produces carry a signature from the start."""
    out = tmp_path / "fresh"
    mv.run(output_dir=out, **_cfg(df, feats))
    assert (out / "resume_manifest.json").is_file()
    assert mv.run(output_dir=out, **_cfg(df, feats))["predictions"] is not None


@pytest.mark.parametrize("bad", ["adapt", "ADOPTED", "yes", 1, None])
def test_an_unrecognised_resume_value_is_refused_early(df, feats, tmp_path, bad):
    with pytest.raises(ValueError, match="is not valid"):
        mv.run(output_dir=tmp_path / "x", resume=bad, **_cfg(df, feats))


def test_adopt_is_accepted_case_insensitively(df, feats, tmp_path):
    out = tmp_path / "legacy"
    mv.run(output_dir=out, **_cfg(df, feats))
    (out / "resume_manifest.json").unlink()
    assert mv.run(output_dir=out, resume="Adopt", **_cfg(df, feats))["predictions"] is not None


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_every_task_has_a_resume_artifact_list(task):
    """Preflight mirrors each pipeline's own ``artifacts_exist=`` expression.

    A task missing from the table would make preflight silently disagree with
    the run about whether a directory can be resumed.
    """
    assert _normalize_task(task) in _RESUME_ARTIFACTS


def test_an_ensemble_manifest_is_looked_for_under_its_method_directory():
    """EnsembleClassifier hands the strategy ``output_dir / method``."""
    assert _resume_dir("runs/x", "soft_voting").as_posix() == "runs/x/soft_voting"
    assert _resume_dir("runs/x", "stacking").as_posix() == "runs/x/stacking"
    assert _resume_dir("runs/x", "text").as_posix() == "runs/x"


@pytest.mark.parametrize("task,extra", [
    ("voting", {}),
    ("stacking", {"init_kwargs": {"n_folds": 2}}),
])
def test_preflight_and_the_run_agree_about_an_ensemble_legacy_dir(
    df, feats, tmp_path, task, extra
):
    specs = [{"model_name": "naive_bayes"}, {"model_name": "random_forest"}]
    out = tmp_path / task
    cfg = dict(task=task, data=df, label_col=LABEL, features=feats,
               tabular_models=specs, **extra)
    mv.run(output_dir=out, **cfg)
    manifest = next(out.rglob("resume_manifest.json"))
    manifest.unlink()

    assert not mv.preflight(output_dir=out, **cfg)["ok"], "preflight missed it"
    with pytest.raises(RuntimeError):
        mv.run(output_dir=out, **cfg)

    assert mv.preflight(output_dir=out, resume="adopt", **cfg)["ok"]
    assert mv.run(output_dir=out, resume="adopt", **cfg)["predictions"] is not None
    assert manifest.is_file()


# ---------------------------------------------------------------------------
# Optimize(cv_folds="auto")
# ---------------------------------------------------------------------------
def _labels(counts: dict) -> np.ndarray:
    return np.array([c for c, n in counts.items() for _ in range(n)])


def test_auto_uses_the_default_when_the_data_allows_it():
    assert resolve_cv_folds(CV_FOLDS_AUTO, _labels({0: 20, 1: 20})) == CV_FOLDS_DEFAULT


def test_auto_drops_to_what_the_rarest_class_supports(caplog):
    with caplog.at_level("WARNING"):
        folds = resolve_cv_folds(CV_FOLDS_AUTO, _labels({0: 20, 1: 2}),
                                 where="the text search")
    assert folds == 2
    said = "\n".join(r.getMessage() for r in caplog.records)
    assert "2 folds" in said and "the text search" in said
    assert "not comparable" in said, "a changed fold count changes the scores"


def test_auto_still_refuses_a_singleton_class():
    """There is no valid fold count for a class with one row."""
    with pytest.raises(ValueError, match="at least 2"):
        resolve_cv_folds(CV_FOLDS_AUTO, _labels({0: 20, 1: 1}))


def test_an_integer_the_data_cannot_honour_raises_and_names_auto():
    with pytest.raises(ValueError) as excinfo:
        resolve_cv_folds(3, _labels({0: 20, 1: 2}), where="the tabular search")
    message = str(excinfo.value)
    assert "minimum class count (2)" in message
    assert 'cv_folds="auto"' in message, "the error should name the way out"


def test_an_integer_the_data_supports_is_returned_unchanged():
    assert resolve_cv_folds(5, _labels({0: 20, 1: 5})) == 5


@pytest.mark.parametrize("bad", [1, 0, -2, "atuo", True, 2.5])
def test_optimize_refuses_an_impossible_cv_folds(bad):
    with pytest.raises(ValueError, match="cv_folds"):
        Optimize(cv_folds=bad)


@pytest.mark.parametrize("good", [2, 3, 10, "auto"])
def test_optimize_accepts_valid_cv_folds(good):
    assert Optimize(cv_folds=good).cv_folds == good


def test_auto_lets_a_search_run_where_an_integer_stops_it(df, feats, tmp_path):
    """The pairing that matters: preflight and the real run agree both ways."""
    thin = _thin(df, "Assault", 3)          # 3 rows total -> 2 in the training split
    strict = _cfg(thin, feats, hyperparams=Optimize(n_trials=2, cv_folds=3))
    auto = _cfg(thin, feats, hyperparams=Optimize(n_trials=2, cv_folds="auto"))

    assert not mv.preflight(**strict)["ok"]
    with pytest.raises(ValueError, match="minimum class count"):
        mv.run(output_dir=tmp_path / "strict", **strict)

    report = mv.preflight(**auto)
    assert report["ok"], report["problems"]
    assert "2-fold" in report["facts"]["search split"]
    out = tmp_path / "auto"
    assert mv.run(output_dir=out, **auto)["predictions"] is not None
    # the artifact must show the fold count that actually ran, not "auto"
    folds = sorted((out / "hpo" / "best_trial").glob("fold_*"))
    assert len(folds) == 2, f"expected 2 fold directories, got {[f.name for f in folds]}"
