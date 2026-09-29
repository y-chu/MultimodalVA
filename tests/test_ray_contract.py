"""What the Ray search backend must produce, checked without a GPU.

Every defect covered here was found by ``tests/hpc_ray_smoke.py`` on the first
real GPU run (2026-09-28). Off a CUDA machine the Ray entry points redirect to
Optuna, so none of it can be caught by running a search here — but the pieces
that go wrong are ordinary functions, and those are testable anywhere.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest


# --- the compat patch must not make NumPy RNGs unpicklable ------------------

def test_rng_still_pickles_after_the_numpy_compat_patch():
    """Regression: the patch used to install a closure, which pickle cannot name.

    ``ensure_numpy_rng_compat()`` replaces ``numpy.random._pickle._bit_generator_ctor``
    and never restores it. While the replacement was a nested ``def``, pickling
    *any* NumPy RNG in the process afterwards raised ``AttributeError: Can't get
    local object``. Ray Tune's searcher checkpointing hit it first, but the blast
    radius is every caller of ``load_joblib_compat()`` — that is, anything that
    loads a saved tabular model.
    """
    from multimodalva.utils.numpy_compat import ensure_numpy_rng_compat

    ensure_numpy_rng_compat()
    blob = pickle.dumps(np.random.default_rng(0))
    assert isinstance(pickle.loads(blob), np.random.Generator)


def test_numpy_compat_ctor_is_module_level():
    """The property that makes it picklable, asserted directly."""
    import numpy.random._pickle as nrp

    from multimodalva.utils.numpy_compat import ensure_numpy_rng_compat

    ensure_numpy_rng_compat()
    ctor = getattr(nrp, "_bit_generator_ctor", None) or getattr(nrp, "__bit_generator_ctor")
    assert "<locals>" not in ctor.__qualname__, (
        f"{ctor.__qualname__} is a local object; pickle references functions by "
        "qualified name, so this breaks pickling of every NumPy RNG"
    )


# --- the Ray trials table must match the contract every reader expects ------

def _ray_like_dataframe():
    """The columns a real ``ResultGrid.get_dataframe()`` carried on a GPU node."""
    row = {
        "accuracy": 0.94, "balanced_accuracy": 0.93, "f1_macro": 0.95,
        "f1_weighted": 0.95, "csmf_accuracy": 0.96, "log_loss": 0.21,
        "fold_0_f1_macro": 0.94, "fold_1_f1_macro": 0.95, "fold_2_f1_macro": 0.96,
        "cv_std_f1_macro": 0.008, "n_cv_folds": 3,
        "timestamp": 1e9, "checkpoint_dir_name": None, "done": True,
        "training_iteration": 1, "trial_id": "2710d69b", "date": "2026-09-28",
        "time_this_iter_s": 5.1, "time_total_s": 5.1, "pid": 66627,
        "hostname": "g008", "node_ip": "10.0.0.1", "time_since_restore": 5.1,
        "iterations_since_restore": 1, "logdir": "/tmp/x",
        "config/n_estimators": 372, "config/learning_rate": 0.009,
    }
    worse = dict(row, f1_macro=0.90, trial_id="bbbb", time_total_s=4.0,
                 **{"config/n_estimators": 100})
    return pd.DataFrame([row, worse])


def test_ray_trials_map_onto_the_reader_contract():
    from multimodalva.utils.ray_compat import ray_trials_to_contract

    out = ray_trials_to_contract(_ray_like_dataframe(), "f1_macro")

    assert "value" in out.columns and out["value"].iloc[0] == pytest.approx(0.95)
    assert "duration" in out.columns
    # Optuna's table carries these and readers use them to tell whether trials
    # overlapped; Ray reports a completion timestamp and a duration instead, so
    # the pair is reconstructed rather than lost.
    assert {"datetime_start", "datetime_complete"} <= set(out.columns)
    assert (out["datetime_complete"] >= out["datetime_start"]).all()
    assert "params_n_estimators" in out.columns
    assert "user_attrs_f1_macro" in out.columns
    assert "user_attrs_cv_std_f1_macro" in out.columns
    assert {f"user_attrs_fold_{i}_f1_macro" for i in range(3)} <= set(out.columns)
    # Ray bookkeeping carries nothing about the search and is dropped.
    for gone in ("trial_id", "logdir", "pid", "hostname", "date", "node_ip"):
        assert gone not in out.columns


def test_validation_reads_a_ray_trials_table_as_hpo_cv(tmp_path):
    """The end the defect actually broke: validation.json said source='none'."""
    from multimodalva.results.validation import _from_hpo_trials
    from multimodalva.utils.ray_compat import ray_trials_to_contract

    csv = tmp_path / "hpo_trials.csv"
    ray_trials_to_contract(_ray_like_dataframe(), "f1_macro").to_csv(csv, index=False)

    found = _from_hpo_trials(csv, "f1_macro")
    assert found is not None
    assert found["source"] == "hpo_cv"
    assert found["n_folds"] == 3
    assert found["objective"] == pytest.approx(0.95)      # the better trial wins
    assert found["scores"]["csmf_accuracy"] == pytest.approx(0.96)


# --- RunConfig: losing storage_path is worse than losing resume -------------

def test_run_config_builders_are_tried_in_turn():
    """Both hpo modules expose the two variants the fallback walks through.

    Ray versions disagree about ``CheckpointConfig(checkpoint_at_end=...)``:
    older ones need it pinned False, ray >= ~2.49 rejects the argument. Before
    the fix only an ImportError fell through to the plain variant, so on 2.52 the
    run ended up with *no* RunConfig — no resume, and the experiment written to
    ``~/ray_results``, which filled a cluster home quota mid-run.
    """
    from multimodalva.tabular import hpo as tab
    from multimodalva.text import hpo as txt

    for mod in (tab, txt):
        assert callable(mod._run_config_with_checkpoint_config)
        assert callable(mod._run_config_plain)


def test_tune_config_does_not_duplicate_the_concurrency_cap():
    """search_alg is already wrapped in a ConcurrencyLimiter.

    Passing the same cap to TuneConfig as well made Ray log "max_concurrent_trials
    will be ignored", which reads in a cluster log like the setting did nothing.
    """
    import inspect

    from multimodalva.tabular import hpo as tab
    from multimodalva.text import hpo as txt

    for mod in (tab, txt):
        src = inspect.getsource(mod)
        assert "max_concurrent_trials=max_concurrent_trials," not in src
        assert "ConcurrencyLimiter(" in src


# --- the Ray trial functions are ordinary functions; call them directly -----
#
# Nothing else in the suite reaches them: off a CUDA machine both entry points
# redirect to Optuna, so a bug inside a trial function only surfaces on a GPU
# node. These call the functions in-process and also pin the rule that a failed
# trial must raise: Ray can then continue other trials while recording this one
# in Result.error, instead of mistaking placeholder scores for success.

def _toy_problem(n_per_class: int = 12):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    from multimodalva import data

    df = data("va_sample", n_per_class=n_per_class)
    names = df["cause_of_death"].to_numpy()
    labels = sorted(set(names))
    label2id = {c: i for i, c in enumerate(labels)}
    y = np.array([label2id[c] for c in names])
    X = df.drop(columns=["cause_of_death"]).select_dtypes(include="number").to_numpy(float)
    if X.shape[1] == 0:
        X = np.random.RandomState(0).rand(len(y), 5)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_splits = [(tr.tolist(), va.tolist()) for tr, va in folds.split(X, y)]
    return X, y, label2id, {i: c for c, i in label2id.items()}, cv_splits


def test_tabular_ray_trial_fn_returns_per_fold_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # the trial fn writes fold dirs into cwd
    X, y, label2id, id2label, cv_splits = _toy_problem()

    from multimodalva.tabular.hpo import _tabular_ray_trial_fn

    out = _tabular_ray_trial_fn(
        {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
        label2id=label2id, id2label=id2label, model_name="lightgbm",
        random_state=42, n_jobs=1, use_gpu=False,
        X_train=X, y_train=y, cv_splits=cv_splits, metric_key="f1_macro",
    )

    assert {f"fold_{i}_f1_macro" for i in range(3)} <= set(out)
    assert "cv_std_f1_macro" in out
    assert out["n_cv_folds"] == 3
    # A successful trial must return a real score.
    assert out["f1_macro"] > 0, f"every fold scored zero — the trial body failed: {out}"


def test_tabular_ray_trial_failure_is_not_silenced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y, label2id, id2label, cv_splits = _toy_problem()

    from multimodalva.tabular.hpo import _tabular_ray_trial_fn

    with pytest.raises(ValueError, match="Unsupported model"):
        _tabular_ray_trial_fn(
            {}, label2id=label2id, id2label=id2label,
            model_name="not_a_model", random_state=42, n_jobs=1,
            use_gpu=False, X_train=X, y_train=y, cv_splits=cv_splits,
            metric_key="f1_macro",
        )


def test_tabular_trial_output_satisfies_the_contract(tmp_path, monkeypatch):
    """Trial function → Ray dataframe → mapper → the columns readers need."""
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    X, y, label2id, id2label, cv_splits = _toy_problem()

    from multimodalva.tabular.hpo import _tabular_ray_trial_fn
    from multimodalva.utils.ray_compat import ray_trials_to_contract

    out = _tabular_ray_trial_fn(
        {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
        label2id=label2id, id2label=id2label, model_name="lightgbm",
        random_state=42, n_jobs=1, use_gpu=False,
        X_train=X, y_train=y, cv_splits=cv_splits, metric_key="f1_macro",
    )
    # Ray turns each returned key into a column of the result dataframe.
    mapped = ray_trials_to_contract(pd.DataFrame([out]), "f1_macro")
    assert {f"user_attrs_fold_{i}_f1_macro" for i in range(3)} <= set(mapped.columns)
    assert "user_attrs_cv_std_f1_macro" in mapped.columns
    assert "value" in mapped.columns


def test_both_ray_trial_fns_take_the_objective_name():
    """The asymmetry that caused it: text had metric_key, tabular had nothing."""
    import inspect

    from multimodalva.tabular.hpo import _tabular_ray_trial_fn
    from multimodalva.text.hpo import _ray_trial_fn

    for fn in (_tabular_ray_trial_fn, _ray_trial_fn):
        assert "metric_key" in inspect.signature(fn).parameters, (
            f"{fn.__name__} cannot name the objective it is scoring, so it cannot "
            "label its per-fold columns"
        )
