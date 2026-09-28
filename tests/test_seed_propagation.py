"""A run-level seed has to reach everything that draws a random number.

Two bugs motivated this file, and neither was visible from the outside — both
produced a perfectly ordinary-looking run whose seed simply did not matter:

* ``train_text()`` was called without ``random_state`` in both ensembles, so a
  text base model trained at ``train_text``'s own default whatever the caller
  asked for. The split and the search honoured the seed; the training did not.
* ``_build_model()`` injected ``random_state`` only when the constructor named
  it. ``XGBClassifier.__init__`` is ``(self, objective, **kwargs)``, so xgboost
  never received a seed and every seed gave byte-identical results.

A seed that is silently ignored is worse than a missing one: a seed-variation
study reads as zero variance, and a "reproducible" run is only reproducible by
accident. These tests read the call sites and the constructors rather than
trusting that a future edit remembers.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import numpy as np
import pytest

import multimodalva.tabular.train as tabular_train

# Functions that train or search and therefore must always be handed the seed.
SEEDED_CALLEES = {
    "train_text",
    "train_tabular",
    "optimize_text",
    "optimize_tabular",
}

PIPELINE_MODULES = [
    "multimodalva/text/text_classifier.py",
    "multimodalva/tabular/tabular_classifier.py",
    "multimodalva/ensemble/voting.py",
    "multimodalva/ensemble/stacking.py",
]


def _package_root() -> pathlib.Path:
    import multimodalva

    return pathlib.Path(multimodalva.__file__).parent.parent


def _seedless_calls(path: pathlib.Path) -> list[str]:
    """Call sites of a training function that do not pass ``random_state``."""
    tree = ast.parse(path.read_text())
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in SEEDED_CALLEES:
            continue
        passed = {kw.arg for kw in node.keywords}
        # ``**kwargs`` forwarding counts: the seed may be inside it.
        forwards = any(kw.arg is None for kw in node.keywords)
        if "random_state" not in passed and not forwards:
            problems.append(f"{path.name}:{node.lineno} {name}() without random_state")
    return problems


@pytest.mark.parametrize("rel_path", PIPELINE_MODULES)
def test_training_calls_pass_the_seed(rel_path):
    path = _package_root() / rel_path
    if not path.is_file():  # pragma: no cover - layout guard
        pytest.skip(f"{rel_path} not found")
    problems = _seedless_calls(path)
    assert not problems, (
        "These calls train or search without being given the run's seed, so the "
        "seed would not affect their result:\n  " + "\n  ".join(problems)
    )


def test_every_tabular_model_that_can_be_seeded_is_seeded():
    """The seed must reach each estimator, including **kwargs constructors."""
    unseeded = []
    for name in tabular_train.TABULAR_MODELS:
        model = tabular_train._build_model(name, {}, 42, n_jobs=1, use_gpu=False)
        params = model.get_params()
        seed_keys = [k for k in params if "random" in k or "seed" in k]
        if not seed_keys:
            continue  # genuinely deterministic (naive_bayes, knn)
        if all(params[k] is None for k in seed_keys):
            unseeded.append(f"{name}: {', '.join(seed_keys)} left as None")
    assert not unseeded, (
        "These models accept a seed but did not get one, so the caller's "
        "random_state has no effect on them:\n  " + "\n  ".join(unseeded)
    )


@pytest.mark.parametrize("model_name", ["xgboost", "random_forest", "lightgbm"])
def test_different_seeds_give_different_models(model_name):
    """The seed must actually change the fit, not merely be stored.

    ``xgboost`` passed the stored-parameter check even while being ignored, so
    this trains twice and compares. Needs row/column subsampling for the seed to
    have anything to act on.
    """
    pytest.importorskip(model_name if model_name != "random_forest" else "sklearn")
    rng = np.random.default_rng(0)
    X = rng.random((200, 10))
    y = rng.integers(0, 3, 200)
    hp = {"n_estimators": 20}
    if model_name in {"xgboost", "lightgbm"}:
        hp.update(subsample=0.6, colsample_bytree=0.6, max_depth=3)
    else:
        hp.update(max_features=0.5)

    def fit(seed):
        m = tabular_train._build_model(model_name, dict(hp), seed, n_jobs=1, use_gpu=False)
        m.fit(X, y)
        return m.predict_proba(X)

    a, b, a_again = fit(42), fit(123), fit(42)
    assert np.array_equal(a, a_again), (
        f"{model_name} is not reproducible at a fixed seed."
    )
    assert not np.array_equal(a, b), (
        f"{model_name} gives identical results for seed 42 and 123, so the "
        "caller's random_state is being ignored."
    )


# ---------------------------------------------------------------------------
# The two-seed contract: split_seed partitions rows, train_seed drives models.
# ---------------------------------------------------------------------------

HIGH_LEVEL_RUNS = {
    "TextClassifier.run": ("multimodalva.text.text_classifier", "TextClassifier", "run"),
    "TabularClassifier.run": ("multimodalva.tabular.tabular_classifier", "TabularClassifier", "run"),
    "DataFusionClassifier.run": ("multimodalva.ensemble.data_fusion_classifier", "DataFusionClassifier", "run"),
    "FeatureFusionClassifier.run": ("multimodalva.ensemble.feature_fusion", "FeatureFusionClassifier", "run"),
    "SoftVotingClassifier.run": ("multimodalva.ensemble.voting", "SoftVotingClassifier", "run"),
    "StackingClassifier.run": ("multimodalva.ensemble.stacking", "StackingClassifier", "run"),
    "StackingClassifier.train_base_models": ("multimodalva.ensemble.stacking", "StackingClassifier", "train_base_models"),
}


def _params(module, cls, meth):
    import importlib

    return inspect.signature(getattr(getattr(importlib.import_module(module), cls), meth)).parameters


@pytest.mark.parametrize("label", list(HIGH_LEVEL_RUNS))
def test_every_entry_point_takes_both_seeds_and_deterministic(label):
    params = _params(*HIGH_LEVEL_RUNS[label])
    missing = [p for p in ("split_seed", "train_seed", "deterministic") if p not in params]
    assert not missing, f"{label} is missing {missing}; every task must take the same seed arguments."


@pytest.mark.parametrize("label", list(HIGH_LEVEL_RUNS))
def test_no_entry_point_keeps_a_removed_seed_argument(label):
    params = _params(*HIGH_LEVEL_RUNS[label])
    stale = [p for p in ("random_state", "set_seed", "automm_seed") if p in params]
    assert not stale, (
        f"{label} still takes {stale}. One seed that meant both the split and "
        "the model is exactly what split_seed/train_seed replaced."
    )


def test_run_signature_has_both_seeds():
    import multimodalva as mv

    params = inspect.signature(mv.run).parameters
    assert {"split_seed", "train_seed", "deterministic"} <= set(params)
    assert "random_state" not in params and "set_seed" not in params


@pytest.mark.parametrize("stale", ["random_state", "set_seed", "automm_seed"])
def test_removed_seed_argument_raises_and_names_the_replacement(stale):
    """A stale seed must fail loudly, not be swallowed by **run_kwargs."""
    import multimodalva as mv

    df = mv.data("va_sample", n_per_class=4)
    with pytest.raises(TypeError) as err:
        mv.run(task="tabular", data=df, label_col="cause_of_death",
               output_dir="/tmp/_mmva_seedguard", **{stale: 1})
    msg = str(err.value)
    assert stale in msg and ("split_seed" in msg or "train_seed" in msg), msg


def test_stacking_later_stages_inherit_stage_one_seeds():
    """Stage 2 in a fresh session must reuse stage 1's seeds, not default 42."""
    from multimodalva.ensemble.stacking import StackingClassifier

    for meth in ("train_meta_learner_stage", "train_class_voter_stage"):
        params = inspect.signature(getattr(StackingClassifier, meth)).parameters
        assert "split_seed" in params and params["split_seed"].default is None, (
            f"{meth} must default split_seed to None so it inherits stage 1's."
        )
    meta = inspect.signature(StackingClassifier.train_meta_learner_stage).parameters
    assert meta["train_seed"].default is None
    assert "fallback_random_state" not in inspect.signature(
        StackingClassifier.train_class_voter_stage).parameters

    clf = StackingClassifier(
        text_models=[], tabular_models=[{"model_name": "random_forest"}],
        output_dir="/tmp/_mmva_inherit",
    )
    clf._split_seed, clf._train_seed = 7, 11
    assert clf._inherit_seeds(None, None) == (7, 11), "unset seeds must inherit stage 1"
    assert clf._inherit_seeds(3, None) == (3, 11), "an explicit seed must still win"


def test_feature_fusion_passes_train_seed_to_every_fit():
    """Both AutoMM fit() branches — plain and search — must be seeded."""
    from multimodalva.ensemble.feature_fusion import FeatureFusionClassifier

    tree = ast.parse(inspect.cleandoc(inspect.getsource(FeatureFusionClassifier.run)))
    fits = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "fit"]
    assert len(fits) >= 2, "expected a plain fit() and a search fit()"
    for call in fits:
        seed = next((kw.value for kw in call.keywords if kw.arg == "seed"), None)
        assert seed is not None, "a predictor.fit() call passes no seed"
        assert getattr(seed, "id", None) == "train_seed", (
            "AutoMM training must be seeded from train_seed"
        )


# --- behaviour: the two seeds control different things ----------------------

def _tabular_run(split_seed, train_seed, out):
    import multimodalva as mv

    df = mv.data("va_sample", n_per_class=25)
    df = df.assign(rowid=[f"R{i:04d}" for i in range(len(df))])
    feats = [c for c in df.columns if c not in ("cause_of_death", "narrative", "rowid", "id")]
    r = mv.run(task="tabular", data=df, label_col="cause_of_death", features=feats,
               model="random_forest", id_col="rowid",
               hyperparams={"n_estimators": 15, "max_features": 0.5},
               split_seed=split_seed, train_seed=train_seed, output_dir=out)
    return r["predictions"].full


def test_train_seed_changes_the_model_but_not_the_split(tmp_path):
    a = _tabular_run(split_seed=1, train_seed=1, out=tmp_path / "a")
    b = _tabular_run(split_seed=1, train_seed=2, out=tmp_path / "b")
    assert a["id"].tolist() == b["id"].tolist(), (
        "train_seed moved the test set; it must only affect the model."
    )
    probs = [c for c in a.columns if c.startswith("prob_")]
    assert not np.array_equal(a[probs].to_numpy(), b[probs].to_numpy()), (
        "train_seed had no effect on the model."
    )


def test_split_seed_changes_the_split(tmp_path):
    a = _tabular_run(split_seed=1, train_seed=1, out=tmp_path / "a")
    b = _tabular_run(split_seed=2, train_seed=1, out=tmp_path / "b")
    assert sorted(a["id"]) != sorted(b["id"]), "split_seed did not change the test set."


def test_same_seeds_reproduce_exactly(tmp_path):
    a = _tabular_run(split_seed=5, train_seed=9, out=tmp_path / "a")
    b = _tabular_run(split_seed=5, train_seed=9, out=tmp_path / "b")
    probs = [c for c in a.columns if c.startswith("prob_")]
    assert a["id"].tolist() == b["id"].tolist()
    assert np.array_equal(a[probs].to_numpy(), b[probs].to_numpy())


def test_search_folds_follow_split_seed_not_train_seed(tmp_path):
    """Folds are a partition, so changing train_seed must not move them.

    If folds followed train_seed, reseeding the model would also change which
    hyperparameters the search picked — conflating two sources of variance —
    and candidates on a validation leaderboard would be scored on different
    folds.

    Per-fold scores normally mix two things: which rows are in each fold, and
    what the model did. To read fold membership alone, this uses naive_bayes
    (no randomness of its own) and a one-value search space (so the sampler's
    seed cannot change the trial's parameters). Then identical per-fold scores
    mean identical folds, and nothing else.

    Also the only test that drives optimize_tabular with use_cv=True; an
    undefined name on that path once went unnoticed because nothing ran it.
    """
    import pandas as pd
    from multimodalva.tabular.hpo import optimize_tabular

    rng = np.random.default_rng(0)
    X = rng.random((120, 6))
    y = np.repeat(np.arange(3), 40)
    id2label = {0: "a", 1: "b", 2: "c"}
    label2id = {v: k for k, v in id2label.items()}

    def fold_scores(split_seed, train_seed, out):
        optimize_tabular(
            X_train=X, y_train=y, label2id=label2id, id2label=id2label,
            model_name="naive_bayes", output_dir=out, n_trials=1,
            search_space={"var_smoothing": ("categorical", [1e-9])},
            use_cv=True, n_cv_folds=3, random_state=train_seed,
            split_seed=split_seed, n_jobs=1,
        )
        trials = pd.read_csv(pathlib.Path(out) / "hpo_trials.csv")
        cols = sorted(c for c in trials.columns if c.startswith("user_attrs_fold_"))
        assert cols, "search recorded no per-fold scores"
        return trials.loc[trials["number"] == 0, cols].round(12).to_numpy()

    base = fold_scores(split_seed=3, train_seed=1, out=tmp_path / "base")
    reseeded_model = fold_scores(split_seed=3, train_seed=99, out=tmp_path / "train")
    resplit = fold_scores(split_seed=4, train_seed=1, out=tmp_path / "split")

    assert np.array_equal(base, reseeded_model), (
        "Changing train_seed moved the search folds. Folds must follow "
        "split_seed, or reseeding the model also changes the chosen "
        "hyperparameters."
    )
    assert not np.array_equal(base, resplit), "split_seed did not move the folds."


# --- deterministic mode: it must turn on, stay on, and turn back off ---------

def test_deterministic_mode_does_not_leak_into_later_runs():
    """These are process-wide flags; deterministic=False must undo True.

    Otherwise one deterministic run in a notebook leaves every later run slow
    and liable to raise, even when deterministic=False is passed explicitly.
    """
    torch = pytest.importorskip("torch")
    from multimodalva.utils.seeds import set_determinism

    try:
        set_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_float32_matmul_precision() == "highest"
        set_determinism(False)
        assert not torch.are_deterministic_algorithms_enabled(), (
            "deterministic=False left deterministic algorithms on from an "
            "earlier run in the same process."
        )
        assert torch.backends.cudnn.deterministic is False
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def test_train_text_does_not_reenable_tf32_under_determinism():
    """train_text() must not switch TF32 back on after set_determinism(True).

    It used to set allow_tf32 = True unconditionally on CUDA, undoing
    deterministic=True the moment training began. That path only runs on a CUDA
    machine, so this checks the source: every assignment that turns TF32 on, and
    every "high" matmul precision, must sit behind the determinism check.
    """
    from multimodalva.text import train as text_train

    src = inspect.getsource(text_train)
    assert "_want_determinism = torch.are_deterministic_algorithms_enabled()" in src
    assert "if _is_cuda and not _want_determinism:" in src, (
        "TF32 is enabled on CUDA without checking whether determinism was asked for."
    )
    assert '"highest" if _want_determinism else "high"' in src, (
        "float32 matmul precision is set to 'high' (TF32) without checking determinism."
    )
    # And nothing else in the module turns TF32 on behind the gate's back.
    tf32_on = [ln.strip() for ln in src.splitlines()
               if "allow_tf32 = True" in ln]
    assert len(tf32_on) == 2, f"unexpected TF32 enablers: {tf32_on}"
