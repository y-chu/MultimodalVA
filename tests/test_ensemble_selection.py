"""Tests for greedy ensemble selection and the nested-CV combiner comparison.

The algorithm is Caruana et al. (2004); these pin the properties the stacking
Stage 2 relies on: convex weights, the best-in-trajectory guarantee, both metric
directions, bagging reproducibility, and output shapes matching the existing
class-aware voter.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from multimodalva.ensemble.ensemble_selection import (
    EnsembleSelection,
    cross_validate_combiner,
)
from multimodalva.ensemble.stacking import (
    StackingClassifier,
    _apply_class_voter,
    _learn_class_voter_weights,
)


def _noisy_library(n_models=5, n=300, c=4, seed=0):
    """OOF probabilities of increasing noise around the truth, plus labels."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, size=n)
    onehot = np.eye(c)[y]
    preds = []
    for m in range(n_models):
        logits = onehot * (2.0 - 0.3 * m) + rng.normal(0, 1.0, size=(n, c))
        p = np.exp(logits)
        preds.append(p / p.sum(axis=1, keepdims=True))
    return np.stack(preds), y


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_bags", [1, 4])
def test_weights_are_nonnegative_and_sum_to_one(n_bags):
    preds, y = _noisy_library()
    es = EnsembleSelection(ensemble_size=30, n_bags=n_bags).fit(preds, y)
    assert (es.weights_ >= 0).all()
    assert es.weights_.sum() == pytest.approx(1.0)
    assert sum(es.weights_by_model_.values()) == pytest.approx(1.0)
    assert set(es.selected_models_) == {k for k, w in es.weights_by_model_.items() if w > 0}


def test_perfect_model_receives_all_weight():
    preds, y = _noisy_library()
    perfect = np.eye(preds.shape[2])[y]
    preds = np.concatenate([preds[:2], perfect[None], preds[2:]])  # perfect at index 2
    for metric in ("accuracy", "f1_macro", "log_loss"):
        es = EnsembleSelection(metric=metric, ensemble_size=20).fit(preds, y)
        assert es.weights_[2] == pytest.approx(1.0), metric
        assert es.selected_models_ == ["model_2"]


def test_best_in_trajectory_is_at_least_best_single():
    preds, y = _noisy_library(seed=3)
    for metric in ("f1_macro", "accuracy", "csmf_accuracy", "log_loss"):
        es = EnsembleSelection(metric=metric, ensemble_size=40).fit(preds, y)
        if es.greater_is_better_:
            assert es.oof_score_ >= es.best_single_score_ - 1e-12, metric
        else:
            assert es.oof_score_ <= es.best_single_score_ + 1e-12, metric


def test_trajectory_and_final_counts_without_best_in_trajectory():
    preds, y = _noisy_library()
    es = EnsembleSelection(ensemble_size=25, use_best_in_trajectory=False).fit(preds, y)
    assert es.trajectory_.shape == (1, 25)
    # Final counts over 25 picks: every weight is a multiple of 1/25.
    assert np.allclose(np.round(es.weights_ * 25), es.weights_ * 25)


def test_sorted_init_seeds_the_top_models():
    preds, y = _noisy_library()
    es = EnsembleSelection(metric="log_loss", ensemble_size=1, sorted_init=3,
                           use_best_in_trajectory=False).fit(preds, y)
    assert es.trajectory_.shape == (1, 2)  # seeded step + 1 greedy step
    # Noise grows with the index, so the three least noisy models are seeded.
    assert (es.weights_[:3] > 0).all()


def test_ties_go_to_the_lowest_index():
    preds, y = _noisy_library(n_models=2)
    dup = np.stack([preds[0], preds[0], preds[1]])
    es = EnsembleSelection(ensemble_size=10).fit(dup, y)
    assert es.weights_[1] == 0.0


# ---------------------------------------------------------------------------
# Metric direction and layouts
# ---------------------------------------------------------------------------

def test_regression_with_rmse():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    preds = np.stack([y + rng.normal(0, s, size=200) for s in (0.3, 0.5, 1.0, 2.0)])
    es = EnsembleSelection(metric=_rmse, greater_is_better=False,
                           problem_type="regression", ensemble_size=30).fit(preds, y)
    assert es.greater_is_better_ is False
    assert es.weights_.sum() == pytest.approx(1.0)
    assert es.oof_score_ <= es.best_single_score_ + 1e-12
    # The noisiest model should not dominate.
    assert es.weights_[3] < es.weights_[0]
    assert es.predict(preds).shape == (200,)


def test_regression_rejects_named_classification_metric():
    with pytest.raises(ValueError, match="callable metric"):
        EnsembleSelection(metric="f1_macro", problem_type="regression").fit(
            np.zeros((2, 5)), np.zeros(5))


def test_contradictory_direction_is_rejected():
    preds, y = _noisy_library()
    with pytest.raises(ValueError, match="contradicts"):
        EnsembleSelection(metric="log_loss", greater_is_better=True).fit(preds, y)


def test_binary_positive_class_layout():
    preds, y = _noisy_library(c=2)
    pos = preds[:, :, 1]  # (n_models, n_samples)
    es2d = EnsembleSelection(ensemble_size=20).fit(pos, y)
    es3d = EnsembleSelection(ensemble_size=20).fit(preds, y)
    np.testing.assert_allclose(es2d.weights_, es3d.weights_)
    assert es2d.predict(pos).shape == (pos.shape[1],)


# ---------------------------------------------------------------------------
# Bagging
# ---------------------------------------------------------------------------

def test_bagged_run_is_reproducible():
    preds, y = _noisy_library(n_models=6)
    a = EnsembleSelection(ensemble_size=20, n_bags=5, random_state=7).fit(preds, y)
    b = EnsembleSelection(ensemble_size=20, n_bags=5, random_state=7).fit(preds, y)
    np.testing.assert_array_equal(a.weights_, b.weights_)
    assert a.bags_ == b.bags_
    assert all(len(bag) == 3 for bag in a.bags_)  # ceil(0.5 * 6)


def test_single_full_bag_equals_unbagged():
    preds, y = _noisy_library()
    base = EnsembleSelection(ensemble_size=30).fit(preds, y)
    one_bag = EnsembleSelection(ensemble_size=30, n_bags=1, bag_fraction=1.0,
                                random_state=123).fit(preds, y)
    np.testing.assert_array_equal(base.weights_, one_bag.weights_)
    np.testing.assert_array_equal(base.trajectory_, one_bag.trajectory_)


# ---------------------------------------------------------------------------
# predict() shapes match the existing combiners
# ---------------------------------------------------------------------------

def test_predict_shape_matches_class_voter():
    preds, y = _noisy_library()
    test = preds[:, :50]
    id2label = {i: str(i) for i in range(preds.shape[2])}
    voter_w = _learn_class_voter_weights(list(preds), y, id2label)
    voter_out = _apply_class_voter(list(test), voter_w)

    es = EnsembleSelection(ensemble_size=20).fit(preds, y)
    out = es.predict(test)
    assert out.shape == voter_out.shape
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_predict_rejects_wrong_model_count():
    preds, y = _noisy_library()
    es = EnsembleSelection(ensemble_size=5).fit(preds, y)
    with pytest.raises(ValueError, match="models"):
        es.predict(preds[:3])


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------

def test_cross_validate_combiner_reports_mean_std_and_is_deterministic():
    preds, y = _noisy_library()
    factory = lambda: EnsembleSelection(ensemble_size=15)  # noqa: E731
    a = cross_validate_combiner(preds, y, factory, n_splits=4, random_state=0)
    b = cross_validate_combiner(preds, y, factory, n_splits=4, random_state=0)
    assert a == b
    assert len(a["fold_scores"]) == a["n_splits_used"] == 4
    assert a["mean"] == pytest.approx(np.mean(a["fold_scores"]))
    assert a["std"] == pytest.approx(np.std(a["fold_scores"]))


def test_cross_validate_combiner_reduces_folds_for_small_classes():
    preds, y = _noisy_library(n=60)
    y = y.copy()
    y[:3] = 3
    y[3:] = np.where(y[3:] == 3, 0, y[3:])  # class 3 has exactly 3 rows
    rep = cross_validate_combiner(preds, y, lambda: EnsembleSelection(ensemble_size=5),
                                  n_splits=5)
    assert rep["n_splits_used"] == 3


def test_cross_validate_combiner_regression_uses_kfold():
    rng = np.random.default_rng(2)
    y = rng.normal(size=100)
    preds = np.stack([y + rng.normal(0, s, size=100) for s in (0.5, 1.0)])
    rep = cross_validate_combiner(
        preds, y,
        lambda: EnsembleSelection(metric=_rmse, greater_is_better=False,
                                  problem_type="regression", ensemble_size=10),
        n_splits=5, metric=_rmse, greater_is_better=False, problem_type="regression",
    )
    assert rep["greater_is_better"] is False
    assert rep["n_splits_used"] == 5


# ---------------------------------------------------------------------------
# StackingClassifier stage wiring (no base models trained)
# ---------------------------------------------------------------------------

@pytest.fixture
def stage1_dir(tmp_path):
    """A minimal Stage 1 output: OOF matrix, labels, label maps, model_sources."""
    preds, y = _noisy_library(n_models=3, n=240, c=3, seed=5)
    class_ids = [10, 20, 30]  # non-contiguous ids, to exercise the id → position map
    oof_y = np.array(class_ids)[y]
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    np.save(oof_dir / "oof_meta_X.npy", np.concatenate(list(preds), axis=1))
    np.save(oof_dir / "oof_y.npy", oof_y)
    names = ["bert", "lightgbm", "lightgbm"]  # duplicate names on purpose
    sources = [
        {"type": "text" if i == 0 else "tabular", "local_index": i,
         "spec": {"model_name": n}, "final_dir": str(tmp_path / f"final/{i}"),
         "col_start": 3 * i, "n_cols": 3}
        for i, n in enumerate(names)
    ]
    (oof_dir / "oof_metadata.json").write_text(json.dumps({"model_sources": sources, "label_col": "cause"}))
    id2label = {cid: f"cause_{cid}" for cid in class_ids}
    (tmp_path / "id2label.json").write_text(json.dumps({str(k): v for k, v in id2label.items()}))
    (tmp_path / "label2id.json").write_text(json.dumps({v: k for k, v in id2label.items()}))
    return tmp_path, preds


def _clf(root):
    return StackingClassifier(text_models=[{"model_name": "bert"}], tabular_models=[],
                              output_dir=root)


def test_train_ensemble_selection_stage_writes_weights(stage1_dir):
    root, _ = stage1_dir
    out = _clf(root).train_ensemble_selection_stage(ensemble_size=20)
    assert set(out["weights"]) == {"0:bert", "1:lightgbm", "2:lightgbm"}
    assert sum(out["weights"].values()) == pytest.approx(1.0)

    es_dir = root / "ensemble_selection"
    assert np.load(es_dir / "ensemble_weights.npy").sum() == pytest.approx(1.0)
    meta = json.loads((es_dir / "ensemble_selection_metadata.json").read_text())
    for key in ("weights_by_model", "oof_score", "best_single_score",
                "simple_average_score", "scores", "inputs"):
        assert key in meta, key
    assert (es_dir / "trajectory.csv").exists()


def test_predict_test_ensemble_selection_combines_with_saved_weights(stage1_dir, monkeypatch):
    import pandas as pd

    root, preds = stage1_dir
    _clf(root).train_ensemble_selection_stage(ensemble_size=20)

    # Fresh instance: weights must be reloaded from disk.
    clf = _clf(root)
    test_probs = preds[:, :30]
    test_df = pd.DataFrame({"cause": ["cause_10"] * 30})
    monkeypatch.setattr(clf, "_predict_test_base_probs",
                        lambda batch_size: (test_probs, test_df, None))
    result = clf.predict_test_ensemble_selection(top_k=2)

    w = np.load(root / "ensemble_selection" / "ensemble_weights.npy")
    expected = np.tensordot(w, test_probs, axes=(0, 0))
    np.testing.assert_allclose(result.full[["prob_10", "prob_20", "prob_30"]].to_numpy(), expected)
    assert (root / "ensemble_selection" / "predictions" / "predictions_top1.csv").exists()


META_CANDIDATES = [
    {"model_name": "logistic_regression", "hyperparams": {"max_iter": 1000, "C": 1.0}},
    {"model_name": "random_forest", "hyperparams": {"n_estimators": 20}},
]


def test_compare_combiners_stage_table(stage1_dir):
    root, _ = stage1_dir
    table = _clf(root).compare_combiners_stage(
        n_combiner_folds=3, meta_learners=META_CANDIDATES,
        ensemble_selection_kwargs={"ensemble_size": 10}, n_jobs=1,
    )
    # One row per meta-learner candidate, named like combiner= names them.
    assert table["combiner"].tolist() == [
        "simple_average", "class_aware_voting", "ensemble_selection",
        "logistic_regression", "random_forest",
    ]
    assert list(table.columns[:5]) == ["combiner", "metric", "cv_mean", "cv_std", "n_folds"]
    assert {"fold_0", "fold_1", "fold_2"} <= set(table.columns)
    assert table[["cv_mean", "cv_std"]].notna().all().all()
    assert (root / "combiner_comparison" / "combiner_comparison.csv").exists()
    payload = json.loads((root / "combiner_comparison" / "combiner_comparison.json").read_text())
    # Every row was scored on the same folds, so fold counts agree.
    assert len({r["n_splits_used"] for r in payload["results"].values()}) == 1
    # best_combiner is the top cv_mean (f1_macro: higher is better), first on ties.
    assert payload["best_combiner"] == table["combiner"].iloc[int(table["cv_mean"].argmax())]


def test_compare_rejects_repeated_meta_learner_names(stage1_dir):
    root, _ = stage1_dir
    with pytest.raises(ValueError, match="more than once"):
        _clf(root).compare_combiners_stage(
            meta_learners=[{"model_name": "random_forest"}, {"model_name": "random_forest"}],
        )


def test_class_voter_combiner_includes_the_fallback():
    """The comparison scores the voter as trained: a fallback means equal weights."""
    from multimodalva.ensemble.stacking import _ClassVoterCombiner, _uniform_soft_vote

    preds, y = _noisy_library(n_models=3, n=240, c=3, seed=5)
    # An impossible margin forces the fallback whenever the CV check runs.
    voter = _ClassVoterCombiner(
        metric="f1", shrinkage=0.1, min_support_for_trust=20.0, fallback_to_soft=True,
        fallback_metric="f1_macro", fallback_cv_folds=3, split_seed=0,
        fallback_tolerance=10.0,
    ).fit(preds, y)
    assert voter.used_uniform_fallback
    np.testing.assert_allclose(voter.predict(preds), _uniform_soft_vote(list(preds)))

    kept = _ClassVoterCombiner(
        metric="f1", shrinkage=0.1, min_support_for_trust=20.0, fallback_to_soft=False,
        fallback_metric="f1_macro", fallback_cv_folds=3, split_seed=0,
        fallback_tolerance=0.0,
    ).fit(preds, y)
    assert not kept.used_uniform_fallback


def test_predict_test_simple_average_is_the_equal_weight_mean(stage1_dir, monkeypatch):
    import pandas as pd

    root, preds = stage1_dir
    clf = _clf(root)
    test_probs = preds[:, :30]
    test_df = pd.DataFrame({"cause": ["cause_10"] * 30})
    monkeypatch.setattr(clf, "_predict_test_base_probs",
                        lambda batch_size: (test_probs, test_df, None))
    result = clf.predict_test_simple_average(top_k=2)
    np.testing.assert_allclose(result.full[["prob_10", "prob_20", "prob_30"]].to_numpy(),
                               test_probs.mean(axis=0))
    assert (root / "simple_average" / "predictions" / "predictions_top1.csv").exists()


@pytest.mark.parametrize("combiner,match", [
    ("nope", "Unknown combiner"),
    (["best", "simple_average"], "best"),
])
def test_run_rejects_bad_combiner_before_stage1(tmp_path, combiner, match):
    import pandas as pd

    clf = _clf(tmp_path)
    with pytest.raises(ValueError, match=match):
        clf.run(pd.DataFrame({"cause": ["a"]}), label_col="cause", combiner=combiner)
    assert not (tmp_path / "data").exists()
