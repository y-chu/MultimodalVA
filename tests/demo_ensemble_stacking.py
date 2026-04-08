"""Small, current demos for stage-wise stacking.

Usage:
    python tests/demo_ensemble_stacking.py meta
    python tests/demo_ensemble_stacking.py class-voter
    python tests/demo_ensemble_stacking.py wrapper

Notes:
    - These demos use only sklearn-style tabular base models by default.
    - ``class-voter`` demonstrates the new Stage 2 class-aware voting path.
"""

from __future__ import annotations

import argparse

from demo_utils import ensure_dir, feature_cols, make_demo_va_df, print_prediction_summary


def _tabular_specs() -> list[dict]:
    return [
        {"model_name": "random_forest", "hyperparams": {"n_estimators": 80, "max_depth": 6}},
        {"model_name": "gbdt", "hyperparams": {"n_estimators": 80, "max_depth": 3}},
    ]


def run_meta_learner() -> None:
    """Run Stage 1 -> Stage 2 meta-learner -> Stage 3 prediction."""
    from multimodalva.ensemble import StackingClassifier

    df = make_demo_va_df(n_samples=140, seed=50)
    clf = StackingClassifier(
        text_models=[],
        tabular_models=_tabular_specs(),
        output_dir=ensure_dir("runs/demo_stacking_meta"),
        n_folds=3,
    )
    clf.train_base_models(
        df=df,
        label_col="cause",
        feature_cols=feature_cols(),
        save_fold_models=False,
        cleanup_fold_files=True,
        n_jobs=-1,
        use_gpu=False,
    )
    clf.train_meta_learner_stage(
        meta_learners={"model_name": "logistic_regression", "hyperparams": {"max_iter": 1000, "C": 1.0}},
        metric="f1_macro",
        meta_cv_folds=3,
        n_jobs=-1,
    )
    result = clf.predict_test(top_k=3)
    print_prediction_summary(result, "StackingClassifier meta-learner")


def run_class_voter() -> None:
    """Run Stage 1 -> Stage 2 class-aware voting -> Stage 3 prediction."""
    from multimodalva.ensemble import StackingClassifier

    df = make_demo_va_df(n_samples=140, seed=51)
    clf = StackingClassifier(
        text_models=[],
        tabular_models=_tabular_specs(),
        output_dir=ensure_dir("runs/demo_stacking_class_voter"),
        n_folds=3,
    )
    clf.train_base_models(
        df=df,
        label_col="cause",
        feature_cols=feature_cols(),
        save_fold_models=False,
        cleanup_fold_files=True,
        n_jobs=-1,
        use_gpu=False,
    )
    info = clf.train_class_voter_stage(metric="recall", alpha=1.0, shrinkage=0.25)
    print("Class-voter OOF scores:", info["scores"])
    result = clf.predict_test_class_voter(top_k=3)
    print_prediction_summary(result, "StackingClassifier class-aware voter")


def run_wrapper() -> None:
    """Run the same class-aware flow through EnsembleClassifier."""
    from multimodalva.ensemble import EnsembleClassifier

    df = make_demo_va_df(n_samples=140, seed=52)
    clf = EnsembleClassifier(
        method="stacking",
        output_dir=ensure_dir("runs/demo_stacking_wrapper"),
        text_models=[],
        tabular_models=_tabular_specs(),
        n_folds=3,
    )
    clf.train_base_models(
        df=df,
        label_col="cause",
        feature_cols=feature_cols(),
        save_fold_models=False,
        cleanup_fold_files=True,
        n_jobs=-1,
        use_gpu=False,
    )
    clf.train_class_voter_stage(metric="recall", alpha=1.0, shrinkage=0.25)
    result = clf.predict_test_class_voter(top_k=3)
    print_prediction_summary(result, "EnsembleClassifier(method='stacking') class-aware voter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="meta",
        choices=["meta", "class-voter", "wrapper"],
        help="Which demo to run.",
    )
    args = parser.parse_args()

    if args.mode == "meta":
        run_meta_learner()
    elif args.mode == "class-voter":
        run_class_voter()
    else:
        run_wrapper()


if __name__ == "__main__":
    main()
