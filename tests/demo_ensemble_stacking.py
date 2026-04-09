"""Demos for the stage-wise stacking ensemble pipeline.

Usage (synthetic demo data):
    python tests/demo_ensemble_stacking.py meta
    python tests/demo_ensemble_stacking.py class-voter
    python tests/demo_ensemble_stacking.py wrapper

Usage (your own CSV):
    python tests/demo_ensemble_stacking.py meta \\
        --data path/to/data.csv --label cause

    python tests/demo_ensemble_stacking.py meta \\
        --data path/to/data.csv --label cause --exclude-cols record_id \\
        --n-folds 5 --n-jobs -1 --output-dir runs/my_stacking

Key flags:
    --data          CSV file path, or "demo" for synthetic data (default: demo)
    --label         Label column name (default: cause)
    --exclude-cols        Comma-separated columns to exclude from features
                    (e.g. ID, date).  Label column is always excluded.
    --n-folds       Number of cross-validation folds for OOF (default: 3)
    --n-jobs        Parallel jobs for tabular models (default: -1 = all cores)
    --output-dir    Where to save model artifacts (default: auto per mode)
    --top-k         Number of top-k predictions to return (default: 3)
    --features      Advanced: explicit feature columns (comma-sep or file path).
                    Overrides --exclude-cols.

Notes:
    - All modes use sklearn-style tabular base models only by default.
    - ``meta`` runs Stage 1 OOF → Stage 2 meta-learner → Stage 3 prediction.
    - ``class-voter`` uses Stage 2 class-aware voting instead of a meta-learner.
    - ``wrapper`` runs the same class-aware flow via EnsembleClassifier.
"""

from __future__ import annotations

import argparse

from demo_utils import ensure_dir, load_data, parse_exclude_cols, print_prediction_summary


def _tabular_specs() -> list[dict]:
    return [
        {"model_name": "random_forest", "hyperparams": {"n_estimators": 80, "max_depth": 6}},
        {"model_name": "gbdt", "hyperparams": {"n_estimators": 80, "max_depth": 3}},
    ]


def run_meta_learner(args: argparse.Namespace) -> None:
    """Stage 1 → Stage 2 meta-learner → Stage 3 prediction."""
    from multimodalva.ensemble import StackingClassifier

    n_samples = 140 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=n_samples or 140, seed=50,
    )
    print(f"Training on {len(feat)} feature columns.")
    out = args.output_dir or "runs/demo_stacking_meta"
    clf = StackingClassifier(
        text_models=[], tabular_models=_tabular_specs(),
        output_dir=ensure_dir(out), n_folds=args.n_folds,
    )
    clf.train_base_models(
        df=df, label_col=args.label, feature_cols=feat,
        save_fold_models=False, cleanup_fold_files=True,
        n_jobs=args.n_jobs, use_gpu=False,
    )
    clf.train_meta_learner_stage(
        meta_learners={"model_name": "logistic_regression", "hyperparams": {"max_iter": 1000, "C": 1.0}},
        metric="f1_macro", meta_cv_folds=args.n_folds, n_jobs=args.n_jobs,
    )
    result = clf.predict_test(top_k=args.top_k)
    print_prediction_summary(result, "StackingClassifier meta-learner")


def run_class_voter(args: argparse.Namespace) -> None:
    """Stage 1 → Stage 2 class-aware voting → Stage 3 prediction."""
    from multimodalva.ensemble import StackingClassifier

    n_samples = 140 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=n_samples or 140, seed=51,
    )
    print(f"Training on {len(feat)} feature columns.")
    out = args.output_dir or "runs/demo_stacking_class_voter"
    clf = StackingClassifier(
        text_models=[], tabular_models=_tabular_specs(),
        output_dir=ensure_dir(out), n_folds=args.n_folds,
    )
    clf.train_base_models(
        df=df, label_col=args.label, feature_cols=feat,
        save_fold_models=False, cleanup_fold_files=True,
        n_jobs=args.n_jobs, use_gpu=False,
    )
    info = clf.train_class_voter_stage(metric="recall", alpha=1.0, shrinkage=0.25)
    print("Class-voter OOF scores:", info["scores"])
    result = clf.predict_test_class_voter(top_k=args.top_k)
    print_prediction_summary(result, "StackingClassifier class-aware voter")


def run_wrapper(args: argparse.Namespace) -> None:
    """Class-aware stacking flow via EnsembleClassifier."""
    from multimodalva.ensemble import EnsembleClassifier

    n_samples = 140 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=n_samples or 140, seed=52,
    )
    print(f"Training on {len(feat)} feature columns.")
    out = args.output_dir or "runs/demo_stacking_wrapper"
    clf = EnsembleClassifier(
        method="stacking",
        output_dir=ensure_dir(out),
        text_models=[], tabular_models=_tabular_specs(),
        n_folds=args.n_folds,
    )
    clf.train_base_models(
        df=df, label_col=args.label, feature_cols=feat,
        save_fold_models=False, cleanup_fold_files=True,
        n_jobs=args.n_jobs, use_gpu=False,
    )
    clf.train_class_voter_stage(metric="recall", alpha=1.0, shrinkage=0.25)
    result = clf.predict_test_class_voter(top_k=args.top_k)
    print_prediction_summary(result, "EnsembleClassifier(method='stacking') class-aware voter")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="meta",
        choices=["meta", "class-voter", "wrapper"],
        help="Stacking variant to demonstrate (default: meta).",
    )
    # Data
    parser.add_argument("--data", default="demo",
                        help='CSV path or "demo" for synthetic data (default: demo).')
    parser.add_argument("--label", default="cause",
                        help="Label column name (default: cause).")
    parser.add_argument("--exclude-cols", default=None, dest="exclude_cols",
                        # --label is always excluded; every other column becomes a feature
                        # unless listed here.
                        help="Comma-separated columns to exclude from features (e.g. 'record_id,date,site'). "
                             "Every column except --label is automatically used as a feature.")
    parser.add_argument("--features", default=None,
                        help="Advanced: explicit feature columns (comma-sep or file path). Overrides --exclude-cols.")
    # Model / training
    parser.add_argument("--n-folds", type=int, default=3, dest="n_folds",
                        help="OOF cross-validation folds (default: 3).")
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs",
                        help="Parallel jobs for tabular models (default: -1).")
    # Output
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Output directory (default: auto per mode).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Number of top-k predictions (default: 3).")

    args = parser.parse_args()

    if args.mode == "meta":
        run_meta_learner(args)
    elif args.mode == "class-voter":
        run_class_voter(args)
    else:
        run_wrapper(args)


if __name__ == "__main__":
    main()
