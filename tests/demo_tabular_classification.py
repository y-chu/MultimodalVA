"""Demos for the tabular classification pipeline.

Usage (synthetic demo data):
    python tests/demo_tabular_classification.py preview
    python tests/demo_tabular_classification.py run
    python tests/demo_tabular_classification.py run --hpo
    python tests/demo_tabular_classification.py run --hpo --model lightgbm --n-trials 20

Usage (your own CSV):
    python tests/demo_tabular_classification.py run \\
        --data path/to/data.csv --label cause

    python tests/demo_tabular_classification.py run \\
        --data path/to/data.csv --label cause --exclude-cols record_id \\
        --model lightgbm --hpo --n-trials 50 --metric f1_macro \\
        --resume --output-dir runs/my_tabular

    # Multiple ID / non-feature columns to exclude:
    python tests/demo_tabular_classification.py run \\
        --data path/to/data.csv --label cause --exclude-cols "id,date,site"

Key flags:
    --data          CSV file path, or "demo" for synthetic data (default: demo)
    --label         Label column name (default: cause)
    --exclude-cols        Comma-separated columns to exclude from features, e.g. ID,
                    date, site columns.  Label and text columns are always
                    excluded automatically.  (default: none)
    --cat-cols      Comma-separated categorical columns (default: "sex" for demo;
                    auto-detected by sklearn for real data when left unset)
    --model         Tabular model alias: random_forest, lightgbm, xgboost,
                    catboost, gbdt, mlp, naive_bayes, knn, svm (default: random_forest)
    --hpo           Enable Optuna hyperparameter search
    --n-trials      HPO trials (default: 4 for demo / recommend 50+ for real data)
    --metric        Optimisation metric: accuracy, f1_macro, f1_weighted,
                    balanced_accuracy, csmf_accuracy, log_loss (default: f1_macro)
    --resume        Resume HPO from existing SQLite study and/or resume training
    --output-dir    Where to save model artifacts (default: runs/demo_tabular)
    --top-k         Number of top-k predictions to return (default: 3)
    --features      Advanced: explicit comma-separated feature column names or
                    path to a text file listing them.  Overrides --exclude-cols.
"""

from __future__ import annotations

import argparse

from demo_utils import (
    ensure_dir,
    load_data,
    parse_cat_cols,
    parse_exclude_cols,
    print_prediction_summary,
)


def preview(args: argparse.Namespace) -> None:
    """Show the dataset head and a minimal run command."""
    df, feat = load_data(
        args.data, args.label,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=12,
    )
    print(df[[*feat[:5], args.label]].head(5).to_string(index=False))
    print(f"\nFeature columns detected ({len(feat)}): {feat[:8]}{'...' if len(feat) > 8 else ''}")
    print(
        "\nMinimal run:\n"
        "  python tests/demo_tabular_classification.py run\n"
        "\nWith your own data:\n"
        "  python tests/demo_tabular_classification.py run \\\n"
        f"    --data {args.data} --label {args.label} --model {args.model}\n"
        "\nWith HPO:\n"
        "  python tests/demo_tabular_classification.py run \\\n"
        f"    --data {args.data} --label {args.label} --model {args.model} --hpo --n-trials 50"
    )


def run_classifier(args: argparse.Namespace) -> None:
    """End-to-end tabular pipeline, with optional HPO."""
    from multimodalva.tabular import TabularClassifier

    use_hpo = args.mode == "hpo" or args.hpo
    n_samples = 120 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=n_samples or 120,
    )
    cat = parse_cat_cols(args.cat_cols, args.data)

    print(f"Training on {len(feat)} feature columns.")
    clf = TabularClassifier(model_name=args.model, output_dir=ensure_dir(args.output_dir))
    run_kwargs: dict = dict(
        df=df,
        feature_cols=feat,
        label_col=args.label,
        cat_cols=cat,
        use_optimize=use_hpo,
        top_k=args.top_k,
    )
    if use_hpo:
        run_kwargs.update(
            n_trials=args.n_trials,
            optimize_metric=args.metric,
            search_space_profile="auto",
            resume_hpo=args.resume,
        )
    else:
        run_kwargs["hyperparams"] = {}

    results = clf.run(**run_kwargs)
    if use_hpo:
        print("Best hyperparameters:", results["best_hyperparams"])
    print_prediction_summary(results["predictions"], f"TabularClassifier ({args.model})")


def run_pipeline_steps(args: argparse.Namespace) -> None:
    """Low-level split → prepare → optimize → train → predict."""
    from multimodalva.tabular.dataset import prepare_dataset
    from multimodalva.tabular.hpo import optimize
    from multimodalva.tabular.predict import predict
    from multimodalva.tabular.train import train
    from multimodalva.utils.split import split

    n_samples = 120 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=n_samples or 120,
    )
    cat = parse_cat_cols(args.cat_cols, args.data)

    print(f"Training on {len(feat)} feature columns.")
    train_df, test_df = split(df, label_col=args.label, test_size=0.2, random_state=42, stratify=True)
    X_train, X_test, y_train, y_test, preprocessor, label2id, id2label, feature_names = prepare_dataset(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feat,
        label_col=args.label,
        cat_cols=cat,
        encode_categoricals="ordinal",
    )
    best_hyperparams, _ = optimize(
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        model_name=args.model,
        output_dir=ensure_dir(f"{args.output_dir}/hpo"),
        n_trials=args.n_trials,
        metric=args.metric,
        search_space_profile="auto",
    )
    _, metadata = train(
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        model_name=args.model,
        output_dir=ensure_dir(f"{args.output_dir}/final"),
        hyperparams=best_hyperparams,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
    result = predict(
        output_dir=f"{args.output_dir}/final",
        X_test=X_test,
        y_test=y_test,
        top_k=args.top_k,
    )
    print("Final hyperparameters:", metadata["hyperparams"])
    print_prediction_summary(result, f"Low-level tabular pipeline ({args.model})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["preview", "run", "hpo", "steps"],
        help="Pipeline variant to demonstrate (default: run).",
    )
    # Data
    parser.add_argument("--data", default="demo",
                        help='CSV path or "demo" for synthetic data (default: demo).')
    parser.add_argument("--label", default="cause",
                        help="Label column name (default: cause).")
    parser.add_argument("--exclude-cols", default=None, dest="exclude_cols",
                        # All columns that are NOT --label and NOT in --exclude-cols are used as features.
                        help="Comma-separated columns to exclude from features (e.g. 'record_id,date,site'). "
                             "Every other column except --label is automatically used as a feature.")
    parser.add_argument("--cat-cols", default=None, dest="cat_cols",
                        help='Comma-separated categorical columns (default: "sex" for demo data).')
    parser.add_argument("--features", default=None,
                        help="Advanced: explicit feature columns (comma-sep or file path). "
                             "Overrides --exclude-cols.")
    # Model
    parser.add_argument("--model", default="random_forest",
                        help="Tabular model alias (default: random_forest).")
    # HPO / training
    parser.add_argument("--hpo", action="store_true",
                        help="Enable Optuna HPO (also activated by mode=hpo).")
    parser.add_argument("--n-trials", type=int, default=4, dest="n_trials",
                        help="HPO trials (default: 4; use 50+ for real data).")
    parser.add_argument("--metric", default="f1_macro",
                        help="HPO optimisation metric (default: f1_macro).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume HPO study and/or training from checkpoint.")
    # Output
    parser.add_argument("--output-dir", default="runs/demo_tabular", dest="output_dir",
                        help="Output directory (default: runs/demo_tabular).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Number of top-k predictions (default: 3).")

    args = parser.parse_args()

    if args.mode == "preview":
        preview(args)
    elif args.mode in ("run", "hpo"):
        run_classifier(args)
    else:
        run_pipeline_steps(args)


if __name__ == "__main__":
    main()
