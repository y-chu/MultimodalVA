"""Demos for the text classification pipeline.

Usage (synthetic demo data):
    python tests/demo_text_classification.py preview
    python tests/demo_text_classification.py run
    python tests/demo_text_classification.py run --hpo
    python tests/demo_text_classification.py run --hpo --n-trials 10
    python tests/demo_text_classification.py steps
    python tests/demo_text_classification.py roberta-pm

Usage (your own CSV):
    python tests/demo_text_classification.py run \\
        --data path/to/data.csv --label cause --text-col narrative

    python tests/demo_text_classification.py run \\
        --data path/to/data.csv --label cause --text-col narrative \\
        --exclude-cols record_id \\
        --model emilyalsentzer/Bio_ClinicalBERT \\
        --hpo --n-trials 20 --metric f1_macro \\
        --resume --output-dir runs/my_text

Key flags:
    --data          CSV file path, or "demo" for synthetic data (default: demo)
    --label         Label column name (default: cause)
    --text-col      Text/narrative column name (default: narrative)
    --exclude-cols        Comma-separated columns to exclude (e.g. ID, date).
                    Label and text columns are always excluded automatically.
    --model         HuggingFace checkpoint or supported shorthand alias
                    (default: emilyalsentzer/Bio_ClinicalBERT)
    --hpo           Enable Optuna hyperparameter search
    --n-trials      HPO trials (default: 2 for demo / recommend 20+ for real data)
    --n-cv-folds    CV folds for HPO scoring (default: 3; use 0 to disable CV)
    --metric        Optimisation metric: accuracy, f1_macro, f1_weighted,
                    balanced_accuracy, csmf_accuracy, log_loss (default: f1_macro)
    --resume        Resume HPO from existing SQLite study and/or resume training
    --max-length    Tokeniser max sequence length (default: 128)
    --output-dir    Where to save model artifacts (default: runs/demo_text)
    --top-k         Number of top-k predictions to return (default: 3)

Notes:
    - ``preview`` is safe — no model download required.
    - ``run``, ``hpo``, and ``steps`` fine-tune a transformer.
    - ``roberta-pm`` resolves and downloads the built-in RoBERTa-PM checkpoint.
"""

from __future__ import annotations

import argparse

from demo_utils import ensure_dir, load_data, parse_exclude_cols, print_prediction_summary


def preview(args: argparse.Namespace) -> None:
    """Show the dataset head and a minimal run command."""
    df, _ = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols), n_samples=12,
    )
    print(df[[args.text_col, args.label]].head(5).to_string(index=False))
    print(
        "\nMinimal run:\n"
        "  python tests/demo_text_classification.py run\n"
        "\nWith your own data:\n"
        "  python tests/demo_text_classification.py run \\\n"
        f"    --data {args.data} --label {args.label} --text-col {args.text_col}\n"
        "\nWith HPO:\n"
        "  python tests/demo_text_classification.py run \\\n"
        f"    --data {args.data} --label {args.label} --text-col {args.text_col} "
        f"--hpo --n-trials 20"
    )


def run_classifier(args: argparse.Namespace) -> None:
    """End-to-end text pipeline, with optional HPO."""
    from multimodalva.text import TextClassifier

    use_hpo = args.mode == "hpo" or args.hpo
    n_samples = 60 if args.data == "demo" else None
    df, _ = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols), n_samples=n_samples or 60,
    )

    clf = TextClassifier(model_name=args.model, output_dir=ensure_dir(args.output_dir))
    run_kwargs: dict = dict(
        df=df,
        text_col=args.text_col,
        label_col=args.label,
        max_length=args.max_length,
        use_fast=True,
        use_optimize=use_hpo,
        batch_size=8,
        top_k=args.top_k,
        use_lora=True,
        resume_training=args.resume,
    )
    if use_hpo:
        run_kwargs.update(
            n_trials=args.n_trials,
            optimize_metric=args.metric,
            resume_hpo=args.resume,
            use_cv=args.n_cv_folds > 0,
            n_cv_folds=args.n_cv_folds,
        )
    else:
        run_kwargs["hyperparams"] = {"epochs": 1, "batch_size": 4, "learning_rate": 2e-5}

    results = clf.run(**run_kwargs)
    if use_hpo:
        print("Best hyperparameters:", results["best_hyperparams"])
    print_prediction_summary(results["predictions"], f"TextClassifier ({args.model})")


def run_pipeline_steps(args: argparse.Namespace) -> None:
    """Low-level split → prepare → optimize → train → predict."""
    from multimodalva.text.dataset import prepare_dataset
    from multimodalva.text.hpo import optimize
    from multimodalva.text.predict import predict
    from multimodalva.text.train import train
    from multimodalva.utils.split import split

    n_samples = 60 if args.data == "demo" else None
    df, _ = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols), n_samples=n_samples or 60,
    )

    train_df, test_df = split(
        df, text_col=args.text_col, label_col=args.label,
        test_size=0.2, random_state=42, stratify=True,
    )
    train_ds, test_ds, label2id, id2label = prepare_dataset(
        train_df, test_df,
        text_col=args.text_col, label_col=args.label,
        model_name=args.model, max_length=args.max_length,
    )
    best_hyperparams, _ = optimize(
        train_dataset=train_ds,
        label2id=label2id, id2label=id2label,
        model_name=args.model,
        output_dir=ensure_dir(f"{args.output_dir}/hpo"),
        n_trials=args.n_trials, metric=args.metric,
        use_lora=True, resume=args.resume,
        use_cv=args.n_cv_folds > 0, n_cv_folds=args.n_cv_folds,
    )
    _, _, metadata = train(
        train_dataset=train_ds,
        label2id=label2id, id2label=id2label,
        model_name=args.model,
        output_dir=ensure_dir(f"{args.output_dir}/final"),
        hyperparams=best_hyperparams,
        val_size=None, use_lora=True,
        early_stopping_patience=2, resume=args.resume,
    )
    result = predict(
        output_dir=f"{args.output_dir}/final",
        test_dataset=test_ds, batch_size=8, top_k=args.top_k,
    )
    print("Final hyperparameters:", metadata["hyperparams"])
    print_prediction_summary(result, f"Low-level text pipeline ({args.model})")


def resolve_roberta_pm(args: argparse.Namespace) -> None:
    """Show how the package resolves the built-in RoBERTa-PM download."""
    from multimodalva.text.train import download_model

    model_path = download_model("roberta-pm", cache_dir=ensure_dir(f"{args.output_dir}/roberta_pm_cache"))
    print("Resolved roberta-pm checkpoint:")
    print(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["preview", "run", "hpo", "steps", "roberta-pm"],
        help="Pipeline variant to demonstrate (default: run).",
    )
    # Data
    parser.add_argument("--data", default="demo",
                        help='CSV path or "demo" for synthetic data (default: demo).')
    parser.add_argument("--label", default="cause",
                        help="Label column name (default: cause).")
    parser.add_argument("--text-col", default="narrative", dest="text_col",
                        help="Text/narrative column name (default: narrative).")
    parser.add_argument("--exclude-cols", default=None, dest="exclude_cols",
                        # --label and --text-col are always excluded; everything else is a feature
                        # unless listed here.
                        help="Comma-separated columns to exclude (e.g. 'record_id,date,site'). "
                             "Every column except --label and --text-col is automatically a feature.")
    # Model
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT",
                        help="HuggingFace checkpoint or alias (default: Bio_ClinicalBERT).")
    parser.add_argument("--max-length", type=int, default=128, dest="max_length",
                        help="Tokeniser max sequence length (default: 128).")
    # HPO / training
    parser.add_argument("--hpo", action="store_true",
                        help="Enable Optuna HPO (also activated by mode=hpo).")
    parser.add_argument("--n-trials", type=int, default=2, dest="n_trials",
                        help="HPO trials (default: 2; use 20+ for real data).")
    parser.add_argument("--n-cv-folds", type=int, default=3, dest="n_cv_folds",
                        help="CV folds for HPO scoring (default: 3; use 0 to disable CV).")
    parser.add_argument("--metric", default="f1_macro",
                        help="HPO optimisation metric (default: f1_macro).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume HPO study and/or training from checkpoint.")
    # Output
    parser.add_argument("--output-dir", default="runs/demo_text", dest="output_dir",
                        help="Output directory (default: runs/demo_text).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Number of top-k predictions (default: 3).")

    args = parser.parse_args()

    if args.mode == "preview":
        preview(args)
    elif args.mode in ("run", "hpo"):
        run_classifier(args)
    elif args.mode == "steps":
        run_pipeline_steps(args)
    else:
        resolve_roberta_pm(args)


if __name__ == "__main__":
    main()
