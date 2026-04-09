"""Demos for the data-fusion ensemble pipeline.

Usage (synthetic demo data):
    python tests/demo_ensemble_data_fusion.py text
    python tests/demo_ensemble_data_fusion.py run
    python tests/demo_ensemble_data_fusion.py run --hpo

Usage (your own CSV):
    python tests/demo_ensemble_data_fusion.py run \\
        --data path/to/data.csv --label cause --text-col narrative

    python tests/demo_ensemble_data_fusion.py run \\
        --data path/to/data.csv --label cause --text-col narrative \\
        --exclude-cols record_id \\
        --model allenai/longformer-base-4096 \\
        --hpo --n-trials 30 --metric f1_macro \\
        --resume --output-dir runs/my_data_fusion

Key flags:
    --data          CSV file path, or "demo" for synthetic data (default: demo)
    --label         Label column name (default: cause)
    --text-col      Text/narrative column name (default: narrative)
    --exclude-cols        Comma-separated columns to exclude from tabular features
                    (e.g. ID, date).  Label and text columns are always excluded.
    --model         Long-context LM checkpoint or alias
                    (default: allenai/longformer-base-4096)
    --hpo           Enable Optuna hyperparameter search
    --n-trials      HPO trials (default: 2 for demo / recommend 30 for real data)
    --metric        Optimisation metric (default: f1_macro)
    --resume        Resume HPO / training from checkpoint
    --max-length    Max token length for fused text (default: 256)
    --output-dir    Where to save model artifacts (default: runs/demo_data_fusion)
    --top-k         Number of top-k predictions to return (default: 3)
    --features      Advanced: explicit feature columns (comma-sep or file path).
                    Overrides --exclude-cols.

Notes:
    - ``text`` only converts tabular features to fused text (no model training).
    - ``run`` fine-tunes a long-context model on the fused narrative + tabular text.
"""

from __future__ import annotations

import argparse

from demo_utils import (
    ensure_dir,
    load_data,
    parse_exclude_cols,
    print_prediction_summary,
)


def preview_fused_text(args: argparse.Namespace) -> None:
    """Show how tabular features become extra narrative text."""
    from multimodalva.ensemble.data_fusion import build_fused_text

    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features, n_samples=8,
    )
    print(f"Using {len(feat)} tabular feature columns.")
    fused = build_fused_text(
        df=df, text_col=args.text_col, feature_cols=feat, group_symptoms=True,
    )
    preview = df[[args.label]].copy()
    preview["fused_text"] = fused
    print(preview.head(4).to_string(index=False))


def run_classifier(args: argparse.Namespace) -> None:
    """End-to-end data-fusion pipeline, with optional HPO."""
    from multimodalva.ensemble import DataFusionClassifier

    use_hpo = args.mode == "hpo" or args.hpo
    n_samples = 60 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features, n_samples=n_samples or 60,
    )

    print(f"Using {len(feat)} tabular feature columns.")
    clf = DataFusionClassifier(model_name=args.model, output_dir=ensure_dir(args.output_dir))
    run_kwargs: dict = dict(
        df=df,
        text_col=args.text_col,
        feature_cols=feat,
        label_col=args.label,
        group_symptoms=True,
        max_length=args.max_length,
        use_optimize=use_hpo,
        use_lora=True,
        resume_training=args.resume,
        batch_size=4,
        top_k=args.top_k,
    )
    if use_hpo:
        run_kwargs.update(
            n_trials=args.n_trials,
            optimize_metric=args.metric,
            resume_hpo=args.resume,
        )
    else:
        run_kwargs["hyperparams"] = {"epochs": 1, "batch_size": 2, "learning_rate": 2e-5}

    results = clf.run(**run_kwargs)
    if use_hpo:
        print("Best hyperparameters:", results["best_hyperparams"])
    print_prediction_summary(results["predictions"], f"DataFusionClassifier ({args.model})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="text",
        choices=["text", "run", "hpo"],
        help="Pipeline variant: text=preview fused text, run=train, hpo=train+HPO (default: text).",
    )
    # Data
    parser.add_argument("--data", default="demo",
                        help='CSV path or "demo" for synthetic data (default: demo).')
    parser.add_argument("--label", default="cause",
                        help="Label column name (default: cause).")
    parser.add_argument("--text-col", default="narrative", dest="text_col",
                        help="Text/narrative column name (default: narrative).")
    parser.add_argument("--exclude-cols", default=None, dest="exclude_cols",
                        # --label and --text-col are always excluded; every other column
                        # becomes a tabular feature unless listed here.
                        help="Comma-separated columns to exclude from tabular features (e.g. 'record_id,date'). "
                             "Every column except --label and --text-col is automatically a feature.")
    parser.add_argument("--features", default=None,
                        help="Advanced: explicit feature columns (comma-sep or file path). Overrides --exclude-cols.")
    # Model
    parser.add_argument("--model", default="allenai/longformer-base-4096",
                        help="Long-context LM checkpoint or alias (default: longformer-base-4096).")
    parser.add_argument("--max-length", type=int, default=256, dest="max_length",
                        help="Max token length for fused text (default: 256).")
    # HPO / training
    parser.add_argument("--hpo", action="store_true",
                        help="Enable Optuna HPO (also activated by mode=hpo).")
    parser.add_argument("--n-trials", type=int, default=2, dest="n_trials",
                        help="HPO trials (default: 2; use 30 for real data).")
    parser.add_argument("--metric", default="f1_macro",
                        help="HPO optimisation metric (default: f1_macro).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume HPO / training from checkpoint.")
    # Output
    parser.add_argument("--output-dir", default="runs/demo_data_fusion", dest="output_dir",
                        help="Output directory (default: runs/demo_data_fusion).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Number of top-k predictions (default: 3).")

    args = parser.parse_args()

    if args.mode == "text":
        preview_fused_text(args)
    else:
        run_classifier(args)


if __name__ == "__main__":
    main()
