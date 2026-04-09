"""Demos for the AutoMM feature-fusion ensemble pipeline.

Usage (synthetic demo data):
    python tests/demo_ensemble_feature_fusion.py preview
    python tests/demo_ensemble_feature_fusion.py run

Usage (your own CSV):
    python tests/demo_ensemble_feature_fusion.py run \\
        --data path/to/data.csv --label cause --text-col narrative

    python tests/demo_ensemble_feature_fusion.py run \\
        --data path/to/data.csv --label cause --text-col narrative \\
        --exclude-cols record_id \\
        --model bioclinicalbert --time-limit 3600 \\
        --output-dir runs/my_feature_fusion

Key flags:
    --data          CSV file path, or "demo" for synthetic data (default: demo)
    --label         Label column name (default: cause)
    --text-col      Text/narrative column name (default: narrative)
    --exclude-cols        Comma-separated columns to exclude from tabular features
                    (e.g. ID, date).  Label and text columns are always excluded.
    --model         AutoMM text backbone alias or HuggingFace checkpoint
                    (default: bioclinicalbert; roberta-pm is also supported)
    --time-limit    AutoMM training budget in seconds (default: 120 for demo)
    --output-dir    Where to save model artifacts (default: runs/demo_feature_fusion)
    --top-k         Number of top-k predictions to return (default: 3)
    --features      Advanced: explicit feature columns (comma-sep or file path).
                    Overrides --exclude-cols.

Notes:
    - ``preview`` is dependency-safe — just shows the multimodal table.
    - ``run`` requires ``pip install autogluon.multimodal``.
"""

from __future__ import annotations

import argparse

from demo_utils import ensure_dir, load_data, parse_exclude_cols, print_prediction_summary


def preview(args: argparse.Namespace) -> None:
    """Show the dataset shape and a minimal AutoMM call."""
    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features, n_samples=10,
    )
    cols = [args.text_col, *feat[:5], args.label]
    print(df[cols].head(4).to_string(index=False))
    print(f"\nTabular feature columns detected ({len(feat)}): {feat[:8]}{'...' if len(feat) > 8 else ''}")
    print(
        "\nMinimal run:\n"
        "  python tests/demo_ensemble_feature_fusion.py run\n"
        "\nWith your own data:\n"
        "  python tests/demo_ensemble_feature_fusion.py run \\\n"
        f"    --data {args.data} --label {args.label} "
        f"--text-col {args.text_col} --model {args.model}"
    )


def run_classifier(args: argparse.Namespace) -> None:
    """Run an AutoMM feature-fusion example."""
    from multimodalva.ensemble import FeatureFusionClassifier

    n_samples = 60 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features, n_samples=n_samples or 60,
    )

    print(f"Using {len(feat)} tabular feature columns.")
    clf = FeatureFusionClassifier(model_name=args.model, output_dir=ensure_dir(args.output_dir))
    results = clf.run(
        df=df,
        text_col=args.text_col,
        feature_cols=feat,
        label_col=args.label,
        time_limit=args.time_limit,
        hyperparameters={"model.hf_text.max_text_len": 128, "optimization.max_epochs": 1},
        use_hpo=False,
        top_k=args.top_k,
    )
    print_prediction_summary(results["predictions"], f"FeatureFusionClassifier ({args.model})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="preview",
        choices=["preview", "run"],
        help="preview=show data, run=train AutoMM model (default: preview).",
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
    parser.add_argument("--model", default="bioclinicalbert",
                        help="AutoMM text backbone alias or checkpoint (default: bioclinicalbert).")
    parser.add_argument("--time-limit", type=int, default=120, dest="time_limit",
                        help="AutoMM training budget in seconds (default: 120).")
    # Output
    parser.add_argument("--output-dir", default="runs/demo_feature_fusion", dest="output_dir",
                        help="Output directory (default: runs/demo_feature_fusion).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Number of top-k predictions (default: 3).")

    args = parser.parse_args()

    if args.mode == "preview":
        preview(args)
    else:
        run_classifier(args)


if __name__ == "__main__":
    main()
