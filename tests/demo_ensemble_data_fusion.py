"""Small demos for the data-fusion pipeline.

Usage:
    python tests/demo_ensemble_data_fusion.py text
    python tests/demo_ensemble_data_fusion.py run

Notes:
    - ``text`` only converts tabular features to fused text and is the safest mode.
    - ``run`` fine-tunes a long-context text model on the fused text.
"""

from __future__ import annotations

import argparse

from demo_utils import ensure_dir, feature_cols, make_demo_va_df, print_prediction_summary


def preview_fused_text() -> None:
    """Show how tabular features become extra narrative text."""
    from multimodalva.ensemble.data_fusion import build_fused_text

    df = make_demo_va_df(n_samples=8, seed=30)
    fused = build_fused_text(
        df=df,
        text_col="narrative",
        feature_cols=feature_cols(),
        group_symptoms=True,
    )
    preview = df[["cause"]].copy()
    preview["fused_text"] = fused
    print(preview.head(4).to_string(index=False))


def run_classifier() -> None:
    """Run the end-to-end data-fusion pipeline on a toy dataset."""
    from multimodalva.ensemble import DataFusionClassifier

    df = make_demo_va_df(n_samples=60, seed=31)
    clf = DataFusionClassifier(
        model_name="allenai/longformer-base-4096",
        output_dir=ensure_dir("runs/demo_data_fusion"),
    )
    results = clf.run(
        df=df,
        text_col="narrative",
        feature_cols=feature_cols(),
        label_col="cause",
        group_symptoms=True,
        max_length=256,
        hyperparams={"epochs": 1, "batch_size": 2, "learning_rate": 2e-5},
        use_optimize=False,
        use_lora=True,
        resume_training=False,
        batch_size=4,
        top_k=3,
    )
    print_prediction_summary(results["predictions"], "DataFusionClassifier.run()")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="text",
        choices=["text", "run"],
        help="Which demo to run.",
    )
    args = parser.parse_args()

    if args.mode == "text":
        preview_fused_text()
    else:
        run_classifier()


if __name__ == "__main__":
    main()
