"""Small demos for AutoMM feature fusion.

Usage:
    python tests/demo_ensemble_feature_fusion.py preview
    python tests/demo_ensemble_feature_fusion.py run

Notes:
    - ``preview`` is dependency-safe and just shows the toy multimodal table.
    - ``run`` requires ``autogluon.multimodal``.
"""

from __future__ import annotations

import argparse

from demo_utils import ensure_dir, feature_cols, make_demo_va_df, print_prediction_summary


def preview() -> None:
    """Show the dataset shape and the minimal AutoMM call."""
    df = make_demo_va_df(n_samples=10, seed=40)
    cols = ["narrative", *feature_cols(), "cause"]
    print(df[cols].head(4).to_string(index=False))
    print(
        "\nMinimal run:\n"
        "clf = FeatureFusionClassifier(model_name='bioclinicalbert', output_dir='runs/demo_feature_fusion')\n"
        "results = clf.run(df=df, text_col='narrative', feature_cols=feature_cols(), label_col='cause')"
    )


def run_classifier(model_name: str = "bioclinicalbert") -> None:
    """Run a small AutoMM feature-fusion example."""
    from multimodalva.ensemble import FeatureFusionClassifier

    df = make_demo_va_df(n_samples=60, seed=41)
    clf = FeatureFusionClassifier(model_name=model_name, output_dir=ensure_dir("runs/demo_feature_fusion"))
    results = clf.run(
        df=df,
        text_col="narrative",
        feature_cols=feature_cols(),
        label_col="cause",
        time_limit=120,
        hyperparameters={"model.hf_text.max_text_len": 128, "optimization.max_epochs": 1},
        use_hpo=False,
        top_k=3,
    )
    print_prediction_summary(results["predictions"], "FeatureFusionClassifier.run()")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="preview",
        choices=["preview", "run"],
        help="Which demo to run.",
    )
    parser.add_argument(
        "--model",
        default="bioclinicalbert",
        help="AutoMM text backbone alias or checkpoint. roberta-pm is supported.",
    )
    args = parser.parse_args()

    if args.mode == "preview":
        preview()
    else:
        run_classifier(args.model)


if __name__ == "__main__":
    main()
