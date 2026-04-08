"""Small, current demos for the tabular classification pipeline.

Usage:
    python tests/demo_tabular_classification.py preview
    python tests/demo_tabular_classification.py run
    python tests/demo_tabular_classification.py hpo
    python tests/demo_tabular_classification.py steps

Notes:
    - Default model is ``random_forest`` so the demo works with core sklearn deps.
    - ``hpo`` demonstrates the adaptive tabular search space defaults.
"""

from __future__ import annotations

import argparse

from multimodalva.tabular import TabularClassifier
from multimodalva.tabular.dataset import prepare_dataset
from multimodalva.tabular.hpo import optimize
from multimodalva.tabular.predict import predict
from multimodalva.tabular.train import train
from multimodalva.utils.split import split

from demo_utils import ensure_dir, feature_cols, make_demo_va_df, print_prediction_summary


DEFAULT_MODEL = "random_forest"


def preview() -> None:
    """Show the toy tabular dataset and the minimal run call."""
    df = make_demo_va_df(n_samples=12, seed=10)
    print(df[feature_cols() + ["cause"]].head(5).to_string(index=False))
    print(
        "\nMinimal run:\n"
        "clf = TabularClassifier(model_name='random_forest', output_dir='runs/demo_tabular')\n"
        "results = clf.run(df=df, feature_cols=feature_cols(), label_col='cause')"
    )


def run_classifier(model_name: str = DEFAULT_MODEL) -> None:
    """Run the simplest end-to-end tabular pipeline."""
    df = make_demo_va_df(n_samples=120, seed=11)
    clf = TabularClassifier(model_name=model_name, output_dir=ensure_dir("runs/demo_tabular"))
    results = clf.run(
        df=df,
        feature_cols=feature_cols(),
        label_col="cause",
        hyperparams={"n_estimators": 80, "max_depth": 6},
        cat_cols=["sex"],
        use_optimize=False,
        top_k=3,
    )
    print_prediction_summary(results["predictions"], "TabularClassifier.run()")


def run_hpo(model_name: str = DEFAULT_MODEL) -> None:
    """Run a very small Optuna search with the adaptive search-space defaults."""
    df = make_demo_va_df(n_samples=120, seed=12)
    clf = TabularClassifier(model_name=model_name, output_dir=ensure_dir("runs/demo_tabular_hpo"))
    results = clf.run(
        df=df,
        feature_cols=feature_cols(),
        label_col="cause",
        cat_cols=["sex"],
        use_optimize=True,
        n_trials=4,
        optimize_metric="f1_macro",
        search_space_profile="auto",
        top_k=3,
    )
    print("Best hyperparameters:", results["best_hyperparams"])
    print_prediction_summary(results["predictions"], "TabularClassifier.run() with HPO")


def run_pipeline_steps(model_name: str = DEFAULT_MODEL) -> None:
    """Demonstrate the low-level split -> prepare -> optimize -> train -> predict flow."""
    df = make_demo_va_df(n_samples=120, seed=13)
    train_df, test_df = split(
        df,
        label_col="cause",
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    X_train, X_test, y_train, y_test, preprocessor, label2id, id2label, feature_names = prepare_dataset(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols(),
        label_col="cause",
        cat_cols=["sex"],
        encode_categoricals="ordinal",
    )
    best_hyperparams, _ = optimize(
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir=ensure_dir("runs/demo_tabular_steps/hpo"),
        n_trials=4,
        metric="f1_macro",
        search_space_profile="auto",
    )
    _, metadata = train(
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir=ensure_dir("runs/demo_tabular_steps/final"),
        hyperparams=best_hyperparams,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
    result = predict(
        output_dir="runs/demo_tabular_steps/final",
        X_test=X_test,
        y_test=y_test,
        top_k=3,
    )
    print("Final training hyperparameters:", metadata["hyperparams"])
    print_prediction_summary(result, "Low-level tabular pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="preview",
        choices=["preview", "run", "hpo", "steps"],
        help="Which demo to run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Supported tabular model alias, for example random_forest, gbdt, lightgbm, xgboost.",
    )
    args = parser.parse_args()

    if args.mode == "preview":
        preview()
    elif args.mode == "run":
        run_classifier(args.model)
    elif args.mode == "hpo":
        run_hpo(args.model)
    else:
        run_pipeline_steps(args.model)


if __name__ == "__main__":
    main()
