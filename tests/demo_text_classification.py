"""Small, current demos for the text classification pipeline.

Usage:
    python tests/demo_text_classification.py preview
    python tests/demo_text_classification.py run
    python tests/demo_text_classification.py hpo
    python tests/demo_text_classification.py steps
    python tests/demo_text_classification.py roberta-pm

Notes:
    - ``preview`` is safe and does not download a model.
    - ``run``, ``hpo``, and ``steps`` fine-tune a transformer on a toy dataset.
    - Default backbone is ClinicalBERT because it is a stable package baseline.
"""

from __future__ import annotations

import argparse

from demo_utils import ensure_dir, make_demo_va_df, print_prediction_summary


DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def preview() -> None:
    """Show the toy dataset and the minimal run call."""
    df = make_demo_va_df(n_samples=12, seed=0)
    print(df[["narrative", "cause"]].head(5).to_string(index=False))
    print(
        "\nMinimal run:\n"
        "clf = TextClassifier(model_name='emilyalsentzer/Bio_ClinicalBERT', output_dir='runs/demo_text')\n"
        "results = clf.run(df=df, text_col='narrative', label_col='cause',\n"
        "                  hyperparams={'epochs': 1, 'batch_size': 4, 'learning_rate': 2e-5})"
    )


def run_classifier(model_name: str = DEFAULT_MODEL) -> None:
    """Run the simplest end-to-end text pipeline."""
    from multimodalva.text import TextClassifier

    df = make_demo_va_df(n_samples=60, seed=1)
    clf = TextClassifier(model_name=model_name, output_dir=ensure_dir("runs/demo_text"))
    results = clf.run(
        df=df,
        text_col="narrative",
        label_col="cause",
        max_length=128,
        hyperparams={"epochs": 1, "batch_size": 4, "learning_rate": 2e-5},
        use_optimize=False,
        batch_size=8,
        top_k=3,
        use_lora=True,
        resume_training=False,
    )
    print_prediction_summary(results["predictions"], "TextClassifier.run()")


def run_hpo(model_name: str = DEFAULT_MODEL) -> None:
    """Run a very small Optuna search, then final training on all train data."""
    from multimodalva.text import TextClassifier

    df = make_demo_va_df(n_samples=60, seed=2)
    clf = TextClassifier(model_name=model_name, output_dir=ensure_dir("runs/demo_text_hpo"))
    results = clf.run(
        df=df,
        text_col="narrative",
        label_col="cause",
        max_length=128,
        use_optimize=True,
        n_trials=2,
        optimize_metric="f1_macro",
        batch_size=8,
        top_k=3,
        use_lora=True,
        resume_hpo=False,
        resume_training=False,
    )
    print("Best hyperparameters:", results["best_hyperparams"])
    print_prediction_summary(results["predictions"], "TextClassifier.run() with HPO")


def run_pipeline_steps(model_name: str = DEFAULT_MODEL) -> None:
    """Demonstrate the low-level split -> prepare -> optimize -> train -> predict flow."""
    from multimodalva.text.dataset import prepare_dataset
    from multimodalva.text.hpo import optimize
    from multimodalva.text.predict import predict
    from multimodalva.text.train import train
    from multimodalva.utils.split import split

    df = make_demo_va_df(n_samples=60, seed=3)
    train_df, test_df = split(
        df,
        text_col="narrative",
        label_col="cause",
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    train_ds, test_ds, label2id, id2label = prepare_dataset(
        train_df,
        test_df,
        text_col="narrative",
        label_col="cause",
        model_name=model_name,
        max_length=128,
    )
    best_hyperparams, _ = optimize(
        train_dataset=train_ds,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir=ensure_dir("runs/demo_text_steps/hpo"),
        n_trials=2,
        metric="f1_macro",
        use_lora=True,
        resume=False,
    )
    _, _, metadata = train(
        train_dataset=train_ds,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir=ensure_dir("runs/demo_text_steps/final"),
        hyperparams=best_hyperparams,
        val_size=None,
        use_lora=True,
        early_stopping_patience=2,
        resume=False,
    )
    result = predict(
        output_dir="runs/demo_text_steps/final",
        test_dataset=test_ds,
        batch_size=8,
        top_k=3,
    )
    print("Final training hyperparameters:", metadata["hyperparams"])
    print_prediction_summary(result, "Low-level text pipeline")


def resolve_roberta_pm() -> None:
    """Show how the package resolves the built-in RoBERTa-PM download."""
    from multimodalva.text.train import download_model

    model_path = download_model("roberta-pm", cache_dir=ensure_dir("runs/demo_roberta_pm_cache"))
    print("Resolved roberta-pm checkpoint:")
    print(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="preview",
        choices=["preview", "run", "hpo", "steps", "roberta-pm"],
        help="Which demo to run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace checkpoint or supported alias for run/hpo/steps.",
    )
    args = parser.parse_args()

    if args.mode == "preview":
        preview()
    elif args.mode == "run":
        run_classifier(args.model)
    elif args.mode == "hpo":
        run_hpo(args.model)
    elif args.mode == "steps":
        run_pipeline_steps(args.model)
    else:
        resolve_roberta_pm()


if __name__ == "__main__":
    main()
