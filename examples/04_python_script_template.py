#!/usr/bin/env python3
r"""
Template 1 of 3 — a Python script calling ``multimodalva.run``.

Copy this file and edit it. Use it when you want the pipeline inside a script
of your own: your data may need cleaning in pandas first, and ``run`` takes the
DataFrame directly.

Run it as-is on the built-in synthetic data:
    python examples/04_python_script_template.py tabular
    python examples/04_python_script_template.py text
    python examples/04_python_script_template.py ensemble
    python examples/04_python_script_template.py hub   # needs a Hugging Face login
    python examples/04_python_script_template.py all   # default; skips 'hub'

To run it on your own data, change four things in any example below:

    data=data("va_sample", ...)  ->  data=pd.read_csv("my_data.csv")
                                     (or simply data="my_data.csv")
    label_col="cause_of_death"   ->  your cause-of-death column
    text_col="narrative"         ->  your narrative column
    features=...                 ->  your indicator columns: a list of names,
                                     a regex as "re:...", or "auto" for every
                                     column that is not the label / text /
                                     filter / split column

Everything else already has a working default. ``model`` and the ``Optimize(...)`` settings are what is most worth revisiting once a first run has
finished; see examples/README.md for what each one changes.
"""

from __future__ import annotations

import sys

from multimodalva import Optimize, data, run

OUT = "examples/_runs"


# ---------------------------------------------------------------------------
# 1. Unimodal text
# ---------------------------------------------------------------------------
def example_text() -> None:
    df = data("va_sample", n_per_class=30)  # -> columns: id, cause_of_death, narrative, i###...
    run(
        task="text",
        data=df,                      # or data="clean.csv"
        label_col="cause_of_death",
        text_col="narrative",
        model="bluebert",             # any alias from `multimodalva list-models`, a HF id, or a local dir
        output_dir=f"{OUT}/text_bluebert",
        hyperparams=Optimize(n_trials=20, metric="f1_macro"),  # search before the final fit
        # advanced knobs pass straight through to TextClassifier.run():
        use_lora=False,
        use_cv=True, n_cv_folds=3,
    )


# ---------------------------------------------------------------------------
# 2. Unimodal tabular
# ---------------------------------------------------------------------------
def example_tabular() -> None:
    df = data("va_sample", n_per_class=30)
    run(
        task="tabular",
        data=df,
        label_col="cause_of_death",
        # feature selection: an explicit list, a regex ("re:..."), or "auto"
        # (every column except label / text / filter / split columns).
        features=r"re:^i\d{3}[a-zA-Z]$",
        model="lightgbm",             # catboost | lightgbm | xgboost | random_forest | mlp | svm | knn | ...
        output_dir=f"{OUT}/tabular_lightgbm",
        hyperparams=Optimize(n_trials=40, metric="f1_macro"),
        encode_categoricals="ordinal",
    )


# ---------------------------------------------------------------------------
# 3. Ensemble (multimodal)
# ---------------------------------------------------------------------------
def example_ensemble() -> None:
    df = data("va_sample", n_per_class=30)
    features = [c for c in df.columns if c.startswith("i") and c[1:4].isdigit()]

    # 3a. Data fusion — tabular rows rendered to text, concatenated with the
    #     narrative, then fine-tuned as one long-context text model.
    run(
        task="data_fusion",
        data=df,
        label_col="cause_of_death",
        text_col="narrative",
        features=features,
        model="clinicalbigbird",      # long-context; BigBird runs natively on Apple Silicon
        output_dir=f"{OUT}/data_fusion",
    )

    # 3b. Feature fusion — AutoGluon AutoMM jointly over text + tabular.
    #     (requires: pip install "multimodalva[feature_fusion]")
    run(
        task="feature_fusion",
        data=df,
        label_col="cause_of_death",
        text_col="narrative",
        features=features,
        model="bioclinicalbert",
        output_dir=f"{OUT}/feature_fusion",
        init_kwargs={"fusion_strategy": "attention", "preset": "medium_quality"},
        time_limit=600,               # forwarded to FeatureFusionClassifier.run()
    )

    # 3c. Soft voting — independent base models, probabilities averaged.
    run(
        task="voting",
        data=df,
        label_col="cause_of_death",
        text_col="narrative",
        features=features,
        output_dir=f"{OUT}/voting",
        text_models=[{"model_name": "bluebert", "max_length": 512}],
        tabular_models=[{"model_name": "lightgbm"}, {"model_name": "random_forest"}],
    )

    # 3d. Stacking — k-fold OOF predictions -> meta-learner (super learner).
    run(
        task="stacking",
        data=df,
        label_col="cause_of_death",
        text_col="narrative",
        features=features,
        output_dir=f"{OUT}/stacking",
        text_models=[{"model_name": "bluebert", "max_length": 512}],
        tabular_models=[{"model_name": "lightgbm"}],
        init_kwargs={"n_folds": 5, "meta_learners": [{"model_name": "logistic_regression"}]},
    )


# ---------------------------------------------------------------------------
# 4. Publishing a trained model to the Hugging Face Hub
#    (only text-based models produce a standalone HF checkpoint: text + data_fusion)
# ---------------------------------------------------------------------------
def example_hub() -> None:
    df = data("va_sample", n_per_class=30)
    run(
        task="text",
        data=df,
        label_col="cause_of_death",
        text_col="narrative",
        model="bluebert",
        output_dir=f"{OUT}/text_for_hub",
        # Hub args pass through to TextClassifier.run(); requires `huggingface-cli login`
        # or hub_token=... . Set hub_private=False to publish publicly.
        push_to_hub=True,
        hub_repo_id="your-username/va-text-bluebert",
        hub_private=True,
    )
    # Data fusion publishes the same way (the card notes it was trained on fused text):
    #   run(task="data_fusion", ..., push_to_hub=True,
    #       hub_repo_id="your-username/va-datafusion")


EXAMPLES = {
    "text": example_text,
    "tabular": example_tabular,
    "ensemble": example_ensemble,
    "hub": example_hub,
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        # Skip 'hub' in 'all' — it would attempt a real upload.
        for name in ("tabular", "text", "ensemble"):
            print(f"\n=== {name} ===")
            EXAMPLES[name]()
    elif which in EXAMPLES:
        EXAMPLES[which]()
    else:
        print(f"Unknown example {which!r}. Choose from: {list(EXAMPLES)} or 'all'.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
