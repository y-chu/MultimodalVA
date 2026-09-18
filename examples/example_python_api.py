#!/usr/bin/env python3
r"""
Interface 1 of 3 — the Python function ``multimodalva.run``.

Best for coders / statisticians working in a notebook or script: you already
have a DataFrame (or a CSV path) and want defaults for everything you don't
care about, with any single parameter overridable.

Every example below is one call. ``data`` accepts a path OR an in-memory
DataFrame, so preprocess however you like first, then hand the frame to ``run``.

Run a single example:
    python examples/example_python_api.py tabular
    python examples/example_python_api.py text
    python examples/example_python_api.py ensemble
    python examples/example_python_api.py all       # default

The examples use the built-in synthetic dataset so they run with no external
data. Swap ``data("va_sample")`` for your own ``pd.read_csv("clean.csv")`` (or
just ``data="clean.csv"``) and set ``label`` / ``text_col`` / ``features`` to
your columns.
"""

from __future__ import annotations

import sys

from multimodalva import data, run

OUT = "examples/_runs"


# ---------------------------------------------------------------------------
# 1. Unimodal text
# ---------------------------------------------------------------------------
def example_text() -> None:
    df = data("va_sample", n_per_class=30)  # -> columns: id, cause_of_death, narrative, i###...
    run(
        task="text",
        data=df,                      # or data="clean.csv"
        label="cause_of_death",
        text_col="narrative",
        model="bluebert",             # any alias from `multimodalva list-models`, a HF id, or a local dir
        output_dir=f"{OUT}/text_bluebert",
        optimize=True,                # Optuna HPO before the final fit
        n_trials=20,
        metric="f1_macro",
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
        label="cause_of_death",
        # feature selection: an explicit list, a regex ("re:..."), or "auto"
        # (every column except label / text / filter / split columns).
        features=r"re:^i\d{3}[a-zA-Z]$",
        model="lightgbm",             # catboost | lightgbm | xgboost | random_forest | mlp | svm | knn | ...
        output_dir=f"{OUT}/tabular_lightgbm",
        optimize=True, n_trials=40, metric="f1_macro",
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
        label="cause_of_death",
        text_col="narrative",
        features=features,
        model="clinicalbigbird",      # long-context; BigBird runs natively on Apple Silicon
        output_dir=f"{OUT}/data_fusion",
        optimize=False,
    )

    # 3b. Feature fusion — AutoGluon AutoMM jointly over text + tabular.
    #     (requires: pip install "multimodalva[feature_fusion]")
    run(
        task="feature_fusion",
        data=df,
        label="cause_of_death",
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
        label="cause_of_death",
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
        label="cause_of_death",
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
        label="cause_of_death",
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
