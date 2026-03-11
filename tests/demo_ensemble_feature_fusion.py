"""
Demo: ensemble strategy 2 — feature-level fusion (AutoGluon AutoMM).

Covers:
    A. FeatureFusionClassifier via EnsembleClassifier dispatcher (recommended entry point)
    B. FeatureFusionClassifier — direct usage, BioClinicalBERT + attention fusion
    C. Ablation: comparing fusion strategies (attention / concat / text_only / tabular_only)
    D. FeatureFusionClassifier — with AutoMM built-in HPO (use_hpo=True)
    E. Post-run analysis: inspect predictions, reload saved model, metric helpers

Run any section by calling its function from __main__.

Overview
--------
Feature-level fusion passes raw text narratives and tabular features jointly to
AutoGluon's MultiModalPredictor (AutoMM).  AutoMM handles tokenisation, tabular
encoding, and cross-modal fusion in a single unified model — no manual feature
engineering required.

Architecture options (``fusion_strategy``):
    "attention"    — Fusion Transformer: self-attention over the joint sequence of
                     the text CLS token + tabular embeddings.  Captures cross-modal
                     interactions (e.g. "fever" in narrative ↔ age + region tabular).
                     Recommended for VA data where narrative and structured indicators
                     carry overlapping and complementary information.

    "concat"       — Fusion MLP: concatenate text CLS + tabular embeddings, then pass
                     through an MLP classifier.  Faster; good baseline.

    "default"      — Let the AutoMM preset decide.  "best_quality" → "attention";
                     "medium_quality" → "concat".

    "text_only" /
    "tabular_only" — Ablations: disable one modality to isolate its contribution.

Text backbone (same shorthands as the text-only pipeline):
    "bioclinicalbert"  emilyalsentzer/Bio_ClinicalBERT   ← recommended for VA data
    "bert"             bert-base-uncased
    "biobert"          dmis-lab/biobert-base-cased-v1.2
    "longformer"       allenai/longformer-base-4096       ← for very long narratives
    or any full HuggingFace Hub ID / local path.

Saving and reloading:
    AutoMM saves to a structured directory — NOT pickle/joblib.
    Artifacts: ``output_dir/automm_model/`` (PyTorch weights + config + tokenizer).
    Reload::

        from autogluon.multimodal import MultiModalPredictor
        predictor = MultiModalPredictor.load("runs/feature_fusion/.../automm_model")
        proba = predictor.predict_proba(new_df)

HPO:
    Without  ``use_hpo``: single fixed training run.
    With     ``use_hpo=True``: AutoMM runs ``n_hpo_trials`` trials via Ray Tune (Bayes
             search by default) over ``DEFAULT_HPO_SPACE``.  Best config saved to
             ``best_hpo_config.json``.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Shared toy dataset
# ---------------------------------------------------------------------------

def make_toy_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic VA DataFrame with both narrative text and tabular features.

    In a real study, the narrative is an interviewer-written free-text account
    and the tabular columns are binary symptom indicators from the questionnaire.
    """
    rng    = np.random.default_rng(seed)
    causes = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]

    texts = {
        "Malaria":   "The deceased had fever and chills for several days before death.",
        "Pneumonia": "Patient had difficulty breathing and productive cough.",
        "HIV/AIDS":  "Chronic weight loss, night sweats and recurrent infections.",
        "Diarrhoea": "Watery stool for one week, signs of severe dehydration.",
        "Maternal":  "Died during or shortly after childbirth with heavy bleeding.",
    }
    labels     = rng.choice(causes, size=n)
    narratives = [texts[c] + f" (case {i})" for i, c in enumerate(labels)]

    # Binary symptom indicators (0/1) — positivity rate correlates with cause
    cause_idx   = np.array([causes.index(c) for c in labels])
    fever       = (rng.random(n) < (0.8 - cause_idx * 0.1)).astype(int)
    cough       = (rng.random(n) < (0.2 + cause_idx * 0.15)).astype(int)
    diarrhoea   = (rng.random(n) < np.where(labels == "Diarrhoea", 0.9, 0.1)).astype(int)
    weight_loss = (rng.random(n) < np.where(labels == "HIV/AIDS",  0.85, 0.1)).astype(int)
    pregnant    = (rng.random(n) < np.where(labels == "Maternal",  0.95, 0.02)).astype(int)
    age_group   = rng.choice(["<5", "5-14", "15-49", "50-69", "70+"], size=n)

    return pd.DataFrame({
        "open_narrative": narratives,
        "fever":          fever,
        "cough":          cough,
        "diarrhoea":      diarrhoea,
        "weight_loss":    weight_loss,
        "pregnant":       pregnant,
        "age_group":      age_group,
        "cause":          labels,
    })


# Text column and tabular feature columns
TEXT_COL     = "open_narrative"
FEATURE_COLS = ["fever", "cough", "diarrhoea", "weight_loss", "pregnant", "age_group"]
LABEL_COL    = "cause"


# ---------------------------------------------------------------------------
# A. EnsembleClassifier dispatcher — simplest entry point
# ---------------------------------------------------------------------------

def demo_via_ensemble_classifier():
    """Run feature fusion through the unified EnsembleClassifier wrapper.

    Constructor kwargs are forwarded to FeatureFusionClassifier.__init__();
    run() kwargs are forwarded to FeatureFusionClassifier.run() unchanged.
    """
    from multimodalva.ensemble import EnsembleClassifier

    df = make_toy_df()

    clf = EnsembleClassifier(
        method="feature_fusion",
        output_dir="runs/ensemble",
        model_name="bioclinicalbert",     # emilyalsentzer/Bio_ClinicalBERT
        fusion_strategy="attention",      # Fusion Transformer (recommended for VA)
        preset="best_quality",
        eval_metric="f1_macro",           # recommended for imbalanced VA data
    )

    results = clf.run(
        df=df,
        text_col=TEXT_COL,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        time_limit=3600,                  # 1 hour; increase for large datasets
        hyperparameters={
            # Fix max_text_len for BERT-based backbones (512 token limit)
            "model.hf_text.max_text_len": 512,
            # Optional: tune learning rate / epochs directly for a fixed run
            # "optimization.learning_rate": 2e-5,
            # "optimization.max_epochs":    10,
        },
        use_hpo=False,
        top_k=3,
    )

    _print_results(results, "Feature fusion (EnsembleClassifier)")

    # Access the underlying FeatureFusionClassifier for strategy-specific attrs
    inner = clf.classifier
    print(f"\nInstance attributes after run():")
    print(f"  clf.classifier.train_df  : {len(inner.train_df)} rows")
    print(f"  clf.classifier.test_df   : {len(inner.test_df)} rows")
    print(f"  clf.classifier.label2id  : {inner.label2id}")
    print(f"  clf.classifier.predictor : {type(inner.predictor).__name__}")

    # Model is automatically saved; how to reload:
    model_dir = inner.output_dir / "automm_model"
    print(f"\nModel saved to: {model_dir}")
    print("  Reload:  from autogluon.multimodal import MultiModalPredictor")
    print(f"           predictor = MultiModalPredictor.load('{model_dir}')")

    return clf, results


# ---------------------------------------------------------------------------
# B. FeatureFusionClassifier — direct usage
# ---------------------------------------------------------------------------

def demo_direct():
    """FeatureFusionClassifier — direct class usage; all options explicit.

    Uses BioClinicalBERT backbone with "attention" fusion strategy.
    Trained model saved to output_dir/automm_model/ for future reload.
    """
    from multimodalva.ensemble import FeatureFusionClassifier

    df = make_toy_df(n=300)

    clf = FeatureFusionClassifier(
        output_dir="runs/feature_fusion/direct",
        model_name="bioclinicalbert",     # or full ID: "emilyalsentzer/Bio_ClinicalBERT"
        fusion_strategy="attention",      # Fusion Transformer — captures cross-modal interactions
        preset="best_quality",            # "medium_quality" trains faster; "best_quality" is most accurate
        eval_metric="f1_macro",
    )

    results = clf.run(
        df=df,
        text_col=TEXT_COL,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        # --- split ---
        test_size=0.2,
        random_state=42,
        stratify=True,
        # --- training ---
        time_limit=3600,
        hyperparameters={
            "model.hf_text.max_text_len": 512,      # BioClinicalBERT: 512 token limit
            # Optionally override AutoMM defaults directly:
            # "optimization.learning_rate": 2e-5,
            # "optimization.max_epochs":    10,
            # "optimization.weight_decay":  1e-4,
        },
        use_hpo=False,
        # --- inference ---
        top_k=3,
    )

    _print_results(results, "Feature fusion — direct (BioClinicalBERT + attention)")
    _print_prediction_result(results["predictions"])

    # Instance attributes set by run()
    print("\nInstance attributes after run():")
    print(f"  clf.train_df       : {len(clf.train_df)} rows")
    print(f"  clf.test_df        : {len(clf.test_df)} rows")
    print(f"  clf.label2id       : {clf.label2id}")
    print(f"  clf.id2label       : {clf.id2label}")
    print(f"  clf.predictor      : {type(clf.predictor).__name__}")
    print(f"  clf.best_hpo_config: {clf.best_hpo_config}  (None — HPO was off)")

    # Saved artifacts
    print(f"\nSaved artifacts in {clf.output_dir}/:")
    print("  automm_model/        — AutoGluon native model directory (weights + config + tokenizer)")
    print("  label2id.json        — label → integer ID mapping")
    print("  id2label.json        — integer ID → label mapping")
    print("  training_metadata.json — full run config + label maps")

    return clf, results


# ---------------------------------------------------------------------------
# C. Ablation: comparing fusion strategies
# ---------------------------------------------------------------------------

def demo_ablation():
    """Train with each fusion strategy and compare test-set accuracy.

    Useful for deciding whether attention fusion genuinely helps over
    text-only or tabular-only baselines for a given dataset.

    Note: All strategies use the same BioClinicalBERT backbone and identical
    training budget, so the comparison is fair.
    """
    from multimodalva.ensemble import FeatureFusionClassifier

    df = make_toy_df(n=300)

    strategies = [
        ("attention",    "Fusion Transformer (cross-modal attention)"),
        ("concat",       "Fusion MLP (concatenated embeddings)"),
        ("text_only",    "Ablation: text narrative only"),
        ("tabular_only", "Ablation: tabular features only"),
    ]

    summary = []
    for strategy, description in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy} — {description}")
        print("="*60)

        clf = FeatureFusionClassifier(
            output_dir=f"runs/feature_fusion/ablation/{strategy}",
            model_name="bioclinicalbert",
            fusion_strategy=strategy,
            preset="best_quality",
            eval_metric="f1_macro",
        )

        # Shorter time_limit for ablation comparisons
        results = clf.run(
            df=df,
            text_col=TEXT_COL,
            feature_cols=FEATURE_COLS,
            label_col=LABEL_COL,
            time_limit=600,           # 10 min per strategy for quick comparison
            hyperparameters={"model.hf_text.max_text_len": 512},
            top_k=3,
        )

        top1 = results["predictions"].top1
        acc  = (top1["true_label"] == top1["predicted_label"]).mean()
        print(f"  Accuracy: {acc:.3f}")
        summary.append({"strategy": strategy, "description": description, "accuracy": acc})

    print("\n=== Ablation summary ===")
    summary_df = pd.DataFrame(summary).sort_values("accuracy", ascending=False)
    print(summary_df.to_string(index=False))

    return summary_df


# ---------------------------------------------------------------------------
# D. FeatureFusionClassifier — with AutoMM built-in HPO
# ---------------------------------------------------------------------------

def demo_hpo():
    """FeatureFusionClassifier with AutoMM's built-in HPO.

    When ``use_hpo=True``, AutoMM internally runs Ray Tune (Bayesian search
    by default) over ``DEFAULT_HPO_SPACE``.  No external Optuna loop needed.

    DEFAULT_HPO_SPACE covers:
        optimization.learning_rate        — float_log [1e-5, 1e-3]
        optimization.max_epochs           — int       [3, 15]
        env.batch_size                    — categorical [8, 16, 32]
        optimization.weight_decay         — float_log [1e-6, 1e-2]
        optimization.top_k_average_method — categorical ["best", "greedy_soup"]

    Custom ``hpo_search_space`` is *merged over* DEFAULT_HPO_SPACE —
    only supply keys you want to change.

    Best config is retrieved from ``predictor.fit_summary()`` (graceful
    fallback if not available) and saved to ``best_hpo_config.json``.
    """
    from multimodalva.ensemble import FeatureFusionClassifier
    from multimodalva.ensemble.feature_fusion import DEFAULT_HPO_SPACE

    df = make_toy_df(n=300)

    print("Default HPO search space:")
    for k, v in DEFAULT_HPO_SPACE.items():
        print(f"  {k:<45}: {v}")

    clf = FeatureFusionClassifier(
        output_dir="runs/feature_fusion/hpo",
        model_name="bioclinicalbert",
        fusion_strategy="attention",
        preset="best_quality",
        eval_metric="f1_macro",
    )

    # Narrow the search space for this toy experiment
    custom_space = {
        "optimization.learning_rate": ("float_log", 1e-5, 5e-5),   # tighter LR range
        "optimization.max_epochs":    ("int",        3,   8),        # fewer epochs
        "env.batch_size":             ("categorical", [8, 16]),       # exclude batch=32
    }

    results = clf.run(
        df=df,
        text_col=TEXT_COL,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        time_limit=3600,
        hyperparameters={
            "model.hf_text.max_text_len": 512,  # fixed — not part of the HPO search
        },
        # --- HPO ---
        use_hpo=True,
        n_hpo_trials=10,              # increase to 20–50 for real experiments
        hpo_search_space=custom_space,
        hpo_scheduler="local",        # "local" = single machine Ray Tune
        hpo_searcher="bayes",         # Bayesian optimisation (recommended)
        # hpo_searcher="random",      # faster but less sample-efficient
        top_k=3,
    )

    _print_results(results, "Feature fusion — HPO (BioClinicalBERT + attention)")

    print(f"\n  best_hpo_config : {results['best_hpo_config']}")
    print(f"\nBest HPO config saved to: {clf.output_dir / 'best_hpo_config.json'}")

    # To re-run with the best config (no HPO):
    if results["best_hpo_config"]:
        print("\nTo reproduce the best run without HPO:")
        print("  clf2 = FeatureFusionClassifier(...)")
        print("  clf2.run(..., hyperparameters=results['best_hpo_config'], use_hpo=False)")

    return clf, results


# ---------------------------------------------------------------------------
# E. Post-run analysis helpers
# ---------------------------------------------------------------------------

def demo_reload_and_predict(model_dir: str, new_df: pd.DataFrame):
    """Reload a saved AutoMM model and predict on new data.

    AutoMM uses its own native format — reload with ``MultiModalPredictor.load()``,
    NOT pickle/joblib.

    Args:
        model_dir: Path to ``output_dir/automm_model/`` saved during training.
        new_df:    DataFrame with the same columns as the training data
                   (excluding the label column).

    Example::

        import json, pathlib
        output_dir = pathlib.Path("runs/feature_fusion/direct")

        # Load label map (needed to decode probability columns)
        with open(output_dir / "id2label.json") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}

        proba = demo_reload_and_predict(
            model_dir=str(output_dir / "automm_model"),
            new_df=new_df_without_label,
        )
    """
    from autogluon.multimodal import MultiModalPredictor

    print(f"Reloading model from: {model_dir}")
    predictor = MultiModalPredictor.load(model_dir)

    proba_df = predictor.predict_proba(new_df)
    print(f"\npredict_proba() output — shape: {proba_df.shape}")
    print(f"  Columns (label strings): {list(proba_df.columns)}")
    print(proba_df.head(3).to_string())

    return proba_df


def _print_results(results: dict, label: str = ""):
    """Print a concise results summary."""
    pred = results["predictions"]
    top1 = pred.top1
    n    = len(top1)
    acc  = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"\n=== {label} ===")
    print(f"  Test samples    : {n}")
    print(f"  Accuracy        : {acc:.3f}")
    print(f"  Output dir      : {results['output_dir']}")
    print(f"  best_hpo_config : {results['best_hpo_config']}")
    print(f"\n  Top-1 sample (first 3 rows):")
    print(top1.head(3).to_string(index=False))


def _print_prediction_result(result):
    """Show all three prediction DataFrames from a PredictionResult."""
    print("\n--- top1: true label, predicted label, probability ---")
    print(result.top1.head(5).to_string(index=False))

    print("\n--- topk: top-3 predictions per sample ---")
    print(result.topk.head(3).to_string(index=False))

    print("\n--- full: probability for every class (integer IDs as columns) ---")
    # Rename prob_0, prob_1 → cause name for readability
    full_named = result.full.rename(
        columns={f"prob_{i}": f"prob_{v}" for i, v in result.id2label.items()}
    )
    print(full_named.head(3).to_string(index=False))


def demo_score_predictions(results: dict):
    """Score a completed run with all four supported metrics.

    Uses ``score_predictions()`` from ``multimodalva.utils`` — the same helper
    used in text and tabular pipelines.
    """
    from multimodalva.utils import score_predictions

    top1 = results["predictions"].top1
    for metric in ["accuracy", "f1_macro", "f1_weighted", "csmf_accuracy"]:
        score = score_predictions(top1, metric=metric)
        print(f"  {metric:<20}: {score:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---------- quick sanity checks (no model download required) ----------
    print("Creating toy dataset ...")
    df = make_toy_df()
    print(f"  Shape : {df.shape}")
    print(f"  Causes: {df['cause'].value_counts().to_dict()}")
    print(f"  Cols  : {list(df.columns)}")

    # Uncomment the section(s) you want to run (require model download):

    # print("\nRunning demo A: via EnsembleClassifier dispatcher ...")
    # demo_via_ensemble_classifier()

    # print("\nRunning demo B: FeatureFusionClassifier direct usage ...")
    # demo_direct()

    # print("\nRunning demo C: ablation across fusion strategies ...")
    # demo_ablation()

    # print("\nRunning demo D: with AutoMM built-in HPO ...")
    # demo_hpo()

    # ----- demo E: reload and score (run after demo B or D) -----
    # import json, pathlib
    # output_dir = pathlib.Path("runs/feature_fusion/direct")
    # with open(output_dir / "id2label.json") as f:
    #     id2label = {int(k): v for k, v in json.load(f).items()}
    # from multimodalva.utils.split import split
    # _, test_df = split(df, label_col=LABEL_COL)
    # demo_reload_and_predict(
    #     model_dir=str(output_dir / "automm_model"),
    #     new_df=test_df.drop(columns=[LABEL_COL]),
    # )
