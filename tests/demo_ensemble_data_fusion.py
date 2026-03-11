"""
Demo: ensemble strategy 1 — data-level fusion with multimodalva.

Covers:
    A. DataFusionClassifier via EnsembleClassifier dispatcher (recommended entry point)
    B. DataFusionClassifier — direct usage with custom fusion options
    C. Standalone text conversion: tabular_to_text() and build_fused_text()
    D. DataFusionClassifier — with Optuna HPO (use_optimize=True)
    E. Post-run analysis: predictions, fused text inspection

Run any section by calling its function from __main__.

Overview
--------
Data fusion converts each row's tabular features into a natural-language sentence,
then prepends the result to the free-text narrative.  The combined text is fine-tuned
with a long-context language model (Longformer / BigBird / Clinical-Longformer).

    Before fusion:
        narrative  = "The deceased had fever and chills."
        fever      = 1
        cough      = 0
        diarrhoea  = 0

    After fusion (with_neg=True, group_symptoms=True):
        "had fever.
         had no cough or diarrhoea.

         The deceased had fever and chills."

Recommended models (in order of preference for VA data):
    "allenai/longformer-base-4096"   — general-purpose; up to 4 096 tokens
    "yikuan8/Clinical-Longformer"    — domain-adapted on clinical notes
    "google/bigbird-roberta-base"    — block-sparse attention; up to 4 096 tokens

    Note: BERT (bert-base-uncased etc.) can be used when the fused text
    stays under 512 tokens (e.g. with_neg=False + group_symptoms=True +
    few features), but Longformer is strongly recommended for real VA data
    where tabular description + narrative frequently exceed 512 tokens.

qdesc usage
-----------
qdesc (question-description CSV) drives the natural-language rendering of
each indicator variable.  For real VA data (IVSS/WHO standard indicators):

    qdesc = load_qdesc()     # auto-loads from utils/qdesc.csv (cached)

For custom or non-standard datasets (like this toy demo):

    qdesc = pd.DataFrame()   # empty → template-only mode; all columns
                             # rendered via templates dict or binary_map
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
    """Create a synthetic VA DataFrame with BOTH narrative text and tabular features.

    In a real study, the narrative is an interviewer-written free-text account
    and the tabular columns are binary symptom indicators from the questionnaire.
    """
    rng = np.random.default_rng(seed)
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

    # Binary symptom indicators (0/1) — positivity rate varies by cause
    cause_idx   = np.array([causes.index(c) for c in labels])
    fever       = (rng.random(n) < (0.8 - cause_idx * 0.1)).astype(int)
    cough       = (rng.random(n) < (0.2 + cause_idx * 0.15)).astype(int)
    diarrhoea   = (rng.random(n) < np.where(labels == "Diarrhoea", 0.9, 0.1)).astype(int)
    weight_loss = (rng.random(n) < np.where(labels == "HIV/AIDS", 0.85, 0.1)).astype(int)
    pregnant    = (rng.random(n) < np.where(labels == "Maternal", 0.95, 0.02)).astype(int)

    return pd.DataFrame({
        "open_narrative": narratives,
        "fever":          fever,
        "cough":          cough,
        "diarrhoea":      diarrhoea,
        "weight_loss":    weight_loss,
        "pregnant":       pregnant,
        "cause":          labels,
    })


# Binary tabular columns passed to feature_cols
FEATURE_COLS = ["fever", "cough", "diarrhoea", "weight_loss", "pregnant"]


# ---------------------------------------------------------------------------
# A. EnsembleClassifier dispatcher — simplest entry point
# ---------------------------------------------------------------------------

def demo_via_ensemble_classifier():
    """Run data fusion through the unified EnsembleClassifier wrapper.

    Passes model_name as a constructor kwarg; all run() kwargs forwarded
    to DataFusionClassifier.run() unchanged.
    """
    from multimodalva.ensemble import EnsembleClassifier

    df = make_toy_df()

    clf = EnsembleClassifier(
        method="data_fusion",
        output_dir="runs/ensemble",
        model_name="allenai/longformer-base-4096",  # Longformer: handles fused text > 512 tokens
        # model_name="bert-base-uncased",           # ok only if fused text stays < 512 tokens
    )

    results = clf.run(
        df=df,
        text_col="open_narrative",
        feature_cols=FEATURE_COLS,
        label_col="cause",
        # --- tabular-to-text options ---
        qdesc=pd.DataFrame(),     # empty → template-only mode (no qdesc.csv lookup)
                                  # real VA data: pass None (auto-load) or load_qdesc()
        binary_map={0: "no", 1: "yes"},
        with_neg=True,            # include "had no fever" for negative indicators
        group_symptoms=True,      # group symptoms by verb → ~40–50% fewer tokens
        separator="\n\n",
        # --- tokenization ---
        max_length=1024,          # Longformer default; use 4096 for very long docs
        # --- training ---
        use_optimize=False,
        hyperparams={
            "learning_rate":               1e-5,
            "batch_size":                  4,    # small batch for long sequences
            "epochs":                      3,
            "warmup_ratio":                0.1,
            "gradient_accumulation_steps": 4,    # effective batch = 4 × 4 = 16
            "freeze_layers":               2,
        },
        gradient_checkpointing=True,   # strongly recommended for 1024+ token sequences
        early_stopping_patience=3,
        # --- inference ---
        batch_size=8,
        top_k=3,
    )

    _print_results(results, "Data fusion (EnsembleClassifier)")

    # Access the underlying DataFusionClassifier for strategy-specific attributes
    inner = clf.classifier
    print(f"\nTrain rows : {len(inner.train_df)}")
    print(f"Test rows  : {len(inner.test_df)}")
    print(f"Classes    : {inner.label2id}")

    return clf, results


# ---------------------------------------------------------------------------
# B. DataFusionClassifier — direct usage
# ---------------------------------------------------------------------------

def demo_direct():
    """DataFusionClassifier — direct class usage; all fusion options explicit."""
    from multimodalva.ensemble import DataFusionClassifier

    df = make_toy_df(n=300)

    clf = DataFusionClassifier(
        model_name="allenai/longformer-base-4096",
        output_dir="runs/data_fusion/direct",
    )

    results = clf.run(
        df=df,
        text_col="open_narrative",
        feature_cols=FEATURE_COLS,
        label_col="cause",
        # --- tabular-to-text ---
        qdesc=pd.DataFrame(),
        # templates: per-column format strings for non-binary columns.
        # Example (numeric): {"age": "Patient age: {value} years."}
        templates=None,
        binary_map={0: "no", 1: "yes"},
        with_neg=True,
        group_symptoms=True,
        separator="\n\n",
        fused_col="fused_text",   # temporary column name used internally
        # --- split ---
        test_size=0.2,
        random_state=42,
        stratify=True,
        # --- tokenization ---
        max_length=1024,
        # --- training ---
        use_optimize=False,
        hyperparams={
            "learning_rate":               1e-5,
            "batch_size":                  4,
            "epochs":                      3,
            "warmup_ratio":                0.1,
            "gradient_accumulation_steps": 4,
            "freeze_layers":               0,    # full fine-tuning; Longformer often benefits
        },
        use_lora=False,
        gradient_checkpointing=True,
        early_stopping_patience=3,
        # --- inference ---
        batch_size=8,
        top_k=3,
    )

    _print_results(results, "Data fusion (direct)")

    # Instance attributes populated by run()
    print(f"\nInstance attributes after run():")
    print(f"  clf.train_df       : {len(clf.train_df)} rows")
    print(f"  clf.test_df        : {len(clf.test_df)} rows")
    print(f"  clf.label2id       : {clf.label2id}")
    print(f"  clf.best_hyperparams: {clf.best_hyperparams}")
    print(f"  clf.study          : {clf.study}  (Optuna study; None when use_optimize=False)")

    return clf, results


# ---------------------------------------------------------------------------
# C. Standalone text conversion: tabular_to_text() and build_fused_text()
# ---------------------------------------------------------------------------

def demo_text_conversion():
    """Inspect the tabular-to-text conversion without running any model.

    Useful for verifying that the fused text looks sensible before committing
    to a long training run.
    """
    from multimodalva.ensemble.data_fusion import (
        build_fused_text,
        tabular_to_text,
        DEFAULT_SEPARATOR,
    )

    df = make_toy_df(n=6)

    # --- build_fused_text: apply to every row, return pd.Series ---
    fused = build_fused_text(
        df,
        text_col="open_narrative",
        feature_cols=FEATURE_COLS,
        qdesc=pd.DataFrame(),          # template-only mode
        binary_map={0: "no", 1: "yes"},
        with_neg=True,
        group_symptoms=True,
        separator=DEFAULT_SEPARATOR,   # "\n\n"
    )

    print("=== Fused text — first 3 rows ===")
    for i, text in enumerate(fused.head(3)):
        label = df.iloc[i]["cause"]
        print(f"\n[{i}] cause={label}")
        print(text)

    # --- tabular_to_text: single row ---
    row = df.iloc[0]
    print(f"\n=== tabular_to_text options (row 0 — cause={row['cause']}) ===")

    # Option 1: with_neg=True, group_symptoms=True (default; fewest tokens)
    t1 = tabular_to_text(
        row, feature_cols=FEATURE_COLS,
        qdesc=pd.DataFrame(), binary_map={0: "no", 1: "yes"},
        with_neg=True, group_symptoms=True,
    )
    print(f"\nwith_neg=True,  group_symptoms=True :\n  {t1}")

    # Option 2: with_neg=False (positive indicators only; good for BERT token budget)
    t2 = tabular_to_text(
        row, feature_cols=FEATURE_COLS,
        qdesc=pd.DataFrame(), binary_map={0: "no", 1: "yes"},
        with_neg=False, group_symptoms=True,
    )
    print(f"\nwith_neg=False, group_symptoms=True :\n  {t2}")

    # Option 3: group_symptoms=False (one sentence per indicator; more verbose)
    t3 = tabular_to_text(
        row, feature_cols=FEATURE_COLS,
        qdesc=pd.DataFrame(), binary_map={0: "no", 1: "yes"},
        with_neg=True, group_symptoms=False,
    )
    print(f"\nwith_neg=True,  group_symptoms=False:\n  {t3}")

    # --- Token length check ---
    # Estimate approximate token count for fused text (rough: 1 token ≈ 4 chars)
    lengths = fused.str.len()
    print(f"\nFused text lengths (chars): mean={lengths.mean():.0f}  max={lengths.max()}")
    print("Rough token estimate (÷4):  mean≈{:.0f}  max≈{:.0f}".format(
        lengths.mean() / 4, lengths.max() / 4
    ))
    print("→ Use max_length=512 only if max tokens < 512; otherwise use 1024+ (Longformer).")

    # --- Real VA data: load qdesc for richer rendering ---
    # qdesc drives natural-language rendering for standard IVSS indicators:
    #   demographics (age/sex) → "He was male, 15 to 49 years old."
    #   symptoms               → "He had fever and cough."  /  "He had no diarrhoea."
    #
    # from multimodalva.ensemble.data_fusion import load_qdesc
    # qdesc = load_qdesc()     # auto-loads utils/qdesc.csv (cached on first call)
    # fused_real = build_fused_text(df, ..., qdesc=qdesc)

    return fused


# ---------------------------------------------------------------------------
# D. DataFusionClassifier — Optuna HPO
# ---------------------------------------------------------------------------

def demo_hpo():
    """DataFusionClassifier with Optuna HPO.

    DEFAULT_SEARCH_SPACE (from data_fusion_classifier.py) is calibrated for
    long-context models and differs from the text-pipeline BERT defaults:

        learning_rate              — 5e-6 → 2e-5 (tighter; Longformer destabilises above ~2e-5)
        batch_size                 — 2 or 4 (GPU memory for 1024+ token sequences)
        epochs                     — 3, 5, or 8 (longer sequences need more passes)
        warmup_ratio               — 0.05 → 0.2 (wider warmup stabilises long-span attention)
        gradient_accumulation_steps — 4 or 8 (effective batch ~16–32)
        freeze_layers              — 0, 2, or 4

    Custom search_space merges over DEFAULT_SEARCH_SPACE — only supply keys to change.
    """
    from multimodalva.ensemble import DataFusionClassifier
    from multimodalva.ensemble.data_fusion_classifier import DEFAULT_SEARCH_SPACE

    df = make_toy_df(n=300)

    print("Default long-context search space:")
    for k, v in DEFAULT_SEARCH_SPACE.items():
        print(f"  {k:<35}: {v}")

    clf = DataFusionClassifier(
        model_name="allenai/longformer-base-4096",
        output_dir="runs/data_fusion/hpo",
    )

    # Narrow the search space for this toy experiment
    custom_space = {
        "learning_rate": ("float_log", 1e-5, 2e-5),
        "freeze_layers": ("categorical", [0, 2]),
    }

    results = clf.run(
        df=df,
        text_col="open_narrative",
        feature_cols=FEATURE_COLS,
        label_col="cause",
        qdesc=pd.DataFrame(),
        binary_map={0: "no", 1: "yes"},
        with_neg=True,
        group_symptoms=True,
        max_length=1024,
        # --- HPO ---
        use_optimize=True,
        n_trials=5,                      # increase to 20+ for real experiments
        optimize_metric="csmf_accuracy", # WHO/InsilicoVA standard; best for imbalanced VA data
        # optimize_metric="f1_macro",    # alternative
        search_space=custom_space,
        gradient_checkpointing=True,
        early_stopping_patience=3,
        # --- inference ---
        batch_size=8,
        top_k=3,
    )

    _print_results(results, "Data fusion (HPO)")

    if clf.study is not None:
        _analyze_study(clf.study)

    return clf, results


# ---------------------------------------------------------------------------
# E. Post-run analysis helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict, label: str = ""):
    pred = results["predictions"]
    top1 = pred.top1
    n    = len(top1)
    acc  = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"\n=== {label} ===")
    print(f"  Test samples : {n}")
    print(f"  Accuracy     : {acc:.3f}")
    print(f"  Output dir   : {results['output_dir']}")
    print(f"  Best HP      : {results['best_hyperparams']}")
    print(f"  Top-1 sample :\n{top1.head(3).to_string(index=False)}")


def _print_prediction_result(result):
    """Show all three prediction DataFrames."""
    print("\n--- top1 (true label, predicted label, probability) ---")
    print(result.top1.head(5).to_string(index=False))

    print("\n--- topk (top-3 classes + probabilities per sample) ---")
    print(result.topk.head(3).to_string(index=False))

    print("\n--- full (probability for every class; integer IDs as columns) ---")
    full_named = result.full.rename(
        columns={f"prob_{i}": f"prob_{v}" for i, v in result.id2label.items()}
    )
    print(full_named.head(3).to_string(index=False))


def _analyze_study(study):
    """Print Optuna study summary with per-trial metrics."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = study.trials_dataframe()
    completed = df[df["state"] == "COMPLETE"]

    print("\n=== HPO study summary ===")
    print(f"  Total trials : {len(df)}")
    print(f"  Completed    : {len(completed)}")
    print(f"  Best value   : {study.best_value:.4f}")
    print(f"  Best params  : {study.best_params}")

    # All 4 metrics stored as user_attrs per trial
    metric_cols = [c for c in df.columns if c.startswith("user_attrs_")]
    if metric_cols and len(completed) > 0:
        print("\n  Per-trial metrics (top 5 by objective):")
        display_cols = ["number", "value"] + metric_cols + ["duration"]
        available    = [c for c in display_cols if c in df.columns]
        print(
            completed.sort_values("value", ascending=False)[available]
            .head(5)
            .to_string(index=False)
        )


def demo_reload_study(study_name: str, storage_path: str):
    """Reload a saved Optuna study from its SQLite file.

    Example:
        demo_reload_study(
            study_name="data_fusion_hpo",
            storage_path="sqlite:///runs/data_fusion/hpo/hpo/hpo_longformer-base-4096.db",
        )
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=storage_path)
    _analyze_study(study)
    return study


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Section C runs without downloading any model — good for a quick sanity check.
    print("Running demo C: standalone text conversion ...")
    demo_text_conversion()

    # Uncomment to run full training demos (require model download):
    # print("\nRunning demo A: via EnsembleClassifier dispatcher ...")
    # demo_via_ensemble_classifier()

    # print("\nRunning demo B: DataFusionClassifier direct usage ...")
    # demo_direct()

    # print("\nRunning demo D: Optuna HPO ...")
    # demo_hpo()
