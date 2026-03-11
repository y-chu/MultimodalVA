"""
Demo: ensemble strategy 3 — decision-level fusion via soft voting.

Covers:
    A. Mode 1 — vote_from_results(): combine pre-computed PredictionResult objects
       (runs without any model download — uses synthetic predictions)
    B. Mode 2 — EnsembleClassifier dispatcher with method="soft_voting"
    C. Mode 2 — SoftVotingClassifier direct: all 5 text base models
    D. Mode 2 — SoftVotingClassifier direct: all 5 tabular base models
    E. Mode 2 — Mixed text + tabular ensemble (bioclinicalbert + lightgbm + catboost)
    F. Mode 2 — Per-model Optuna HPO (use_optimize=True)
    G. Post-run analysis: inspect base predictions, weighted vote, performance leaderboard

Run a single section::

    python tests/demo_ensemble_voting.py A    # runs without model download
    python tests/demo_ensemble_voting.py C    # all text models (requires GPU/download)

Run all sections::

    python tests/demo_ensemble_voting.py


Overview — Soft Voting
----------------------
Trains each base model independently on the same train split.  At inference
time, combines their probability matrices via (optionally weighted) averaging
and takes the argmax as the final prediction.

Key properties:
- Requires ≥ 2 base models total (text + tabular combined).
- Text and tabular base models share the same label2id / id2label, guaranteeing
  that prob_0, prob_1, ... refer to identical classes across all models.
- Per-model Optuna HPO: add ``"use_optimize": True`` to any model spec.
- Text-only, tabular-only, or mixed ensembles are all supported.

Text model shorthands (same as text-only pipeline):
    "bioclinicalbert"   emilyalsentzer/Bio_ClinicalBERT       ← recommended for VA
    "bluebert"          bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12
    "biomedbert"        microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    "bert"              bert-base-uncased
    "roberta-pm"        PubMedRoBERTa (downloaded via text.train.download_model)

Tabular model aliases (same as tabular-only pipeline):
    "lightgbm"          LGBMClassifier       — gradient boosting on trees; fast + accurate
    "catboost"          CatBoostClassifier   — gradient boosting; handles categoricals natively
    "gbdt"              GradientBoostingClassifier (sklearn) — note: user may call this "cbdt"
    "xgboost"           XGBClassifier        — parallel gradient boosting
    "mlp"               MLPClassifier        — feed-forward neural net; scale_numeric=True recommended

Saving:
    Base model artifacts: output_dir/base_models/text_<i>/ and tabular_<i>/
    Ensemble metadata:    output_dir/training_metadata.json
                          output_dir/label2id.json
                          output_dir/id2label.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Shared toy dataset
# ---------------------------------------------------------------------------

def make_toy_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic VA DataFrame with text narratives and tabular features.

    In a real study:
    - ``open_narrative``: interviewer free-text account of the death circumstances
    - Tabular columns: binary symptom indicators from the structured questionnaire

    Cause-feature correlations are intentionally strong so that even the toy
    models can demonstrate ensemble gains.
    """
    rng    = np.random.default_rng(seed)
    causes = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]

    texts = {
        "Malaria":   "The deceased had fever and chills for several days before death.",
        "Pneumonia": "Patient had difficulty breathing and persistent productive cough.",
        "HIV/AIDS":  "Chronic weight loss, night sweats, and recurrent opportunistic infections.",
        "Diarrhoea": "Watery stool for one week with signs of severe dehydration.",
        "Maternal":  "Died during or shortly after childbirth complicated by heavy bleeding.",
    }
    labels     = rng.choice(causes, size=n)
    narratives = [texts[c] + f" (case {i})" for i, c in enumerate(labels)]

    cause_idx   = np.array([causes.index(c) for c in labels])
    fever       = (rng.random(n) < (0.85 - cause_idx * 0.1)).astype(int)
    cough       = (rng.random(n) < (0.15 + cause_idx * 0.15)).astype(int)
    diarrhoea   = (rng.random(n) < np.where(labels == "Diarrhoea", 0.92, 0.08)).astype(int)
    weight_loss = (rng.random(n) < np.where(labels == "HIV/AIDS",  0.88, 0.08)).astype(int)
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


TEXT_COL     = "open_narrative"
FEATURE_COLS = ["fever", "cough", "diarrhoea", "weight_loss", "pregnant", "age_group"]
LABEL_COL    = "cause"
CAUSES       = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]
ID2LABEL     = {i: c for i, c in enumerate(CAUSES)}
LABEL2ID     = {c: i for i, c in enumerate(CAUSES)}


# ---------------------------------------------------------------------------
# A. Mode 1 — vote_from_results(): no training required
# ---------------------------------------------------------------------------

def _make_synthetic_result(true_labels: list[str], seed: int = 0):
    """Build a synthetic PredictionResult for Mode 1 demo (no model needed)."""
    from multimodalva.utils.types import PredictionResult

    rng = np.random.default_rng(seed)
    n   = len(true_labels)
    n_c = len(CAUSES)

    # Softmax-ish probabilities: dominant class signal + noise
    true_ids    = np.array([LABEL2ID[l] for l in true_labels])
    logits      = rng.random((n, n_c)) * 0.5
    logits[np.arange(n), true_ids] += 2.5 + rng.random(n) * 0.5  # true class boost
    exp_logits  = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba       = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    sorted_ids  = sorted(ID2LABEL.keys())
    top1_pos    = np.argmax(proba, axis=1)
    top1_ids    = [sorted_ids[p] for p in top1_pos]

    top1_df = pd.DataFrame({
        "true_label":      true_labels,
        "predicted_label": [ID2LABEL[ci] for ci in top1_ids],
        "predicted_prob":  proba[np.arange(n), top1_pos],
    })

    full_data = {"true_label": true_labels}
    for j, cid in enumerate(sorted_ids):
        full_data[f"prob_{cid}"] = proba[:, j]
    full_df = pd.DataFrame(full_data)

    top_k = 3
    top_indices = np.argsort(proba, axis=1)[:, ::-1][:, :top_k]
    topk_data = {"true_label": true_labels}
    for j in range(top_k):
        col_pos   = top_indices[:, j]
        class_ids = [sorted_ids[p] for p in col_pos]
        topk_data[f"top{j+1}_label"] = [ID2LABEL[ci] for ci in class_ids]
        topk_data[f"top{j+1}_prob"]  = proba[np.arange(n), col_pos]
    topk_df = pd.DataFrame(topk_data)

    return PredictionResult(top1=top1_df, full=full_df, topk=topk_df, id2label=ID2LABEL)


def demo_mode1_vote_from_results():
    """Mode 1: vote over pre-computed PredictionResult objects — no training required.

    This is the typical workflow when you have already trained text and tabular
    models separately (e.g. TextClassifier, TabularClassifier) and want to combine
    their predictions post-hoc.

    Requirements:
    - All PredictionResult objects must share the same id2label mapping.
    - This guarantees that prob_0, prob_1, ... refer to identical classes in every
      model's .full DataFrame — safe for column-wise averaging.

    In a real study you would load results from disk::

        import json, pickle
        with open("runs/text/id2label.json") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        text_result    = text_clf.predictions    # or load from disk
        tabular_result = tabular_clf.predictions

        from multimodalva.ensemble.voting import vote_from_results
        final = vote_from_results([text_result, tabular_result], id2label=id2label)
    """
    from multimodalva.ensemble.voting import vote_from_results, soft_vote
    from multimodalva.utils import csmf_accuracy

    print("=== Mode 1: vote_from_results() ===")
    print("Building synthetic PredictionResult objects (no model download needed) ...")

    rng = np.random.default_rng(42)
    n   = 200
    true_labels = list(rng.choice(CAUSES, size=n))

    # Simulate 5 independently trained models (3 text, 2 tabular)
    bert_result    = _make_synthetic_result(true_labels, seed=0)   # bioclinicalbert
    bluebert_result = _make_synthetic_result(true_labels, seed=1)  # bluebert
    bio_result     = _make_synthetic_result(true_labels, seed=2)   # biomedbert
    lgbm_result    = _make_synthetic_result(true_labels, seed=3)   # lightgbm
    cat_result     = _make_synthetic_result(true_labels, seed=4)   # catboost

    # --- A1. Equal-weight vote over all 5 models ----------------------------
    print("\n--- A1. Equal-weight vote (5 models) ---")
    all_results = [bert_result, bluebert_result, bio_result, lgbm_result, cat_result]
    final = vote_from_results(all_results, id2label=ID2LABEL, top_k=3)
    _print_result_summary(final, "5-model equal-weight vote")

    # --- A2. Weighted vote: trust text models more ---------------------------
    print("\n--- A2. Weighted vote: [0.25, 0.25, 0.2, 0.15, 0.15] ---")
    weighted = vote_from_results(
        all_results,
        id2label=ID2LABEL,
        weights=[0.25, 0.25, 0.20, 0.15, 0.15],   # bioclinicalbert + bluebert weighted highest
        top_k=3,
    )
    _print_result_summary(weighted, "5-model weighted vote")

    # --- A3. Compare individual models vs. ensemble -------------------------
    print("\n--- A3. Individual model accuracy vs. ensemble ---")
    model_names = ["bioclinicalbert", "bluebert", "biomedbert", "lightgbm", "catboost"]
    rows = []
    for name, r in zip(model_names, all_results):
        acc   = (r.top1["true_label"] == r.top1["predicted_label"]).mean()
        csmfa = csmf_accuracy(r.top1["true_label"], r.top1["predicted_label"])
        rows.append({"model": name, "accuracy": acc, "csmf_accuracy": csmfa})
    acc_ens   = (final.top1["true_label"] == final.top1["predicted_label"]).mean()
    csmfa_ens = csmf_accuracy(final.top1["true_label"], final.top1["predicted_label"])
    rows.append({"model": "ensemble (equal)", "accuracy": acc_ens, "csmf_accuracy": csmfa_ens})
    acc_w     = (weighted.top1["true_label"] == weighted.top1["predicted_label"]).mean()
    csmfa_w   = csmf_accuracy(weighted.top1["true_label"], weighted.top1["predicted_label"])
    rows.append({"model": "ensemble (weighted)", "accuracy": acc_w, "csmf_accuracy": csmfa_w})

    cmp_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print(cmp_df.to_string(index=False, float_format="{:.4f}".format))

    # --- A4. Inspect the final PredictionResult structure -------------------
    print("\n--- A4. PredictionResult structure ---")
    _print_prediction_result(final, n_rows=3)

    # --- A5. Directly calling soft_vote() on raw numpy arrays ---------------
    print("\n--- A5. soft_vote() directly on numpy matrices ---")
    prob_matrices = [
        r.full[[f"prob_{i}" for i in sorted(ID2LABEL.keys())]].to_numpy(dtype=float)
        for r in all_results
    ]
    combined = soft_vote(prob_matrices)                          # equal weights
    combined_w = soft_vote(prob_matrices, weights=[0.3, 0.25, 0.2, 0.15, 0.1])
    print(f"  soft_vote() output shape: {combined.shape}   (n_samples × n_classes)")
    print(f"  Row sums (should be 1.0): {combined.sum(axis=1)[:5].round(6)}")
    print(f"  Weighted — row sums     : {combined_w.sum(axis=1)[:5].round(6)}")

    return final


# ---------------------------------------------------------------------------
# B. Mode 2 — EnsembleClassifier dispatcher
# ---------------------------------------------------------------------------

def demo_via_ensemble_classifier():
    """Mode 2: soft voting through the unified EnsembleClassifier dispatcher.

    Constructor kwargs are forwarded to SoftVotingClassifier.__init__();
    run() kwargs are forwarded to SoftVotingClassifier.run() unchanged.

    This is the recommended entry point for most users — same interface
    as all other ensemble strategies.
    """
    from multimodalva.ensemble import EnsembleClassifier

    df = make_toy_df()

    clf = EnsembleClassifier(
        method="soft_voting",
        output_dir="runs/ensemble/voting/dispatcher",
        # SoftVotingClassifier kwargs:
        text_models=[
            {
                "model_name":    "bioclinicalbert",
                "hyperparams":   {"epochs": 5, "learning_rate": 2e-5, "batch_size": 16},
                "max_length":    512,
                "use_lora":      False,
                "use_optimize":  False,
            },
        ],
        tabular_models=[
            {
                "model_name":          "lightgbm",
                "hyperparams":         {"n_estimators": 300, "learning_rate": 0.05},
                "encode_categoricals": "ordinal",
                "scale_numeric":       False,
                "use_optimize":        False,
            },
            {
                "model_name":  "catboost",
                "hyperparams": {"iterations": 300, "learning_rate": 0.05, "depth": 6},
                "use_optimize": False,
            },
        ],
        weights=None,   # equal weights; set e.g. [0.5, 0.25, 0.25] to trust BERT more
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
        # --- text training ---
        val_size=0.1,                     # internal validation fraction for text models
        early_stopping_patience=3,
        gradient_checkpointing=False,
        # --- tabular training ---
        n_jobs=-1,
        use_gpu=None,                     # None = auto-detect CUDA
        # --- inference ---
        top_k=3,
        batch_size=32,
    )

    _print_results(results, "EnsembleClassifier dispatcher (soft_voting)")

    # Access the underlying SoftVotingClassifier via clf.classifier
    inner = clf.classifier
    print(f"\nInstance attributes via clf.classifier:")
    print(f"  train_df          : {len(inner.train_df)} rows")
    print(f"  test_df           : {len(inner.test_df)} rows")
    print(f"  label2id          : {inner.label2id}")
    print(f"  base_predictions  : {len(inner.base_predictions)} results")

    print(f"\nBase model accuracy breakdown:")
    all_specs = inner.text_models + inner.tabular_models
    for i, (spec, bpred) in enumerate(zip(all_specs, inner.base_predictions)):
        acc = (bpred.top1["true_label"] == bpred.top1["predicted_label"]).mean()
        print(f"  [{i}] {spec['model_name']:<20} acc = {acc:.4f}")

    return clf, results


# ---------------------------------------------------------------------------
# C. Mode 2 — all 5 text base models
# ---------------------------------------------------------------------------

def demo_all_text_models():
    """Soft voting over all 5 text base models.

    Text base models:
        bioclinicalbert — emilyalsentzer/Bio_ClinicalBERT         (recommended for VA)
        bluebert        — bionlp/bluebert_pubmed_mimic_uncased_...
        biomedbert      — microsoft/BiomedNLP-BiomedBERT-base-...
        bert            — bert-base-uncased
        roberta-pm      — PubMedRoBERTa (downloaded via download_model)

    Note: roberta-pm is downloaded on first use via text.train.download_model().
    Set ``"model_name": "roberta-pm"`` — the pipeline handles download automatically.

    All models use the same train/test split and label2id — guarantees that
    prob_0, prob_1, ... refer to identical classes for safe averaging.
    """
    from multimodalva.ensemble import SoftVotingClassifier

    df = make_toy_df(n=400)

    clf = SoftVotingClassifier(
        text_models=[
            {
                "model_name":    "bioclinicalbert",   # emilyalsentzer/Bio_ClinicalBERT
                "hyperparams":   {"epochs": 5, "learning_rate": 2e-5, "batch_size": 16},
                "max_length":    512,
                "use_lora":      False,
                "use_optimize":  False,
            },
            {
                "model_name":    "bluebert",          # bionlp/bluebert_pubmed_mimic_uncased_...
                "hyperparams":   {"epochs": 5, "learning_rate": 2e-5, "batch_size": 16},
                "max_length":    512,
                "use_optimize":  False,
            },
            {
                "model_name":    "biomedbert",        # microsoft/BiomedNLP-BiomedBERT-...
                "hyperparams":   {"epochs": 5, "learning_rate": 2e-5, "batch_size": 16},
                "max_length":    512,
                "use_optimize":  False,
            },
            {
                "model_name":    "bert",              # bert-base-uncased
                "hyperparams":   {"epochs": 5, "learning_rate": 3e-5, "batch_size": 16},
                "max_length":    512,
                "use_optimize":  False,
            },
            {
                "model_name":    "roberta-pm",        # PubMedRoBERTa (auto-downloaded)
                "hyperparams":   {"epochs": 5, "learning_rate": 2e-5, "batch_size": 16},
                "max_length":    512,
                "use_optimize":  False,
            },
        ],
        tabular_models=[],          # text-only ensemble
        output_dir="runs/ensemble/voting/text_only",
        weights=None,               # equal weights; override if one model is stronger
    )

    results = clf.run(
        df=df,
        text_col=TEXT_COL,
        feature_cols=None,          # not needed for text-only
        label_col=LABEL_COL,
        test_size=0.2,
        random_state=42,
        val_size=0.1,               # internal val for text training
        early_stopping_patience=3,
        gradient_checkpointing=True,  # enable for large models / limited VRAM
        top_k=3,
        batch_size=32,
    )

    _print_results(results, "All 5 text models — equal-weight soft vote")
    _print_base_model_breakdown(results, clf.text_models, [])

    # Show per-model strengths via weighted voting
    print("\n--- Applying custom weights after training ---")
    from multimodalva.ensemble.voting import vote_from_results
    weighted = vote_from_results(
        clf.base_predictions,
        id2label=clf.id2label,
        # Down-weight bert (general domain); trust clinical/biomedical models more
        weights=[0.25, 0.22, 0.22, 0.12, 0.19],
        top_k=3,
    )
    acc_w = (weighted.top1["true_label"] == weighted.top1["predicted_label"]).mean()
    print(f"  Weighted ensemble accuracy: {acc_w:.4f}")

    return clf, results


# ---------------------------------------------------------------------------
# D. Mode 2 — all 5 tabular base models
# ---------------------------------------------------------------------------

def demo_all_tabular_models():
    """Soft voting over all 5 tabular base models.

    Tabular base models:
        lightgbm  — LGBMClassifier         (gradient boosting; fast + strong)
        catboost  — CatBoostClassifier      (gradient boosting; handles categoricals)
        gbdt      — GradientBoostingClassifier (sklearn; note: alias "gbdt", not "cbdt")
        xgboost   — XGBClassifier           (parallel gradient boosting)
        mlp       — MLPClassifier           (feed-forward NN; scale_numeric=True recommended)

    All models share the same preprocessed features.  Both ordinal-encoded and
    one-hot encoded feature sets are demonstrated via per-model encode_categoricals.

    Note: ``scale_numeric=True`` is applied to mlp only (recommended for neural nets).
    Tree-based models (lightgbm, catboost, gbdt, xgboost) do not need scaling.
    """
    from multimodalva.ensemble import SoftVotingClassifier

    df = make_toy_df(n=400)

    clf = SoftVotingClassifier(
        text_models=[],             # tabular-only ensemble
        tabular_models=[
            {
                "model_name":          "lightgbm",
                "hyperparams":         {
                    "n_estimators":    300,
                    "learning_rate":   0.05,
                    "max_depth":       7,
                    "num_leaves":      63,
                    "min_child_samples": 20,
                },
                "encode_categoricals": "ordinal",
                "scale_numeric":       False,
                "use_optimize":        False,
            },
            {
                "model_name":  "catboost",
                "hyperparams": {
                    "iterations":     300,
                    "learning_rate":  0.05,
                    "depth":          6,
                    "l2_leaf_reg":    3.0,
                },
                "encode_categoricals": "ordinal",   # CatBoost handles ordinals natively
                "scale_numeric":       False,
                "use_optimize":        False,
            },
            {
                # Note: the alias is "gbdt" (GradientBoostingClassifier), not "cbdt"
                "model_name":  "gbdt",
                "hyperparams": {
                    "n_estimators":   200,
                    "learning_rate":  0.1,
                    "max_depth":      4,
                    "subsample":      0.8,
                },
                "encode_categoricals": "ordinal",
                "scale_numeric":       False,
                "use_optimize":        False,
            },
            {
                "model_name":  "xgboost",
                "hyperparams": {
                    "n_estimators":    300,
                    "learning_rate":   0.05,
                    "max_depth":       6,
                    "subsample":       0.8,
                    "colsample_bytree": 0.8,
                },
                "encode_categoricals": "ordinal",
                "scale_numeric":       False,
                "use_optimize":        False,
            },
            {
                "model_name":  "mlp",
                "hyperparams": {
                    "hidden_layer_sizes": (256, 128, 64),
                    "learning_rate_init": 1e-3,
                    "max_iter":           500,
                    "alpha":              1e-4,
                },
                "encode_categoricals": "onehot",    # onehot works better for MLP
                "scale_numeric":       True,        # StandardScaler required for MLP
                "use_optimize":        False,
            },
        ],
        output_dir="runs/ensemble/voting/tabular_only",
        weights=None,
    )

    results = clf.run(
        df=df,
        text_col=None,              # not needed for tabular-only
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        test_size=0.2,
        random_state=42,
        n_jobs=-1,
        use_gpu=None,               # auto-detect; set True for catboost/xgboost/lightgbm on GPU
        top_k=3,
    )

    _print_results(results, "All 5 tabular models — equal-weight soft vote")
    _print_base_model_breakdown(results, [], clf.tabular_models)

    return clf, results


# ---------------------------------------------------------------------------
# E. Mode 2 — mixed text + tabular ensemble
# ---------------------------------------------------------------------------

def demo_mixed_ensemble():
    """Soft voting over a mixed text + tabular ensemble.

    Combines:
        Text:    bioclinicalbert (clinical BERT, 512 tokens)
        Tabular: lightgbm + catboost

    This is the canonical multimodal VA setup: the narrative carries
    qualitative context (symptom timing, severity phrasing) while the
    tabular indicators provide structured, machine-readable signals.

    The ensemble outperforms either modality alone when the two modalities
    are complementary rather than redundant.
    """
    from multimodalva.ensemble import SoftVotingClassifier

    df = make_toy_df(n=400)

    clf = SoftVotingClassifier(
        text_models=[
            {
                "model_name":    "bioclinicalbert",
                "hyperparams":   {
                    "epochs":         5,
                    "learning_rate":  2e-5,
                    "batch_size":     16,
                    "warmup_ratio":   0.1,
                    "weight_decay":   0.01,
                    "freeze_layers":  2,
                },
                "max_length":    512,
                "use_lora":      False,
                "use_optimize":  False,
            },
        ],
        tabular_models=[
            {
                "model_name":          "lightgbm",
                "hyperparams":         {"n_estimators": 300, "learning_rate": 0.05},
                "encode_categoricals": "ordinal",
                "scale_numeric":       False,
                "use_optimize":        False,
            },
            {
                "model_name":  "catboost",
                "hyperparams": {"iterations": 300, "learning_rate": 0.05, "depth": 6},
                "use_optimize": False,
            },
        ],
        output_dir="runs/ensemble/voting/mixed",
        # Trust the text model slightly more — text carries richer semantic information
        weights=[0.50, 0.25, 0.25],
    )

    results = clf.run(
        df=df,
        text_col=TEXT_COL,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        test_size=0.2,
        random_state=42,
        val_size=0.1,
        early_stopping_patience=3,
        n_jobs=-1,
        use_gpu=None,
        top_k=3,
        batch_size=32,
    )

    _print_results(results, "Mixed ensemble: bioclinicalbert + lightgbm + catboost")
    _print_base_model_breakdown(results, clf.text_models, clf.tabular_models)

    # Compare modality contributions
    print("\n--- Modality ablation (post-hoc, no retraining) ---")
    from multimodalva.ensemble.voting import vote_from_results

    id2label = clf.id2label
    bp       = clf.base_predictions   # [bioclinicalbert, lightgbm, catboost]

    text_only_r = vote_from_results(bp[0:1], id2label=id2label, top_k=3)  # single model
    tab_only_r  = vote_from_results(bp[1:3], id2label=id2label, top_k=3)  # lgbm + catboost

    acc_text  = (text_only_r.top1["true_label"] == text_only_r.top1["predicted_label"]).mean()
    acc_tab   = (tab_only_r.top1["true_label"]  == tab_only_r.top1["predicted_label"]).mean()
    acc_mixed = (results["predictions"].top1["true_label"] == results["predictions"].top1["predicted_label"]).mean()

    rows = [
        {"setup": "text only (bioclinicalbert)",    "accuracy": acc_text},
        {"setup": "tabular only (lgbm + catboost)", "accuracy": acc_tab},
        {"setup": "mixed ensemble (all 3)",         "accuracy": acc_mixed},
    ]
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.4f}".format))

    return clf, results


# ---------------------------------------------------------------------------
# F. Mode 2 — per-model Optuna HPO (use_optimize=True)
# ---------------------------------------------------------------------------

def demo_with_hpo():
    """Soft voting with per-model Optuna HPO.

    Set ``"use_optimize": True`` in any base model spec to run Optuna
    before training that model.  Each model runs its own independent
    HPO loop; the best hyperparameters are then used for final training.

    HPO artifacts per model:
        output_dir/base_models/text_0/hpo/    — Optuna study (SQLite) + trials CSV
        output_dir/base_models/tabular_0/hpo/ — same

    Workflow per model with use_optimize=True:
        optimize() → best_hyperparams  (internal 80/20 sub-split of train set)
        train(best_hyperparams, val_size=None)   (train on full train set)
        predict()

    Workflow per model with use_optimize=False:
        train(hyperparams, val_size=0.1)
        predict()

    Note: n_trials can be customised per model via the "n_trials" spec key.
    optimize_metric can differ per model — e.g. "csmf_accuracy" for text
    (recommended for imbalanced VA data) and "f1_macro" for tabular.
    """
    from multimodalva.ensemble import SoftVotingClassifier

    df = make_toy_df(n=400)

    clf = SoftVotingClassifier(
        text_models=[
            {
                "model_name":      "bioclinicalbert",
                "use_optimize":    True,              # Optuna HPO before training
                "n_trials":        10,                # increase to 20–50 for real experiments
                "optimize_metric": "csmf_accuracy",  # WHO standard — recommended for VA
                "search_space":    None,              # None = DEFAULT_SEARCH_SPACE
                # Fixed HP keys here are IGNORED when use_optimize=True;
                # HPO determines all hyperparameters.
                "max_length":      512,
                "use_lora":        False,
            },
        ],
        tabular_models=[
            {
                "model_name":          "lightgbm",
                "use_optimize":        True,
                "n_trials":            15,
                "optimize_metric":     "f1_macro",
                "search_space":        None,          # uses DEFAULT_SEARCH_SPACES["lightgbm"]
                "encode_categoricals": "ordinal",
                "scale_numeric":       False,
            },
            {
                # No HPO for the second tabular model — use fixed hyperparams
                "model_name":  "xgboost",
                "hyperparams": {"n_estimators": 200, "learning_rate": 0.1},
                "use_optimize": False,
                "encode_categoricals": "ordinal",
            },
        ],
        output_dir="runs/ensemble/voting/with_hpo",
        weights=None,
    )

    results = clf.run(
        df=df,
        text_col=TEXT_COL,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        test_size=0.2,
        random_state=42,
        val_size=0.1,
        early_stopping_patience=3,
        n_jobs=-1,
        use_gpu=None,
        top_k=3,
        batch_size=32,
    )

    _print_results(results, "Soft voting with per-model Optuna HPO")
    _print_base_model_breakdown(results, clf.text_models, clf.tabular_models)

    # HPO study artifacts
    print("\nHPO study artifacts:")
    for i in range(len(clf.text_models)):
        hpo_dir = clf.output_dir / "base_models" / f"text_{i}" / "hpo"
        print(f"  text_{i} HPO:    {hpo_dir}")
    for i in range(len(clf.tabular_models)):
        hpo_dir = clf.output_dir / "base_models" / f"tabular_{i}" / "hpo"
        print(f"  tabular_{i} HPO: {hpo_dir}")

    print("\nTo reload and inspect an HPO study:")
    print("  import optuna")
    print("  study = optuna.load_study(study_name='<model_name>',")
    print("                            storage='sqlite:///hpo_<model_name>.db')")
    print("  print(study.best_params)")

    return clf, results


# ---------------------------------------------------------------------------
# G. Post-run analysis
# ---------------------------------------------------------------------------

def demo_post_run_analysis(clf, results: dict):
    """Post-run analysis: inspect base predictions, metrics, and save/reload results.

    Args:
        clf:     SoftVotingClassifier instance (after run()).
        results: dict returned by clf.run().
    """
    from multimodalva.utils import score_predictions, csmf_accuracy, cccsmf_accuracy
    from multimodalva.results import performance_leaderboard, topk_accuracy, confusion_heatmap

    print("=== G. Post-run analysis ===")

    # --- G1. Inspect individual base model predictions ----------------------
    print("\n--- G1. Per-base-model accuracy ---")
    all_specs = clf.text_models + clf.tabular_models
    rows = []
    for i, (spec, bpred) in enumerate(zip(all_specs, clf.base_predictions)):
        top1     = bpred.top1
        acc      = (top1["true_label"] == top1["predicted_label"]).mean()
        csmfa    = csmf_accuracy(top1["true_label"], top1["predicted_label"])
        cccsmfa  = cccsmf_accuracy(top1["true_label"], top1["predicted_label"])
        rows.append({
            "model":          spec["model_name"],
            "accuracy":       round(acc, 4),
            "csmf_accuracy":  round(csmfa, 4),
            "cccsmf_accuracy": round(cccsmfa, 4),
        })

    final_top1 = results["predictions"].top1
    rows.append({
        "model":          "ensemble (voted)",
        "accuracy":       round((final_top1["true_label"] == final_top1["predicted_label"]).mean(), 4),
        "csmf_accuracy":  round(csmf_accuracy(final_top1["true_label"], final_top1["predicted_label"]), 4),
        "cccsmf_accuracy": round(cccsmf_accuracy(final_top1["true_label"], final_top1["predicted_label"]), 4),
    })
    print(pd.DataFrame(rows).to_string(index=False))

    # --- G2. All metrics via score_predictions() ----------------------------
    print("\n--- G2. Ensemble metrics (score_predictions) ---")
    for metric in ["accuracy", "f1_macro", "f1_weighted", "csmf_accuracy"]:
        score = score_predictions(results["predictions"].top1, metric=metric)
        print(f"  {metric:<20}: {score:.4f}")

    # --- G3. Performance leaderboard across all base models + ensemble ------
    print("\n--- G3. Performance leaderboard (via results.performance_leaderboard) ---")
    all_top1 = [r.top1.rename(columns={"predicted_label": spec["model_name"]})
                for spec, r in zip(all_specs, clf.base_predictions)]
    leaderboard_df = all_top1[0][["true_label", all_specs[0]["model_name"]]].copy()
    for spec, r in zip(all_specs[1:], clf.base_predictions[1:]):
        leaderboard_df[spec["model_name"]] = r.top1["predicted_label"].values
    leaderboard_df["ensemble"] = results["predictions"].top1["predicted_label"].values

    model_cols = [spec["model_name"] for spec in all_specs] + ["ensemble"]
    lb = performance_leaderboard(
        leaderboard_df,
        true_col="true_label",
        model_cols=model_cols,
        metrics=["accuracy", "f1_macro", "csmf_accuracy", "cccsmf_accuracy"],
        sort_by="f1_macro",
        percentage=True,
    )
    print(lb.to_string(float_format="{:.2f}".format))

    # --- G4. Top-k accuracy table -------------------------------------------
    print("\n--- G4. Top-k accuracy (ensemble) ---")
    topk_table = topk_accuracy(results["predictions"].topk, max_k=3)
    print(topk_table.to_string(index=False))

    # --- G5. Confusion heatmap (ensemble predictions) -----------------------
    print("\n--- G5. Saving confusion heatmap ---")
    save_path = Path(clf.output_dir) / "confusion_heatmap.png"
    fig, ax = confusion_heatmap(
        y_true=results["predictions"].top1["true_label"],
        y_pred=results["predictions"].top1["predicted_label"],
        normalize=True,
        title="Soft Voting Ensemble — Confusion Matrix",
        save_path=save_path,
    )
    print(f"  Heatmap saved to: {save_path}")

    # --- G6. Load saved metadata and label maps from disk -------------------
    print("\n--- G6. Reload metadata from disk ---")
    meta_path = Path(clf.output_dir) / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  n_train          : {meta['n_train']}")
        print(f"  n_test           : {meta['n_test']}")
        print(f"  n_classes        : {meta['n_classes']}")
        print(f"  n_text_models    : {meta['n_text_models']}")
        print(f"  n_tabular_models : {meta['n_tabular_models']}")
        print(f"  weights          : {meta['weights']}")
    else:
        print(f"  (metadata not found at {meta_path} — run demo_mixed_ensemble() first)")

    return lb


# ---------------------------------------------------------------------------
# Shared print helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict, label: str = ""):
    """Print a concise results summary."""
    pred = results["predictions"]
    top1 = pred.top1
    n    = len(top1)
    acc  = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"\n=== {label} ===")
    print(f"  Test samples     : {n}")
    print(f"  Ensemble accuracy: {acc:.4f}")
    print(f"  Output dir       : {results['output_dir']}")
    print(f"\n  Top-1 sample (first 3 rows):")
    print(top1.head(3).to_string(index=False))


def _print_prediction_result(result, n_rows: int = 3):
    """Show all three DataFrames from a PredictionResult."""
    print("\n--- top1: true / predicted / probability ---")
    print(result.top1.head(n_rows).to_string(index=False))

    print(f"\n--- topk: top-{result.topk.shape[1] // 2} predictions per sample ---")
    print(result.topk.head(n_rows).to_string(index=False))

    print("\n--- full: probability per class (rename prob_i → cause name) ---")
    full_named = result.full.rename(
        columns={f"prob_{i}": f"prob_{v}" for i, v in result.id2label.items()}
    )
    print(full_named.head(n_rows).to_string(index=False))


def _print_base_model_breakdown(results: dict, text_specs: list, tabular_specs: list):
    """Print per-model accuracy vs. ensemble accuracy."""
    print("\n--- Base model accuracy breakdown ---")
    all_specs = text_specs + tabular_specs
    bp        = results["base_predictions"]
    rows = []
    for spec, bpred in zip(all_specs, bp):
        acc = (bpred.top1["true_label"] == bpred.top1["predicted_label"]).mean()
        rows.append({"model": spec["model_name"], "accuracy": round(acc, 4)})
    final_top1 = results["predictions"].top1
    ens_acc = (final_top1["true_label"] == final_top1["predicted_label"]).mean()
    rows.append({"model": "ensemble (voted)", "accuracy": round(ens_acc, 4)})
    print(pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = {
    "A": ("Mode 1 — vote_from_results() (no model needed)",  demo_mode1_vote_from_results),
    "B": ("EnsembleClassifier dispatcher",                   demo_via_ensemble_classifier),
    "C": ("All 5 text models",                               demo_all_text_models),
    "D": ("All 5 tabular models",                            demo_all_tabular_models),
    "E": ("Mixed text + tabular ensemble",                   demo_mixed_ensemble),
    "F": ("Per-model Optuna HPO",                            demo_with_hpo),
}


if __name__ == "__main__":
    # ---------- mode 1 sanity check (no model download required) ----------
    print("Creating toy dataset ...")
    df = make_toy_df()
    print(f"  Shape : {df.shape}")
    print(f"  Causes: {df['cause'].value_counts().to_dict()}")
    print(f"  Cols  : {list(df.columns)}")

    args = sys.argv[1:]

    if not args:
        # Run section A (no-download) by default; uncomment others to activate
        print("\nRunning demo A (no model download) ...")
        demo_mode1_vote_from_results()

        print("\n\nTo run sections that require model downloads:")
        for key, (desc, _) in SECTIONS.items():
            if key != "A":
                print(f"  python tests/demo_ensemble_voting.py {key}   # {desc}")
    else:
        key = args[0].upper()
        if key not in SECTIONS:
            print(f"Unknown section '{key}'. Available: {list(SECTIONS.keys())}")
            sys.exit(1)

        desc, fn = SECTIONS[key]
        print(f"\nRunning section {key}: {desc}")

        if key == "G":
            # Section G requires a completed run; use demo_mixed_ensemble() to get one
            print("  (Section G requires a completed run — running demo_mixed_ensemble() first)")
            clf, results = demo_mixed_ensemble()
            demo_post_run_analysis(clf, results)
        else:
            result = fn()
            if key in ("B", "C", "D", "E", "F") and isinstance(result, tuple):
                clf, results = result
                print("\nRunning post-run analysis (section G) ...")
                demo_post_run_analysis(clf, results)
