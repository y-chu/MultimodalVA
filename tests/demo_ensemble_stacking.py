"""
Demo: ensemble strategy 4 — decision-level fusion via stacking (super learner).

Covers:
    A. Standalone generate_oof_predictions() with optional Optuna HPO
    B. Synthetic OOF + meta-learner selection in isolation (no model download)
    C. StackingClassifier via EnsembleClassifier dispatcher (tabular-only)
    D. StackingClassifier direct — tabular-only (lightgbm + random_forest)
    E. StackingClassifier direct — mixed text + tabular (bioclinicalbert + lightgbm)
    F. Multi-meta-learner candidate selection (logistic regression vs lightgbm vs random_forest)
    G. Stage-by-stage execution (cross-session pattern: Stage 1 → Stage 2 → Stage 3)
    H. Post-run analysis: OOF meta-features, meta-scores, PredictionResult helpers

Run a single section::

    python tests/demo_ensemble_stacking.py A    # standalone OOF (no HPO)
    python tests/demo_ensemble_stacking.py A hpo  # standalone OOF with Optuna HPO
    python tests/demo_ensemble_stacking.py B    # no model download needed
    python tests/demo_ensemble_stacking.py D    # tabular-only (no GPU needed)
    python tests/demo_ensemble_stacking.py E    # mixed (requires model download)

Run all sections::

    python tests/demo_ensemble_stacking.py


Overview — Stacking (Super Learner)
-------------------------------------
Stacking is a two-stage ensemble method.  In Stage 1, each base model is trained
k times via cross-validation; for each fold, the held-out probability predictions
(out-of-fold / OOF) are collected.  After all folds, the OOF probability matrices
are concatenated into a meta-feature matrix covering the full training set.  A
meta-learner is then trained on this matrix to map base-model probability stacks
to final class labels (Stage 2).  At inference time (Stage 3), final base models
(retrained on all training data) predict on the test set; their probability stacks
are passed through the fitted meta-learner.

Why OOF predictions avoid leakage:
    Each training sample's meta-features come from a model that never saw that
    sample during training — analogous to how ``cross_val_predict`` works.  This
    prevents the meta-learner from learning to trust whichever base model best
    memorised its training data.

Meta-learner candidate selection:
    When multiple meta-learner specs are provided, each is scored via stratified
    k-fold CV on the OOF meta-features (a second level of cross-validation).
    Only then is the best candidate fitted on ALL OOF data.  This keeps the test
    set completely blind — it is used exactly once in Stage 3.

Resumability:
    Each fold and final model writes a completion marker (oof_probs.npy /
    training_metadata.json).  Re-running the same stage method after an
    interruption automatically skips completed components.

Disk cleanup:
    ``cleanup_fold_files=True`` (default): after all folds are complete and
    ``oof_meta_X.npy`` is assembled, the per-fold ``fold_*/`` directories are
    deleted.  The resume check switches to ``oof_meta_X.npy`` existence, so
    cleanup is safe.

Output directory layout::

    output_dir/
    ├── data/          train_df.csv, test_df.csv, X_test.npy, y_test.npy
    ├── oof/           oof_meta_X.npy, oof_y.npy, meta_feature_names.json,
    │                  oof_metadata.json  (fold_*/ deleted after assembly)
    ├── hpo/           per-model Optuna artifacts (if use_optimize=True)
    ├── final/         text_i/ and tabular_i/ final base models
    ├── meta_learner/  meta_learner.joblib, meta_scores.json,
    │                  meta_learner_metadata.json
    ├── predictions/   top1.csv, topk.csv, full.csv
    ├── label2id.json
    ├── id2label.json
    └── training_metadata.json
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

    Cause-feature correlations are intentionally strong so that even toy
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

# Sorted alphabetically — matches what prepare_dataset() produces from this dataset
CAUSES   = ["Diarrhoea", "HIV/AIDS", "Malaria", "Maternal", "Pneumonia"]
LABEL2ID = {c: i for i, c in enumerate(CAUSES)}
ID2LABEL = {i: c for i, c in enumerate(CAUSES)}


# ---------------------------------------------------------------------------
# A. Standalone generate_oof_predictions() with optional HPO
# ---------------------------------------------------------------------------

def demo_generate_oof(use_hpo: bool = False):
    """Use the standalone generate_oof_predictions() to build the OOF meta-feature matrix.

    This is the lowest-level public API — it gives full control over the OOF
    loop without routing through StackingClassifier.  Useful when you want to:

    - Run the OOF loop as a standalone batch job and wire the result into your
      own meta-learner training code
    - Mix generate_oof_predictions() with a custom HPO loop
    - Inspect or modify oof_meta_X before passing it to any meta-learner

    Pipeline steps (tabular-only; no model download):

        split()                         -> train_df, test_df
        tabular.prepare_dataset()       -> X_train, y_train, label2id, id2label
        [tabular.hpo.optimize()]        -> tabular_best_hp  (optional)
        generate_oof_predictions()      -> oof_meta_X, oof_y
        StackingClassifier.__init__()   -> inject OOF state
        clf.train_meta_learner_stage()  -> meta_learner fitted on full OOF

    HPO note (use_hpo=True):
        ``tabular.hpo.optimize()`` is run ONCE on the full training set before
        the OOF loop.  The best HP is then reused for every fold and the final
        full-data model — the standard approach to prevent per-fold leakage and
        reduce compute.

    Args:
        use_hpo: If True, runs Optuna HPO before the OOF loop and passes
                 ``tabular_best_hp`` to generate_oof_predictions().
                 If False, fixed hyperparams from the spec are used.
    """
    from multimodalva.utils.split import split
    from multimodalva.tabular.dataset import prepare_dataset as tab_prepare
    from multimodalva.ensemble.stacking import generate_oof_predictions, StackingClassifier

    df = make_toy_df(n=300)

    tabular_specs = [
        {
            "model_name": "lightgbm",
            "hyperparams": {
                "n_estimators":  200,
                "learning_rate": 0.05,
                "max_depth":     6,
            },
        },
        {
            "model_name": "random_forest",
            "hyperparams": {"n_estimators": 200},
        },
    ]

    # --- Step 1: split -------------------------------------------------------
    train_df, test_df = split(df, label_col=LABEL_COL)
    print(f"Split: {len(train_df)} train / {len(test_df)} test")

    # --- Step 2: tabular preprocessing --------------------------------------
    (X_train, _, y_train, _,
     _, label2id, id2label, _) = tab_prepare(
        train_df, test_df,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        encode_categoricals="ordinal",
        scale_numeric=False,
    )
    print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")
    print(f"Classes: {label2id}")

    # --- Step 3 (optional): HPO to get best hyperparams ---------------------
    tabular_best_hp = None

    if use_hpo:
        from multimodalva.tabular.hpo import optimize as tab_optimize

        print("\nRunning Optuna HPO on LightGBM (5 trials) ...")
        lgbm_best_hp, lgbm_study = tab_optimize(
            X_train=X_train, y_train=y_train,
            label2id=label2id, id2label=id2label,
            model_name="lightgbm",
            output_dir="runs/stacking/oof_standalone/hpo/tabular_0",
            n_trials=5,             # increase to 20-50 for real experiments
            metric="f1_macro",
            random_state=42,
            n_jobs=-1,
        )
        print(f"  LightGBM best HP : {lgbm_best_hp}")
        print(f"  Best value       : {lgbm_study.best_value:.4f}")

        print("\nRunning Optuna HPO on RandomForest (5 trials) ...")
        rf_best_hp, _ = tab_optimize(
            X_train=X_train, y_train=y_train,
            label2id=label2id, id2label=id2label,
            model_name="random_forest",
            output_dir="runs/stacking/oof_standalone/hpo/tabular_1",
            n_trials=5,
            metric="f1_macro",
            random_state=42,
            n_jobs=-1,
        )
        print(f"  RandomForest best HP : {rf_best_hp}")

        # tabular_best_hp is a list aligned with tabular_specs order
        tabular_best_hp = [lgbm_best_hp, rf_best_hp]
    else:
        print("\nSkipping HPO — using fixed hyperparams from specs.")
        print("Set use_hpo=True or pass tabular_best_hp explicitly to use Optuna results.")

    # --- Step 4: generate OOF predictions -----------------------------------
    # text_model_specs=[] and train_text_dataset=None for tabular-only.
    # tabular_best_hp=None falls back to spec["hyperparams"] for each model.
    print("\nGenerating OOF predictions (5 folds x 2 tabular models) ...")
    oof_meta_X, oof_y = generate_oof_predictions(
        text_model_specs=[],
        tabular_model_specs=tabular_specs,
        train_text_dataset=None,     # tabular-only: no text dataset
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        n_folds=5,
        random_state=42,
        output_dir="runs/stacking/oof_standalone/oof",
        n_jobs=-1,
        use_gpu=None,
        resume=True,                 # skip completed folds on re-run
        save_fold_models=False,      # discard fold weights after OOF extraction
        cleanup_fold_files=True,     # delete fold_*/ dirs after assembly
        tabular_best_hp=tabular_best_hp,  # None -> use spec["hyperparams"]
    )

    # --- Step 5: inspect the result -----------------------------------------
    import json as _json
    oof_dir = Path("runs/stacking/oof_standalone/oof")
    with open(oof_dir / "meta_feature_names.json") as f:
        meta_names = _json.load(f)

    print(f"\nOOF meta-feature matrix: {oof_meta_X.shape}")
    print(f"  Rows    : one per training sample ({len(y_train)} total)")
    print(f"  Columns : {len(meta_names)}  (2 models x 5 classes)")
    print(f"\n  Column names:")
    for col in meta_names:
        print(f"    {col}")

    print(f"\n  First 3 rows (oof_meta_X):")
    print(pd.DataFrame(oof_meta_X, columns=meta_names).head(3).to_string())

    print(f"\n  Label distribution (oof_y): {np.bincount(oof_y)}")

    # --- Step 6: wire into StackingClassifier for meta-learner training -----
    # Inject OOF state to skip Stage 1 entirely — same pattern as Section B.
    print("\n--- Wiring oof_meta_X into StackingClassifier for meta-learner training ---")
    clf = StackingClassifier(
        text_models=[],
        tabular_models=tabular_specs,
        output_dir="runs/stacking/oof_standalone",
        meta_learners=[
            {"model_name": "logistic_regression", "hyperparams": {"C": 1.0}},
            {"model_name": "lightgbm", "hyperparams": {"n_estimators": 50}},
        ],
        meta_select_metric="f1_macro",
        n_folds=5,
    )
    clf.oof_meta_X = oof_meta_X
    clf.oof_y      = oof_y
    clf.label2id   = label2id
    clf.id2label   = id2label

    stage2 = clf.train_meta_learner_stage(meta_cv_folds=3)

    print(f"\nMeta-learner scores (CV f1_macro on OOF):")
    for name, score in stage2["meta_scores"].items():
        marker = " <- selected" if name == stage2["best_meta_name"] else ""
        print(f"  {name:<40}: {score}{marker}")
    print(f"\nFitted meta-learner : {type(clf.meta_learner).__name__}")

    return oof_meta_X, oof_y, clf


# ---------------------------------------------------------------------------
# B. Synthetic OOF + meta-learner selection in isolation (no model download)
# ---------------------------------------------------------------------------

def demo_synthetic_meta_learner():
    """Train and select a meta-learner on synthetic OOF meta-features.

    Runs without downloading or training any ML model.  Demonstrates the
    meta-learner selection logic directly:

    - Synthetic OOF probabilities for two tabular base models (lightgbm + random_forest)
    - Three meta-learner candidates scored via stratified 3-fold CV on OOF data
    - Best candidate fitted on full OOF and saved to disk

    This is useful for:
    - Understanding the meta-learner CV-selection mechanism
    - Testing Stage 2 in isolation before running full stacking
    - Rapid prototyping of custom meta-learner candidates

    Key insight — CV for ranking vs. fitting:
        The k-fold CV here is ONLY for ranking candidates (when len(specs) > 1).
        The final meta-learner is ALWAYS fitted on ALL OOF data via
        ``best_model.fit(oof_meta_X, oof_y)``.  When only one candidate is
        provided, CV is skipped entirely.
    """
    from multimodalva.ensemble import StackingClassifier

    rng       = np.random.default_rng(0)
    n_train   = 240
    n_classes = 5

    # Simulate OOF probability matrices from two tabular base models.
    # Dirichlet samples give valid probability rows (non-negative, sum to 1).
    oof_lgbm = rng.dirichlet(alpha=np.ones(n_classes) * 2, size=n_train)
    oof_rf   = rng.dirichlet(alpha=np.ones(n_classes) * 2, size=n_train)
    oof_meta_X = np.hstack([oof_lgbm, oof_rf])   # (240, 10)
    oof_y      = rng.integers(0, n_classes, size=n_train)

    print(f"Synthetic OOF meta-feature matrix: {oof_meta_X.shape}")
    print(f"  Columns: [lgbm_prob_0..4, rf_prob_0..4]  (2 models x 5 classes)")
    print(f"  Label distribution: {np.bincount(oof_y)}")

    # Instantiate classifier and inject synthetic OOF state directly.
    # Setting oof_meta_X bypasses the disk-load in _ensure_oof_loaded(),
    # so Stage 2 can run without any Stage 1 artifacts on disk.
    clf = StackingClassifier(
        text_models=[],
        tabular_models=[
            {"model_name": "lightgbm"},
            {"model_name": "random_forest"},
        ],
        output_dir="runs/stacking/synthetic_meta",
        meta_learners=[
            # Candidate 1: standard logistic regression
            {"model_name": "logistic_regression", "hyperparams": {"C": 1.0}},
            # Candidate 2: stronger regularisation (smaller C)
            {"model_name": "logistic_regression", "hyperparams": {"C": 0.1}},
            # Candidate 3: tree-based meta-learner (nonlinear combinations)
            {"model_name": "random_forest", "hyperparams": {"n_estimators": 50}},
        ],
        meta_select_metric="f1_macro",
    )

    # Inject synthetic state — bypasses Stage 1
    clf.oof_meta_X = oof_meta_X
    clf.oof_y      = oof_y
    clf.label2id   = LABEL2ID
    clf.id2label   = ID2LABEL

    stage2 = clf.train_meta_learner_stage(meta_cv_folds=3)

    print("\n=== Meta-learner candidate scores (f1_macro, 3-fold CV on OOF) ===")
    for name, score in stage2["meta_scores"].items():
        marker = " <- selected" if name == stage2["best_meta_name"] else ""
        print(f"  {name:<42}: {score}{marker}")

    print(f"\nFitted meta-learner : {type(clf.meta_learner).__name__}")
    print(f"Saved to            : {clf.output_dir / 'meta_learner' / 'meta_learner.joblib'}")
    print()
    print("Note: CV scores are used for candidate ranking only.")
    print("      The winner is re-fitted on ALL OOF data — not on a single fold.")

    return clf, stage2


# ---------------------------------------------------------------------------
# C. EnsembleClassifier dispatcher (tabular-only, no GPU needed)
# ---------------------------------------------------------------------------

def demo_via_ensemble_classifier():
    """Run stacking through the unified EnsembleClassifier wrapper.

    Constructor kwargs are forwarded to StackingClassifier.__init__();
    run() kwargs are forwarded to StackingClassifier.run() unchanged.
    Uses tabular-only base models — no GPU or model download required.
    """
    from multimodalva.ensemble import EnsembleClassifier

    df = make_toy_df()

    clf = EnsembleClassifier(
        method="stacking",
        output_dir="runs/ensemble/stacking_dispatcher",
        text_models=[],
        tabular_models=[
            {"model_name": "lightgbm",     "hyperparams": {"n_estimators": 200, "learning_rate": 0.05}},
            {"model_name": "random_forest", "hyperparams": {"n_estimators": 200}},
        ],
        meta_learners=[{"model_name": "logistic_regression"}],
        meta_select_metric="f1_macro",
        n_folds=5,
    )

    results = clf.run(
        df=df,
        label_col=LABEL_COL,
        feature_cols=FEATURE_COLS,   # tabular-only: text_col omitted
        encode_categoricals="ordinal",
        meta_cv_folds=3,
        top_k=3,
    )

    _print_results(results, "Stacking (EnsembleClassifier dispatcher, tabular-only)")

    # Access the inner StackingClassifier for strategy-specific attrs
    inner = clf.classifier
    print(f"\nInstance attributes after run():")
    print(f"  clf.classifier.oof_meta_X shape : {inner.oof_meta_X.shape}")
    print(f"  clf.classifier.meta_scores      : {inner.meta_scores}")
    print(f"  clf.classifier.meta_learner     : {type(inner.meta_learner).__name__}")
    print(f"  clf.classifier.label2id         : {inner.label2id}")

    return clf, results


# ---------------------------------------------------------------------------
# D. StackingClassifier direct — tabular-only
# ---------------------------------------------------------------------------

def demo_tabular_only():
    """StackingClassifier with two tabular base models; logistic meta-learner.

    Tabular-only stacking requires no GPU and no model download.  The OOF loop
    trains lightgbm and random_forest on each of 5 folds; their held-out
    probability matrices are stacked and fed to a logistic regression meta-learner.

    Base model spec keys for tabular models:
        model_name         — alias: lightgbm, catboost, random_forest, xgboost, mlp, ...
        hyperparams        — dict forwarded directly to the model constructor
        use_optimize       — run Optuna HPO once before OOF loop (default False)
        n_trials           — Optuna trials (default 20; only if use_optimize=True)
        optimize_metric    — metric for Optuna (default "f1_macro")
    """
    from multimodalva.ensemble import StackingClassifier

    df = make_toy_df(n=300)

    clf = StackingClassifier(
        text_models=[],
        tabular_models=[
            {
                "model_name": "lightgbm",
                "hyperparams": {
                    "n_estimators":      200,
                    "learning_rate":     0.05,
                    "max_depth":         6,
                    "num_leaves":        31,
                    "min_child_samples": 20,
                },
            },
            {
                "model_name": "random_forest",
                "hyperparams": {
                    "n_estimators": 200,
                    "max_depth":    None,   # unlimited depth
                    "max_features": "sqrt",
                },
            },
        ],
        output_dir="runs/stacking/tabular_only",
        meta_learners=[{"model_name": "logistic_regression", "hyperparams": {"C": 1.0}}],
        meta_select_metric="f1_macro",
        n_folds=5,
    )

    results = clf.run(
        df=df,
        label_col=LABEL_COL,
        feature_cols=FEATURE_COLS,
        encode_categoricals="ordinal",  # OrdinalEncoder — compatible with LightGBM + RF
        scale_numeric=False,            # trees do not require feature scaling
        n_jobs=-1,                      # all CPU cores for tabular models
        use_gpu=None,                   # auto-detect CUDA (None = auto)
        save_fold_models=False,         # discard fold weights after OOF extraction
        cleanup_fold_files=True,        # delete fold_*/ dirs after oof_meta_X.npy assembled
        meta_cv_folds=3,
        top_k=3,
    )

    _print_results(results, "Stacking — tabular-only (LightGBM + RandomForest)")
    _print_prediction_result(results["predictions"])

    print(f"\nOOF meta-feature matrix shape: {clf.oof_meta_X.shape}")
    print(f"  = n_train x (2 models x 5 classes)")
    print(f"  Row i: OOF probability predictions from LightGBM + RandomForest")
    print(f"         for training sample i (from the fold where i was held out)")

    _show_dir_layout(clf.output_dir)

    return clf, results


# ---------------------------------------------------------------------------
# E. StackingClassifier direct — mixed text + tabular
# ---------------------------------------------------------------------------

def demo_mixed():
    """StackingClassifier with one text model + one tabular model.

    Uses BioClinicalBERT (text) and LightGBM (tabular) as base models.
    Their OOF probability stacks are combined by a logistic regression meta-learner.

    Text base model spec keys:
        model_name                — shorthand (e.g. "bioclinicalbert") or full HF ID
        hyperparams               — dict: epochs, batch_size, learning_rate, freeze_layers, ...
        max_length                — tokeniser max length (default 512)
        use_lora                  — LoRA fine-tuning (default False)
        use_optimize              — Optuna HPO once before OOF loop (default False)
        early_stopping_patience   — patience for text fold training (default 3)
        gradient_checkpointing    — reduce VRAM at the cost of speed (default False)

    The text and tabular pipelines share the same label2id / id2label built once
    inside train_base_models(), so prob_0, prob_1, ... refer to identical causes
    across both modalities — a prerequisite for valid meta-feature concatenation.
    """
    from multimodalva.ensemble import StackingClassifier

    df = make_toy_df(n=300)

    clf = StackingClassifier(
        text_models=[
            {
                "model_name": "bioclinicalbert",   # emilyalsentzer/Bio_ClinicalBERT
                "hyperparams": {
                    "epochs":        3,
                    "batch_size":    16,
                    "learning_rate": 2e-5,
                    "freeze_layers": 2,
                },
                "max_length": 512,
                "use_lora":   False,
                "early_stopping_patience": 3,
            },
        ],
        tabular_models=[
            {
                "model_name": "lightgbm",
                "hyperparams": {"n_estimators": 200, "learning_rate": 0.05},
            },
        ],
        output_dir="runs/stacking/mixed",
        meta_learners=[{"model_name": "logistic_regression"}],
        meta_select_metric="f1_macro",
        n_folds=5,
    )

    results = clf.run(
        df=df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        feature_cols=FEATURE_COLS,
        # text training
        val_size=0.1,
        early_stopping_patience=3,
        # tabular
        encode_categoricals="ordinal",
        n_jobs=-1,
        use_gpu=None,
        # fold storage
        save_fold_models=False,
        cleanup_fold_files=True,
        # meta-learner
        meta_cv_folds=3,
        top_k=3,
    )

    _print_results(results, "Stacking — mixed (BioClinicalBERT + LightGBM)")
    _print_prediction_result(results["predictions"])

    print(f"\nOOF meta-feature matrix: {clf.oof_meta_X.shape}")
    print(f"  Columns: [bert_prob_0..4 (5 cols) | lgbm_prob_0..4 (5 cols)] = 10 total")

    return clf, results


# ---------------------------------------------------------------------------
# F. Multi-meta-learner candidate selection
# ---------------------------------------------------------------------------

def demo_multi_meta_learner():
    """Compare multiple meta-learner candidates; StackingClassifier selects the best.

    When more than one spec is passed to ``meta_learners``, each candidate is
    scored via stratified k-fold CV on the OOF meta-features.  The highest-scoring
    model is then fitted on ALL OOF data and saved.

    Correct approach (implemented here):
        Score candidates on OOF via CV → select best → fit best on ALL OOF data.

    Why not score on the test set?
        Using test predictions to rank meta-learner candidates would contaminate the
        test set (turning it into a de facto validation set), inflate the reported
        score via multiple comparisons, and invalidate the final evaluation.
        The test set is used exactly once — in Stage 3 after all selection is done.

    Why CV rather than in-sample scoring on OOF?
        In-sample scoring (fit + predict on the same OOF data) is biased toward
        complex models.  A random forest might appear to get 0.98 in-sample but
        0.71 on held-out data.  CV on OOF gives honest out-of-sample estimates
        for valid candidate ranking.
    """
    from multimodalva.ensemble import StackingClassifier

    df = make_toy_df(n=300)

    clf = StackingClassifier(
        text_models=[],
        tabular_models=[
            {"model_name": "lightgbm",     "hyperparams": {"n_estimators": 200}},
            {"model_name": "random_forest", "hyperparams": {"n_estimators": 200}},
            {"model_name": "catboost",      "hyperparams": {"iterations":   200}},
        ],
        output_dir="runs/stacking/multi_meta",
        meta_learners=[
            # Logistic regression variants (differ in regularisation strength)
            {"model_name": "logistic_regression", "hyperparams": {"C": 1.0}},
            {"model_name": "logistic_regression", "hyperparams": {"C": 0.1}},
            # Tree-based meta-learner (captures nonlinear interactions between models)
            {"model_name": "lightgbm",
             "hyperparams": {"n_estimators": 50, "learning_rate": 0.1}},
        ],
        meta_select_metric="f1_macro",
        n_folds=5,
    )

    results = clf.run(
        df=df,
        label_col=LABEL_COL,
        feature_cols=FEATURE_COLS,
        encode_categoricals="ordinal",
        n_jobs=-1,
        meta_cv_folds=3,
        top_k=3,
    )

    _print_results(results, "Stacking — 3 base models + multi-meta-learner selection")

    print("\n=== Meta-learner candidate scores (CV f1_macro on OOF) ===")
    meta_scores = results["meta_scores"]
    ranked = sorted(meta_scores.items(), key=lambda x: x[1] or 0, reverse=True)
    for name, score in ranked:
        marker = " <- selected" if score == max(v for v in meta_scores.values() if v) else ""
        print(f"  {name:<42}: {score}{marker}")
    print()
    print("The selected candidate is re-fitted on ALL OOF data.")

    return clf, results


# ---------------------------------------------------------------------------
# G. Stage-by-stage execution (cross-session pattern)
# ---------------------------------------------------------------------------

def demo_stage_by_stage():
    """Run each stage explicitly; stages can span separate Python sessions.

    Stacking training is often too long for a single session.  The three-stage
    design lets you checkpoint between stages:

        Session 1: clf.train_base_models(df, ...)    -> fold models + final models
        Session 2: clf.train_meta_learner_stage(...) -> meta-learner selected + saved
        Session 3: clf.predict_test(...)             -> test predictions assembled

    In sessions 2 and 3, reconstruct the same StackingClassifier (same specs +
    same output_dir) and call the stage method directly.  All artifacts from the
    earlier stage are loaded from disk automatically.

    This example runs all three stages in sequence but uses separate StackingClassifier
    objects to simulate the cross-session pattern.
    """
    from multimodalva.ensemble import StackingClassifier

    df = make_toy_df(n=300)

    OUTPUT_DIR   = "runs/stacking/stage_by_stage"
    text_specs   = []
    tabular_specs = [
        {"model_name": "lightgbm",     "hyperparams": {"n_estimators": 100}},
        {"model_name": "random_forest", "hyperparams": {"n_estimators": 100}},
    ]
    meta_specs = [{"model_name": "logistic_regression"}]

    # -------------------------------------------------------------------------
    # Session 1: Stage 1 — train base models + generate OOF
    # -------------------------------------------------------------------------
    print("=== Session 1: Stage 1 — train base models ===")
    clf1 = StackingClassifier(
        text_models=text_specs,
        tabular_models=tabular_specs,
        output_dir=OUTPUT_DIR,
        meta_learners=meta_specs,
        n_folds=5,
        resume=True,   # skip completed folds on re-run
    )
    stage1 = clf1.train_base_models(
        df=df,
        label_col=LABEL_COL,
        feature_cols=FEATURE_COLS,
        encode_categoricals="ordinal",
        save_fold_models=False,    # discard fold weights (saves disk)
        cleanup_fold_files=True,   # delete fold_*/ dirs after OOF assembly
    )
    print(f"  OOF meta-feature matrix : {stage1['oof_meta_X'].shape}")
    print(f"  Artifacts saved to      : {stage1['output_dir']}")
    print(f"  Artifacts written:")
    print(f"    data/train_df.csv, data/test_df.csv, data/X_test.npy, data/y_test.npy")
    print(f"    oof/oof_meta_X.npy, oof/oof_y.npy, oof/oof_metadata.json")
    print(f"    final/tabular_0/, final/tabular_1/")

    # -------------------------------------------------------------------------
    # Session 2: Stage 2 — meta-learner (fresh clf, same output_dir)
    # -------------------------------------------------------------------------
    print("\n=== Session 2: Stage 2 — train meta-learner ===")
    clf2 = StackingClassifier(
        text_models=text_specs,
        tabular_models=tabular_specs,
        output_dir=OUTPUT_DIR,   # same path -> OOF loaded from disk automatically
        meta_learners=meta_specs,
        n_folds=5,
    )
    # _ensure_oof_loaded() is called inside train_meta_learner_stage();
    # it reads oof_meta_X.npy + oof_y.npy + label maps from disk.
    stage2 = clf2.train_meta_learner_stage(
        meta_cv_folds=3,
        metric="f1_macro",
    )
    print(f"  Best meta-learner : {stage2['best_meta_name']}")
    print(f"  Meta scores       : {stage2['meta_scores']}")
    print(f"  Saved to          : {OUTPUT_DIR}/meta_learner/meta_learner.joblib")

    # -------------------------------------------------------------------------
    # Session 3: Stage 3 — predict test (fresh clf, same output_dir)
    # -------------------------------------------------------------------------
    print("\n=== Session 3: Stage 3 — predict test set ===")
    clf3 = StackingClassifier(
        text_models=text_specs,
        tabular_models=tabular_specs,
        output_dir=OUTPUT_DIR,   # same path -> meta-learner + label maps loaded from disk
        meta_learners=meta_specs,
        n_folds=5,
    )
    # predict_test() loads: meta_learner.joblib (Stage 2 artifact),
    #                       data/test_df.csv, data/X_test.npy, data/y_test.npy (Stage 1 artifacts),
    #                       final/tabular_{i}/ (Stage 1 artifacts).
    predictions = clf3.predict_test(top_k=3)

    top1 = predictions.top1
    acc  = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"\n  Test accuracy : {acc:.4f}")
    print(f"\n  Top-5 rows:")
    print(top1.head(5).to_string(index=False))

    return predictions


# ---------------------------------------------------------------------------
# H. Post-run analysis
# ---------------------------------------------------------------------------

def demo_analysis(output_dir: str = "runs/stacking/tabular_only"):
    """Inspect OOF meta-features, meta-scores, and score predictions.

    Run this section after demo_tabular_only() (section C) or
    demo_stage_by_stage() (section F) has completed.
    Reads artifacts directly from disk — no model required.

    Args:
        output_dir: Path to the StackingClassifier output directory.
    """
    from multimodalva.utils import score_predictions

    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f"Output dir not found: {output_dir}")
        print("Run section C or F first to generate artifacts.")
        return

    # --- OOF meta-features ---
    oof_dir = output_dir / "oof"
    meta_X  = np.load(oof_dir / "oof_meta_X.npy")
    oof_y   = np.load(oof_dir / "oof_y.npy")

    with open(oof_dir / "meta_feature_names.json") as f:
        meta_names = json.load(f)
    with open(oof_dir / "oof_metadata.json") as f:
        oof_meta = json.load(f)

    print("=== OOF meta-feature matrix ===")
    print(f"  Shape          : {meta_X.shape}  (n_train x n_models*n_classes)")
    print(f"  Folds          : {oof_meta['n_folds']}")
    print(f"  Text models    : {oof_meta['n_text_models']}")
    print(f"  Tabular models : {oof_meta['n_tabular_models']}")
    print(f"  N classes      : {oof_meta['n_classes']}")
    print(f"  First 4 column names:")
    for col in meta_names[:4]:
        print(f"    {col}")
    if len(meta_names) > 4:
        print(f"  ... ({len(meta_names)} total)")

    # --- Label maps ---
    with open(output_dir / "id2label.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    print(f"\nLabel map: {id2label}")

    # --- Meta-learner selection ---
    meta_dir = output_dir / "meta_learner"
    with open(meta_dir / "meta_scores.json") as f:
        meta_scores = json.load(f)
    with open(meta_dir / "meta_learner_metadata.json") as f:
        meta_meta = json.load(f)

    print(f"\n=== Meta-learner selection ===")
    print(f"  Selection metric : {meta_meta['meta_select_metric']}")
    print(f"  CV folds         : {meta_meta['meta_cv_folds']}")
    for name, score in meta_scores.items():
        marker = " <- selected" if name == meta_meta["best_meta_name"] else ""
        print(f"  {name:<30}: {score}{marker}")

    # --- Test prediction scores ---
    pred_dir = output_dir / "predictions"
    if pred_dir.exists():
        top1 = pd.read_csv(pred_dir / "top1.csv")
        topk = pd.read_csv(pred_dir / "topk.csv")
        print(f"\n=== Test prediction scores ===")
        print(f"  Test samples: {len(top1)}")
        for metric in ["accuracy", "f1_macro", "f1_weighted", "csmf_accuracy"]:
            score = score_predictions(top1, metric=metric)
            print(f"  {metric:<20}: {score:.4f}")

        print(f"\n  Top-5 rows (top1):")
        print(top1.head(5).to_string(index=False))

        print(f"\n  Top-3 rows (topk):")
        print(topk.head(3).to_string(index=False))

    # --- OOF meta-feature as DataFrame (for inspection) ---
    oof_df = pd.DataFrame(meta_X, columns=meta_names)
    oof_df["true_label"] = [id2label[int(y)] for y in oof_y]
    print(f"\n=== OOF meta-feature DataFrame (first 3 rows) ===")
    print(oof_df.head(3).to_string())

    return oof_df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict, label: str = ""):
    """Print a concise results summary."""
    pred = results["predictions"]
    top1 = pred.top1
    n    = len(top1)
    acc  = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"\n=== {label} ===")
    print(f"  Test samples   : {n}")
    print(f"  Accuracy       : {acc:.3f}")
    print(f"  Meta-learner   : {type(results['meta_learner']).__name__}")
    print(f"  Meta scores    : {results['meta_scores']}")
    print(f"  Output dir     : {results['output_dir']}")
    print(f"\n  Top-1 sample (first 3 rows):")
    print(top1.head(3).to_string(index=False))


def _print_prediction_result(result):
    """Show all three prediction DataFrames from a PredictionResult."""
    print("\n--- top1: true label, predicted label, probability ---")
    print(result.top1.head(5).to_string(index=False))

    print("\n--- topk: top-3 predictions per sample ---")
    print(result.topk.head(3).to_string(index=False))

    print("\n--- full: probability for every class (integer IDs as columns) ---")
    # Rename prob_0, prob_1 -> cause name for readability
    full_named = result.full.rename(
        columns={f"prob_{i}": f"prob_{v}" for i, v in result.id2label.items()}
    )
    print(full_named.head(3).to_string(index=False))


def _show_dir_layout(output_dir: Path):
    """Print which saved artifact paths exist after a run."""
    paths = [
        "data/train_df.csv",
        "data/test_df.csv",
        "data/X_test.npy",
        "data/y_test.npy",
        "oof/oof_meta_X.npy",
        "oof/oof_y.npy",
        "oof/meta_feature_names.json",
        "oof/oof_metadata.json",
        "final/tabular_0/model.joblib",
        "final/tabular_1/model.joblib",
        "meta_learner/meta_learner.joblib",
        "meta_learner/meta_scores.json",
        "meta_learner/meta_learner_metadata.json",
        "predictions/top1.csv",
        "predictions/topk.csv",
        "predictions/full.csv",
        "label2id.json",
        "id2label.json",
        "training_metadata.json",
    ]
    print(f"\nSaved artifacts in {output_dir}/:")
    for p in paths:
        exists = "[x]" if (output_dir / p).exists() else "[ ]"
        print(f"  {exists} {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    section = sys.argv[1].upper() if len(sys.argv) > 1 else None

    # Quick sanity check (always runs, no model required)
    df = make_toy_df()
    print(f"Toy dataset: {df.shape}")
    print(f"Causes: {df['cause'].value_counts().to_dict()}")
    print(f"Columns: {list(df.columns)}")

    if section == "A" or section is None:
        print("\n" + "=" * 60)
        print("Section A: standalone generate_oof_predictions() with optional HPO")
        print("=" * 60)
        # Pass "hpo" as second argv to enable Optuna HPO before OOF loop:
        #   python tests/demo_ensemble_stacking.py A hpo
        use_hpo = len(sys.argv) > 2 and sys.argv[2].lower() == "hpo"
        demo_generate_oof(use_hpo=use_hpo)

    if section == "B" or section is None:
        print("\n" + "=" * 60)
        print("Section B: synthetic OOF + meta-learner (no model download)")
        print("=" * 60)
        demo_synthetic_meta_learner()

    if section == "C" or section is None:
        print("\n" + "=" * 60)
        print("Section C: EnsembleClassifier dispatcher (tabular-only)")
        print("=" * 60)
        demo_via_ensemble_classifier()

    if section == "D" or section is None:
        print("\n" + "=" * 60)
        print("Section D: StackingClassifier direct — tabular-only")
        print("=" * 60)
        demo_tabular_only()

    if section == "E" or section is None:
        print("\n" + "=" * 60)
        print("Section E: mixed text + tabular (requires model download)")
        print("=" * 60)
        demo_mixed()

    if section == "F" or section is None:
        print("\n" + "=" * 60)
        print("Section F: multi-meta-learner candidate selection")
        print("=" * 60)
        demo_multi_meta_learner()

    if section == "G" or section is None:
        print("\n" + "=" * 60)
        print("Section G: stage-by-stage execution (cross-session pattern)")
        print("=" * 60)
        demo_stage_by_stage()

    if section == "H" or section is None:
        print("\n" + "=" * 60)
        print("Section H: post-run analysis (run after D or G)")
        print("=" * 60)
        # Try D's output first, fall back to G's
        for candidate in ["runs/stacking/tabular_only", "runs/stacking/stage_by_stage"]:
            if Path(candidate).exists():
                demo_analysis(candidate)
                break
        else:
            print("No output dir found. Run section D or G first.")
