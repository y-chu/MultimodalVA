"""
Demo: result visualization — training diagnostics and prediction summary.

Covers:
    A. plot_loss_curves          — training/evaluation loss from a HuggingFace Trainer log
    B. hpo_leaderboard           — ranked HPO trial table from an Optuna study or DataFrame
    C. performance_leaderboard   — multi-model metric comparison table
    D. topk_from_full            — re-derive top-k labels/probs from the full probability matrix
    E. topk_accuracy             — cumulative accuracy at k = 1 … K
    F. confusion_heatmap         — per-model heatmap with count / row-normalised / abbreviated variants
    G. cause_accuracy_heatmap    — cause × model accuracy grid; top-1 or top-k; column grouping
    H. cause_accuracy_diff_heatmap — difference vs. a baseline model; diverging colormap centred at 0
    I. plot_topk_accuracy          — single-model bar, multi-model grouped, multi-model facet (2-col grid)

All sections use synthetic data so no model training is required.

Usage
-----
Run the full file to execute all sections sequentially:

    python tests/demo_results.py

Or call individual sections:

    python tests/demo_results.py G
    python tests/demo_results.py H

Imports
-------
    from multimodalva.results import (
        plot_loss_curves,
        hpo_leaderboard,
        performance_leaderboard,
        topk_from_full,
        topk_accuracy,
        confusion_heatmap,
        cause_accuracy_heatmap,
        cause_accuracy_diff_heatmap,
    )
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

CAUSES = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]
LABEL2ID = {c: i for i, c in enumerate(CAUSES)}
ID2LABEL = {i: c for i, c in enumerate(CAUSES)}


# ---------------------------------------------------------------------------
# Shared synthetic data builders
# ---------------------------------------------------------------------------

def _make_log_history(
    n_steps: int = 500,
    eval_every: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Simulate a HuggingFace Trainer log_history."""
    rng = np.random.default_rng(seed)
    history = []
    train_loss = 2.0
    for step in range(1, n_steps + 1):
        # Training loss: exponential decay + small noise
        train_loss = train_loss * 0.99 + rng.normal(0, 0.02)
        history.append({"step": step, "loss": max(0.05, train_loss), "epoch": step / 100})
        # Eval loss: slightly higher, evaluated less frequently
        if step % eval_every == 0:
            eval_loss = max(0.08, train_loss * 1.05 + rng.normal(0, 0.03))
            history.append({"step": step, "eval_loss": eval_loss, "epoch": step / 100})
    return history


def _make_hpo_df(n_trials: int = 20, seed: int = 42) -> pd.DataFrame:
    """Simulate a ``study.trials_dataframe()`` output."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_trials):
        lr        = float(10 ** rng.uniform(-5, -3))
        epochs    = int(rng.integers(3, 10))
        batch     = int(rng.choice([8, 16, 32]))
        freeze    = int(rng.choice([0, 2, 4, 6]))
        f1_macro  = float(rng.uniform(0.40, 0.78))
        rows.append({
            "number":                  i,
            "state":                   "COMPLETE",
            "value":                   round(f1_macro, 4),
            "duration":                float(rng.uniform(60, 600)),
            "user_attrs_accuracy":     round(f1_macro + rng.uniform(-0.05, 0.05), 4),
            "user_attrs_f1_macro":     round(f1_macro, 4),
            "user_attrs_f1_weighted":  round(f1_macro + rng.uniform(-0.02, 0.02), 4),
            "user_attrs_csmf_accuracy":round(f1_macro + rng.uniform(-0.03, 0.03), 4),
            "params_learning_rate":    round(lr, 8),
            "params_epochs":           epochs,
            "params_batch_size":       batch,
            "params_freeze_layers":    freeze,
        })
    # Sprinkle a few failed trials
    rows[3]["state"]  = "FAIL"
    rows[11]["state"] = "FAIL"
    return pd.DataFrame(rows)


def _make_predictions_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Simulate a wide prediction DataFrame for multiple models.

    Columns: true_label, pred_bert, pred_lightgbm, pred_ensemble
    Each model has a different accuracy level so the leaderboard is interesting.
    """
    rng    = np.random.default_rng(seed)
    y_true = rng.choice(CAUSES, size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08])

    def _noisy_pred(y, accuracy):
        """Flip each label to a random wrong class with probability 1-accuracy."""
        out = []
        for label in y:
            if rng.random() < accuracy:
                out.append(label)
            else:
                others = [c for c in CAUSES if c != label]
                out.append(rng.choice(others))
        return np.array(out)

    return pd.DataFrame({
        "true_label":    y_true,
        "pred_bert":     _noisy_pred(y_true, accuracy=0.72),
        "pred_lightgbm": _noisy_pred(y_true, accuracy=0.63),
        "pred_ensemble": _noisy_pred(y_true, accuracy=0.78),
    })


def _make_full_proba_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Simulate ``result.full`` — true_label + one prob_<id> column per class.

    Probabilities are Dirichlet-sampled and then biased toward the true class
    to simulate a reasonably calibrated classifier.
    """
    rng    = np.random.default_rng(seed)
    n_cls  = len(CAUSES)
    y_true = rng.choice(CAUSES, size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08])

    rows = []
    for label in y_true:
        true_id = LABEL2ID[label]
        alpha   = np.ones(n_cls) * 0.3
        alpha[true_id] = 3.0            # higher weight on the true class
        probs = rng.dirichlet(alpha)
        rows.append(probs)

    prob_matrix = np.stack(rows)
    df = pd.DataFrame(
        prob_matrix,
        columns=[f"prob_{i}" for i in range(n_cls)],
    )
    df.insert(0, "true_label", y_true)
    return df


def _make_wide_pred_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Wide prediction DataFrame: true_label + top-1 predictions for 4 models.

    Models and approximate top-1 accuracy:
        insilicova  — baseline-level (55 %)
        lightgbm    — tabular model  (63 %)
        bert        — text model     (72 %)
        ensemble    — best combined  (78 %)

    All four models are scored against the same y_true so per-cause accuracies
    are comparable across columns.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.choice(CAUSES, size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08])

    def _noisy_pred(y, accuracy):
        out = []
        for label in y:
            if rng.random() < accuracy:
                out.append(label)
            else:
                others = [c for c in CAUSES if c != label]
                out.append(rng.choice(others))
        return np.array(out)

    return pd.DataFrame({
        "true_label": y_true,
        "insilicova": _noisy_pred(y_true, accuracy=0.55),
        "lightgbm":   _noisy_pred(y_true, accuracy=0.63),
        "bert":       _noisy_pred(y_true, accuracy=0.72),
        "ensemble":   _noisy_pred(y_true, accuracy=0.78),
    })


def _make_model_full_dfs(n: int = 300) -> dict[str, pd.DataFrame]:
    """Per-model full probability DataFrames, all sharing the same y_true.

    Returns a dict: model_name → DataFrame with columns
    [true_label, prob_0, prob_1, ..., prob_{C-1}].

    Higher ``concentration`` → predicted probabilities are more peaked on the
    true class → higher effective accuracy.  Models in order of quality:
        insilicova (concentration 1.5) < lightgbm (2.5) < bert (3.5) < ensemble (5.0)
    """
    rng_true = np.random.default_rng(99)
    y_true = rng_true.choice(CAUSES, size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08])

    model_configs = {
        "insilicova": (42, 1.5),
        "lightgbm":   (43, 2.5),
        "bert":       (44, 3.5),
        "ensemble":   (45, 5.0),
    }

    result: dict[str, pd.DataFrame] = {}
    for model_name, (seed, concentration) in model_configs.items():
        rng = np.random.default_rng(seed)
        rows = []
        for label in y_true:
            true_id = LABEL2ID[label]
            alpha = np.ones(len(CAUSES)) * 0.3
            alpha[true_id] = concentration
            probs = rng.dirichlet(alpha)
            rows.append(probs)
        prob_matrix = np.stack(rows)
        df = pd.DataFrame(
            prob_matrix, columns=[f"prob_{i}" for i in range(len(CAUSES))],
        )
        df.insert(0, "true_label", y_true)
        result[model_name] = df
    return result


# ---------------------------------------------------------------------------
# A. Loss curves
# ---------------------------------------------------------------------------

def demo_loss_curves() -> None:
    """Section A: plot training and evaluation loss curves.

    Demonstrates:
    - Default view (all steps)
    - Skipping early noisy steps via skip_steps
    """
    from multimodalva.results import plot_loss_curves

    print("\n" + "=" * 60)
    print("A. Loss Curves")
    print("=" * 60)

    log_history = _make_log_history(n_steps=500, eval_every=50)
    n_entries   = len(log_history)
    train_only  = sum(1 for e in log_history if "loss" in e and "eval_loss" not in e)
    eval_only   = sum(1 for e in log_history if "eval_loss" in e)
    print(f"  Log history: {n_entries} entries  ({train_only} train steps, {eval_only} eval checkpoints)")

    # A1: full loss curves
    print("\n  A1: Full loss curves (all 500 steps)")
    plot_loss_curves(log_history)

    # A2: skip early noisy steps
    print("\n  A2: Loss curves after step 100 (skip noisy warm-up)")
    plot_loss_curves(log_history, skip_steps=100)


# ---------------------------------------------------------------------------
# B. HPO leaderboard
# ---------------------------------------------------------------------------

def demo_hpo_leaderboard() -> None:
    """Section B: format HPO trials as a ranked leaderboard.

    Demonstrates:
    - Default sort (by Optuna objective value = f1_macro)
    - Sort by a different metric (csmf_accuracy)
    - Restrict to top-N trials
    - Exclude hyperparameter columns for a compact view
    """
    from multimodalva.results import hpo_leaderboard

    print("\n" + "=" * 60)
    print("B. HPO Leaderboard")
    print("=" * 60)

    trials_df = _make_hpo_df(n_trials=20)
    print(f"  Trials DataFrame: {len(trials_df)} rows  "
          f"({(trials_df['state'] == 'COMPLETE').sum()} completed, "
          f"{(trials_df['state'] == 'FAIL').sum()} failed)")

    # B1: default sort (by 'value' = f1_macro objective)
    board = hpo_leaderboard(trials_df)
    print("\n  B1: All completed trials, sorted by objective (f1_macro):")
    print(board.to_string(index=False))

    # B2: sort by csmf_accuracy, show top 5
    board_csmf = hpo_leaderboard(trials_df, sort_by="csmf_accuracy", top_n=5)
    print("\n  B2: Top-5 trials by CSMF accuracy:")
    print(board_csmf.to_string(index=False))

    # B3: compact view — no hyperparameter columns
    board_compact = hpo_leaderboard(trials_df, top_n=10, param_cols=False)
    print("\n  B3: Top-10 compact (metrics only, no params):")
    print(board_compact.to_string(index=False))

    # B4: also accepts an optuna.Study — show how it would be called
    print("\n  B4: Accepts optuna.Study directly (if study is available):")
    print("      board = hpo_leaderboard(study)  # study from text/tabular hpo.py")
    print("      board = hpo_leaderboard(study, sort_by='csmf_accuracy', top_n=5)")


# ---------------------------------------------------------------------------
# C. Performance leaderboard
# ---------------------------------------------------------------------------

def demo_performance_leaderboard() -> None:
    """Section C: multi-model performance comparison table.

    Demonstrates:
    - Default metrics (accuracy, balanced_accuracy, f1_macro, f1_weighted, csmf_accuracy)
    - Custom metric subset
    - Top-k accuracy columns (top2_accuracy, top3_accuracy) via topk_dfs
    - Building the wide DataFrame from individual PredictionResults
    """
    from multimodalva.results import performance_leaderboard, topk_from_full

    print("\n" + "=" * 60)
    print("C. Performance Leaderboard")
    print("=" * 60)

    pred_df = _make_predictions_df(n=300)
    print(f"  Predictions DataFrame: {len(pred_df)} samples, "
          f"models: {[c for c in pred_df.columns if c != 'true_label']}")

    # C1: default — all 10 metrics, sort by f1_macro
    board = performance_leaderboard(pred_df, true_col="true_label")
    print("\n  C1: Default leaderboard (all 10 metrics, sorted by f1_macro):")
    print(board.to_string())

    # C2: custom metric subset
    board_custom = performance_leaderboard(
        pred_df,
        true_col="true_label",
        metrics=["accuracy", "f1_macro", "csmf_accuracy"],
        sort_by="csmf_accuracy",
    )
    print("\n  C2: Custom metrics (accuracy, f1_macro, csmf_accuracy), sorted by CSMF:")
    print(board_custom.to_string())

    # C3: raw proportions (percentage=False)
    board_raw = performance_leaderboard(
        pred_df,
        true_col="true_label",
        metrics=["accuracy", "f1_macro"],
        percentage=False,
    )
    print("\n  C3: Raw proportions (percentage=False):")
    print(board_raw.to_string())

    # C4: top-k accuracy columns — top2 and top3 appended after default metrics
    # Build one full probability DataFrame per model (simulates result.full)
    full_bert     = _make_full_proba_df(n=300, seed=42)
    full_lightgbm = _make_full_proba_df(n=300, seed=7)
    full_ensemble = _make_full_proba_df(n=300, seed=99)

    # Derive top-3 topk DataFrames from full probability matrices
    topk_dfs = {
        "pred_bert":     topk_from_full(full_bert,     ID2LABEL, k=3),
        "pred_lightgbm": topk_from_full(full_lightgbm, ID2LABEL, k=3),
        "pred_ensemble": topk_from_full(full_ensemble, ID2LABEL, k=3),
    }

    board_topk = performance_leaderboard(
        pred_df,
        true_col="true_label",
        metrics=["accuracy", "f1_macro", "csmf_accuracy"],
        topk_dfs=topk_dfs,
        top_k=3,   # appends top2_accuracy and top3_accuracy columns
    )
    print("\n  C4: With top-k accuracy (top2 and top3) appended:")
    print(board_topk.to_string())

    # C5: sort by top3_accuracy
    board_by_top3 = performance_leaderboard(
        pred_df,
        true_col="true_label",
        metrics=["accuracy", "f1_macro"],
        topk_dfs=topk_dfs,
        top_k=3,
        sort_by="top3_accuracy",
    )
    print("\n  C5: Sorted by top3_accuracy:")
    print(board_by_top3.to_string())

    # C6: building wide df from PredictionResult objects
    print("\n  C6: Building wide df from PredictionResults (pattern):")
    print("""
    # After running text and tabular classifiers:
    pred_wide = pd.DataFrame({
        "true_label":    text_result.top1["true_label"],
        "pred_bert":     text_result.top1["predicted_label"],
        "pred_lightgbm": tabular_result.top1["predicted_label"],
        "pred_ensemble": ensemble_result.top1["predicted_label"],
    })
    topk_dfs = {
        "pred_bert":     text_result.topk,
        "pred_lightgbm": tabular_result.topk,
        "pred_ensemble": ensemble_result.topk,
    }
    board = performance_leaderboard(
        pred_wide, true_col="true_label",
        topk_dfs=topk_dfs, top_k=3,
    )
    """)


# ---------------------------------------------------------------------------
# D. Top-K extraction from full probability distribution
# ---------------------------------------------------------------------------

def demo_topk_from_full() -> None:
    """Section D: re-derive top-k labels and probabilities from result.full.

    Demonstrates:
    - Default top-3 extraction
    - Custom k (top-5, top-1)
    - Works on any probability matrix with prob_<id> columns
    """
    from multimodalva.results import topk_from_full

    print("\n" + "=" * 60)
    print("D. Top-K Extraction from Full Probability Matrix")
    print("=" * 60)

    full_df = _make_full_proba_df(n=300)
    n_cls   = len(CAUSES)
    print(f"  Full DataFrame: {len(full_df)} rows, {n_cls} probability columns (prob_0 … prob_{n_cls-1})")
    print(f"  id2label: {ID2LABEL}")

    # D1: default top-3
    topk3 = topk_from_full(full_df, ID2LABEL, k=3)
    print(f"\n  D1: Top-3 extraction — shape {topk3.shape}")
    print(topk3.head(6).to_string(index=False))

    # D2: top-5 (all classes)
    topk5 = topk_from_full(full_df, ID2LABEL, k=5)
    print(f"\n  D2: Top-5 extraction — shape {topk5.shape}")
    print(topk5.head(4).to_string(index=False))

    # D3: top-1 only (equivalent to top1 DataFrame but from full probs)
    topk1 = topk_from_full(full_df, ID2LABEL, k=1)
    acc = (topk1["true_label"] == topk1["top1_label"]).mean()
    print(f"\n  D3: Top-1 extraction — accuracy = {acc:.3f}")

    # D4: pattern — re-derive top-k on a result that was originally predicted with top_k=3
    print("\n  D4: Re-derive top-k with a different k (pattern):")
    print("""
    # result was obtained with predict(..., top_k=3) — only top-3 stored in result.topk
    # To get top-5 retrospectively, use the full probability matrix:
    topk5 = topk_from_full(result.full, result.id2label, k=5)
    """)


# ---------------------------------------------------------------------------
# E. Top-K accuracy
# ---------------------------------------------------------------------------

def demo_topk_accuracy() -> None:
    """Section E: cumulative accuracy at k = 1 … K.

    Demonstrates:
    - From result.topk (direct)
    - From topk_from_full output
    - Restricting to max_k
    """
    from multimodalva.results import topk_from_full, topk_accuracy

    print("\n" + "=" * 60)
    print("E. Top-K Accuracy")
    print("=" * 60)

    full_df = _make_full_proba_df(n=300)

    # Build topk DataFrame from full probs (mirrors real usage after predict())
    topk5 = topk_from_full(full_df, ID2LABEL, k=5)

    # E1: full k=1..5 table
    acc_table = topk_accuracy(topk5)
    print("\n  E1: Cumulative accuracy at k = 1 … 5:")
    print(acc_table.to_string(index=False))

    # E2: restrict to max_k=3
    acc_table3 = topk_accuracy(topk5, max_k=3)
    print("\n  E2: Restricted to max_k=3:")
    print(acc_table3.to_string(index=False))

    # E3: pass result.topk directly (pattern)
    print("\n  E3: From PredictionResult.topk (pattern):")
    print("""
    result = clf.run(df, ...)["predictions"]   # top_k=3 by default
    acc_table = topk_accuracy(result.topk)
    # k=1: standard accuracy; k=3: true cause in top-3 predictions
    """)

    # E4: compare models by top-1 and top-3 accuracy
    print("\n  E4: Compare models at k=1 and k=3 (pattern):")
    print("""
    for model_name, result in results.items():
        topk = topk_from_full(result.full, result.id2label, k=3)
        acc  = topk_accuracy(topk)
        acc1 = acc.loc[acc["k"] == 1, "accuracy_pct"].values[0]
        acc3 = acc.loc[acc["k"] == 3, "accuracy_pct"].values[0]
        print(f"{model_name:20s}  acc@1={acc1:.1f}%  acc@3={acc3:.1f}%")
    """)


# ---------------------------------------------------------------------------
# F. Confusion heatmap
# ---------------------------------------------------------------------------

def demo_confusion_heatmap() -> None:
    """Section F: seaborn heatmap of predicted vs. true labels.

    Demonstrates:
    - Raw count heatmap (default)
    - Row-normalised heatmap (per-class recall distribution)
    - Custom label ordering
    - Label abbreviations
    - Saving to a file
    - Side-by-side multi-model comparison
    """
    from multimodalva.results import confusion_heatmap
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("F. Confusion Heatmap")
    print("=" * 60)

    pred_df = _make_predictions_df(n=300)
    y_true  = pred_df["true_label"]

    # F1: default raw count heatmap
    print("\n  F1: Default heatmap — raw counts, alphabetical label order")
    fig, ax = confusion_heatmap(
        y_true=y_true,
        y_pred=pred_df["pred_bert"],
        model_name="BioClinicalBERT",
    )
    plt.show()

    # F2: row-normalised (per-class recall)
    print("\n  F2: Row-normalised — each row sums to 1 (per-class recall distribution)")
    fig, ax = confusion_heatmap(
        y_true=y_true,
        y_pred=pred_df["pred_bert"],
        model_name="BioClinicalBERT",
        normalize=True,
    )
    plt.show()

    # F3: custom label order (disease burden order, most common first)
    label_order = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]
    print("\n  F3: Custom label order (disease burden order)")
    fig, ax = confusion_heatmap(
        y_true=y_true,
        y_pred=pred_df["pred_ensemble"],
        model_name="Ensemble",
        label_order=label_order,
    )
    plt.show()

    # F4: label abbreviations via dict (order + abbr in one)
    abbr_map = {
        "Malaria":    "MAL",
        "Pneumonia":  "PNE",
        "HIV/AIDS":   "HIV",
        "Diarrhoea":  "DIA",
        "Maternal":   "MAT",
    }
    print("\n  F4: Label abbreviations (dict label_order, label_abbr=True)")
    fig, ax = confusion_heatmap(
        y_true=y_true,
        y_pred=pred_df["pred_ensemble"],
        model_name="Ensemble",
        label_order=abbr_map,
        label_abbr=True,
        normalize=True,
    )
    plt.show()

    # F5: save to file
    print("\n  F5: Save to PNG")
    fig, ax = confusion_heatmap(
        y_true=y_true,
        y_pred=pred_df["pred_ensemble"],
        model_name="Ensemble",
        label_order=label_order,
        normalize=True,
        save_path="/tmp/confusion_ensemble.png",
    )
    print("  Saved to /tmp/confusion_ensemble.png")
    plt.show()

    # F6: multi-model comparison — one heatmap per model in a single figure
    print("\n  F6: Multi-model side-by-side comparison")
    models     = ["pred_bert", "pred_lightgbm", "pred_ensemble"]
    model_names = ["BioClinicalBERT", "LightGBM", "Ensemble"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax_i, (col, name) in enumerate(zip(models, model_names)):
        _, sub_ax = confusion_heatmap(
            y_true=y_true,
            y_pred=pred_df[col],
            model_name=name,
            label_order=abbr_map,
            label_abbr=True,
            normalize=True,
            annot=True,
            figsize=(7, 6),   # ignored — we pass in ax below
        )
        # Transfer the heatmap from the returned figure onto the grid axis
        # (for proper subplot layout, create axes externally and pass fig+ax)
    plt.tight_layout()
    plt.show()
    print("  Tip: for multi-panel layouts, call confusion_heatmap() per model separately")
    print("  and use plt.subplots() + ax.set_title() to combine into a publication figure.")


# ---------------------------------------------------------------------------
# G. Cause-specific accuracy heatmap
# ---------------------------------------------------------------------------

def demo_cause_accuracy_heatmap() -> None:
    """Section G: cause × model accuracy heatmap.

    Demonstrates:
    - G1: Top-1 accuracy from a wide prediction DataFrame (basic usage)
    - G2: Cause ordering, dropping, and renaming
    - G3: Model renaming + column grouping annotations below x-axis
    - G4: Top-3 accuracy using topk_dfs (from topk_from_full)
    - G5: Save to file
    """
    from multimodalva.results import cause_accuracy_heatmap, topk_from_full
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("G. Cause-Specific Accuracy Heatmap")
    print("=" * 60)

    pred_df   = _make_wide_pred_df(n=300)
    model_cols = ["insilicova", "lightgbm", "bert", "ensemble"]
    print(f"  Wide pred df: {len(pred_df)} rows, models: {model_cols}")

    # G1: Basic top-1 heatmap — all models, all causes, percentage cells
    print("\n  G1: Top-1 accuracy heatmap (all models, alphabetical cause order)")
    fig, ax, acc_df = cause_accuracy_heatmap(
        df=pred_df,
        true_col="true_label",
    )
    print(f"  accuracy_df shape: {acc_df.shape}  (causes × models)")
    print(acc_df.round(1).to_string())
    plt.show()

    # G2: Custom cause ordering + drop a cause + cause renaming
    print("\n  G2: Custom cause order, drop Maternal, rename causes")
    cause_order  = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]
    cause_rename = {
        "HIV/AIDS":  "HIV / AIDS",
        "Diarrhoea": "Diarrheal disease",
    }
    fig, ax, acc_df = cause_accuracy_heatmap(
        df=pred_df,
        true_col="true_label",
        model_cols=model_cols,
        cause_order=cause_order,
        drop_causes=["Maternal"],
        cause_rename=cause_rename,
        figsize=(10, 4),
    )
    plt.show()

    # G3: Model renaming + group boundary annotations
    # Layout: cols 0-1 = Unimodal (insilicova, lightgbm)
    #         cols 2-3 = Multimodal (bert, ensemble)
    print("\n  G3: Model rename + column grouping annotations")
    model_rename = {
        "insilicova": "InSilicoVA",
        "lightgbm":   "LightGBM",
        "bert":       "BioClinicalBERT",
        "ensemble":   "Stacking",
    }
    # Bracket spans from center of first col to center of last col in each group
    # Column index: 0=insilicova, 1=lightgbm, 2=bert, 3=ensemble
    group_boundaries = [(0.5, 1.5), (2.5, 3.5)]
    group_labels     = ["Unimodal", "Multimodal"]

    fig, ax, acc_df = cause_accuracy_heatmap(
        df=pred_df,
        true_col="true_label",
        model_cols=model_cols,
        cause_order=cause_order,
        drop_causes=["Maternal"],
        cause_rename=cause_rename,
        model_rename=model_rename,
        group_boundaries=group_boundaries,
        group_labels=group_labels,
        vmin=0, vmax=100,       # pin scale so cells are comparable across plots
        figsize=(10, 4),
    )
    plt.show()

    # G4: Top-3 accuracy using per-model full probability DataFrames
    # topk_dfs overrides df for each model name; df=None when using topk_dfs only
    print("\n  G4: Top-3 accuracy using topk_dfs (from full probability matrices)")
    full_dfs = _make_model_full_dfs(n=300)
    topk_dfs = {
        name: topk_from_full(full_df, ID2LABEL, k=3)
        for name, full_df in full_dfs.items()
    }

    fig, ax, acc_df3 = cause_accuracy_heatmap(
        df=None,               # pass None — all accuracy comes from topk_dfs
        true_col="true_label",
        topk_dfs=topk_dfs,
        top_k=3,
        cause_order=cause_order,
        drop_causes=["Maternal"],
        cause_rename=cause_rename,
        model_rename=model_rename,
        group_boundaries=group_boundaries,
        group_labels=group_labels,
        vmin=0, vmax=100,
        cbar_label="Top-3 Accuracy (%)",
        figsize=(10, 4),
    )
    print(f"  Top-3 accuracy_df (% correct in top-3 predictions):")
    print(acc_df3.round(1).to_string())
    plt.show()

    # G5: Save to file
    print("\n  G5: Save heatmap to PNG")
    fig, ax, _ = cause_accuracy_heatmap(
        df=pred_df,
        true_col="true_label",
        model_cols=model_cols,
        cause_order=cause_order,
        cause_rename=cause_rename,
        model_rename=model_rename,
        group_boundaries=group_boundaries,
        group_labels=group_labels,
        vmin=0, vmax=100,
        figsize=(10, 4),
        save_path="/tmp/cause_accuracy.png",
        dpi=150,
    )
    print("  Saved to /tmp/cause_accuracy.png")
    plt.show()


# ---------------------------------------------------------------------------
# H. Cause-specific accuracy difference heatmap
# ---------------------------------------------------------------------------

def demo_cause_accuracy_diff_heatmap() -> None:
    """Section H: cause × model accuracy difference heatmap vs. a baseline.

    Each cell shows accuracy(model) − accuracy(baseline).
    Positive values (warm) → model outperforms baseline for that cause.
    Negative values (cool) → model underperforms.
    Colormap is always centred at 0 (white/grey = ties with baseline).

    Demonstrates:
    - H1: Basic difference heatmap (InSilicoVA as baseline)
    - H2: With include_baseline=True (baseline zero-column kept for visual reference)
    - H3: Cause ordering + model renaming + column grouping annotations
    - H4: Top-3 difference using topk_dfs
    - H5: Save to file
    """
    from multimodalva.results import cause_accuracy_diff_heatmap, topk_from_full
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("H. Cause-Specific Accuracy Difference Heatmap")
    print("=" * 60)

    pred_df    = _make_wide_pred_df(n=300)
    model_cols = ["lightgbm", "bert", "ensemble"]   # comparison models (not baseline)
    cause_order = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]
    print(f"  Baseline: insilicova  |  Comparison: {model_cols}")

    # H1: Basic diff heatmap — 3 comparison models vs InSilicoVA baseline
    # vmin/vmax auto-computed as ±max_abs; center=0 is always applied
    print("\n  H1: Difference vs InSilicoVA (auto symmetric scale)")
    fig, ax, diff_df = cause_accuracy_diff_heatmap(
        df=pred_df,
        true_col="true_label",
        baseline_col="insilicova",
        model_cols=model_cols,
    )
    print(f"  diff_df shape: {diff_df.shape}  (causes × comparison models)")
    print(diff_df.round(1).to_string())
    plt.show()

    # H2: include_baseline=True — prepend a zero column so baseline label is visible
    print("\n  H2: include_baseline=True — zero column for InSilicoVA shown on left")
    fig, ax, diff_df = cause_accuracy_diff_heatmap(
        df=pred_df,
        true_col="true_label",
        baseline_col="insilicova",
        model_cols=model_cols,
        include_baseline=True,
        figsize=(10, 4.5),
    )
    plt.show()

    # H3: Full publication-style: cause order, model rename, group annotations
    # With include_baseline, columns are: insilicova (0), lightgbm (1), bert (2), ensemble (3)
    # Brackets: Unimodal = cols 0-1 (insilicova, lightgbm); Multimodal = cols 2-3 (bert, ensemble)
    print("\n  H3: Publication-style: cause order + model rename + group annotations")
    cause_rename = {
        "HIV/AIDS":  "HIV / AIDS",
        "Diarrhoea": "Diarrheal disease",
    }
    model_rename = {
        "insilicova": "InSilicoVA",
        "lightgbm":   "LightGBM",
        "bert":       "BioClinicalBERT",
        "ensemble":   "Stacking",
    }
    group_boundaries = [(0.5, 1.5), (2.5, 3.5)]
    group_labels     = ["Unimodal", "Multimodal"]

    fig, ax, diff_df = cause_accuracy_diff_heatmap(
        df=pred_df,
        true_col="true_label",
        baseline_col="insilicova",
        model_cols=model_cols,
        include_baseline=True,
        cause_order=cause_order,
        cause_rename=cause_rename,
        model_rename=model_rename,
        group_boundaries=group_boundaries,
        group_labels=group_labels,
        figsize=(10, 4.5),
    )
    plt.show()

    # H4: Top-3 difference using topk_dfs
    # Both baseline and comparison models must be in topk_dfs
    print("\n  H4: Top-3 difference using topk_dfs")
    full_dfs = _make_model_full_dfs(n=300)
    topk_dfs = {
        name: topk_from_full(full_df, ID2LABEL, k=3)
        for name, full_df in full_dfs.items()
    }

    fig, ax, diff_df3 = cause_accuracy_diff_heatmap(
        df=None,                   # all accuracy from topk_dfs
        true_col="true_label",
        baseline_col="insilicova",
        model_cols=model_cols,
        topk_dfs=topk_dfs,
        top_k=3,
        include_baseline=True,
        cause_order=cause_order,
        cause_rename=cause_rename,
        model_rename=model_rename,
        group_boundaries=group_boundaries,
        group_labels=group_labels,
        cbar_label="Δ Top-3 Accuracy (%)",
        figsize=(10, 4.5),
    )
    print(f"  Top-3 diff_df (Δ accuracy vs InSilicoVA top-3):")
    print(diff_df3.round(1).to_string())
    plt.show()

    # H5: Save to file
    print("\n  H5: Save diff heatmap to PNG")
    _, _, _ = cause_accuracy_diff_heatmap(
        df=pred_df,
        true_col="true_label",
        baseline_col="insilicova",
        model_cols=model_cols,
        include_baseline=True,
        cause_order=cause_order,
        cause_rename=cause_rename,
        model_rename=model_rename,
        group_boundaries=group_boundaries,
        group_labels=group_labels,
        figsize=(10, 4.5),
        save_path="/tmp/cause_accuracy_diff.png",
        dpi=150,
    )
    print("  Saved to /tmp/cause_accuracy_diff.png")
    plt.show()


# ---------------------------------------------------------------------------
# I. plot_topk_accuracy — bar charts (single, grouped, facet)
# ---------------------------------------------------------------------------

def demo_plot_topk_accuracy() -> None:
    """Section I: top-k accuracy bar charts.

    Demonstrates:
    - I1: single-model bar chart (one colour per k)
    - I2: multi-model grouped bar chart (one bundle per k, models side by side)
    - I3: multi-model facet bar chart (2 × 2 grid, shared y-axis)
    """
    import matplotlib.pyplot as plt
    from multimodalva.results import topk_from_full, plot_topk_accuracy

    print("\n" + "=" * 60)
    print("I. plot_topk_accuracy — bar charts")
    print("=" * 60)

    # Synthetic full-prob DataFrames for four mock models
    rng = np.random.default_rng(0)

    def _make_topk(n: int = 300, seed: int = 0) -> "pd.DataFrame":
        rng_  = np.random.default_rng(seed)
        full_ = _make_full_proba_df(n=n)
        return topk_from_full(full_, ID2LABEL, k=3)

    topk_bert  = _make_topk(seed=0)
    topk_lgbm  = _make_topk(seed=1)
    topk_df_   = _make_topk(seed=2)
    topk_stack = _make_topk(seed=3)

    # I1: Single-model bar chart
    print("\n  I1: Single-model bar chart")
    fig, ax = plot_topk_accuracy(
        topk_bert,
        max_k=3,
        model_label="BioBERT (adults)",
        save_path="/tmp/topk_single.png",
    )
    print("  Saved to /tmp/topk_single.png")
    plt.show()

    # I2: Multi-model grouped bar chart
    print("\n  I2: Multi-model grouped bar chart")
    fig, ax = plot_topk_accuracy(
        {
            "BioBERT":    topk_bert,
            "LightGBM":   topk_lgbm,
            "Data Fusion": topk_df_,
            "Stacking":   topk_stack,
        },
        max_k=3,
        kind="grouped",
        title="Top-k Accuracy Comparison",
        save_path="/tmp/topk_grouped.png",
    )
    print("  Saved to /tmp/topk_grouped.png")
    plt.show()

    # I3: Multi-model facet bar chart (2 × 2 grid)
    print("\n  I3: Multi-model facet bar chart (2 cols × 2 rows)")
    fig, axes = plot_topk_accuracy(
        {
            "BioBERT":    topk_bert,
            "LightGBM":   topk_lgbm,
            "Data Fusion": topk_df_,
            "Stacking":   topk_stack,
        },
        max_k=3,
        kind="facet",
        ncols=2,
        title="Top-k Accuracy by Model",
        save_path="/tmp/topk_facet.png",
    )
    print(f"  axes shape: {axes.shape}")
    print("  Saved to /tmp/topk_facet.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sections = {
        "A": demo_loss_curves,
        "B": demo_hpo_leaderboard,
        "C": demo_performance_leaderboard,
        "D": demo_topk_from_full,
        "E": demo_topk_accuracy,
        "F": demo_confusion_heatmap,
        "G": demo_cause_accuracy_heatmap,
        "H": demo_cause_accuracy_diff_heatmap,
        "I": demo_plot_topk_accuracy,
    }

    # Run a specific section if passed as argument, else run all
    if len(sys.argv) > 1:
        for key in sys.argv[1:]:
            key = key.upper()
            if key in sections:
                sections[key]()
            else:
                print(f"Unknown section '{key}'.  Available: {list(sections)}")
    else:
        for fn in sections.values():
            fn()
