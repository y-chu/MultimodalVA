"""
Demo: tabular-only cause-of-death classification with multimodalva.

Covers:
    A. TabularClassifier wrapper — fixed hyperparams (LightGBM)
    B. TabularClassifier wrapper — Optuna HPO with custom search space
    C. Individual pipeline steps (split → prepare_dataset → train → predict)
    D. Model comparison: run multiple algorithms on the same data
    E. Post-run analysis: predictions, HPO study, SHAP feature names
    F. Ray Tune HPO — distributed HPO across multiple CPUs/GPUs or cluster nodes

Run any section by calling its function from __main__.

Note: no GPU is required; all models run on CPU by default.
      Pass use_gpu=True to TabularClassifier.run() to enable CUDA for
      lightgbm, xgboost, and catboost if a GPU is available.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Shared toy dataset (replace with real VA data in practice)
# ---------------------------------------------------------------------------

def make_toy_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic verbal autopsy DataFrame with tabular features.

    Simulates the kind of structured indicators collected in a VA interview:
    age, sex, symptom duration, binary symptom flags, and geographic region.
    In a real study these would come from a standardised VA questionnaire
    (WHO 2016, PHMRC, etc.).
    """
    rng = np.random.default_rng(seed)

    causes = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]
    n_causes = len(causes)
    labels = rng.choice(causes, size=n)

    # Numeric features — values vary by cause to give the model a signal
    cause_idx = np.array([causes.index(c) for c in labels])
    age          = np.clip(rng.normal(40 - cause_idx * 5, 15, n), 0, 90).astype(int)
    duration_days = np.clip(rng.normal(7 + cause_idx * 3, 5, n), 1, 60).astype(int)
    fever         = (rng.random(n) < (0.8 - cause_idx * 0.1)).astype(int)
    cough         = (rng.random(n) < (0.2 + cause_idx * 0.15)).astype(int)
    diarrhoea     = (rng.random(n) < np.where(labels == "Diarrhoea", 0.9, 0.1)).astype(int)
    weight_loss   = (rng.random(n) < np.where(labels == "HIV/AIDS", 0.85, 0.1)).astype(int)
    pregnant      = (rng.random(n) < np.where(labels == "Maternal", 0.95, 0.02)).astype(int)

    # Categorical features
    sex    = rng.choice(["male", "female"], size=n)
    region = rng.choice(["urban", "rural", "peri-urban"], size=n)

    return pd.DataFrame({
        "age":           age,
        "duration_days": duration_days,
        "fever":         fever,
        "cough":         cough,
        "diarrhoea":     diarrhoea,
        "weight_loss":   weight_loss,
        "pregnant":      pregnant,
        "sex":           sex,
        "region":        region,
        "cause":         labels,
    })


FEATURE_COLS = [
    "age", "duration_days",
    "fever", "cough", "diarrhoea", "weight_loss", "pregnant",  # binary symptoms
    "sex", "region",                                            # categorical
]


# ---------------------------------------------------------------------------
# A. TabularClassifier — fixed hyperparams (LightGBM)
# ---------------------------------------------------------------------------

def demo_fixed_hyperparams():
    """End-to-end run with explicit LightGBM hyperparams; no HPO."""
    from multimodalva.tabular import TabularClassifier

    df = make_toy_df()

    clf = TabularClassifier(
        model_name="lightgbm",
        output_dir="runs/tabular_fixed",
    )

    results = clf.run(
        df=df,
        feature_cols=FEATURE_COLS,
        label_col="cause",
        # --- split ---
        test_size=0.2,
        random_state=42,
        stratify=True,
        # --- feature engineering ---
        encode_categoricals="ordinal",  # OrdinalEncoder; NaN-safe for LightGBM's native NaN handling
        scale_numeric=False,            # tree models do not need feature scaling
        # --- training ---
        use_optimize=False,
        hyperparams={
            "n_estimators":      300,
            "learning_rate":     0.05,
            "max_depth":         -1,    # -1 = no depth limit; complexity governed by num_leaves
            "num_leaves":        63,
            "min_child_samples": 20,    # larger = fewer splits on small leaf groups; good for rare causes
            "verbose":           -1,    # suppress LightGBM output
        },
        # --- inference ---
        top_k=3,
        n_jobs=-1,    # use all CPU cores
        use_gpu=None, # None = auto-detect CUDA; False to force CPU
    )

    _print_results(results, "LightGBM — fixed hyperparams")
    return clf, results


# ---------------------------------------------------------------------------
# B. TabularClassifier — HPO with custom search space
# ---------------------------------------------------------------------------

def demo_hpo():
    """End-to-end run with Optuna HPO, custom search space, and CSMF metric."""
    from multimodalva.tabular import TabularClassifier
    from multimodalva.tabular.hpo import DEFAULT_SEARCH_SPACES

    df = make_toy_df(n=400)

    clf = TabularClassifier(
        model_name="lightgbm",
        output_dir="runs/tabular_hpo",
    )

    # Custom search space overrides / extends DEFAULT_SEARCH_SPACES["lightgbm"].
    # Only supply the keys you want to change; the rest come from the default.
    # Spec format:
    #   ("float_log", low, high)   — log-uniform float (best for learning rates)
    #   ("float",     low, high)   — uniform float
    #   ("int",       low, high)   — uniform int
    #   ("categorical", [values])  — discrete choices
    custom_space = {
        "learning_rate":     ("float_log", 1e-3, 0.1),   # narrow range for small dataset
        "n_estimators":      ("int",       100, 500),
        "num_leaves":        ("int",       20, 80),
        "min_child_samples": ("int",       10, 50),       # prevent splits on rare-cause subgroups
    }

    results = clf.run(
        df=df,
        feature_cols=FEATURE_COLS,
        label_col="cause",
        # --- split ---
        test_size=0.2,
        random_state=42,
        stratify=True,
        # --- feature engineering ---
        encode_categoricals="ordinal",
        scale_numeric=False,
        # --- HPO ---
        use_optimize=True,
        n_trials=15,                        # increase to 30–50 for real experiments
        optimize_metric="csmf_accuracy",    # WHO/InsilicoVA standard; use for imbalanced VA data
        # optimize_metric="f1_macro",       # alternative for imbalanced class distributions
        search_space=custom_space,
        # --- inference ---
        top_k=3,
        n_jobs=-1,
    )

    _print_results(results, "LightGBM — HPO (CSMF)")

    # Access the Optuna study stored on the classifier instance
    if clf.study is not None:
        _analyze_study(clf.study)

    return clf, results


# ---------------------------------------------------------------------------
# C. Individual pipeline steps
# ---------------------------------------------------------------------------

def demo_pipeline_steps():
    """Call each pipeline step directly for maximum control."""
    from multimodalva.utils.split import split
    from multimodalva.tabular.dataset import prepare_dataset
    from multimodalva.tabular.train import train, DEFAULT_HYPERPARAMS, SUPPORTED_MODELS
    from multimodalva.tabular.predict import predict

    df = make_toy_df()
    model_name = "lightgbm"

    # Step 1 — split
    # label_col is the first keyword arg; always use keyword form.
    train_df, test_df = split(
        df,
        label_col="cause",
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    # Step 2 — feature engineering + label encoding
    # Build label maps ONCE from the union of train+test labels, then reuse
    # across all sub-pipelines to guarantee consistent prob column ordering.
    (
        X_train, X_test,
        y_train, y_test,
        preprocessor,
        label2id, id2label,
        feature_names,
    ) = prepare_dataset(
        train_df, test_df,
        feature_cols=FEATURE_COLS,
        label_col="cause",
        encode_categoricals="ordinal",  # OrdinalEncoder — safe for LightGBM native NaN handling
        scale_numeric=False,
    )
    print(f"Classes ({len(label2id)}): {label2id}")
    print(f"Feature names: {feature_names}")

    # Step 3 — train
    # hyperparams are merged over DEFAULT_HYPERPARAMS["lightgbm"]
    hp = {
        **DEFAULT_HYPERPARAMS[model_name],
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 20,
        "verbose":           -1,
    }
    model, metadata = train(
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir="runs/tabular_steps/final",
        hyperparams=hp,
        preprocessor=preprocessor,   # bundled into model.joblib for later scoring on raw data
        feature_names=feature_names, # saved as feature_names.json for SHAP
        random_state=42,
        n_jobs=-1,
        use_gpu=None,                # None = auto-detect CUDA
    )
    print("Training complete. Artifacts saved to:", metadata["output_dir"])

    # Step 4 — predict (disk mode: reload model from output_dir)
    result = predict(
        output_dir="runs/tabular_steps/final",
        X_test=X_test,
        y_test=y_test,
        top_k=3,
        save_dir="runs/tabular_steps/predictions",  # writes top1/full/topk CSVs
    )
    _print_prediction_result(result)

    # --- In-memory predict (skip disk reload) ---
    # Useful immediately after train() — avoids writing/reading model files.
    result_mem = predict(
        output_dir=None,
        X_test=X_test,
        y_test=y_test,
        model=model,
        id2label=id2label,
    )
    print("In-memory prediction rows:", len(result_mem.top1))

    return result


# ---------------------------------------------------------------------------
# D. Model comparison: same data, multiple algorithms
# ---------------------------------------------------------------------------

def demo_model_comparison():
    """Run several models on identical train/test splits and compare metrics.

    Useful for choosing a model class before running full HPO.
    Share the same label2id/id2label across models so prob columns align.
    """
    from multimodalva.utils.split import split
    from multimodalva.tabular.dataset import prepare_dataset
    from multimodalva.tabular.train import train
    from multimodalva.tabular.predict import predict
    from multimodalva.utils.metrics import score_predictions

    df = make_toy_df(n=400)

    # Split once — all models see the same train/test rows
    train_df, test_df = split(df, label_col="cause", test_size=0.2, random_state=42)

    # Prepare once — same label maps for all models
    (
        X_train, X_test,
        y_train, y_test,
        preprocessor,
        label2id, id2label,
        feature_names,
    ) = prepare_dataset(
        train_df, test_df,
        feature_cols=FEATURE_COLS,
        label_col="cause",
        encode_categoricals="ordinal",
        scale_numeric=False,
    )

    models_to_compare = {
        "lightgbm":      {"n_estimators": 200, "learning_rate": 0.05, "verbose": -1},
        "random_forest": {"n_estimators": 200, "max_depth": None},
        "xgboost":       {"n_estimators": 200, "learning_rate": 0.05, "verbosity": 0},
        "gbdt":          {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    }

    print(f"\n{'Model':<16} {'Accuracy':>10} {'F1 Macro':>10} {'CSMF Acc':>10}")
    print("-" * 50)

    for model_name, hp in models_to_compare.items():
        model, metadata = train(
            X_train=X_train, y_train=y_train,
            label2id=label2id, id2label=id2label,
            model_name=model_name,
            output_dir=f"runs/tabular_compare/{model_name}",
            hyperparams=hp,
            preprocessor=preprocessor,
            feature_names=feature_names,
            random_state=42, n_jobs=-1,
        )
        result = predict(
            output_dir=f"runs/tabular_compare/{model_name}",
            X_test=X_test, y_test=y_test, top_k=3,
        )
        acc   = score_predictions(result.top1, "accuracy")
        f1    = score_predictions(result.top1, "f1_macro")
        csmf  = score_predictions(result.top1, "csmf_accuracy")
        print(f"{model_name:<16} {acc:>10.3f} {f1:>10.3f} {csmf:>10.3f}")

    print()


# ---------------------------------------------------------------------------
# E. Post-run analysis helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict, label: str = ""):
    pred = results["predictions"]
    top1 = pred.top1
    n = len(top1)
    acc = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"\n=== {label} ===")
    print(f"  Test samples : {n}")
    print(f"  Accuracy     : {acc:.3f}")
    print(f"  Output dir   : {results['output_dir']}")
    print(f"  Best HP      : {results['best_hyperparams']}")
    print(f"  Top-1 sample :\n{top1.head(3).to_string(index=False)}")


def _print_prediction_result(result):
    """Show all three output DataFrames from predict()."""
    print("\n--- top1 (true label, predicted label, probability) ---")
    print(result.top1.head(5).to_string(index=False))

    print("\n--- topk (top-3 classes + probabilities per sample) ---")
    print(result.topk.head(3).to_string(index=False))

    print("\n--- full (probability for every class; integer IDs as column names) ---")
    # Rename integer columns to label strings for readability
    full_named = result.full.rename(
        columns={f"prob_{i}": f"prob_{v}" for i, v in result.id2label.items()}
    )
    print(full_named.head(3).to_string(index=False))


def _analyze_study(study):
    """Print an Optuna study summary and trial-level metrics."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = study.trials_dataframe()

    print("\n=== HPO study summary ===")
    print(f"  Total trials   : {len(df)}")
    completed = df[df["state"] == "COMPLETE"]
    pruned    = df[df["state"] == "PRUNED"]
    failed    = df[df["state"] == "FAIL"]
    print(f"  Completed      : {len(completed)}")
    print(f"  Pruned         : {len(pruned)}")
    print(f"  Failed         : {len(failed)}")
    print(f"  Best value     : {study.best_value:.4f}")
    print(f"  Best params    : {study.best_params}")

    # All 4 metrics for every completed trial (stored as user_attrs_* columns)
    metric_cols = [c for c in df.columns if c.startswith("user_attrs_")]
    if metric_cols and len(completed) > 0:
        print("\n  Per-trial metrics (top 5 by objective):")
        display_cols = ["number", "value"] + metric_cols + ["duration"]
        available = [c for c in display_cols if c in df.columns]
        print(
            completed.sort_values("value", ascending=False)[available]
            .head(5)
            .to_string(index=False)
        )


def demo_reload_study(study_name: str, storage_path: str):
    """Load a previously saved Optuna study from its SQLite file.

    Example:
        demo_reload_study(
            study_name="tabular_hpo",
            storage_path="sqlite:///runs/tabular_hpo/hpo/hpo_lightgbm.db",
        )
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(study_name=study_name, storage=storage_path)
    _analyze_study(study)

    # Re-run additional trials on top of existing ones:
    # study.optimize(objective_fn, n_trials=10)
    return study


# ---------------------------------------------------------------------------
# F. Ray Tune HPO — distributed HPO across multiple CPUs/GPUs or cluster nodes
# ---------------------------------------------------------------------------

def demo_hpo_ray():
    """Distributed HPO via Ray Tune — use on multi-core servers or HPC clusters.

    Calls optimize_ray() directly (pipeline-steps style) rather than through
    TabularClassifier, since TabularClassifier.run() uses Optuna internally.

    When to prefer this over demo_hpo() (Optuna):
        - Machine with ≥16 cores  →  run 4+ trials in parallel (4 cores/trial).
        - SLURM/Kubernetes cluster  →  pass ray_address="auto".
    Few cores (≤8): Optuna (demo_hpo) is simpler — Ray overhead adds no benefit.

    GPU models (catboost, lightgbm, xgboost):
        Set num_gpus_per_trial=1.0 AND use_gpu=True together.
        All other models (gbdt, mlp, random_forest, naive_bayes, knn, svm)
        are CPU-only; leave num_gpus_per_trial=0.0 (default).

    n_jobs note: automatically capped to num_cpus_per_trial to prevent each
    worker from claiming all cores when multiple trials run concurrently.

    Requires:  pip install 'ray[tune]' optuna
    """
    from multimodalva.utils.split import split
    from multimodalva.tabular.dataset import prepare_dataset
    from multimodalva.tabular.hpo import optimize_ray
    from multimodalva.tabular.train import train
    from multimodalva.tabular.predict import predict
    from multimodalva.utils.metrics import score_predictions

    df = make_toy_df(n=400)
    model_name = "lightgbm"

    # Step 1 — split
    train_df, test_df = split(df, label_col="cause", test_size=0.2, random_state=42)

    # Step 2 — feature engineering + label encoding
    # Build label maps ONCE and reuse; ensures consistent prob column ordering.
    (
        X_train, X_test,
        y_train, y_test,
        preprocessor,
        label2id, id2label,
        feature_names,
    ) = prepare_dataset(
        train_df, test_df,
        feature_cols=FEATURE_COLS,
        label_col="cause",
        encode_categoricals="ordinal",
        scale_numeric=False,
    )

    # Step 3 — Ray Tune HPO
    best_hyperparams, results = optimize_ray(
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir="runs/tabular_hpo_ray",
        n_trials=15,                      # increase to 30–50 for real experiments
        metric="csmf_accuracy",           # WHO standard; use for imbalanced VA data
        # --- CPU/GPU resources per trial ---
        num_gpus_per_trial=0.0,           # 0.0 = CPU-only (correct for lightgbm/random_forest/etc.)
        # num_gpus_per_trial=1.0,         # enable together with use_gpu=True for GPU tree models
        num_cpus_per_trial=4,             # n_jobs auto-capped to this value per trial
        max_concurrent_trials=None,       # None = Ray fills all available CPU/GPU slots
        # --- Cluster address ---
        ray_address=None,                 # None = local Ray; "auto" = existing cluster
        # ray_address="auto",             # use after: ray start --head --num-cpus=32
        # --- Optional search space override ---
        # search_space={"n_estimators": ("int", 100, 500)},
        val_size=0.2,
        random_state=42,
        n_jobs=-1,     # auto-capped to num_cpus_per_trial (=4) when running concurrently
        use_gpu=None,  # None = auto-detect; set True + num_gpus_per_trial=1.0 for GPU
        save_trials_csv=True,             # writes hpo_trials_ray.csv to output_dir
    )
    print("Best hyperparams (Ray):", best_hyperparams)

    # Analyse all trial results — columns include metric names + config/* keys
    trials_df = results.get_dataframe()
    print(f"\nTrial results ({len(trials_df)} trials) — columns: {list(trials_df.columns)}")
    metric_cols = [c for c in ("csmf_accuracy", "accuracy", "f1_macro") if c in trials_df.columns]
    if metric_cols:
        print(trials_df[metric_cols].describe().to_string())

    # Step 4 — train final model with best hyperparams
    model, metadata = train(
        X_train=X_train,
        y_train=y_train,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir="runs/tabular_hpo_ray/final",
        hyperparams=best_hyperparams,
        preprocessor=preprocessor,
        feature_names=feature_names,
        random_state=42,
        n_jobs=-1,
    )
    print("Training complete. Artifacts:", metadata["output_dir"])

    # Step 5 — predict (in-memory; no disk reload needed)
    result = predict(
        output_dir=None,
        X_test=X_test,
        y_test=y_test,
        model=model,
        id2label=id2label,
    )
    _print_prediction_result(result)

    for metric in ("accuracy", "f1_macro", "csmf_accuracy"):
        print(f"  {metric:<20}: {score_predictions(result.top1, metric):.4f}")

    return best_hyperparams, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running demo A: LightGBM with fixed hyperparams ...")
    demo_fixed_hyperparams()

    # Uncomment to run other demos:
    # print("\nRunning demo B: Optuna HPO ...")
    # demo_hpo()

    # print("\nRunning demo C: individual pipeline steps ...")
    # demo_pipeline_steps()

    # print("\nRunning demo D: model comparison ...")
    # demo_model_comparison()

    # print("\nRunning demo F: Ray Tune HPO (requires ray[tune] + multi-core server/cluster) ...")
    # demo_hpo_ray()
