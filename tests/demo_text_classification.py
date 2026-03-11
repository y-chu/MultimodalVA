"""
Demo: text-only cause-of-death classification with multimodalva.

Covers:
    A. TextClassifier wrapper — fixed hyperparams
    B. TextClassifier wrapper — Optuna HPO with custom search space
    C. Individual pipeline steps (split → prepare_dataset → train → predict)
    D. Advanced train() options: LoRA, class weights, label smoothing, layer freezing
    E. Post-run analysis: predictions, HPO study, trial results
    F. Remote / non-Hub models via download_model()
    G. Ray Tune HPO — distributed HPO across multiple GPUs or cluster nodes

Run any section by calling its function from __main__.
"""

from __future__ import annotations

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Shared toy dataset (replace with real VA data in practice)
# ---------------------------------------------------------------------------

def make_toy_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic verbal autopsy DataFrame for demonstration."""
    import numpy as np
    rng = np.random.default_rng(seed)
    causes = ["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"]
    texts = {
        "Malaria":    "The deceased had fever and chills for several days before death.",
        "Pneumonia":  "Patient had difficulty breathing and productive cough.",
        "HIV/AIDS":   "Chronic weight loss, night sweats and recurrent infections were reported.",
        "Diarrhoea":  "Watery stool for one week, signs of severe dehydration.",
        "Maternal":   "Died during or shortly after childbirth with heavy bleeding.",
    }
    labels = rng.choice(causes, size=n)
    narratives = [texts[c] + f" (case {i})" for i, c in enumerate(labels)]
    return pd.DataFrame({"narrative": narratives, "cause": labels})


# ---------------------------------------------------------------------------
# A. TextClassifier — fixed hyperparams
# ---------------------------------------------------------------------------

def demo_fixed_hyperparams():
    """End-to-end run with explicit hyperparameters; no HPO."""
    from multimodalva.text import TextClassifier

    df = make_toy_df()

    clf = TextClassifier(
        model_name="bert-base-uncased",
        output_dir="runs/text_fixed",
    )

    results = clf.run(
        df=df,
        text_col="narrative",
        label_col="cause",
        # --- split ---
        test_size=0.2,
        random_state=42,
        stratify=True,
        # --- tokenization ---
        max_length=128,          # 512 for real narratives; 128 is fine for this demo
        # --- training ---
        use_optimize=False,
        hyperparams={
            "learning_rate": 3e-5,
            "batch_size": 16,
            "epochs": 3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "gradient_accumulation_steps": 1,
            "freeze_layers": 2,       # freeze embeddings + bottom 2 BERT layers
            "label_smoothing": 0.05,  # reduces overconfidence; helpful for imbalanced classes
            "max_grad_norm": 1.0,
            "dataloader_num_workers": 2,
            # Uncomment to enable class-weight rebalancing:
            # "class_weights": "balanced",  # auto-computed from training labels
            # "class_weights": [1.0, 2.0, 1.5, 2.0, 1.0],  # manual, ordered by class ID
        },
        # --- inference ---
        batch_size=32,
    )

    _print_results(results, "Fixed hyperparams")
    return clf, results


# ---------------------------------------------------------------------------
# B. TextClassifier — HPO with custom search space
# ---------------------------------------------------------------------------

def demo_hpo():
    """End-to-end run with Optuna HPO, custom search space, and CSMF metric."""
    from multimodalva.text import TextClassifier
    from multimodalva.text.hpo import DEFAULT_SEARCH_SPACE

    df = make_toy_df(n=300)

    clf = TextClassifier(
        model_name="bert-base-uncased",
        output_dir="runs/text_hpo",
    )

    # Custom search space overrides / extends DEFAULT_SEARCH_SPACE.
    # Only supply the keys you want to change; the rest come from DEFAULT_SEARCH_SPACE.
    # See hpo.py for the full list of supported spec types:
    #   ("float_log", low, high)   — log-uniform float (best for LR)
    #   ("float",     low, high)   — uniform float
    #   ("int",       low, high)   — uniform int
    #   ("categorical", [values])  — discrete choices
    custom_space = {
        "learning_rate": ("float_log", 1e-5, 5e-5),   # widen range
        "batch_size":    ("categorical", [8, 16]),
        "epochs":        ("categorical", [3, 5]),
        "freeze_layers": ("categorical", [0, 2, 4]),   # exclude aggressive 6-layer freeze
        # Narrow weight decay for small dataset:
        "weight_decay":  ("float", 0.0, 0.05),
    }

    results = clf.run(
        df=df,
        text_col="narrative",
        label_col="cause",
        # --- split ---
        test_size=0.2,
        random_state=42,
        stratify=True,
        # --- tokenization ---
        max_length=128,
        # --- HPO ---
        use_optimize=True,
        n_trials=10,              # increase to 20–50 for real experiments
        optimize_metric="csmf_accuracy",  # WHO/InsilicoVA standard for VA data
        # optimize_metric="f1_macro",     # alternative for imbalanced classes
        search_space=custom_space,
        # --- inference ---
        batch_size=32,
    )

    _print_results(results, "HPO (CSMF)")

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
    from multimodalva.text.dataset import prepare_dataset
    from multimodalva.text.train import train, DEFAULT_HYPERPARAMS, SUPPORTED_MODELS
    from multimodalva.text.predict import predict

    df = make_toy_df()
    model_name = SUPPORTED_MODELS["biobert"]  # or any HuggingFace Hub ID

    # Step 1 — split
    train_df, test_df = split(
        df,
        text_col="narrative",
        label_col="cause",
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    # Step 2 — tokenize
    # label maps are built from the union of train+test labels — build ONCE
    # and reuse across all sub-pipelines for consistent column ordering.
    train_ds, test_ds, label2id, id2label = prepare_dataset(
        train_df, test_df,
        text_col="narrative",
        label_col="cause",
        model_name=model_name,
        max_length=512,       # use model's native max; None = model's built-in maximum
    )
    print(f"Classes ({len(label2id)}): {label2id}")

    # Step 3 — train
    # Merge custom hyperparams over defaults
    hp = {
        **DEFAULT_HYPERPARAMS,
        "learning_rate": 2e-5,
        "epochs": 4,
        "freeze_layers": 4,          # moderate freeze for small dataset
        "class_weights": "balanced", # auto-compute from training label counts
        "label_smoothing": 0.05,
    }
    trainer, tokenizer, metadata = train(
        train_dataset=train_ds,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir="runs/text_steps/final",
        hyperparams=hp,
        val_size=0.1,         # internal stratified val split for early stopping
        use_lora=False,
        gradient_checkpointing=False,
        early_stopping_patience=3,    # stop if eval loss doesn't improve for 3 epochs
        resume=True,          # resume from latest checkpoint if run was interrupted
    )
    print("Training complete. log_history entries:", len(metadata["log_history"]))

    # Step 4 — predict (disk mode)
    result = predict(
        output_dir="runs/text_steps/final",
        test_dataset=test_ds,
        batch_size=32,
        top_k=3,
        save_dir="runs/text_steps/predictions",  # writes top1/full/topk CSVs
    )
    _print_prediction_result(result)

    # --- In-memory predict (skip disk reload) ---
    # Useful immediately after train() — avoids writing/reading model files.
    result_mem = predict(
        output_dir=None,
        test_dataset=test_ds,
        batch_size=32,
        model=trainer.model,
        tokenizer=tokenizer,
        id2label=metadata["id2label"],
    )
    print("In-memory prediction rows:", len(result_mem.top1))

    return result


# ---------------------------------------------------------------------------
# D. Advanced train() options: LoRA + gradient checkpointing
# ---------------------------------------------------------------------------

def demo_lora():
    """Fine-tune with LoRA adapters — much fewer trainable parameters."""
    from multimodalva.utils.split import split
    from multimodalva.text.dataset import prepare_dataset
    from multimodalva.text.train import train, SUPPORTED_MODELS

    df = make_toy_df()
    model_name = SUPPORTED_MODELS["biobert"]

    train_df, test_df = split(df, label_col="cause", text_col="narrative", test_size=0.2)
    train_ds, test_ds, label2id, id2label = prepare_dataset(
        train_df, test_df, "narrative", "cause", model_name
    )

    # LoRA hyperparameters are passed alongside standard ones.
    # hpo.py LORA_SEARCH_SPACE covers lora_r, lora_alpha, lora_dropout automatically
    # when use_lora=True is passed to optimize().
    trainer, tokenizer, metadata = train(
        train_dataset=train_ds,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir="runs/text_lora/final",
        hyperparams={
            "learning_rate": 3e-4,    # LoRA can use a higher LR than full fine-tuning
            "batch_size": 16,
            "epochs": 5,
            "lora_r": 16,             # rank; higher = more capacity
            "lora_alpha": 64,         # scaling = alpha / r; 4× r is a common heuristic
            "lora_dropout": 0.05,
            "freeze_layers": 0,       # LoRA handles regularisation; no layer freezing needed
        },
        val_size=0.1,
        use_lora=True,                 # enable LoRA adapters
        gradient_checkpointing=True,   # reduce GPU memory at cost of slightly slower training
        early_stopping_patience=3,
    )
    print("LoRA training done. Output dir:", metadata["output_dir"])
    return trainer, metadata


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

    # All 4 metrics for every completed trial
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

    # Reload study from SQLite (demonstrates persistence)
    # study2 = optuna.load_study(study_name=study.study_name, storage=study._storage)


def demo_reload_study(study_name: str, storage_path: str):
    """Load a previously saved Optuna study from its SQLite file.

    Example:
        demo_reload_study(
            study_name="text_hpo",
            storage_path="sqlite:///runs/text_hpo/hpo/hpo_bert-base-uncased.db",
        )
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(study_name=study_name, storage=storage_path)
    _analyze_study(study)

    # Re-run additional trials on top of existing ones
    # study.optimize(objective_fn, n_trials=10)
    return study


# ---------------------------------------------------------------------------
# F. Remote model download (not on HuggingFace Hub)
# ---------------------------------------------------------------------------

def demo_remote_model():
    """Download a model not hosted on the HuggingFace Hub, then train."""
    from multimodalva.text.train import download_model, REMOTE_MODELS

    print("Available remote models:", list(REMOTE_MODELS))

    # Downloads and extracts the archive; returns local path
    # model_path = download_model("roberta-pm", cache_dir="models/roberta-pm")

    # Then pass model_path anywhere model_name is accepted:
    # train(..., model_name=model_path)
    # prepare_dataset(..., model_name=model_path)

    print("(download_model call commented out to avoid network I/O in this demo)")


# ---------------------------------------------------------------------------
# G. Ray Tune HPO — distributed HPO across multiple GPUs or cluster nodes
# ---------------------------------------------------------------------------

def demo_hpo_ray():
    """Distributed HPO via Ray Tune — use when you have multiple GPUs or an HPC cluster.

    Calls optimize_ray() directly (pipeline-steps style) rather than through
    TextClassifier, since TextClassifier.run() uses Optuna internally.

    When to prefer this over demo_hpo() (Optuna):
        - You have ≥2 GPUs on one machine  →  N concurrent trials, one GPU each.
        - You are on a SLURM/Kubernetes cluster  →  pass ray_address="auto" after
          running `ray start --head --num-gpus=N` on the head node.
    Single GPU (MPS or CUDA):  Optuna (demo_hpo) is better — Ray overhead adds
    no benefit when only one trial can run at a time.

    Requires:  pip install 'ray[tune]' optuna
    """
    from multimodalva.utils.split import split
    from multimodalva.text.dataset import prepare_dataset
    from multimodalva.text.hpo import optimize_ray
    from multimodalva.text.train import train
    from multimodalva.text.predict import predict

    df = make_toy_df(n=300)
    model_name = "bert-base-uncased"

    # Step 1 — split
    train_df, test_df = split(
        df, label_col="cause", text_col="narrative", test_size=0.2, random_state=42
    )

    # Step 2 — tokenize
    # Build label maps ONCE and reuse; ensures consistent prob column ordering.
    train_ds, test_ds, label2id, id2label = prepare_dataset(
        train_df, test_df,
        text_col="narrative",
        label_col="cause",
        model_name=model_name,
        max_length=128,
    )

    # Step 3 — Ray Tune HPO
    best_hyperparams, results = optimize_ray(
        train_dataset=train_ds,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir="runs/text_hpo_ray",
        n_trials=10,                      # increase to 20–50 for real experiments
        metric="csmf_accuracy",           # WHO standard; use for imbalanced VA data
        # --- Ray resource allocation ---
        num_gpus_per_trial=1.0,           # one GPU per trial; concurrent trials = #GPUs
        # num_gpus_per_trial=0.5,         # pack 2 trials per GPU (only if VRAM allows)
        num_cpus_per_trial=4,
        max_concurrent_trials=None,       # None = Ray auto-fills all available GPUs
        # --- Cluster address ---
        ray_address=None,                 # None = local Ray; "auto" = existing cluster
        # ray_address="auto",             # use after: ray start --head --num-gpus=4
        # --- Optional search space override ---
        # search_space={"learning_rate": ("float_log", 1e-5, 5e-5)},
        use_lora=False,
        gradient_checkpointing=False,
        early_stopping_patience=3,
        val_size=0.2,
        random_state=42,
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
    # val_size=None: epochs already determined by HPO; train on all training data.
    trainer, tokenizer, metadata = train(
        train_dataset=train_ds,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        output_dir="runs/text_hpo_ray/final",
        hyperparams=best_hyperparams,
        val_size=None,
        gradient_checkpointing=False,
        early_stopping_patience=None,
        resume=False,
    )

    # Step 5 — predict (in-memory; no disk reload needed)
    result = predict(
        output_dir=None,
        test_dataset=test_ds,
        model=trainer.model,
        tokenizer=tokenizer,
        id2label=id2label,
    )
    _print_prediction_result(result)
    return best_hyperparams, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running demo A: fixed hyperparams ...")
    demo_fixed_hyperparams()

    # Uncomment to run other demos:
    # print("\nRunning demo B: Optuna HPO ...")
    # demo_hpo()

    # print("\nRunning demo C: individual pipeline steps ...")
    # demo_pipeline_steps()

    # print("\nRunning demo D: LoRA ...")
    # demo_lora()

    # demo_remote_model()

    # print("\nRunning demo G: Ray Tune HPO (requires ray[tune] + multiple GPUs/cluster) ...")
    # demo_hpo_ray()
