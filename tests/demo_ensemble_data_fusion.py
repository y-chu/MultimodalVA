"""Demos for the data-fusion ensemble pipeline.

Usage (synthetic demo data):
    python tests/demo_ensemble_data_fusion.py text
    python tests/demo_ensemble_data_fusion.py run
    python tests/demo_ensemble_data_fusion.py run --hpo
    python tests/demo_ensemble_data_fusion.py benchmark

Usage (your own CSV):
    python tests/demo_ensemble_data_fusion.py run \\
        --data path/to/data.csv --label cause --text-col narrative

    python tests/demo_ensemble_data_fusion.py run \\
        --data path/to/data.csv --label cause --text-col narrative \\
        --exclude-cols record_id \\
        --model allenai/longformer-base-4096 \\
        --hpo --n-trials 30 --metric f1_macro \\
        --resume --output-dir runs/my_data_fusion

    python tests/demo_ensemble_data_fusion.py benchmark \\
        --max-length 256 --output-dir runs/benchmark

Key flags:
    --data          CSV file path, or "demo" for synthetic data (default: demo)
    --label         Label column name (default: cause)
    --text-col      Text/narrative column name (default: narrative)
    --exclude-cols        Comma-separated columns to exclude from tabular features
                    (e.g. ID, date).  Label and text columns are always excluded.
    --model         Long-context LM checkpoint or alias
                    (default: allenai/longformer-base-4096)
    --hpo           Enable Optuna hyperparameter search
    --n-trials      HPO trials (default: 2 for demo / recommend 30 for real data)
    --n-cv-folds    CV folds for HPO scoring (default: 3; use 0 to disable CV)
    --metric        Optimisation metric (default: f1_macro)
    --resume        Resume HPO / training from checkpoint
    --max-length    Max token length for fused text (default: 256)
    --output-dir    Where to save model artifacts (default: runs/demo_data_fusion)
    --top-k         Number of top-k predictions to return (default: 3)
    --features      Advanced: explicit feature columns (comma-sep or file path).
                    Overrides --exclude-cols.
    --models        benchmark only: comma-sep model aliases or checkpoints to compare.
                    Default: longformer,clinicallongformer,bigbird,clinicalbigbird

Notes:
    - ``text`` only converts tabular features to fused text (no model training).
    - ``run`` fine-tunes a long-context model on the fused narrative + tabular text.
    - ``benchmark`` times one training epoch across multiple model families on the
      current device (MPS / CPU / CUDA) and prints a comparison table.
      Use this to decide which model family to run HPO on your hardware.

MPS (Apple Silicon) guidance:
    BigBird is the recommended family for local M-chip runs.  train() auto-detects MPS
    and sets attention_type="original_full" so BigBird uses standard dense attention
    (MPS-native, no PYTORCH_ENABLE_MPS_FALLBACK required).
    Longformer requires PYTORCH_ENABLE_MPS_FALLBACK=1 and is often slower than CPU on
    MPS due to scatter/gather ops falling back to CPU with continuous tensor copies.
"""

from __future__ import annotations

import argparse

from demo_utils import (
    ensure_dir,
    load_data,
    parse_exclude_cols,
    print_prediction_summary,
)


# Long-context model families available for data fusion.
# Keys are display names; values are the model_name strings passed to DataFusionClassifier.
_BENCHMARK_MODELS: dict[str, str] = {
    "longformer":         "allenai/longformer-base-4096",
    "clinicallongformer": "yikuan8/Clinical-Longformer",
    "bigbird":            "google/bigbird-roberta-base",
    "clinicalbigbird":    "yikuan8/Clinical-BigBird",
}


def preview_fused_text(args: argparse.Namespace) -> None:
    """Show how tabular features become extra narrative text."""
    from multimodalva.ensemble.data_fusion import build_fused_text

    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features, n_samples=8,
    )
    print(f"Using {len(feat)} tabular feature columns.")
    fused = build_fused_text(
        df=df, text_col=args.text_col, feature_cols=feat, group_symptoms=True,
    )
    preview = df[[args.label]].copy()
    preview["fused_text"] = fused
    print(preview.head(4).to_string(index=False))


def run_classifier(args: argparse.Namespace) -> None:
    """End-to-end data-fusion pipeline, with optional HPO."""
    from multimodalva.ensemble import DataFusionClassifier

    use_hpo = args.mode == "hpo" or args.hpo
    n_samples = 60 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features, n_samples=n_samples or 60,
    )

    print(f"Using {len(feat)} tabular feature columns.")
    clf = DataFusionClassifier(model_name=args.model, output_dir=ensure_dir(args.output_dir))
    run_kwargs: dict = dict(
        df=df,
        text_col=args.text_col,
        feature_cols=feat,
        label_col=args.label,
        group_symptoms=True,
        use_fast=True,
        max_length=args.max_length,
        use_optimize=use_hpo,
        use_lora=True,
        resume_training=args.resume,
        batch_size=4,
        top_k=args.top_k,
    )
    if use_hpo:
        run_kwargs.update(
            n_trials=args.n_trials,
            optimize_metric=args.metric,
            resume_hpo=args.resume,
            use_cv=args.n_cv_folds > 0,
            n_cv_folds=args.n_cv_folds,
        )
    else:
        run_kwargs["hyperparams"] = {"epochs": 1, "batch_size": 2, "learning_rate": 2e-5}

    results = clf.run(**run_kwargs)
    if use_hpo:
        print("Best hyperparameters:", results["best_hyperparams"])
    print_prediction_summary(results["predictions"], f"DataFusionClassifier ({args.model})")


def run_benchmark(args: argparse.Namespace) -> None:
    """Time one training epoch across long-context model families on the current device.

    Runs a tiny synthetic dataset (60 rows, 1 epoch, batch_size=2, no LoRA, no HPO)
    through each requested model family, collects wall-clock time, and prints a
    comparison table.  Use this to pick the fastest model for your hardware before
    launching a full HPO run.

    On MPS (Apple Silicon):
      - BigBird auto-configures to attention_type="original_full" (MPS-native).
      - Longformer warns and may require PYTORCH_ENABLE_MPS_FALLBACK=1.
    """
    import time
    import torch
    from multimodalva.ensemble import DataFusionClassifier

    # Resolve which models to benchmark.
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        model_map: dict[str, str] = {}
        for key in requested:
            if key in _BENCHMARK_MODELS:
                model_map[key] = _BENCHMARK_MODELS[key]
            else:
                # Accept raw HuggingFace IDs / local paths too.
                model_map[key] = key
    else:
        model_map = dict(_BENCHMARK_MODELS)

    # Device info.
    if torch.cuda.is_available():
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_label = "MPS (Apple Silicon)"
    else:
        device_label = "CPU"

    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=60,
        seed=42,
    )
    print(f"Benchmark: {len(df)} rows, {len(feat)} feature columns, device={device_label}")
    print(f"max_length={args.max_length}, epochs=1, batch_size=2, use_lora=False\n")

    fixed_hp = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 2e-5,
        "gradient_accumulation_steps": 1,
        "freeze_layers": 0,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
    }

    results_table: list[dict] = []
    base_dir = args.output_dir or "runs/benchmark_data_fusion"

    for display_name, checkpoint in model_map.items():
        out = ensure_dir(f"{base_dir}/{display_name}")
        print(f"  [{display_name}]  {checkpoint}")
        t0 = time.perf_counter()
        try:
            clf = DataFusionClassifier(model_name=checkpoint, output_dir=out)
            clf.run(
                df=df,
                text_col=args.text_col,
                feature_cols=feat,
                label_col=args.label,
                group_symptoms=True,
                use_fast=True,
                max_length=args.max_length,
                use_optimize=False,
                use_lora=False,
                gradient_checkpointing=False,
                resume_training=False,
                batch_size=4,   # inference batch
                top_k=1,
                hyperparams=fixed_hp,
            )
            elapsed = time.perf_counter() - t0
            status = "ok"
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            status = f"ERROR: {exc}"
            print(f"    -> {status}")

        results_table.append({
            "model": display_name,
            "checkpoint": checkpoint,
            "elapsed_s": elapsed,
            "status": status,
        })
        print(f"    elapsed: {elapsed:.1f}s  status: {status}\n")

    # Print comparison table.
    print("=" * 72)
    print(f"Benchmark summary — device: {device_label}")
    print(f"{'Model':<22} {'Elapsed (s)':>12}  {'Status'}")
    print("-" * 72)
    ok_rows = [r for r in results_table if r["status"] == "ok"]
    fastest = min(ok_rows, key=lambda r: r["elapsed_s"])["elapsed_s"] if ok_rows else None
    for r in results_table:
        marker = ""
        if r["status"] == "ok" and fastest is not None:
            ratio = r["elapsed_s"] / fastest
            marker = f"  (×{ratio:.1f})" if ratio > 1.05 else "  ← fastest"
        print(f"{r['model']:<22} {r['elapsed_s']:>12.1f}  {r['status']}{marker}")
    print("=" * 72)

    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        print(
            "\nMPS note: BigBird models use attention_type='original_full' (MPS-native).\n"
            "Longformer models require PYTORCH_ENABLE_MPS_FALLBACK=1 and may be\n"
            "slower than CPU due to MPS↔CPU tensor copies for scatter/gather ops."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="text",
        choices=["text", "run", "hpo", "benchmark"],
        help=(
            "Pipeline variant: text=preview fused text, run=train, "
            "hpo=train+HPO, benchmark=time model families (default: text)."
        ),
    )
    # Data
    parser.add_argument("--data", default="demo",
                        help='CSV path or "demo" for synthetic data (default: demo).')
    parser.add_argument("--label", default="cause",
                        help="Label column name (default: cause).")
    parser.add_argument("--text-col", default="narrative", dest="text_col",
                        help="Text/narrative column name (default: narrative).")
    parser.add_argument("--exclude-cols", default=None, dest="exclude_cols",
                        # --label and --text-col are always excluded; every other column
                        # becomes a tabular feature unless listed here.
                        help="Comma-separated columns to exclude from tabular features (e.g. 'record_id,date'). "
                             "Every column except --label and --text-col is automatically a feature.")
    parser.add_argument("--features", default=None,
                        help="Advanced: explicit feature columns (comma-sep or file path). Overrides --exclude-cols.")
    # Model (run / hpo modes)
    parser.add_argument("--model", default="allenai/longformer-base-4096",
                        help="Long-context LM checkpoint or alias (default: longformer-base-4096).")
    parser.add_argument("--max-length", type=int, default=256, dest="max_length",
                        help="Max token length for fused text (default: 256).")
    # Benchmark mode
    parser.add_argument(
        "--models", default=None,
        help=(
            "benchmark only: comma-sep model aliases or checkpoints to compare "
            "(default: longformer,clinicallongformer,bigbird,clinicalbigbird). "
            "Aliases: " + ", ".join(_BENCHMARK_MODELS.keys())
        ),
    )
    # HPO / training
    parser.add_argument("--hpo", action="store_true",
                        help="Enable Optuna HPO (also activated by mode=hpo).")
    parser.add_argument("--n-trials", type=int, default=2, dest="n_trials",
                        help="HPO trials (default: 2; use 30 for real data).")
    parser.add_argument("--n-cv-folds", type=int, default=3, dest="n_cv_folds",
                        help="CV folds for HPO scoring (default: 3; use 0 to disable CV).")
    parser.add_argument("--metric", default="f1_macro",
                        help="HPO optimisation metric (default: f1_macro).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume HPO / training from checkpoint.")
    # Output
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Output directory (default: runs/demo_data_fusion for run/hpo; "
                             "runs/benchmark_data_fusion for benchmark).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Number of top-k predictions (default: 3).")

    args = parser.parse_args()

    # Apply per-mode output_dir defaults.
    if args.output_dir is None:
        if args.mode == "benchmark":
            args.output_dir = "runs/benchmark_data_fusion"
        else:
            args.output_dir = "runs/demo_data_fusion"

    if args.mode == "text":
        preview_fused_text(args)
    elif args.mode == "benchmark":
        run_benchmark(args)
    else:
        run_classifier(args)


if __name__ == "__main__":
    main()
