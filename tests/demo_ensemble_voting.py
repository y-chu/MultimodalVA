"""Demos for the soft-voting ensemble pipeline.

Usage (synthetic demo data):
    python tests/demo_ensemble_voting.py synthetic
    python tests/demo_ensemble_voting.py tabular
    python tests/demo_ensemble_voting.py mixed

Usage (your own CSV — tabular and mixed modes):
    python tests/demo_ensemble_voting.py tabular \\
        --data path/to/data.csv --label cause

    python tests/demo_ensemble_voting.py mixed \\
        --data path/to/data.csv --label cause --text-col narrative \\
        --exclude-cols record_id --output-dir runs/my_voting

Key flags:
    --data          CSV file path, or "demo" for synthetic data (default: demo)
    --label         Label column name (default: cause)
    --text-col      Text column for mixed mode (default: narrative)
    --exclude-cols        Comma-separated columns to exclude from tabular features
                    (e.g. ID, date).  Label and text columns are always excluded.
    --output-dir    Where to save model artifacts (default: auto per mode)
    --top-k         Number of top-k predictions to return (default: 3)
    --features      Advanced: explicit feature columns (comma-sep or file path).
                    Overrides --exclude-cols.

Notes:
    - ``synthetic`` averages pre-built PredictionResult objects (no training).
    - ``tabular`` uses only sklearn-style base models.
    - ``mixed`` adds a text model and may download transformer weights.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from demo_utils import ensure_dir, load_data, parse_exclude_cols, print_prediction_summary


ID2LABEL = {0: "HIV/AIDS", 1: "Pneumonia", 2: "Traffic / transport accident"}


def _synthetic_result(name: str, seed: int):
    from multimodalva.utils.types import PredictionResult

    rng = np.random.default_rng(seed)
    true_ids = rng.choice([0, 1, 2], size=18, p=[0.4, 0.35, 0.25])
    probs = np.zeros((len(true_ids), len(ID2LABEL)), dtype=float)
    for i, true_id in enumerate(true_ids):
        row = rng.dirichlet(np.ones(len(ID2LABEL)))
        row[true_id] += 1.5 if name == "tabular" else 0.9
        probs[i] = row / row.sum()

    true_labels = [ID2LABEL[int(i)] for i in true_ids]
    full = pd.DataFrame({"true_label": true_labels})
    for class_id in sorted(ID2LABEL):
        full[f"prob_{class_id}"] = probs[:, class_id]

    pred_ids = probs.argmax(axis=1)
    top1 = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": [ID2LABEL[int(i)] for i in pred_ids],
        "predicted_prob": probs.max(axis=1),
    })
    topk = pd.DataFrame({
        "true_label": true_labels,
        "top1_label": [ID2LABEL[int(i)] for i in pred_ids],
        "top1_prob": probs.max(axis=1),
    })
    return PredictionResult(top1=top1, full=full, topk=topk, id2label=ID2LABEL)


def synthetic_vote(args: argparse.Namespace) -> None:
    """Average a few pre-built PredictionResult objects (no training)."""
    from multimodalva.ensemble.voting import vote_from_results

    results = [
        _synthetic_result("tabular", 0),
        _synthetic_result("text", 1),
        _synthetic_result("text", 2),
    ]
    voted = vote_from_results(results, id2label=ID2LABEL, weights=[0.5, 0.25, 0.25], top_k=args.top_k)
    print_prediction_summary(voted, "vote_from_results() synthetic soft vote")


def tabular_vote(args: argparse.Namespace) -> None:
    """Soft voting with sklearn-style tabular base models only."""
    from multimodalva.ensemble.voting import SoftVotingClassifier

    n_samples = 140 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=n_samples or 140, seed=21,
    )
    print(f"Training on {len(feat)} feature columns.")
    out = args.output_dir or "runs/demo_voting_tabular"
    clf = SoftVotingClassifier(
        text_models=[],
        tabular_models=[
            {"model_name": "random_forest", "hyperparams": {"n_estimators": 80, "max_depth": 6}},
            {"model_name": "gbdt", "hyperparams": {"n_estimators": 80, "max_depth": 3}},
        ],
        output_dir=ensure_dir(out),
    )
    results = clf.run(df=df, label_col=args.label, feature_cols=feat, top_k=args.top_k)
    print_prediction_summary(results["predictions"], "SoftVotingClassifier tabular-only")


def mixed_vote(args: argparse.Namespace) -> None:
    """Soft voting with text + tabular base models via EnsembleClassifier."""
    from multimodalva.ensemble import EnsembleClassifier

    n_samples = 80 if args.data == "demo" else None
    df, feat = load_data(
        args.data, args.label, text_col=args.text_col,
        exclude_cols=parse_exclude_cols(args.exclude_cols),
        features_arg=args.features,
        n_samples=n_samples or 80, seed=22,
    )
    print(f"Training on {len(feat)} tabular feature columns.")
    out = args.output_dir or "runs/demo_voting_mixed"
    clf = EnsembleClassifier(
        method="soft_voting",
        output_dir=ensure_dir(out),
        text_models=[{
            "model_name": "emilyalsentzer/Bio_ClinicalBERT",
            "hyperparams": {"epochs": 1, "batch_size": 4, "learning_rate": 2e-5},
        }],
        tabular_models=[
            {"model_name": "random_forest", "hyperparams": {"n_estimators": 80, "max_depth": 6}},
        ],
    )
    results = clf.run(
        df=df, text_col=args.text_col, feature_cols=feat,
        label_col=args.label, batch_size=8, top_k=args.top_k,
    )
    print_prediction_summary(results["predictions"], "EnsembleClassifier(method='soft_voting')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="synthetic",
        choices=["synthetic", "tabular", "mixed"],
        help="Voting variant to demonstrate (default: synthetic).",
    )
    # Data
    parser.add_argument("--data", default="demo",
                        help='CSV path or "demo" for synthetic data (default: demo).')
    parser.add_argument("--label", default="cause",
                        help="Label column name (default: cause).")
    parser.add_argument("--text-col", default="narrative", dest="text_col",
                        help="Text column for mixed mode (default: narrative).")
    parser.add_argument("--exclude-cols", default=None, dest="exclude_cols",
                        # --label and --text-col are always excluded; every other column
                        # becomes a tabular feature unless listed here.
                        help="Comma-separated columns to exclude from tabular features (e.g. 'record_id,date'). "
                             "Every column except --label and --text-col is automatically a feature.")
    parser.add_argument("--features", default=None,
                        help="Advanced: explicit feature columns (comma-sep or file path). Overrides --exclude-cols.")
    # Output
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Output directory (default: auto per mode).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Number of top-k predictions (default: 3).")

    args = parser.parse_args()

    if args.mode == "synthetic":
        synthetic_vote(args)
    elif args.mode == "tabular":
        tabular_vote(args)
    else:
        mixed_vote(args)


if __name__ == "__main__":
    main()
