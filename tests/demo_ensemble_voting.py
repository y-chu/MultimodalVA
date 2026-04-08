"""Small demos for soft voting.

Usage:
    python tests/demo_ensemble_voting.py synthetic
    python tests/demo_ensemble_voting.py tabular
    python tests/demo_ensemble_voting.py mixed

Notes:
    - ``synthetic`` requires no model training and is the safest starting point.
    - ``tabular`` uses only sklearn-style base models.
    - ``mixed`` adds a text model and may download transformer weights.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from demo_utils import ensure_dir, feature_cols, make_demo_va_df, print_prediction_summary


ID2LABEL = {0: "HIV/AIDS", 1: "Pneumonia", 2: "Traffic / transport accident"}


def _synthetic_result(name: str, seed: int) -> PredictionResult:
    """Create a small PredictionResult for vote_from_results()."""
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
    top1 = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": [ID2LABEL[int(i)] for i in pred_ids],
            "predicted_prob": probs.max(axis=1),
        }
    )
    topk = pd.DataFrame(
        {
            "true_label": true_labels,
            "top1_label": [ID2LABEL[int(i)] for i in pred_ids],
            "top1_prob": probs.max(axis=1),
        }
    )
    return PredictionResult(top1=top1, full=full, topk=topk, id2label=ID2LABEL)


def synthetic_vote() -> None:
    """Average a few saved-like prediction objects."""
    from multimodalva.ensemble.voting import vote_from_results

    results = [
        _synthetic_result("tabular", 0),
        _synthetic_result("text", 1),
        _synthetic_result("text", 2),
    ]
    voted = vote_from_results(results, id2label=ID2LABEL, weights=[0.5, 0.25, 0.25], top_k=3)
    print_prediction_summary(voted, "vote_from_results() synthetic soft vote")


def tabular_vote() -> None:
    """Run soft voting with only sklearn-style tabular base models."""
    from multimodalva.ensemble.voting import SoftVotingClassifier

    df = make_demo_va_df(n_samples=140, seed=21)
    clf = SoftVotingClassifier(
        text_models=[],
        tabular_models=[
            {"model_name": "random_forest", "hyperparams": {"n_estimators": 80, "max_depth": 6}},
            {"model_name": "gbdt", "hyperparams": {"n_estimators": 80, "max_depth": 3}},
        ],
        output_dir=ensure_dir("runs/demo_voting_tabular"),
    )
    results = clf.run(
        df=df,
        label_col="cause",
        feature_cols=feature_cols(),
        top_k=3,
    )
    print_prediction_summary(results["predictions"], "SoftVotingClassifier tabular-only")


def mixed_vote() -> None:
    """Run soft voting through the unified ensemble wrapper with text + tabular models."""
    from multimodalva.ensemble import EnsembleClassifier

    df = make_demo_va_df(n_samples=80, seed=22)
    clf = EnsembleClassifier(
        method="soft_voting",
        output_dir=ensure_dir("runs/demo_voting_mixed"),
        text_models=[
            {
                "model_name": "emilyalsentzer/Bio_ClinicalBERT",
                "hyperparams": {"epochs": 1, "batch_size": 4, "learning_rate": 2e-5},
            }
        ],
        tabular_models=[
            {"model_name": "random_forest", "hyperparams": {"n_estimators": 80, "max_depth": 6}},
        ],
    )
    results = clf.run(
        df=df,
        text_col="narrative",
        feature_cols=feature_cols(),
        label_col="cause",
        batch_size=8,
        top_k=3,
    )
    print_prediction_summary(results["predictions"], "EnsembleClassifier(method='soft_voting')")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="synthetic",
        choices=["synthetic", "tabular", "mixed"],
        help="Which demo to run.",
    )
    args = parser.parse_args()

    if args.mode == "synthetic":
        synthetic_vote()
    elif args.mode == "tabular":
        tabular_vote()
    else:
        mixed_vote()


if __name__ == "__main__":
    main()
