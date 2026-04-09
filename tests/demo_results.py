"""Demos for result tables and visualizations.

Usage:
    python tests/demo_results.py leaderboard
    python tests/demo_results.py topk
    python tests/demo_results.py heatmaps

Key flags:
    --output-dir    Where to save figure files (default: runs/demo_results)

All demos use synthetic prediction objects to illustrate result utilities.
Figures are saved under ``<output-dir>/figures/``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from multimodalva.results import (
    cause_accuracy_diff_heatmap,
    cause_accuracy_heatmap,
    confusion_heatmap,
    performance_leaderboard,
    plot_topk_accuracy,
    topk_from_full,
)

from demo_utils import CAUSES, ensure_dir


def _make_predictions(n_samples: int = 60, seed: int = 60):
    rng = np.random.default_rng(seed)
    active_causes = CAUSES[:3]
    id2label = {i: label for i, label in enumerate(active_causes)}
    true_ids = rng.choice(list(id2label), size=n_samples, p=[0.4, 0.35, 0.25])
    true_labels = [id2label[int(i)] for i in true_ids]

    probs_a = np.array([rng.dirichlet([6 if j == y else 1.2 for j in range(3)]) for y in true_ids])
    probs_b = np.array([rng.dirichlet([4 if j == y else 1.4 for j in range(3)]) for y in true_ids])
    probs_c = 0.55 * probs_a + 0.45 * probs_b

    def build_full(probs: np.ndarray) -> pd.DataFrame:
        full = pd.DataFrame({"true_label": true_labels})
        for class_id in sorted(id2label):
            full[f"prob_{class_id}"] = probs[:, class_id]
        return full

    full_a = build_full(probs_a)
    full_b = build_full(probs_b)
    full_c = build_full(probs_c)

    pred_df = pd.DataFrame({
        "true_label": true_labels,
        "tabular_model": [id2label[int(i)] for i in probs_a.argmax(axis=1)],
        "text_model": [id2label[int(i)] for i in probs_b.argmax(axis=1)],
        "ensemble_model": [id2label[int(i)] for i in probs_c.argmax(axis=1)],
    })
    topk_dfs = {
        "tabular_model": topk_from_full(full_a, id2label, k=3),
        "text_model": topk_from_full(full_b, id2label, k=3),
        "ensemble_model": topk_from_full(full_c, id2label, k=3),
    }
    prob_dfs = {
        "tabular_model": full_a,
        "text_model": full_b,
        "ensemble_model": full_c,
    }
    return pred_df, prob_dfs, topk_dfs


def leaderboard_demo(args: argparse.Namespace) -> None:
    pred_df, prob_dfs, topk_dfs = _make_predictions()
    board = performance_leaderboard(
        df=pred_df,
        true_col="true_label",
        metrics=["accuracy", "balanced_accuracy", "f1_macro", "log_loss"],
        prob_dfs=prob_dfs,
        id2label={0: CAUSES[0], 1: CAUSES[1], 2: CAUSES[2]},
        topk_dfs=topk_dfs,
        top_k=3,
        sort_by="f1_macro",
    )
    print(board.round(2).to_string())


def topk_demo(args: argparse.Namespace) -> None:
    _, _, topk_dfs = _make_predictions()
    fig_dir = ensure_dir(f"{args.output_dir}/figures")
    fig, _ = plot_topk_accuracy(
        data=topk_dfs,
        kind="grouped",
        max_k=3,
        title="Synthetic top-k accuracy",
        save_path=str(fig_dir / "topk_accuracy.png"),
    )
    print(f"Saved top-k accuracy plot to {fig_dir / 'topk_accuracy.png'}")
    fig.clear()


def heatmap_demo(args: argparse.Namespace) -> None:
    pred_df, _, topk_dfs = _make_predictions()
    fig_dir = ensure_dir(f"{args.output_dir}/figures")

    _, _, acc_df = cause_accuracy_heatmap(
        df=pred_df,
        true_col="true_label",
        model_cols=["tabular_model", "text_model", "ensemble_model"],
        top_k=1,
        group_boundaries=[(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
        group_labels=["Tabular", "Text", "Ensemble"],
        save_path=str(fig_dir / "cause_accuracy_heatmap.png"),
    )
    print("Cause accuracy matrix:")
    print(acc_df.round(1).to_string())

    confusion_heatmap(
        y_true=pred_df["true_label"],
        y_pred=pred_df["ensemble_model"],
        model_name="ensemble_model",
        save_path=str(fig_dir / "confusion_heatmap.png"),
    )
    cause_accuracy_diff_heatmap(
        df=pred_df,
        true_col="true_label",
        baseline_col="text_model",
        model_cols=["tabular_model", "ensemble_model"],
        group_boundaries=[(0.5, 1.5), (1.5, 2.5)],
        group_labels=["Tabular", "Ensemble"],
        save_path=str(fig_dir / "cause_accuracy_diff_heatmap.png"),
    )
    print(f"Saved heatmaps to {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="leaderboard",
        choices=["leaderboard", "topk", "heatmaps"],
        help="Which result demo to run (default: leaderboard).",
    )
    parser.add_argument("--output-dir", default="runs/demo_results", dest="output_dir",
                        help="Directory for saved figures (default: runs/demo_results).")

    args = parser.parse_args()

    if args.mode == "leaderboard":
        leaderboard_demo(args)
    elif args.mode == "topk":
        topk_demo(args)
    else:
        heatmap_demo(args)


if __name__ == "__main__":
    main()
