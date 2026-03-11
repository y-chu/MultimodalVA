"""
Results subpackage: training diagnostics and prediction visualization.

Sub-modules:
    - vis_train:   Training/HPO diagnostics
                   (plot_loss_curves, hpo_leaderboard)
    - vis_predict: Prediction summary and visualization
                   (performance_leaderboard, topk_from_full,
                    topk_accuracy, cause_accuracy_heatmap,
                    cause_accuracy_diff_heatmap, confusion_heatmap)
"""

from .vis_train import plot_loss_curves, hpo_leaderboard
from .vis_predict import (
    performance_leaderboard,
    topk_from_full,
    topk_accuracy,
    cause_accuracy_heatmap,
    cause_accuracy_diff_heatmap,
    confusion_heatmap,
)

__all__ = [
    "plot_loss_curves",
    "hpo_leaderboard",
    "performance_leaderboard",
    "topk_from_full",
    "topk_accuracy",
    "cause_accuracy_heatmap",
    "cause_accuracy_diff_heatmap",
    "confusion_heatmap",
]
