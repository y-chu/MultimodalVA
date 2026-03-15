"""
Results subpackage: training diagnostics and prediction visualization.

Sub-modules:
    - vis_train:   Training/HPO diagnostics
                   (plot_loss_curves, hpo_leaderboard,
                    oov_rate, hpo_loss_trend, hpo_convergence_plot,
                    hpo_metric_variance,
                    hyperparameter_importance, train_eval_gap,
                    loss_curve_diagnostics)
    - vis_predict: Prediction summary and visualization
                   (performance_leaderboard, topk_from_full,
                    topk_accuracy, cause_accuracy_heatmap,
                    cause_accuracy_diff_heatmap, confusion_heatmap)
"""

from .vis_train import (
    plot_loss_curves,
    hpo_leaderboard,
    oov_rate,
    hpo_loss_trend,
    hpo_convergence_plot,
    hpo_metric_variance,
    hyperparameter_importance,
    train_eval_gap,
    loss_curve_diagnostics,
)
from .vis_predict import (
    performance_leaderboard,
    topk_from_full,
    topk_accuracy,
    cause_accuracy_heatmap,
    cause_accuracy_diff_heatmap,
    confusion_heatmap,
)

__all__ = [
    # vis_train
    "plot_loss_curves",
    "hpo_leaderboard",
    "oov_rate",
    "hpo_loss_trend",
    "hpo_convergence_plot",
    "hpo_metric_variance",
    "hyperparameter_importance",
    "train_eval_gap",
    "loss_curve_diagnostics",
    # vis_predict
    "performance_leaderboard",
    "topk_from_full",
    "topk_accuracy",
    "cause_accuracy_heatmap",
    "cause_accuracy_diff_heatmap",
    "confusion_heatmap",
]
