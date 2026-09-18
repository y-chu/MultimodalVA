"""
Results subpackage: training diagnostics and prediction visualization.

Sub-modules:
    - palettes:    Color constants (TOPK_BAR_COLORS, HEATMAP_SEQ,
                    HEATMAP_DIV, HEATMAP_CLINICAL)
    - vis_train:   Training/HPO diagnostics
                   (plot_loss_curves, hpo_leaderboard,
                    oov_rate, hpo_loss_trend, hpo_convergence_plot,
                    hpo_metric_variance,
                    hyperparameter_importance, plot_param_importances,
                    train_eval_gap, loss_curve_diagnostics)
    - bootstrap:   Bootstrap confidence intervals
                   (bootstrap_ci, paired_bootstrap_ci, predictions_frame)
    - vis_predict: Prediction summary and visualization
                   (performance_leaderboard, topk_from_full,
                    topk_accuracy, plot_topk_accuracy,
                    cause_accuracy_heatmap,
                    cause_accuracy_diff_heatmap, csmf_scatterplot,
                    confusion_heatmap)
"""

from .palettes import (
    TOPK_BAR_COLORS,
    HEATMAP_SEQ,
    HEATMAP_DIV,
    HEATMAP_CLINICAL,
)
from .vis_train import (
    plot_loss_curves,
    hpo_leaderboard,
    oov_rate,
    hpo_loss_trend,
    hpo_convergence_plot,
    hpo_metric_variance,
    hyperparameter_importance,
    plot_param_importances,
    train_eval_gap,
    loss_curve_diagnostics,
)
from .calibration import (
    classwise_bin_data,
    ece_score,
    mce_score,
    brier_multiclass,
    calibration_summary,
)
from .bootstrap import (
    bootstrap_ci,
    paired_bootstrap_ci,
    predictions_frame,
)
from .vis_predict import (
    performance_leaderboard,
    topk_from_full,
    topk_accuracy,
    plot_topk_accuracy,
    cause_accuracy_heatmap,
    cause_accuracy_diff_heatmap,
    csmf_scatterplot,
    confusion_heatmap,
)

__all__ = [
    "classwise_bin_data",
    "ece_score",
    "mce_score",
    "brier_multiclass",
    "calibration_summary",
    # bootstrap
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "predictions_frame",
    # palettes
    "TOPK_BAR_COLORS",
    "HEATMAP_SEQ",
    "HEATMAP_DIV",
    "HEATMAP_CLINICAL",
    # vis_train
    "plot_loss_curves",
    "hpo_leaderboard",
    "oov_rate",
    "hpo_loss_trend",
    "hpo_convergence_plot",
    "hpo_metric_variance",
    "hyperparameter_importance",
    "plot_param_importances",
    "train_eval_gap",
    "loss_curve_diagnostics",
    # vis_predict
    "performance_leaderboard",
    "topk_from_full",
    "topk_accuracy",
    "plot_topk_accuracy",
    "cause_accuracy_heatmap",
    "cause_accuracy_diff_heatmap",
    "csmf_scatterplot",
    "confusion_heatmap",
]
