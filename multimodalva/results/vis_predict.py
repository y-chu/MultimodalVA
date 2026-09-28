"""
Result summary and visualization for predictions.

Public API
----------
    performance_leaderboard(df, true_col, model_cols, metrics, sort_by,
                            topk_dfs, top_k)
        Multi-model metric table from a wide prediction DataFrame.
        Pass ``topk_dfs`` and ``top_k`` (default 3) to append
        ``top2_accuracy`` and ``top3_accuracy`` columns.

    hpo_leaderboard(source, sort_by, top_n, ascending)
        Formatted HPO trial table from an Optuna study or trials DataFrame.

    topk_from_full(full_df, id2label, k)
        Extract top-k labels and their probabilities from a full probability
        DataFrame (result.full).  Useful for re-deriving top-k with a
        different k, or for any raw probability matrix.

    topk_accuracy(topk_df, max_k)
        Accuracy at k=1..K: true label is among the top-k predicted causes.

    plot_topk_accuracy(data, max_k, kind, ncols, ...)
        Bar chart of top-k accuracy.  Single-model (one bar per k) or
        multi-model in ``"grouped"`` (bundles by k) or ``"facet"``
        (one subplot per model, 2-column grid by default) layout.
        Input: ``result.topk`` or ``dict[model_name, topk_df]``.
        Returns ``(fig, ax)`` for single/grouped, ``(fig, axes)`` for facet.

    cause_accuracy_heatmap(df, true_col, model_cols, top_k, topk_dfs, ...)
        Cause-specific accuracy heatmap across selected models.
        Rows = causes, columns = models, cells = % correct.
        Supports top-1 (default) or top-k accuracy; optional cause ordering,
        renaming, filtering, and column grouping annotations.

    cause_accuracy_diff_heatmap(df, true_col, baseline_col, model_cols, ...)
        Same layout as cause_accuracy_heatmap but cells show the difference
        accuracy(model) − accuracy(baseline).  Diverging colormap centred at 0.
        Optional include_baseline column (zeros) to keep baseline label visible.

    csmf_scatterplot(y_true, y_pred, ...)
        Scatter plot of true vs. modelled CSMF for one model.
        Each cause is shown as a labelled point with a 45-degree equality line.

    confusion_heatmap(y_true, y_pred, label_order, ...)
        Seaborn heatmap of predicted vs. true labels for one model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..utils.metrics import log_loss_from_full, REPORT_METRICS
from .palettes import TOPK_BAR_COLORS, HEATMAP_SEQ, HEATMAP_DIV, HEATMAP_CLINICAL

logger = logging.getLogger(__name__)

# Metrics that require a full probability DataFrame rather than predicted labels.
_PROB_METRICS: frozenset[str] = frozenset({"log_loss"})

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

def _make_metric_registry() -> dict:
    """Build the supported metric name → callable map (lazy sklearn import)."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from ..utils.metrics import csmf_accuracy, cccsmf_accuracy

    return {
        "accuracy":           lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy":  lambda yt, yp: balanced_accuracy_score(yt, yp),
        "f1_macro":           lambda yt, yp: f1_score(yt, yp, average="macro",     zero_division=0),
        "f1_weighted":        lambda yt, yp: f1_score(yt, yp, average="weighted",  zero_division=0),
        "precision_macro":    lambda yt, yp: precision_score(yt, yp, average="macro",    zero_division=0),
        "precision_weighted": lambda yt, yp: precision_score(yt, yp, average="weighted", zero_division=0),
        "recall_macro":       lambda yt, yp: recall_score(yt, yp, average="macro",    zero_division=0),
        "recall_weighted":    lambda yt, yp: recall_score(yt, yp, average="weighted", zero_division=0),
        "csmf_accuracy":      csmf_accuracy,
        "cccsmf_accuracy":    cccsmf_accuracy,
    }



# ---------------------------------------------------------------------------
# 1. Performance leaderboard
# ---------------------------------------------------------------------------

def performance_leaderboard(
    df: pd.DataFrame,
    true_col: str,
    model_cols: list[str] | None = None,
    metrics: list[str] | None = None,
    sort_by: str = "f1_macro",
    ascending: bool = False,
    percentage: bool = True,
    prob_dfs: "dict[str, pd.DataFrame] | None" = None,
    id2label: dict | None = None,
    topk_dfs: "dict[str, pd.DataFrame] | None" = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Build a multi-model performance leaderboard from a wide prediction DataFrame.

    The input ``df`` should have one column of true labels and one column per
    model's predicted labels::

        id | true_label | pred_bert | pred_lightgbm | pred_ensemble
        -- | ---------- | --------- | ------------- | -------------
        0  | Malaria    | Malaria   | Pneumonia     | Malaria
        1  | HIV/AIDS   | HIV/AIDS  | HIV/AIDS      | HIV/AIDS

    Metrics are computed for each model column against ``true_col`` and returned
    as a single ranked DataFrame.

    To include ``"log_loss"`` (case-level calibration), pass ``prob_dfs`` and
    ``id2label`` alongside the prediction DataFrame::

        prob_dfs = {
            "pred_bert":     bert_result.full,
            "pred_lightgbm": lgbm_result.full,
        }
        board = performance_leaderboard(
            df, "true_label",
            metrics=[*REPORT_METRICS, "log_loss"],
            prob_dfs=prob_dfs,
            id2label=bert_result.id2label,
            sort_by="log_loss", ascending=True,
        )

    To include top-k accuracy columns (``top2_accuracy``, ``top3_accuracy``, …),
    pass ``topk_dfs`` with a topk DataFrame per model (from ``result.topk`` or
    :func:`topk_from_full`) and set ``top_k``::

        topk_dfs = {
            "pred_bert":     bert_result.topk,
            "pred_lightgbm": topk_from_full(lgbm_result.full, lgbm_result.id2label, k=3),
        }
        board = performance_leaderboard(
            df, "true_label",
            topk_dfs=topk_dfs,
            top_k=3,   # adds top2_accuracy and top3_accuracy columns
        )

    Args:
        df:          DataFrame containing true labels and one or more model
                     prediction columns.
        true_col:    Name of the ground-truth label column.
        model_cols:  Columns to evaluate.  If ``None``, all columns except
                     ``true_col`` and ``"id"`` are used.
        metrics:     Metrics to compute.  Default: all 10 label-based metrics —
                     ``accuracy``, ``balanced_accuracy``,
                     ``f1_macro``, ``f1_weighted``,
                     ``precision_macro``, ``precision_weighted``,
                     ``recall_macro``, ``recall_weighted``,
                     ``csmf_accuracy``, ``cccsmf_accuracy``.
                     Add ``"log_loss"`` explicitly to include calibration (requires
                     ``prob_dfs`` and ``id2label``).
        sort_by:     Metric to sort the leaderboard by.  Default ``"f1_macro"``.
                     Use ``ascending=True`` when sorting by ``"log_loss"``.
                     Also accepts ``"top2_accuracy"``, ``"top3_accuracy"``, etc.
                     when ``topk_dfs`` is provided.
        ascending:   Sort direction.  Default ``False`` (highest first).
        percentage:  Multiply label-based metric values by 100 for readability.
                     Default True.  **Not applied to** ``log_loss`` (it is not
                     bounded to [0, 1]).
        prob_dfs:    Dict mapping model column name → ``result.full`` DataFrame
                     (columns: ``true_label``, ``prob_0``, ``prob_1``, …).
                     Required when ``"log_loss"`` is in ``metrics``.
        id2label:    Integer class ID → label string mapping shared across all
                     models.  Required when ``"log_loss"`` is in ``metrics``.
        topk_dfs:    Dict mapping model column name → topk DataFrame
                     (``result.topk`` or output of :func:`topk_from_full`).
                     When provided, columns ``top2_accuracy`` through
                     ``top{top_k}_accuracy`` are appended to the leaderboard.
                     Models absent from ``topk_dfs`` will show ``NaN`` for these
                     columns.
        top_k:       Highest rank to include in top-k accuracy columns.
                     Default 3 → adds ``top2_accuracy`` and ``top3_accuracy``.
                     Set to 1 to skip top-k columns even when ``topk_dfs`` is
                     provided.  Ignored when ``topk_dfs`` is ``None``.

    Returns:
        DataFrame indexed by model name with one column per metric, sorted by
        ``sort_by``.  Top-k accuracy columns (if any) are appended after the
        requested ``metrics`` columns.

    Raises:
        ValueError: If ``true_col`` is not in ``df`` or an unknown metric is
                    requested.
    """
    if true_col not in df.columns:
        raise ValueError(f"true_col {true_col!r} not found in DataFrame columns.")

    if model_cols is None:
        model_cols = [c for c in df.columns if c not in {true_col, "id"}]
    if not model_cols:
        raise ValueError("No model columns found.  Pass model_cols explicitly.")

    metrics = metrics or list(REPORT_METRICS)
    registry = _make_metric_registry()

    label_metrics = [m for m in metrics if m not in _PROB_METRICS]
    prob_metrics  = [m for m in metrics if m in _PROB_METRICS]

    unknown = [m for m in label_metrics if m not in registry]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}.  "
            f"Available label-based: {list(registry)}.  "
            f"Probability-based (requires prob_dfs + id2label): {sorted(_PROB_METRICS)}."
        )

    # Top-k accuracy column names that will be appended (k=2..top_k)
    topk_metric_names: list[str] = (
        [f"top{k}_accuracy" for k in range(2, top_k + 1)]
        if topk_dfs is not None and top_k >= 2
        else []
    )

    all_valid_sort_targets = set(metrics) | set(topk_metric_names)
    if sort_by not in all_valid_sort_targets:
        raise ValueError(
            f"sort_by={sort_by!r} must be in the requested metrics list or a "
            f"top-k accuracy column.  Available: {sorted(all_valid_sort_targets)}."
        )

    if prob_metrics and (prob_dfs is None or id2label is None):
        logger.warning(
            "Metrics %s require prob_dfs and id2label — these columns will be NaN.  "
            "Pass prob_dfs={model_col: result.full, ...} and id2label=result.id2label.",
            prob_metrics,
        )

    y_true = df[true_col]
    rows = []
    for col in model_cols:
        y_pred = df[col]
        row: dict = {"model": col, "n": len(y_true)}

        for m in label_metrics:
            try:
                val = registry[m](y_true, y_pred)
                row[m] = round(val * 100, 2) if percentage else round(val, 4)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not compute %s for model %r: %s", m, col, exc)
                row[m] = float("nan")

        for m in prob_metrics:
            if m == "log_loss":
                if prob_dfs is not None and col in prob_dfs and id2label is not None:
                    try:
                        row[m] = round(log_loss_from_full(prob_dfs[col], id2label), 4)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not compute log_loss for model %r: %s", col, exc)
                        row[m] = float("nan")
                else:
                    row[m] = float("nan")

        # Top-k accuracy (k = 2 .. top_k)
        if topk_metric_names:
            if topk_dfs is not None and col in topk_dfs:
                try:
                    acc_df = topk_accuracy(topk_dfs[col], max_k=top_k)
                    # acc_df rows: k=1,2,...,top_k — index by k
                    acc_by_k: dict[int, float] = dict(
                        zip(acc_df["k"].tolist(), acc_df["accuracy"].tolist())
                    )
                    for k in range(2, top_k + 1):
                        metric_name = f"top{k}_accuracy"
                        val = acc_by_k.get(k, float("nan"))
                        row[metric_name] = round(val * 100, 2) if percentage else round(val, 4)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not compute top-k accuracy for model %r: %s", col, exc)
                    for metric_name in topk_metric_names:
                        row[metric_name] = float("nan")
            else:
                for metric_name in topk_metric_names:
                    row[metric_name] = float("nan")

        rows.append(row)

    board = pd.DataFrame(rows).set_index("model")
    # Reorder columns: requested metrics first, then top-k accuracy columns
    col_order = [m for m in metrics if m in board.columns] + topk_metric_names
    board = board[["n"] + col_order].sort_values(sort_by, ascending=ascending)
    return board


# ---------------------------------------------------------------------------
# 3. Top-K extraction from full probability distribution
# ---------------------------------------------------------------------------

def topk_from_full(
    full_df: pd.DataFrame,
    id2label: dict,
    k: int = 3,
) -> pd.DataFrame:
    """Extract the top-k labels and their probabilities from a full probability DataFrame.

    Works directly on ``result.full`` from a ``PredictionResult``, which has one
    ``prob_<id>`` column per class.  Useful when you want to re-derive top-k with a
    different k, when combining predictions from multiple models, or when working
    with any raw probability matrix that follows the same column convention.

    Args:
        full_df:   The ``full`` DataFrame from a ``PredictionResult``.
                   Expected columns::

                       true_label | prob_0 | prob_1 | prob_2 | ...

                   The integer suffix in each ``prob_<id>`` column must match
                   the keys of ``id2label``.  Pass ``result.full`` directly.
        id2label:  Integer class ID → label string mapping.
                   Pass ``result.id2label`` directly.
        k:         Number of top causes to extract per sample.  Must be ≤ the
                   number of classes.  Default 3.

    Returns:
        DataFrame with columns::

            true_label | top1_label | top1_prob | top2_label | top2_prob | ... | topK_label | topK_prob

        Rows correspond 1-to-1 with rows in ``full_df``.  The ``topj_label``
        column contains the label string for the j-th most probable class;
        ``topj_prob`` contains the corresponding probability.

    Raises:
        ValueError: If ``full_df`` contains no ``prob_*`` columns, if ``k``
                    exceeds the number of classes, or if ``id2label`` keys do
                    not match ``full_df`` probability columns.

    Example::

        from multimodalva.results import topk_from_full

        # Derive top-5 from an existing result (originally predicted with top_k=3)
        topk5 = topk_from_full(result.full, result.id2label, k=5)

        # Also works on a merged probability matrix from multiple runs
        topk3 = topk_from_full(merged_full_df, id2label, k=3)
    """
    # --- locate probability columns in canonical integer-ID order ----------
    prob_cols = {
        int(c[5:]): c
        for c in full_df.columns
        if c.startswith("prob_") and c[5:].lstrip("-").isdigit()
    }
    if not prob_cols:
        raise ValueError(
            "No probability columns found in full_df.  "
            "Expected columns named 'prob_0', 'prob_1', ... matching id2label keys."
        )

    n_classes = len(prob_cols)
    if k < 1 or k > n_classes:
        raise ValueError(
            f"k={k} out of range.  full_df has {n_classes} probability columns "
            f"(prob_0 … prob_{n_classes - 1})."
        )

    # Validate id2label coverage
    missing_ids = [i for i in prob_cols if i not in id2label]
    if missing_ids:
        raise ValueError(
            f"id2label is missing entries for class IDs: {missing_ids}.  "
            "Pass the id2label from the same PredictionResult."
        )

    # --- build probability matrix (n_samples × n_classes) in ID order ------
    sorted_ids  = sorted(prob_cols)                   # ascending integer IDs
    prob_matrix = full_df[[prob_cols[i] for i in sorted_ids]].to_numpy(dtype=float)

    # top-k indices (descending probability)
    top_indices = np.argsort(prob_matrix, axis=1)[:, ::-1][:, :k]  # (n, k)

    # --- assemble output DataFrame -----------------------------------------
    n = len(full_df)
    true_labels = full_df["true_label"].tolist() if "true_label" in full_df.columns else [None] * n

    data: dict = {"true_label": true_labels}
    for j in range(k):
        col_pos   = top_indices[:, j]                    # position in sorted_ids
        class_ids = [sorted_ids[p] for p in col_pos]     # actual integer class IDs
        data[f"top{j + 1}_label"] = [id2label[ci] for ci in class_ids]
        data[f"top{j + 1}_prob"]  = prob_matrix[np.arange(n), col_pos]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 5. Top-K accuracy
# ---------------------------------------------------------------------------

def topk_accuracy(
    topk_df: pd.DataFrame,
    max_k: int | None = None,
) -> pd.DataFrame:
    """Compute cause-of-death accuracy at each k (top-1 through top-K).

    "Accuracy at k" means the fraction of samples where the true cause-of-death
    appears anywhere in the model's top-k predictions.  Top-1 is equivalent to
    standard accuracy; top-K is the maximum achievable accuracy given the model's
    ranking.

    Args:
        topk_df:  The ``topk`` DataFrame from a ``PredictionResult``.
                  Expected columns::

                      true_label | top1_label | top1_prob | top2_label | top2_prob | ...

                  Pass ``result.topk`` directly.
        max_k:    Evaluate up to this k.  Must not exceed the number of
                  top-label columns in ``topk_df``.  ``None`` = use all
                  available k.

    Returns:
        DataFrame with columns:

            ``k``           — rank cutoff (1, 2, 3, ...)
            ``n_correct``   — number of samples with true label in top k
            ``n_total``     — total samples
            ``accuracy``    — n_correct / n_total
            ``accuracy_pct``— accuracy × 100 (for display)

    Raises:
        ValueError: If ``topk_df`` is missing expected columns.
    """
    if "true_label" not in topk_df.columns:
        raise ValueError("topk_df must contain a 'true_label' column.")

    # Detect top-label columns: top1_label, top2_label, ...
    label_cols = sorted(
        [c for c in topk_df.columns if c.endswith("_label") and c.startswith("top")],
        key=lambda c: int(c[3:c.index("_label")]),  # sort by numeric k
    )
    if not label_cols:
        raise ValueError(
            "No topK_label columns found in topk_df.  "
            "Expected columns like 'top1_label', 'top2_label', ..."
        )

    if max_k is not None:
        if max_k < 1 or max_k > len(label_cols):
            raise ValueError(
                f"max_k={max_k} out of range.  topk_df has {len(label_cols)} top-label columns."
            )
        label_cols = label_cols[:max_k]

    true = topk_df["true_label"].values
    n    = len(true)
    rows = []
    for k, col in enumerate(label_cols, start=1):
        # Accumulate correct mask up to this k
        if k == 1:
            correct_mask = true == topk_df[col].values
        else:
            correct_mask = correct_mask | (true == topk_df[col].values)
        n_correct = int(correct_mask.sum())
        acc       = n_correct / n
        rows.append({
            "k":            k,
            "n_correct":    n_correct,
            "n_total":      n,
            "accuracy":     round(acc, 4),
            "accuracy_pct": round(acc * 100, 2),
        })

    return pd.DataFrame(rows)


def _csmf_dataframe(
    y_true,
    y_pred,
    label_order: "list[str] | None" = None,
    cause_map: "dict[str, str] | None" = None,
    drop_zero_zero: bool = False,
    percentage: bool = True,
) -> pd.DataFrame:
    """Build a cause-wise true/predicted CSMF comparison table."""
    true_s = pd.Series(y_true, dtype="object").astype(str)
    pred_s = pd.Series(y_pred, dtype="object").astype(str)

    if label_order is None:
        causes = sorted(set(true_s.unique()).union(pred_s.unique()))
    else:
        extras = [c for c in sorted(set(true_s.unique()).union(pred_s.unique())) if c not in label_order]
        causes = list(label_order) + extras

    n = len(true_s)
    rows: list[dict] = []
    for cause in causes:
        true_csmf = float((true_s == cause).sum()) / n
        pred_csmf = float((pred_s == cause).sum()) / n
        if drop_zero_zero and true_csmf == 0.0 and pred_csmf == 0.0:
            continue
        rows.append(
            {
                "cause": cause,
                "label": cause_map.get(cause, cause) if cause_map is not None else cause,
                "true_csmf": true_csmf * 100 if percentage else true_csmf,
                "pred_csmf": pred_csmf * 100 if percentage else pred_csmf,
                "abs_error": abs(pred_csmf - true_csmf) * (100 if percentage else 1),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Top-K accuracy bar chart
# ---------------------------------------------------------------------------

# Palettes imported from palettes.py — edit that file to adjust colours.
# TOPK_BAR_COLORS is used as the default categorical palette here.


def plot_topk_accuracy(
    data: "pd.DataFrame | dict[str, pd.DataFrame]",
    max_k: int | None = None,
    kind: str = "grouped",
    ncols: int = 2,
    colors: list[str] | None = None,
    title: str | None = None,
    model_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (0, 105),
    label_fontsize: int = 9,
    save_path: "str | None" = None,
    dpi: int = 150,
) -> tuple:
    """Plot top-k accuracy as a bar chart for one or multiple models.

    Accepts either a single topk DataFrame (single-model bar chart) or a
    dict of topk DataFrames keyed by model name (multi-model grouped or facet).

    Args:
        data:           Single ``topk_df`` (from ``result.topk`` or
                        ``topk_from_full()``) **or** a
                        ``dict[model_name, topk_df]`` for multi-model.
                        ``topk_accuracy()`` is called internally — do **not**
                        pre-compute the accuracy table.
        max_k:          Evaluate up to this k.  ``None`` = all available k.
        kind:           Multi-model layout:

                        - ``"grouped"`` — side-by-side bars within each k
                          bundle; one colour per model.
                        - ``"facet"``   — one subplot per model arranged in a
                          grid; one colour per k level (consistent across
                          subplots); shared y-axis range.

                        Ignored for single-model input.
        ncols:          Number of columns in the facet grid.  Default 2.
                        Ignored for ``kind="grouped"`` and single-model.
        colors:         Bar colours.

                        - Single model or facet: list length ≥ max_k —
                          one colour per k level.
                        - Grouped: list length ≥ number of models — one colour
                          per model.

                        Default: built-in palette
                        (steelblue / darkorange / seagreen / …).
        title:          Single-model: axes title (overridden by
                        ``model_label`` when both are set).
                        Multi-model: ``fig.suptitle``.
        model_label:    Single-model only — axes title string.  Takes
                        precedence over ``title`` for the axes title.
        figsize:        Figure size ``(width, height)`` in inches.
                        Auto-sized when ``None``.
        ylim:           Y-axis limits ``(ymin, ymax)``.  Default ``(0, 105)``.
        label_fontsize: Font size for the percentage labels above each bar.
                        Default 9.
        save_path:      File path to save the figure.  Skipped when ``None``.
        dpi:            Resolution for saved figure.  Default 150.

    Returns:
        ``(fig, ax)``    — single-model **or** ``kind="grouped"`` multi-model.
        ``(fig, axes)``  — ``kind="facet"`` multi-model; ``axes`` is a 2-D
                           NumPy array shaped ``(nrows, ncols)``.

    Raises:
        ValueError: If ``data`` is not a DataFrame or non-empty dict, if
                    ``kind`` is unrecognised, or if models have no common
                    k values.

    Examples::

        from multimodalva.results import plot_topk_accuracy

        # --- Single model ---
        fig, ax = plot_topk_accuracy(
            result.topk, max_k=3, model_label="BioBERT (adults)",
        )

        # --- Multi-model grouped ---
        fig, ax = plot_topk_accuracy(
            {"BioBERT": result_text.topk, "LightGBM": result_tab.topk},
            max_k=3, kind="grouped", title="Top-k Accuracy Comparison",
        )

        # --- Multi-model facet (2 × 2 grid) ---
        fig, axes = plot_topk_accuracy(
            {
                "BioBERT":   result_text.topk,
                "LightGBM":  result_tab.topk,
                "Data Fusion": result_df.topk,
                "Stacking":  result_stk.topk,
            },
            max_k=3, kind="facet", ncols=2,
            title="Top-k Accuracy by Model",
        )
        plt.show()
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    palette = colors or TOPK_BAR_COLORS

    # ------------------------------------------------------------------ #
    # Single-model path
    # ------------------------------------------------------------------ #
    if isinstance(data, pd.DataFrame):
        acc    = topk_accuracy(data, max_k=max_k)
        ks     = acc["k"].astype(str).tolist()
        pcts   = acc["accuracy_pct"].tolist()
        bcolors = [palette[i % len(palette)] for i in range(len(ks))]

        _figsize = figsize or (max(4.0, len(ks) * 1.4), 3.5)
        fig, ax = plt.subplots(figsize=_figsize)
        ax.bar(ks, pcts, color=bcolors)
        for k_str, pct in zip(ks, pcts):
            ax.text(k_str, pct + 0.5, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=label_fontsize)
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(ylim)
        _ax_title = model_label or title
        if _ax_title:
            ax.set_title(_ax_title)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig, ax

    # ------------------------------------------------------------------ #
    # Multi-model path — validate input
    # ------------------------------------------------------------------ #
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(
            "data must be a topk DataFrame (single model) or a non-empty "
            "dict[model_name, topk_df] (multi-model)."
        )

    model_names = list(data.keys())
    acc_tables  = {name: topk_accuracy(df, max_k=max_k) for name, df in data.items()}

    all_ks = sorted(
        set.intersection(*[set(t["k"].tolist()) for t in acc_tables.values()])
    )
    if not all_ks:
        raise ValueError("No common k values across models.")

    # ------------------------------------------------------------------ #
    # Grouped bar chart — one bundle per k, models side by side
    # ------------------------------------------------------------------ #
    if kind == "grouped":
        n_models    = len(model_names)
        n_k         = len(all_ks)
        bar_width   = min(0.7 / n_models, 0.25)
        x_centers   = np.arange(n_k)
        model_colors = [palette[i % len(palette)] for i in range(n_models)]

        _figsize = figsize or (max(5.0, n_k * (n_models * bar_width + 0.8)), 4.0)
        fig, ax = plt.subplots(figsize=_figsize)

        for m_idx, (name, color) in enumerate(zip(model_names, model_colors)):
            acc  = acc_tables[name]
            pcts = [
                float(acc.loc[acc["k"] == k, "accuracy_pct"].values[0])
                for k in all_ks
            ]
            offsets = x_centers + (m_idx - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(offsets, pcts, width=bar_width, color=color, label=name)
            for bar, pct in zip(bars, pcts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=label_fontsize,
                )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"Top-{k}" for k in all_ks])
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(ylim)
        ax.legend(loc="lower right", framealpha=0.8)
        if title:
            ax.set_title(title)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig, ax

    # ------------------------------------------------------------------ #
    # Facet bar chart — one subplot per model, shared y-axis
    # ------------------------------------------------------------------ #
    if kind == "facet":
        n_models = len(model_names)
        nrows    = (n_models + ncols - 1) // ncols
        n_k      = len(all_ks)
        k_colors = [palette[i % len(palette)] for i in range(n_k)]
        k_strs   = [str(k) for k in all_ks]

        _figsize = figsize or (ncols * 4.5, nrows * 3.5)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=_figsize,
            sharey=True,
            squeeze=False,
        )

        for m_idx, name in enumerate(model_names):
            row, col = divmod(m_idx, ncols)
            ax   = axes[row][col]
            acc  = acc_tables[name]
            pcts = [
                float(acc.loc[acc["k"] == k, "accuracy_pct"].values[0])
                for k in all_ks
            ]
            bars = ax.bar(k_strs, pcts, color=k_colors)
            for bar, pct in zip(bars, pcts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=label_fontsize,
                )
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_ylim(ylim)

        # Hide unused axes when n_models doesn't fill the grid
        for empty_idx in range(n_models, nrows * ncols):
            row, col = divmod(empty_idx, ncols)
            axes[row][col].set_visible(False)

        # Shared axis labels
        fig.supxlabel("k", fontsize=10)
        fig.supylabel("Accuracy (%)", fontsize=10)
        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes


def csmf_scatterplot(
    y_true,
    y_pred,
    label_order: "list[str] | None" = None,
    cause_map: "dict[str, str] | None" = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    point_color: str = "#00468B",
    point_size: float = 55,
    alpha: float = 0.85,
    annotate: bool = True,
    label_fontsize: int = 8,
    percentage: bool = True,
    drop_zero_zero: bool = False,
    max_val: float | None = None,
    crowded_threshold: float | None = None,
    figsize: tuple[float, float] = (7.2, 6.6),
    save_path: "str | None" = None,
    dpi: int = 150,
):
    """Plot true CSMF vs modelled CSMF with one labelled point per cause.

    Labels are placed as close to their points as possible (≤ 18 pt offset).
    Causes with both true and predicted CSMF below ``crowded_threshold`` are
    marked with uppercase letters (A, B, C…) and listed in a footnote to avoid
    crowding near the origin.  Any larger cause that still cannot fit a full
    label also falls back to a letter tag.

    Args:
        y_true:             Ground-truth cause labels.
        y_pred:             Model-predicted cause labels.
        label_order:        Optional cause order for the underlying CSMF table.
        cause_map:          Optional mapping from raw cause names to shorter
                            display labels used in annotations.
        title:              Plot title.
        xlabel:             X-axis label. Defaults to ``"True CSMF (%)"``.
        ylabel:             Y-axis label. Defaults to ``"Modelled CSMF (%)"``.
        point_color:        Marker colour.
        point_size:         Marker size passed to ``ax.scatter``.
        alpha:              Marker opacity.
        annotate:           Whether to label each point with the cause name.
        label_fontsize:     Font size for point labels.
        percentage:         Plot CSMF values as percentages when True, else
                            proportions in [0, 1].
        drop_zero_zero:     Drop causes with both true and predicted CSMF equal
                            to zero.
        max_val:            Optional common axis maximum.
        crowded_threshold:  Causes with both true and pred CSMF at or below
                            this value are tagged with a letter immediately
                            instead of attempting a full label. Defaults to 5
                            (5 % CSMF when ``percentage=True``, 0.05 otherwise).
                            Pass 0 to disable letter-tagging entirely.
        figsize:            Figure size in inches.
        save_path:          Optional output path for the figure.
        dpi:                Save DPI.

    Returns:
        ``(fig, ax, csmf_df)`` where ``csmf_df`` contains one row per cause
        with columns ``cause``, ``label``, ``true_csmf``, ``pred_csmf``,
        ``abs_error``.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    csmf_df = _csmf_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_order=label_order,
        cause_map=cause_map,
        drop_zero_zero=drop_zero_zero,
        percentage=percentage,
    )
    if csmf_df.empty:
        raise ValueError("No causes available to plot after filtering.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        csmf_df["true_csmf"],
        csmf_df["pred_csmf"],
        s=point_size,
        color=point_color,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    upper = float(max(csmf_df["true_csmf"].max(), csmf_df["pred_csmf"].max()))
    axis_max = max_val if max_val is not None else upper * (1.12 if annotate else 1.08)
    if axis_max <= 0:
        axis_max = 1.0 if not percentage else 100.0

    # Extend axes slightly into negative space so the dense cluster near the
    # origin has breathing room and labels can sit below/left of their points
    # without being clipped.  Shade the impossible (negative CSMF) L-region in
    # light grey and hide tick labels there so it does not read as real data.
    neg_margin = axis_max * 0.04
    ax.set_xlim(-neg_margin, axis_max)
    ax.set_ylim(-neg_margin, axis_max)
    ax.axvspan(-neg_margin, 0, facecolor="#EEEEEE", edgecolor="none", zorder=0)
    ax.axhspan(-neg_margin, 0, facecolor="#EEEEEE", edgecolor="none", zorder=0)
    from matplotlib.ticker import FuncFormatter as _FuncFormatter
    _hide_neg = _FuncFormatter(lambda v, _p: "" if v < -1e-9 else f"{v:g}")
    ax.xaxis.set_major_formatter(_hide_neg)
    ax.yaxis.set_major_formatter(_hide_neg)
    ax.plot([0, axis_max], [0, axis_max], linestyle="--", linewidth=1.2, color="#7E6148", zorder=2)

    footnotes: list[str] = []
    if annotate:
        # Default: causes with both true and pred CSMF at or below 5% (5 % units
        # if percentage=True, 0.05 if percentage=False) are letter-tagged in a
        # footnote.  Larger causes get full labels placed next to their points.
        if crowded_threshold is not None:
            _threshold = crowded_threshold
        else:
            _threshold = 5.0 if percentage else 0.05
        footnotes = _place_scatter_labels(
            ax=ax,
            csmf_df=csmf_df,
            axis_max=axis_max,
            label_fontsize=label_fontsize,
            crowded_threshold=_threshold,
            point_size=point_size,
        )

    ax.set_xlabel(xlabel or ("True CSMF (%)" if percentage else "True CSMF"))
    ax.set_ylabel(ylabel or ("Modelled CSMF (%)" if percentage else "Modelled CSMF"))
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.45, zorder=1)
    ax.set_axisbelow(True)

    if footnotes:
        # Wrap footnotes into rows of 4 so the note block stays compact.
        per_row = 4
        rows = [footnotes[i : i + per_row] for i in range(0, len(footnotes), per_row)]
        fn_text = "Note:  " + "\n       ".join("   ".join(r) for r in rows)
        n_rows = len(rows)
        bottom_frac = 0.045 + 0.038 * (n_rows - 1)  # ~4.5 % per line
        fig.text(
            0.02,
            0.01,
            fn_text,
            ha="left",
            va="bottom",
            fontsize=max(7, label_fontsize - 1),
            color="#1B1919",
        )
        plt.tight_layout(rect=(0, bottom_frac, 1, 1))
    else:
        plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("CSMF scatter plot saved to %s", save_path)

    return fig, ax, csmf_df


def _place_scatter_labels(
    ax,
    csmf_df: pd.DataFrame,
    axis_max: float,
    label_fontsize: int,
    crowded_threshold: float,
    point_size: float = 55,
) -> list[str]:
    """Place labels next to points; letter-tag everything ≤ crowded_threshold.

    Strategy
    --------
    * Causes with both true and pred CSMF at or below *crowded_threshold* are
      tagged with uppercase letters (A, B, C…) drawn at the point and listed
      in the returned footnotes.
    * All other causes get their full label placed adjacent to the point with
      a small perpendicular-to-diagonal offset, *no connector line by default*.
    * adjustText is then run only to push apart any labels that would still
      overlap.  After it settles, a connector line is drawn **only for labels
      that adjustText had to move significantly** from their adjacent
      starting position — quiet, uncluttered points keep clean adjacent
      labels; only crowded clusters show a short link line.
    """
    from adjustText import adjust_text

    _LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def _tag(i: int) -> str:
        return _LETTERS[i] if i < 26 else f"{_LETTERS[i % 26]}{i // 26 + 1}"

    footnotes: list[str] = []
    texts: list = []          # full-label Text objects
    xs: list[float] = []      # original data-point x for each text
    ys: list[float] = []      # original data-point y for each text
    init_pos: list[tuple[float, float]] = []  # initial (tx, ty) before adjust
    tag_n = 0

    # Sort: largest CSMF first so letter tags A, B, C run from biggest to
    # smallest among the crowded near-origin set.
    work = csmf_df.copy()
    work["_r"] = np.hypot(work["true_csmf"], work["pred_csmf"])
    work = work.sort_values("_r", ascending=False)

    # Initial offset for full labels: small (~2.5 % of axis_max) so the label
    # starts visibly adjacent to the point.  Perpendicular to the diagonal so
    # above-diagonal labels start upper-left, below-diagonal start lower-right.
    _INV_SQRT2 = 1.0 / np.sqrt(2)
    base_offset = axis_max * 0.025

    # Transparent bbox gives adjustText a real text bbox to use for overlap
    # detection while staying invisible to the viewer.
    _bbox = dict(boxstyle="round,pad=0.10",
                 facecolor="none", edgecolor="none")

    for row in work.itertuples(index=False):
        x, y = row.true_csmf, row.pred_csmf
        if x <= crowded_threshold and y <= crowded_threshold:
            tag = _tag(tag_n); tag_n += 1
            footnotes.append(f"{tag} = {row.label}")
            ax.text(x, y, tag, fontsize=label_fontsize,
                    color="#1B1919", zorder=5,
                    ha="center", va="center", bbox=_bbox)
            continue

        # Above diagonal (pred > true) → label upper-left of point.
        # Below diagonal (pred < true) → label lower-right of point.
        sign = 1.0 if (y >= x) else -1.0
        tx = float(np.clip(x - sign * base_offset * _INV_SQRT2,
                           0.0, axis_max * 0.97))
        ty = float(np.clip(y + sign * base_offset * _INV_SQRT2,
                           0.0, axis_max * 0.97))
        t = ax.text(tx, ty, row.label, fontsize=label_fontsize,
                    color="#1B1919", zorder=5,
                    ha="center", va="center", bbox=_bbox)
        texts.append(t)
        xs.append(x)
        ys.append(y)
        init_pos.append((tx, ty))

    if texts:
        # Run adjustText WITHOUT arrows — we draw connectors ourselves below
        # only for labels that actually had to move.
        adjust_text(
            texts,
            x=xs, y=ys,
            ax=ax,
            arrowprops=None,
            expand=(1.05, 1.1),
            force_text=(0.2, 0.3),
            force_points=(0.25, 0.3),
            lim=200,
            verbose=False,
        )

        # Connector threshold: if the label moved more than ~4 % of axis_max
        # from its initial adjacent position, draw a thin line back to the
        # data point.  Otherwise leave it floating clean next to the point.
        move_threshold = axis_max * 0.04
        for t, x_orig, y_orig, (tx0, ty0) in zip(texts, xs, ys, init_pos):
            fx, fy = t.get_position()
            displacement = np.hypot(fx - tx0, fy - ty0)
            if displacement > move_threshold:
                ax.annotate(
                    "",
                    xy=(x_orig, y_orig),
                    xytext=(fx, fy),
                    arrowprops=dict(arrowstyle="-", color="#7E6148",
                                    lw=0.5, shrinkA=2, shrinkB=2),
                    zorder=4,
                )

    return footnotes


# ---------------------------------------------------------------------------
# Private helpers shared by cause_accuracy_heatmap and cause_accuracy_diff_heatmap
# ---------------------------------------------------------------------------

def _compute_cause_accuracy_matrix(
    df: pd.DataFrame | None,
    true_col: str,
    model_cols: list[str] | None,
    top_k: int,
    topk_dfs: dict | None,
    percentage: bool,
    insilicova_first: bool,
) -> pd.DataFrame:
    """Compute per-cause accuracy for each model; return a cause × model DataFrame.

    Args:
        df:          Wide prediction DataFrame (true_col + one pred col per model).
        true_col:    Ground-truth column name in df.
        model_cols:  Columns from df to include.  None = all except true_col / "id".
        top_k:       Rank cutoff for topk_dfs accuracy.  1 = exact match for df.
        topk_dfs:    Dict model_name → topk DataFrame; keys override df entries.
        percentage:  Multiply accuracy by 100.

    Returns:
        DataFrame indexed by cause with one column per model.
    """
    def _reorder_insilicova_first(names: list[str]) -> list[str]:
        if not insilicova_first:
            return names
        ins = [name for name in names if "insilicova" in str(name).lower()]
        rest = [name for name in names if "insilicova" not in str(name).lower()]
        return ins + rest

    model_accs: dict[str, pd.Series] = {}

    # --- step A: accuracy from wide df (exact match, top-1) -----------------
    if df is not None:
        if true_col not in df.columns:
            raise ValueError(f"true_col {true_col!r} not found in df.")
        cols = model_cols if model_cols is not None else [
            c for c in df.columns if c not in {true_col, "id"}
        ]
        cols = _reorder_insilicova_first(list(cols))
        if not cols:
            raise ValueError("No model columns found in df.  Pass model_cols explicitly.")
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Model column {col!r} not found in df.")
            correct = (df[col] == df[true_col]).astype(float)
            acc = correct.groupby(df[true_col].values).mean()
            if percentage:
                acc = acc * 100
            model_accs[col] = acc

    # --- step B: accuracy from topk_dfs (top-k in-rank check) ---------------
    if topk_dfs is not None:
        for model_name, topk_df in topk_dfs.items():
            if "true_label" not in topk_df.columns:
                raise ValueError(
                    f"topk_df for model {model_name!r} must contain a 'true_label' column."
                )
            label_cols = sorted(
                [c for c in topk_df.columns if c.startswith("top") and c.endswith("_label")],
                key=lambda c: int(c[3: c.index("_label")]),
            )
            if not label_cols:
                raise ValueError(
                    f"topk_df for model {model_name!r} has no 'topK_label' columns.  "
                    "Build it with topk_from_full()."
                )
            k_eff = min(top_k, len(label_cols))
            label_cols = label_cols[:k_eff]

            true_labels = topk_df["true_label"]
            in_topk = pd.concat(
                [topk_df[lc] == true_labels for lc in label_cols], axis=1
            ).any(axis=1)

            temp = pd.DataFrame({
                "true_label": true_labels.values,
                "in_topk":    in_topk.values,
            })
            acc = temp.groupby("true_label")["in_topk"].mean()
            if percentage:
                acc = acc * 100
            model_accs[model_name] = acc  # overrides df entry for same model name

    if not model_accs:
        raise ValueError("No model accuracy data could be computed.")

    ordered_names = _reorder_insilicova_first(list(model_accs))
    return pd.DataFrame(model_accs)[ordered_names]


def _resolve_true_label_counts(
    df: pd.DataFrame | None,
    true_col: str,
    topk_dfs: dict[str, "pd.DataFrame"] | None,
) -> pd.Series:
    """Resolve per-cause sample counts for y-axis label annotations."""
    if df is not None:
        if true_col not in df.columns:
            raise ValueError(
                f"true_col {true_col!r} not found in df columns; cannot compute label counts."
            )
        return df[true_col].value_counts()

    if not topk_dfs:
        raise ValueError(
            "show_n=True requires df or topk_dfs to compute per-cause sample counts."
        )

    first_topk = next(iter(topk_dfs.values()))
    if "true_label" not in first_topk.columns:
        raise ValueError(
            "topk_dfs entries must contain 'true_label' when show_n=True."
        )
    return first_topk["true_label"].value_counts()


def _group_boundaries_from_sizes(group_sizes: list[int], n_cols: int) -> list[tuple[float, float]]:
    """Convert group sizes to start/end spans in heatmap column-centre coordinates.

    Example:
        group_sizes=[5, 4] -> [(0.5, 4.5), (5.5, 8.5)]

    This means:
        - label 1 centered over columns 1–5, line from mid(col1) to mid(col5)
        - label 2 centered over columns 6–9, line from mid(col6) to mid(col9)
    """
    if not group_sizes:
        raise ValueError("group_sizes must contain at least one positive integer.")
    if any((not isinstance(size, (int, np.integer))) or int(size) < 1 for size in group_sizes):
        raise ValueError(
            f"group_sizes must be positive integers, got {group_sizes!r}."
        )

    total = int(sum(int(size) for size in group_sizes))
    if total != n_cols:
        raise ValueError(
            f"group_sizes sum to {total}, but the heatmap has {n_cols} columns."
        )

    boundaries: list[tuple[float, float]] = []
    start = 0.5
    for size in group_sizes:
        size = int(size)
        end = start + size - 1
        boundaries.append((start, end))
        start = end + 1
    return boundaries


def _draw_group_annotations(
    ax,
    group_boundaries: list[tuple[float, float]],
    group_labels: list[str],
    fontsize: int = 12,
) -> None:
    """Draw bracket lines and labels above the heatmap to annotate column groups.

    Uses pure data coordinates — ``ax.hlines()`` for the bracket lines and
    ``ax.text()`` for labels — so x positions align exactly with seaborn heatmap
    column indices (e.g. 0.5 = centre of column 0, 2.5 = centre of column 2).

    Seaborn heatmap inverts the y-axis (``ylim = (n_rows, 0)``).  Annotations
    are placed at negative y values, which sit above the first data row in this
    inverted space.  ``ax.set_ylim`` is extended so they are visible without
    relying on ``clip_on=False`` overlays.

    Group-boundary semantics:
    - Explicit centre span mode: pass the centres of the first and last columns
      in a group. Example: ``(0.5, 4.5)`` spans columns 1–5 exactly.
    - Contiguous-boundary mode: pass the centre of the first column in a group
      and the centre of the first column in the next group. Example:
      ``(0.5, 5.5)`` means the group spans columns 1–5.
    - 1-based inclusive column mode: pass integer-like pairs such as ``(1, 5)``
      to mean columns 1–5. This is normalized internally to ``(0.5, 4.5)``.

    The helper auto-detects contiguous-group mode whenever adjacent groups share
    a boundary (i.e. previous ``end`` equals next ``start``).

    Args:
        ax:               Matplotlib Axes (must already contain a seaborn heatmap).
        group_boundaries: List of ``(xmin, xmax)`` column-centre pairs using the
                          same data coordinates as the heatmap cells
                          (e.g. ``(0.5, 4.5)`` spans the first through fifth
                          columns without extending into neighboring groups).
        group_labels:     One label per group; must match len(group_boundaries).
        fontsize:         Label font size.  Default 12.
    """
    if len(group_boundaries) != len(group_labels):
        raise ValueError(
            "group_boundaries and group_labels must have the same length."
        )

    for i, boundary in enumerate(group_boundaries):
        if not (hasattr(boundary, "__len__") and len(boundary) == 2):
            raise ValueError(
                f"group_boundaries[{i}] must be a 2-element (start, end) tuple, "
                f"got {boundary!r} with {len(boundary) if hasattr(boundary, '__len__') else 'unknown'} elements. "
                "Example: group_boundaries=[(0.5, 2.5), (3.5, 5.5)]"
            )

    # Annotation geometry in data (y) coordinates.
    # Seaborn heatmap y-axis is inverted: ylim = (n_rows, 0).
    # Negative y → above the first data row (top of the heatmap).
    y_bar  = -0.35   # horizontal bracket line
    y_text = -0.65   # label just above the line
    y_top  = -1.4    # new ylim top — enough room for line + label

    y_bottom, _ = ax.get_ylim()   # n_rows (bottom of inverted axis)

    def _looks_like_column_numbers(start: float, end: float) -> bool:
        return (
            np.isclose(start, round(start))
            and np.isclose(end, round(end))
            and start >= 1
            and end >= 1
        )

    normalized_boundaries: list[tuple[float, float]] = []
    for start, end in group_boundaries:
        if _looks_like_column_numbers(start, end):
            normalized_boundaries.append((start - 0.5, end - 0.5))
        else:
            normalized_boundaries.append((start, end))

    single_col_half_width = 0.45
    contiguous_mode = any(
        np.isclose(normalized_boundaries[i][1], normalized_boundaries[i + 1][0])
        for i in range(len(normalized_boundaries) - 1)
    )

    for (start, end), label in zip(normalized_boundaries, group_labels):
        if end < start:
            raise ValueError(
                f"Each group boundary must satisfy start <= end, got {(start, end)!r}."
            )

        if contiguous_mode:
            if np.isclose(end - start, 1.0):
                xmin = start - single_col_half_width
                xmax = start + single_col_half_width
            else:
                xmin = start
                xmax = end - 1.0
        elif np.isclose(start, end):
            xmin = start - single_col_half_width
            xmax = end + single_col_half_width
        else:
            xmin = start
            xmax = end

        ax.hlines(y=y_bar, xmin=xmin, xmax=xmax, color="black", linewidth=2)
        ax.text(
            (xmin + xmax) / 2, y_text, label,
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
        )

    # Extend the y-axis upward (negative in inverted space) to reveal the annotations.
    ax.set_ylim(y_bottom, y_top)


# ---------------------------------------------------------------------------
# 6. Cause-specific accuracy heatmap
# ---------------------------------------------------------------------------

def cause_accuracy_heatmap(
    df: pd.DataFrame | None,
    true_col: str,
    model_cols: list[str] | None = None,
    top_k: int = 1,
    topk_dfs: dict[str, "pd.DataFrame"] | None = None,
    # --- display ordering / filtering ---
    cause_order: list[str] | None = None,
    drop_causes: list[str] | None = None,
    cause_rename: dict[str, str] | None = None,
    model_rename: dict[str, str] | None = None,
    # --- column grouping annotations below x-axis ---
    group_boundaries: list[tuple[float, float]] | None = None,
    group_sizes: list[int] | None = None,
    group_labels: list[str] | None = None,
    # --- aesthetics ---
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    percentage: bool = True,
    annot: bool = True,
    fmt: str = ".1f",
    figsize: tuple[float, float] = (9, 4.5),
    cbar_label: str | None = None,
    x_rotation: int = 45,
    y_rotation: int = 0,
    show_n: bool = False,
    insilicova_first: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> tuple:
    """Cause-specific accuracy heatmap across selected models.

    Rows = causes, columns = models.  Each cell shows the fraction (or
    percentage) of samples of that cause that are correctly classified by that
    model.

    Supports two accuracy modes:

    - **top-1** (default): correct when the top-predicted label equals the
      true label.  Pass ``df`` with one prediction column per model.
    - **top-k**: correct when the true label appears anywhere in the model's
      top-k predictions.  Pass ``topk_dfs`` (a dict of topk DataFrames, one
      per model, as returned by :func:`topk_from_full`) and set ``top_k``.

    Args:
        df:             Wide DataFrame with ``true_col`` and one prediction
                        column per model (for ``top_k=1``).  Pass ``None``
                        when using ``topk_dfs`` exclusively.
        true_col:       Name of the ground-truth label column in ``df``
                        (also the key in ``topk_dfs`` DataFrames is always
                        ``"true_label"``).
        model_cols:     Columns from ``df`` to evaluate.  ``None`` uses all
                        columns except ``true_col`` and ``"id"``.  Ignored
                        when ``topk_dfs`` is provided without ``df``.
        top_k:          Rank cutoff for accuracy.  Default 1 (exact match).
                        When > 1, ``topk_dfs`` must be provided.
        topk_dfs:       Dict mapping model name → topk DataFrame (from
                        :func:`topk_from_full`).  When provided, overrides
                        ``df``-based accuracy for those models; ``top_k``
                        determines how many top-label columns to check.
                        If a model appears in both ``df`` and ``topk_dfs``,
                        ``topk_dfs`` takes precedence.

        cause_order:    Reindex rows to this cause ordering.  Causes absent
                        from the data are silently dropped; causes present in
                        the data but absent from ``cause_order`` are excluded.
                        ``None`` = alphabetical order from the data.
        drop_causes:    Causes to exclude from the heatmap (applied after
                        ``cause_order``).
        cause_rename:   Dict ``{original → display}`` applied to row labels.
        model_rename:   Dict ``{original → display}`` applied to column labels.

        group_boundaries: List of ``(start, end)`` pairs in heatmap column-centre
                          coordinates marking model groups. Two conventions are
                          accepted:
                          1. explicit span, e.g. ``(1.5, 5.5)`` for columns 1–5
                          2. contiguous-group boundaries, e.g. ``(1.5, 6.5)``
                             for columns 1–5 when the next group starts at 6.5
                          Single-column groups can be passed as ``(0.5, 0.5)``
                          in explicit span mode or ``(0.5, 1.5)`` in contiguous
                          mode. Group lines are drawn only over their own columns.
        group_sizes:      Alternative to ``group_boundaries``. One integer per
                          group giving the number of displayed model columns in
                          that group. Example: ``group_sizes=[5, 4]`` generates
                          spans ``[(0.5, 4.5), (5.5, 8.5)]`` automatically.
        group_labels:   One label string per entry in ``group_boundaries``.

        cmap:           Colormap.  Default: custom light-cream → dark-red
                        ``LinearSegmentedColormap`` calibrated for accuracy.
        vmin:           Colormap lower bound.  Default: min of data.
        vmax:           Colormap upper bound.  Default: max of data.
                        Tip: pass ``vmin=0, vmax=100`` when ``percentage=True``
                        to anchor the full range.
        percentage:     Multiply accuracy by 100 for display.  Default True.
        annot:          Annotate cells.  Default True.
        fmt:            ``seaborn.heatmap`` ``fmt`` string.  Default ``".1f"``.
        figsize:        Figure size ``(width, height)``.  Default ``(9, 4.5)``.
        cbar_label:     Colorbar label.  Default: ``"Accuracy (%)"`` or
                        ``"Accuracy"`` depending on ``percentage``.
        x_rotation:     X-axis tick rotation.  Default 45.
        y_rotation:     Y-axis tick rotation.  Default 0.
        show_n:         Append true-label sample size to each y-axis tick,
                        e.g. ``"Malaria (n=42)"``.  Counts are taken from
                        ``df[true_col]`` when ``df`` is provided, otherwise
                        from the ``"true_label"`` column of the first entry
                        in ``topk_dfs``.  Default False.
        insilicova_first:
                        When True (default), any model whose name contains
                        ``"insilicova"`` is moved to the far-left side of the
                        heatmap before plotting and grouping.
        save_path:      Save figure to this path before returning.  Supports
                        any matplotlib extension (.png, .pdf, .svg).
        dpi:            Resolution when saving.  Default 150.

    Returns:
        ``(fig, ax, accuracy_df)`` — the matplotlib Figure and Axes, and the
        cause × model accuracy DataFrame used for plotting (before renaming).

    Raises:
        ValueError: On invalid inputs (missing columns, top_k without
                    topk_dfs, no models found).

    Example — top-1 from a wide prediction DataFrame::

        from multimodalva.results import cause_accuracy_heatmap

        fig, ax, acc = cause_accuracy_heatmap(
            df=pred_df,
            true_col="true_label",
            model_cols=["bert", "lgbm", "ensemble"],
            cause_order=["Malaria", "Pneumonia", "HIV/AIDS"],
            model_rename={"bert": "BioClinicalBERT", "lgbm": "LightGBM"},
            group_boundaries=[(0.5, 1.5), (2.5, 2.5)],
            group_labels=["Uni-modal", "Multi-modal"],
            save_path="figures/cause_accuracy.png",
        )

    Example — top-3 from PredictionResult objects::

        from multimodalva.results import topk_from_full, cause_accuracy_heatmap

        topk_dfs = {
            "bert": topk_from_full(bert_result.full, bert_result.id2label, k=3),
            "lgbm": topk_from_full(lgbm_result.full, lgbm_result.id2label, k=3),
        }
        fig, ax, acc = cause_accuracy_heatmap(
            df=None,
            true_col="true_label",
            topk_dfs=topk_dfs,
            top_k=3,
        )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if df is None and topk_dfs is None:
        raise ValueError("At least one of df or topk_dfs must be provided.")
    if top_k > 1 and topk_dfs is None:
        raise ValueError(
            "top_k > 1 requires topk_dfs.  "
            "Build per-model topk DataFrames with topk_from_full(result.full, result.id2label, k=top_k)."
        )
    if top_k < 1:
        raise ValueError(f"top_k must be ≥ 1, got {top_k}.")
    if group_boundaries is not None and group_sizes is not None:
        raise ValueError("Pass only one of group_boundaries or group_sizes, not both.")

    # ------------------------------------------------------------------
    # Steps 1–3: compute per-cause accuracy for each model
    # ------------------------------------------------------------------
    accuracy_df = _compute_cause_accuracy_matrix(
        df=df, true_col=true_col, model_cols=model_cols,
        top_k=top_k, topk_dfs=topk_dfs, percentage=percentage,
        insilicova_first=insilicova_first,
    )

    # ------------------------------------------------------------------
    # Step 4: cause ordering / filtering
    # ------------------------------------------------------------------
    if cause_order is not None:
        accuracy_df = accuracy_df.reindex(cause_order)

    if drop_causes:
        accuracy_df = accuracy_df.drop(index=drop_causes, errors="ignore")

    # Drop rows that are entirely NaN (causes not present in any model)
    accuracy_df = accuracy_df.dropna(how="all")

    # ------------------------------------------------------------------
    # Step 4b: compute per-cause sample counts (used when show_n=True)
    # ------------------------------------------------------------------
    if show_n:
        label_counts = _resolve_true_label_counts(df=df, true_col=true_col, topk_dfs=topk_dfs)
    else:
        label_counts = None

    # ------------------------------------------------------------------
    # Step 5: rename for display (work on a copy to keep accuracy_df clean)
    # ------------------------------------------------------------------
    display_df = accuracy_df.copy()
    if cause_rename:
        display_df = display_df.rename(index=cause_rename)
    if model_rename:
        display_df = display_df.rename(columns=model_rename)

    if group_sizes is not None:
        group_boundaries = _group_boundaries_from_sizes(group_sizes, len(display_df.columns))

    # ------------------------------------------------------------------
    # Step 6: colormap
    # ------------------------------------------------------------------
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "va_cause_accuracy",
            HEATMAP_CLINICAL,
        )

    if cbar_label is None:
        cbar_label = "Accuracy (%)" if percentage else "Accuracy"

    # ------------------------------------------------------------------
    # Step 7: plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    heatmap_kws: dict = dict(
        annot=annot,
        fmt=fmt,
        ax=ax,
        cmap=cmap,
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.85},
    )
    if vmin is not None:
        heatmap_kws["vmin"] = vmin
    if vmax is not None:
        heatmap_kws["vmax"] = vmax

    sns.heatmap(display_df, **heatmap_kws)

    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=x_rotation)
    ax.tick_params(axis="y", rotation=y_rotation)

    if show_n:
        ytick_labels = [
            f"{display_cause} (n={label_counts.get(orig_cause, 0)})"
            for orig_cause, display_cause in zip(accuracy_df.index, display_df.index)
        ]
        ax.set_yticklabels(ytick_labels, rotation=y_rotation)

    # ------------------------------------------------------------------
    # Step 8: column grouping annotations above heatmap
    # ------------------------------------------------------------------
    if group_boundaries and group_labels:
        _draw_group_annotations(ax, group_boundaries, group_labels)

    # Group annotations live inside the axes (via ax.set_ylim extension), so
    # tight_layout needs no rect constraint — it lays out the full figure normally.
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Cause accuracy heatmap saved to %s", save_path)

    return fig, ax, accuracy_df


# ---------------------------------------------------------------------------
# 7. Cause-specific accuracy difference heatmap
# ---------------------------------------------------------------------------

def cause_accuracy_diff_heatmap(
    df: pd.DataFrame | None,
    true_col: str,
    baseline_col: str,
    model_cols: list[str] | None = None,
    include_baseline: bool = False,
    top_k: int = 1,
    topk_dfs: dict[str, "pd.DataFrame"] | None = None,
    # --- display ordering / filtering ---
    cause_order: list[str] | None = None,
    drop_causes: list[str] | None = None,
    cause_rename: dict[str, str] | None = None,
    model_rename: dict[str, str] | None = None,
    # --- column grouping annotations below x-axis ---
    group_boundaries: list[tuple[float, float]] | None = None,
    group_sizes: list[int] | None = None,
    group_labels: list[str] | None = None,
    # --- aesthetics ---
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    percentage: bool = True,
    annot: bool = True,
    fmt: str = ".1f",
    figsize: tuple[float, float] = (9, 4.5),
    cbar_label: str | None = None,
    x_rotation: int = 45,
    y_rotation: int = 0,
    show_n: bool = False,
    insilicova_first: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> tuple:
    """Cause-specific accuracy difference heatmap vs. a baseline model.

    Same rows-as-causes, columns-as-models layout as :func:`cause_accuracy_heatmap`,
    but each cell shows the **difference** in per-cause accuracy relative to a
    chosen baseline model: ``accuracy(model) − accuracy(baseline)``.

    - Positive values (warm colours in ``RdBu_r``) → model outperforms baseline for
      that cause.
    - Negative values (cool colours) → model underperforms.
    - The colormap is centred at 0 so the neutral point is always white/grey.

    Args:
        df:              Wide DataFrame with ``true_col`` and one prediction
                         column per model (for ``top_k=1``).  Pass ``None``
                         when using ``topk_dfs`` exclusively.
        true_col:        Name of the ground-truth label column in ``df``.
        baseline_col:    Name of the baseline model.  Must be present as a
                         column in ``df`` or as a key in ``topk_dfs``.
                         Its per-cause accuracy is subtracted from every other
                         model; the column itself is excluded from the heatmap
                         unless ``include_baseline=True``.
        model_cols:      Comparison model columns from ``df`` (excluding
                         ``baseline_col``).  ``None`` = all columns in ``df``
                         except ``true_col``, ``"id"``, and ``baseline_col``.
        include_baseline: When ``True``, prepend a baseline column (all zeros)
                          to the heatmap so the reference label is visible.
                          Default False.
        top_k:           Rank cutoff for accuracy.  Default 1 (exact match).
                         When > 1, ``topk_dfs`` must be provided.
        topk_dfs:        Dict mapping model name → topk DataFrame (from
                         :func:`topk_from_full`).  Overrides ``df``-based
                         accuracy for the same model name.

        cause_order:     Reindex rows to this ordering.  ``None`` = alphabetical.
        drop_causes:     Causes to exclude (applied after ``cause_order``).
        cause_rename:    ``{original → display}`` row-label mapping.
        model_rename:    ``{original → display}`` column-label mapping.
                         Applied to comparison model names **and** to
                         ``baseline_col`` when ``include_baseline=True``.

        group_boundaries: ``(start, end)`` column-centre pairs for bracket
                          annotations above the heatmap. Supports the same
                          explicit-span and contiguous-group conventions as
                          :func:`cause_accuracy_heatmap`. Group lines are drawn
                          only over their own columns.
        group_sizes:      Alternative to ``group_boundaries``. One integer per
                          group giving the number of displayed model columns in
                          that group. Example: ``group_sizes=[5, 4]`` generates
                          spans ``[(0.5, 4.5), (5.5, 8.5)]`` automatically.
        group_labels:    One string per group boundary.

        cmap:            Diverging colormap.  Default ``"RdBu_r"``.
        vmin:            Colormap lower bound.  Defaults to symmetric around the
                         data range (i.e. ``-max_abs``).
        vmax:            Colormap upper bound.  Defaults to ``max_abs``.
                         Pass explicit ``vmin``/``vmax`` to pin the scale.
        percentage:      Compute and display accuracy as percentage.  Default True.
        annot:           Annotate cells.  Default True.
        fmt:             Cell annotation format string.  Default ``".1f"``.
        figsize:         Figure size ``(width, height)``.  Default ``(9, 4.5)``.
        cbar_label:      Colorbar label.  Default: ``"Difference in accuracy (%)"``
                         or ``"Difference in accuracy"`` depending on ``percentage``.
        x_rotation:      X-axis tick rotation.  Default 45.
        y_rotation:      Y-axis tick rotation.  Default 0.
        show_n:          Append true-label sample size to each y-axis tick,
                         e.g. ``"HIV/AIDS (n=123)"``. Counts are taken from
                         ``df[true_col]`` when ``df`` is provided, otherwise
                         from the ``"true_label"`` column of the first entry
                         in ``topk_dfs``. Default False.
        insilicova_first:
                         When True (default), any model whose name contains
                         ``"insilicova"`` is moved to the far-left side of the
                         heatmap before plotting and grouping.
        save_path:       Path to save the figure.  Supports .png, .pdf, .svg.
        dpi:             Save resolution.  Default 150.

    Returns:
        ``(fig, ax, diff_df)`` — Figure, Axes, and the cause × model difference
        DataFrame (before display renaming; ``baseline_col`` excluded unless
        ``include_baseline=True``).

    Raises:
        ValueError: If ``baseline_col`` is not found, ``top_k > 1`` without
                    ``topk_dfs``, or mismatched group parameters.

    Example::

        from multimodalva.results import cause_accuracy_diff_heatmap

        fig, ax, diff = cause_accuracy_diff_heatmap(
            df=pred_df,
            true_col="true_label",
            baseline_col="insilicova",
            model_cols=["lgbm", "bert", "ensemble_data", "ensemble_feat", "ensemble_stack"],
            cause_order=["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"],
            model_rename={
                "lgbm":           "LightGBM",
                "bert":           "BioClinicalBERT",
                "ensemble_data":  "A - Data fusion",
                "ensemble_feat":  "B - Feature fusion",
                "ensemble_stack": "C - Stacking",
            },
            group_boundaries=[(0.5, 1.5), (2.5, 4.5)],
            group_labels=["Unimodal", "Multimodal"],
            save_path="figures/accuracy_diff.png",
        )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if df is None and topk_dfs is None:
        raise ValueError("At least one of df or topk_dfs must be provided.")
    if top_k > 1 and topk_dfs is None:
        raise ValueError(
            "top_k > 1 requires topk_dfs.  "
            "Build per-model topk DataFrames with topk_from_full(result.full, result.id2label, k=top_k)."
        )
    if top_k < 1:
        raise ValueError(f"top_k must be ≥ 1, got {top_k}.")

    # ------------------------------------------------------------------
    # Step 1: build the full accuracy matrix (baseline + all comparison models)
    # ------------------------------------------------------------------
    # Ensure baseline_col is always computed when coming from df
    if df is not None and model_cols is not None:
        all_cols: list[str] | None = (
            [baseline_col] + [c for c in model_cols if c != baseline_col]
        )
    else:
        all_cols = model_cols  # None → compute everything

    full_acc = _compute_cause_accuracy_matrix(
        df=df, true_col=true_col, model_cols=all_cols,
        top_k=top_k, topk_dfs=topk_dfs, percentage=percentage,
        insilicova_first=insilicova_first,
    )

    # ------------------------------------------------------------------
    # Step 2: extract baseline and compute per-cause differences
    # ------------------------------------------------------------------
    if baseline_col not in full_acc.columns:
        raise ValueError(
            f"baseline_col {baseline_col!r} not found in the computed accuracy matrix.  "
            "Check that it appears in df (as a column) or in topk_dfs (as a key)."
        )

    baseline_acc = full_acc[baseline_col]

    # Comparison columns: everything except the baseline
    if model_cols is not None:
        comp_cols = [c for c in model_cols if c != baseline_col]
    else:
        comp_cols = [c for c in full_acc.columns if c != baseline_col]

    if not comp_cols:
        raise ValueError(
            "No comparison models found after excluding baseline_col.  "
            "Pass at least one model in model_cols other than baseline_col."
        )

    diff_df = full_acc[comp_cols].subtract(baseline_acc, axis=0)

    # Optionally prepend baseline column (zeros) so label is visible in plot
    if include_baseline:
        diff_df.insert(0, baseline_col, 0.0)

    if group_boundaries is not None and group_sizes is not None:
        raise ValueError("Pass only one of group_boundaries or group_sizes, not both.")

    # ------------------------------------------------------------------
    # Step 3: cause ordering / filtering
    # ------------------------------------------------------------------
    if cause_order is not None:
        diff_df = diff_df.reindex(cause_order)

    if drop_causes:
        diff_df = diff_df.drop(index=drop_causes, errors="ignore")

    diff_df = diff_df.dropna(how="all")

    # ------------------------------------------------------------------
    # Step 3b: compute per-cause sample counts (used when show_n=True)
    # ------------------------------------------------------------------
    if show_n:
        label_counts = _resolve_true_label_counts(df=df, true_col=true_col, topk_dfs=topk_dfs)
    else:
        label_counts = None

    # ------------------------------------------------------------------
    # Step 4: rename for display (copy to keep diff_df clean)
    # ------------------------------------------------------------------
    display_df = diff_df.copy()
    if cause_rename:
        display_df = display_df.rename(index=cause_rename)
    if model_rename:
        display_df = display_df.rename(columns=model_rename)

    if group_sizes is not None:
        group_boundaries = _group_boundaries_from_sizes(group_sizes, len(display_df.columns))

    # ------------------------------------------------------------------
    # Step 5: symmetric colormap bounds centred at 0
    # ------------------------------------------------------------------
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("va_diff", HEATMAP_DIV)

    if vmin is None and vmax is None:
        max_abs = float(np.nanmax(np.abs(display_df.values)))
        vmin = -max_abs
        vmax =  max_abs

    if cbar_label is None:
        cbar_label = "Difference in accuracy (%)" if percentage else "Difference in accuracy"

    # ------------------------------------------------------------------
    # Step 6: plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    heatmap_kws: dict = dict(
        annot=annot,
        fmt=fmt,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.85},
    )

    sns.heatmap(display_df, **heatmap_kws)

    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=x_rotation)
    ax.tick_params(axis="y", rotation=y_rotation)

    if show_n:
        ytick_labels = [
            f"{display_cause} (n={label_counts.get(orig_cause, 0)})"
            for orig_cause, display_cause in zip(diff_df.index, display_df.index)
        ]
        ax.set_yticklabels(ytick_labels, rotation=y_rotation)

    # ------------------------------------------------------------------
    # Step 7: column grouping annotations above heatmap
    # ------------------------------------------------------------------
    if group_boundaries and group_labels:
        _draw_group_annotations(ax, group_boundaries, group_labels)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Cause accuracy diff heatmap saved to %s", save_path)

    return fig, ax, diff_df


# ---------------------------------------------------------------------------
# 8. Confusion heatmap
# ---------------------------------------------------------------------------

def confusion_heatmap(
    y_true,
    y_pred,
    label_order: list | dict | None = None,
    model_name: str = "Model",
    true_name: str = "True",
    label_abbr: bool = False,
    annot: bool = True,
    figsize: tuple = (10, 8),
    cmap=None,
    x_rotation: int = 90,
    y_rotation: int = 0,
    normalize: bool = False,
    save_path: str | None = None,
):
    """Heatmap of predicted vs. true labels for a single model.

    Rows = true labels (y-axis), columns = predicted labels (x-axis),
    cells = count (or proportion when ``normalize=True``).

    Args:
        y_true:       Ground-truth labels.  Series or array-like.
        y_pred:       Predicted labels.  Same shape as ``y_true``.
        label_order:  Controls the ordering of both axes:

                      - ``list``  — labels in the desired display order.
                        Missing labels get zero-count rows/cols.
                      - ``dict``  — ``{label: abbreviation}``.  Keys set the
                        order; when ``label_abbr=True``, values are used as
                        display names.
                      - ``None``  — alphabetical order.

        model_name:   Name shown on x-axis and in the title.
        true_name:    Name shown on y-axis.
        label_abbr:   Replace full label names with abbreviations from
                      ``label_order`` dict.  Ignored when ``label_order`` is
                      a list.  Default False.
        annot:        Annotate cells with counts/proportions.  Default True.
        figsize:      Figure size ``(width, height)``.  Default ``(10, 8)``.
        cmap:         Colour map.  Default ``"Blues"``.
        x_rotation:   X-axis tick rotation.  Default 90.
        y_rotation:   Y-axis tick rotation.  Default 0.
        normalize:    Normalise each *true-label row* to sum to 1 (shows
                      per-class recall distribution).  Default False.
        save_path:    If given, save figure to this path before showing.
                      Supports any matplotlib-supported extension (.png, .pdf,
                      .svg).  Default None (do not save).

    Returns:
        ``(fig, ax)`` — the matplotlib Figure and Axes objects, so callers can
        apply further customisation or save with a custom DPI.

    Example::

        fig, ax = confusion_heatmap(
            y_true=df["true_label"],
            y_pred=df["pred_bert"],
            label_order=["Malaria", "Pneumonia", "HIV/AIDS", "Diarrhoea", "Maternal"],
            model_name="BioClinicalBERT",
            normalize=True,
        )
        fig.savefig("heatmap_bert.png", dpi=150, bbox_inches="tight")
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score

    y_true = pd.Series(y_true).astype(str).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(str).reset_index(drop=True)

    # --- resolve label ordering and optional abbreviation mapping -----------
    abbr_map: dict | None = None
    if isinstance(label_order, dict):
        ordered_labels = list(label_order.keys())
        if label_abbr:
            abbr_map = {str(k): str(v) for k, v in label_order.items()}
    elif label_order is not None:
        ordered_labels = [str(lbl) for lbl in label_order]
    else:
        # Alphabetical over the union of true + predicted labels
        ordered_labels = sorted(set(y_true) | set(y_pred))

    if abbr_map:
        y_true = y_true.map(abbr_map).fillna(y_true)
        y_pred = y_pred.map(abbr_map).fillna(y_pred)
        ordered_labels = [abbr_map.get(lbl, lbl) for lbl in ordered_labels]

    # --- build cross-tabulation, fill in missing labels --------------------
    data_fig = pd.crosstab(y_true, y_pred)
    for lbl in ordered_labels:
        if lbl not in data_fig.index:
            data_fig.loc[lbl] = 0
        if lbl not in data_fig.columns:
            data_fig[lbl] = 0

    # Reorder to match label_order
    data_fig = data_fig.reindex(index=ordered_labels, columns=ordered_labels, fill_value=0)

    # --- optional row-normalisation (per-class recall distribution) --------
    fmt = "g"
    if normalize:
        row_sums = data_fig.sum(axis=1).replace(0, np.nan)
        data_fig = data_fig.div(row_sums, axis=0).fillna(0)
        fmt = ".2f"

    # --- accuracy for title -------------------------------------------------
    accuracy = accuracy_score(y_true.astype(str), y_pred.astype(str))

    # --- plot ---------------------------------------------------------------
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("va_confusion", HEATMAP_SEQ)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data_fig,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    title_suffix = " (row-normalised)" if normalize else ""
    ax.set_title(
        f"{model_name} — Accuracy: {accuracy * 100:.2f}%{title_suffix}",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel(model_name, fontsize=11)
    ax.set_ylabel(true_name, fontsize=11)
    ax.tick_params(axis="x", rotation=x_rotation)
    ax.tick_params(axis="y", rotation=y_rotation)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Heatmap saved to %s", save_path)

    return fig, ax
