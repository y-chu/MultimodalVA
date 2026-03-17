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

    cause_accuracy_heatmap(df, true_col, model_cols, top_k, topk_dfs, ...)
        Cause-specific accuracy heatmap across selected models.
        Rows = causes, columns = models, cells = % correct.
        Supports top-1 (default) or top-k accuracy; optional cause ordering,
        renaming, filtering, and column grouping annotations.

    cause_accuracy_diff_heatmap(df, true_col, baseline_col, model_cols, ...)
        Same layout as cause_accuracy_heatmap but cells show the difference
        accuracy(model) − accuracy(baseline).  Diverging colormap centred at 0.
        Optional include_baseline column (zeros) to keep baseline label visible.

    confusion_heatmap(y_true, y_pred, label_order, ...)
        Seaborn heatmap of predicted vs. true labels for one model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..utils.metrics import log_loss_from_full

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


_DEFAULT_METRICS = [
    "accuracy", "balanced_accuracy",
    "f1_macro", "f1_weighted",
    "precision_macro", "precision_weighted",
    "recall_macro", "recall_weighted",
    "csmf_accuracy", "cccsmf_accuracy",
]


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
            metrics=[*_DEFAULT_METRICS, "log_loss"],
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

    metrics = metrics or _DEFAULT_METRICS
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
    model_accs: dict[str, pd.Series] = {}

    # --- step A: accuracy from wide df (exact match, top-1) -----------------
    if df is not None:
        if true_col not in df.columns:
            raise ValueError(f"true_col {true_col!r} not found in df.")
        cols = model_cols if model_cols is not None else [
            c for c in df.columns if c not in {true_col, "id"}
        ]
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

    return pd.DataFrame(model_accs)


def _draw_group_annotations(
    ax,
    fig,
    group_boundaries: list[tuple[float, float]],
    group_labels: list[str],
    fontsize: int = 10,
) -> None:
    """Draw bracket lines and labels below the x-axis to annotate column groups.

    Uses ``ax.get_xaxis_transform()`` so x-coordinates are in heatmap column-index
    space (e.g. 0.5 → between column 0 and 1) and y-coordinates are in axes
    fraction (0 = bottom of axes box, negative = below it).  ``clip_on=False``
    lets the annotations extend outside the axes boundaries.

    Args:
        ax:               Matplotlib Axes.
        fig:              Matplotlib Figure (used to adjust bottom margin).
        group_boundaries: List of ``(start, end)`` column-index pairs.
        group_labels:     One label per group; must match len(group_boundaries).
        fontsize:         Label font size.  Default 10.
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
                "Example: group_boundaries=[(0.5, 1.5), (2.5, 3.5)]"
            )

    transform = ax.get_xaxis_transform()  # x=data coords, y=axes fraction
    y_line = -0.05   # just below the axes bottom edge (axes fraction)
    y_text = -0.14   # text sits further below the line

    for (start, end), label in zip(group_boundaries, group_labels):
        ax.plot(
            [start, end], [y_line, y_line],
            color="black", linewidth=2,
            transform=transform, clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            (start + end) / 2, y_text, label,
            transform=transform,
            ha="center", va="top",
            fontsize=fontsize, clip_on=False,
        )

    # Reserve bottom margin so annotations are not cropped on save
    fig.subplots_adjust(bottom=0.22)


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

        group_boundaries: List of ``(start, end)`` pairs in heatmap column-index
                          coordinates marking model groups.  For example,
                          ``[(0.5, 3.5), (4.5, 6.5)]`` draws a bracket under
                          columns 1–3 and 5–6 (0-based, 0.5-offset for centering).
                          Drawn as horizontal lines just below the x-axis ticks.
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

    # ------------------------------------------------------------------
    # Steps 1–3: compute per-cause accuracy for each model
    # ------------------------------------------------------------------
    accuracy_df = _compute_cause_accuracy_matrix(
        df=df, true_col=true_col, model_cols=model_cols,
        top_k=top_k, topk_dfs=topk_dfs, percentage=percentage,
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
    # Step 5: rename for display (work on a copy to keep accuracy_df clean)
    # ------------------------------------------------------------------
    display_df = accuracy_df.copy()
    if cause_rename:
        display_df = display_df.rename(index=cause_rename)
    if model_rename:
        display_df = display_df.rename(columns=model_rename)

    # ------------------------------------------------------------------
    # Step 6: colormap
    # ------------------------------------------------------------------
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "va_cause_accuracy",
            [(255 / 255, 245 / 255, 242 / 255), "#D8453B"],
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

    # ------------------------------------------------------------------
    # Step 8: column grouping annotations below x-axis
    # ------------------------------------------------------------------
    if group_boundaries and group_labels:
        _draw_group_annotations(ax, fig, group_boundaries, group_labels)

    plt.tight_layout(rect=[0, 0.05 if group_boundaries else 0, 1, 1])

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
    group_labels: list[str] | None = None,
    # --- aesthetics ---
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    percentage: bool = True,
    annot: bool = True,
    fmt: str = ".1f",
    figsize: tuple[float, float] = (9, 4.5),
    cbar_label: str | None = None,
    x_rotation: int = 45,
    y_rotation: int = 0,
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

        group_boundaries: ``(start, end)`` column-index pairs for bracket
                          annotations below the x-axis ticks.
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

    # ------------------------------------------------------------------
    # Step 3: cause ordering / filtering
    # ------------------------------------------------------------------
    if cause_order is not None:
        diff_df = diff_df.reindex(cause_order)

    if drop_causes:
        diff_df = diff_df.drop(index=drop_causes, errors="ignore")

    diff_df = diff_df.dropna(how="all")

    # ------------------------------------------------------------------
    # Step 4: rename for display (copy to keep diff_df clean)
    # ------------------------------------------------------------------
    display_df = diff_df.copy()
    if cause_rename:
        display_df = display_df.rename(index=cause_rename)
    if model_rename:
        display_df = display_df.rename(columns=model_rename)

    # ------------------------------------------------------------------
    # Step 5: symmetric colormap bounds centred at 0
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 7: column grouping annotations below x-axis
    # ------------------------------------------------------------------
    if group_boundaries and group_labels:
        _draw_group_annotations(ax, fig, group_boundaries, group_labels)

    plt.tight_layout(rect=[0, 0.05 if group_boundaries else 0, 1, 1])

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
    cmap: str = "Blues",
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
