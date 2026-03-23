"""
Training and HPO result visualization.

Public API
----------
    plot_loss_curves(log_history, skip_steps, warmup_steps)
        Training/evaluation loss curves from a HuggingFace Trainer log.
        Optional warmup boundary annotation and post-warmup slope check.

    hpo_leaderboard(source, sort_by, top_n, ascending, param_cols)
        Formatted HPO trial table from an Optuna study or trials DataFrame.

    oov_rate(texts, tokenizer, plot, top_k)
        Tokenizer [UNK] rate across a text corpus — diagnoses checkpoint
        domain mismatch before or after training.

    hpo_loss_trend(source, metric, plot)
        How the Optuna objective (or any stored metric) evolves across trials.
        Reveals whether HPO is converging.

    hpo_convergence_plot(source, metric, save_path, plot)
        Line + marker convergence plot; optional file save.  x = sequential
        trial index; horizontal dashed line at the best value seen.

    hpo_metric_variance(source, metric_cols, plot)
        Distribution of every metric stored in an Optuna study.
        Box + swarm plot; returns per-metric mean / std / CV summary.

    hyperparameter_importance(source, evaluator, plot)
        Ranked hyperparameter importance from an Optuna study.
        Uses optuna.importance when available, otherwise Pearson correlation.

    plot_param_importances(source, study_name, metric, kind, params, save_path, plot)
        Interactive Plotly visualization via optuna.visualization.
        Accepts a Study object, a path to an SQLite .db file, or a
        {"study_name": ..., "storage": ...} dict.  Supports five plot kinds:
        "importance" (default), "parallel_coordinate", "contour", "slice",
        "optimization_history".  Saves to .html (interactive) or .png/.pdf
        (requires kaleido).

    train_eval_gap(log_history, skip_steps, plot)
        Gap between training loss and evaluation loss at matched steps.
        Large positive gaps indicate overfitting.

    loss_curve_diagnostics(log_history, warmup_steps)
        Quantitative checks on a loss curve: post-warmup slope, stabilisation,
        and train/eval gap — returns a diagnostics dict without plotting.

"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    log_history: list[dict],
    skip_steps: int = 0,
    warmup_steps: int = 0,
) -> None:
    """Plot training and evaluation loss curves from a HuggingFace log history.

    The log_history comes directly from train() metadata:
        metadata["log_history"]
    or from a live trainer:
        trainer.state.log_history

    When ``warmup_steps`` is set, a vertical dashed line marks the end of the
    warmup phase.  A short annotation reports whether training loss drops
    steeply in the first 20 % of post-warmup steps and then stabilises — the
    healthy pattern for a well-fitted checkpoint.  See also
    ``loss_curve_diagnostics()`` for a programmatic version of these checks.

    Args:
        log_history:   List of dicts logged by the HuggingFace Trainer.
        skip_steps:    Ignore all entries at or below this step number.
                       Useful for skipping noisy early steps. Default 0.
        warmup_steps:  Number of LR warmup steps.  When > 0, draws a vertical
                       dashed line and annotates the post-warmup slope check.
                       Default 0 (no annotation).
    """
    df = pd.DataFrame(log_history)

    df_loss = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
    df_eval = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()

    if skip_steps > 0:
        df_loss = df_loss[df_loss["step"] > skip_steps]
        df_eval = df_eval[df_eval["step"] > skip_steps]

    if df_loss.empty and df_eval.empty:
        raise ValueError("No loss values found in log_history.")

    _, ax = plt.subplots(figsize=(10, 6))
    if not df_loss.empty:
        ax.plot(df_loss["step"], df_loss["loss"], label="Training Loss")
    if not df_eval.empty:
        ax.plot(df_eval["step"], df_eval["eval_loss"], label="Evaluation Loss")

    if warmup_steps > 0 and not df_loss.empty:
        ax.axvline(warmup_steps, color="grey", linestyle="--", linewidth=1,
                   label=f"Warmup end ({warmup_steps})")

        # Post-warmup slope check: compare mean loss in first 20 % vs last 20 %
        # of the post-warmup window
        post = df_loss[df_loss["step"] > warmup_steps].sort_values("step")
        if len(post) >= 4:
            split = max(1, len(post) // 5)
            early_mean = post["loss"].iloc[:split].mean()
            late_mean  = post["loss"].iloc[-split:].mean()
            drop_pct   = (early_mean - late_mean) / (early_mean + 1e-12) * 100
            if drop_pct > 15:
                note = f"✓ Post-warmup drop {drop_pct:.0f} % (steep then stabilise)"
                colour = "green"
            elif drop_pct > 0:
                note = f"△ Post-warmup drop {drop_pct:.0f} % (modest)"
                colour = "orange"
            else:
                note = f"✗ Loss not decreasing post-warmup ({drop_pct:.0f} %)"
                colour = "red"
            ax.annotate(
                note,
                xy=(0.02, 0.05),
                xycoords="axes fraction",
                color=colour,
                fontsize=9,
            )

    title = "Loss Curves" + (f" (after step {skip_steps})" if skip_steps > 0 else "")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. HPO leaderboard
# ---------------------------------------------------------------------------

def hpo_leaderboard(
    source,
    sort_by: str | None = None,
    top_n: int | None = None,
    ascending: bool = False,
    param_cols: bool = True,
) -> pd.DataFrame:
    """Format HPO trial results as a ranked leaderboard.

    Works with both Optuna ``Study`` objects and the DataFrame returned by
    ``study.trials_dataframe()``.

    Args:
        source:    An ``optuna.Study`` or a ``pd.DataFrame`` from
                   ``study.trials_dataframe()``.
        sort_by:   Column to sort by.  Defaults to ``"value"`` (Optuna
                   objective column) when present; falls back to the first
                   available metric column for Ray Tune DataFrames (which
                   have no ``"value"`` column).  Can be set explicitly to any
                   metric name (e.g. ``"f1_macro"``, ``"csmf_accuracy"``).
        top_n:     Return only the top N trials.  ``None`` = all trials.
        ascending: Sort direction.  Default ``False`` (highest score first).
        param_cols: Include hyperparameter columns (``params_*``) in the
                    output.  Default True.

    Returns:
        DataFrame of completed trials, sorted and optionally trimmed.  Metric
        columns are renamed from ``user_attrs_metric`` → ``metric`` for
        readability.  Columns:

            ``trial``         — trial number
            ``value``         — Optuna objective value
            ``duration``      — wall-clock time per trial
            ``<metrics>``     — all 4 metrics stored as user attributes
            ``params_<name>`` — hyperparameter values (when param_cols=True)

    Raises:
        ValueError: If ``sort_by`` column is not found in the trials DataFrame.
    """
    if hasattr(source, "trials_dataframe"):
        # Optuna Study object
        trials_df = source.trials_dataframe()
    elif isinstance(source, pd.DataFrame):
        trials_df = source.copy()
    else:
        raise TypeError(
            f"source must be an optuna.Study or pd.DataFrame, got {type(source).__name__}."
        )

    # Keep only completed trials
    if "state" in trials_df.columns:
        trials_df = trials_df[trials_df["state"] == "COMPLETE"].copy()

    if trials_df.empty:
        logger.warning("No completed trials found.")
        return trials_df

    # Rename user_attrs_* → metric name for readability
    user_attr_rename = {
        c: c.replace("user_attrs_", "")
        for c in trials_df.columns
        if c.startswith("user_attrs_")
    }
    trials_df = trials_df.rename(columns=user_attr_rename)

    # Rename "number" → "trial"; for Ray Tune CSVs (no "number") create sequential index
    if "number" in trials_df.columns:
        trials_df = trials_df.rename(columns={"number": "trial"})
    elif "trial" not in trials_df.columns:
        trials_df = trials_df.reset_index(drop=True)
        trials_df.insert(0, "trial", trials_df.index)

    # Select columns: trial, value, duration, metrics, [params_*]
    priority_cols = ["trial", "value", "duration"]
    metric_cols   = list(user_attr_rename.values())
    param_columns = [c for c in trials_df.columns if c.startswith("params_")]

    keep = [c for c in priority_cols if c in trials_df.columns]
    keep += [c for c in metric_cols  if c in trials_df.columns]
    if param_cols:
        keep += param_columns
    # Any remaining columns not yet included
    keep += [
        c for c in trials_df.columns
        if c not in keep and c not in {"state", "datetime_start", "datetime_complete"}
    ]

    trials_df = trials_df[keep]

    # Determine sort column.
    # "value" is the Optuna objective column.  Ray Tune DataFrames omit it and
    # expose metrics directly (e.g. "f1_macro").  Fall back to the first available
    # metric column; skip sorting if none can be found.
    if sort_by is None:
        if "value" in trials_df.columns:
            sort_col = "value"
        elif metric_cols and metric_cols[0] in trials_df.columns:
            sort_col = metric_cols[0]
            logger.debug(
                "No 'value' column found; sorting by first metric column %r.", sort_col
            )
        else:
            sort_col = None
            logger.warning(
                "No sort column found (no 'value' and no user-attr metrics). "
                "Returning trials in original order."
            )
    elif sort_by in trials_df.columns:
        sort_col = sort_by
    else:
        raise ValueError(
            f"sort_by={sort_by!r} not found in trials.  "
            f"Available columns: {list(trials_df.columns)}."
        )

    if sort_col is not None:
        trials_df = trials_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    else:
        trials_df = trials_df.reset_index(drop=True)

    if top_n is not None:
        trials_df = trials_df.head(top_n)

    return trials_df


# ---------------------------------------------------------------------------
# Helper shared by HPO diagnostic functions
# ---------------------------------------------------------------------------

def _load_trials(source) -> pd.DataFrame:
    """Return a completed-trials DataFrame from an Optuna study or DataFrame."""
    if hasattr(source, "trials_dataframe"):
        df = source.trials_dataframe()
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError(
            f"source must be an optuna.Study or pd.DataFrame, "
            f"got {type(source).__name__}."
        )
    if "state" in df.columns:
        df = df[df["state"] == "COMPLETE"].copy()
    # Rename user_attrs_* → short name
    df = df.rename(columns={
        c: c.replace("user_attrs_", "")
        for c in df.columns
        if c.startswith("user_attrs_")
    })
    if "number" in df.columns:
        df = df.rename(columns={"number": "trial"})
    elif "trial" not in df.columns:
        # Ray Tune CSVs have no "number" column; use a sequential index so
        # callers that sort/select on "trial" always find the column.
        df = df.reset_index(drop=True)
        df.insert(0, "trial", df.index)
    return df


# ---------------------------------------------------------------------------
# 3. Tokenizer OOV / [UNK] rate
# ---------------------------------------------------------------------------

def oov_rate(
    texts: list[str],
    tokenizer,
    *,
    plot: bool = True,
    top_k: int = 20,
) -> dict:
    """Estimate the [UNK] token rate to diagnose checkpoint domain mismatch.

    Tokenizes ``texts`` with the supplied HuggingFace tokenizer and counts how
    many tokens are mapped to the unknown-token ID.  A high OOV rate (> ~5 %)
    suggests the model checkpoint was pre-trained on a very different domain
    and may benefit from domain-adaptive pre-training or a different backbone.

    Args:
        texts:     List of raw text strings (e.g. ``train_df["narrative"].tolist()``).
        tokenizer: A loaded HuggingFace ``PreTrainedTokenizer`` / ``PreTrainedTokenizerFast``.
                   Load with ``AutoTokenizer.from_pretrained(model_name)``.
        plot:      Show a bar chart of UNK rate per document (top_k highest).
                   Default True.
        top_k:     Number of highest-OOV documents to show in the bar chart.
                   Default 20.

    Returns:
        dict with keys:

            ``total_tokens``  — total token count across all texts
            ``unk_tokens``    — number of [UNK] tokens
            ``unk_rate``      — fraction of tokens that are [UNK]  (0–1)
            ``per_text``      — list of per-text OOV fractions (same order as ``texts``)

    Raises:
        ValueError: If ``texts`` is empty.

    Example::

        from transformers import AutoTokenizer
        from multimodalva.results import oov_rate

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
        stats = oov_rate(train_df["narrative"].dropna().tolist(), tokenizer)
        print(f"OOV rate: {stats['unk_rate']:.2%}")
    """
    if not texts:
        raise ValueError("texts is empty.")

    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        logger.warning(
            "Tokenizer has no unk_token_id (e.g. SentencePiece / BPE). "
            "OOV rate is not meaningful — returning zeros."
        )
        return {"total_tokens": 0, "unk_tokens": 0, "unk_rate": 0.0, "per_text": [0.0] * len(texts)}

    per_text: list[float] = []
    total_tokens = 0
    total_unk    = 0

    for text in texts:
        ids    = tokenizer.encode(str(text), add_special_tokens=True)
        n_unk  = sum(1 for t in ids if t == unk_id)
        total_tokens += len(ids)
        total_unk    += n_unk
        per_text.append(n_unk / len(ids) if ids else 0.0)

    unk_rate = total_unk / total_tokens if total_tokens else 0.0

    logger.info(
        "OOV rate: %.2f%% (%d / %d tokens are [UNK])",
        unk_rate * 100, total_unk, total_tokens,
    )

    if plot:
        top_idx  = np.argsort(per_text)[::-1][:top_k]
        top_vals = [per_text[i] * 100 for i in top_idx]
        labels   = [f"doc {i}" for i in top_idx]

        _, ax = plt.subplots(figsize=(max(8, top_k * 0.5), 4))
        bars = ax.bar(range(len(top_vals)), top_vals, color="steelblue")
        ax.axhline(unk_rate * 100, color="red", linestyle="--",
                   label=f"Mean {unk_rate:.2%}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("[UNK] rate (%)")
        ax.set_title(
            f"Top-{len(top_vals)} documents by [UNK] rate  "
            f"(corpus mean: {unk_rate:.2%})"
        )
        ax.legend()
        # Annotate each bar with its value
        for bar, val in zip(bars, top_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7,
            )
        plt.tight_layout()
        plt.show()

    return {
        "total_tokens": total_tokens,
        "unk_tokens":   total_unk,
        "unk_rate":     unk_rate,
        "per_text":     per_text,
    }


# ---------------------------------------------------------------------------
# 4. HPO loss / metric trend across trials
# ---------------------------------------------------------------------------

def hpo_loss_trend(
    source,
    *,
    metric: str | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    """Plot how the Optuna objective (or a stored metric) evolves across trials.

    Reveals whether HPO is converging: a healthy run shows the objective
    improving over early trials and then flattening as the search space is
    exhausted.  A flat or worsening trend may indicate a search space that
    is too narrow or too wide.

    Args:
        source:  An ``optuna.Study`` or ``pd.DataFrame`` from
                 ``study.trials_dataframe()``.
        metric:  Column name to plot.  Defaults to ``"value"`` (the Optuna
                 objective).  Can be any user attribute stored per trial
                 (e.g. ``"accuracy"``, ``"f1_macro"``).
        plot:    Show the trend chart.  Default True.

    Returns:
        DataFrame with columns ``trial``, ``<metric>``, and
        ``best_so_far`` (running maximum).

    Raises:
        ValueError: If ``metric`` column is not found in the trials DataFrame.
    """
    trials_df = _load_trials(source)
    if trials_df.empty:
        logger.warning("No completed trials to plot.")
        return trials_df

    col = metric or "value"
    if col not in trials_df.columns:
        if metric is not None:
            raise ValueError(
                f"metric={col!r} not found.  "
                f"Available columns: {list(trials_df.columns)}."
            )
        # Auto-fallback for Ray Tune DataFrames, which expose metric columns
        # directly and have no "value" (Optuna objective) column.
        _candidates = ["accuracy", "f1_macro", "f1_weighted", "csmf_accuracy",
                       "balanced_accuracy", "log_loss"]
        col = next((m for m in _candidates if m in trials_df.columns), None)
        if col is None:
            logger.warning(
                "No 'value' column and no standard metric columns found.  "
                "Available columns: %s", list(trials_df.columns),
            )
            return pd.DataFrame()
        logger.info(
            "No 'value' column found (Ray Tune DataFrame?); "
            "using %r as the trend metric.", col,
        )

    out = trials_df[["trial", col]].dropna(subset=[col]).sort_values("trial").copy()
    out["best_so_far"] = out[col].cummax()

    if plot:
        _, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(out["trial"], out[col], s=30, alpha=0.7, label=col)
        ax.plot(out["trial"], out["best_so_far"], color="red",
                linewidth=1.5, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel(col)
        ax.set_title(f"HPO metric trend — {col}")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4b. HPO convergence plot (line + markers, optional save)
# ---------------------------------------------------------------------------

def hpo_convergence_plot(
    source,
    *,
    metric: str | None = None,
    save_path=None,
    plot: bool = True,
) -> "plt.Figure":
    """Line + marker plot of HPO trial values — the canonical convergence view.

    Each trial is plotted in the order it was run (x = sequential index,
    not trial number) so the curve directly shows how the search improves
    over time.  A horizontal dashed line marks the best value seen.

    Compared to ``hpo_loss_trend()``, which uses a scatter + cumulative-max
    overlay, this function uses a connected ``"o-"`` style and supports
    saving to disk — matching the typical offline reporting workflow.

    Args:
        source:     An ``optuna.Study`` or ``pd.DataFrame`` from
                    ``study.trials_dataframe()``.
        metric:     Column to plot.  Defaults to ``"value"`` (the Optuna
                    objective).  Any user attribute works (e.g.
                    ``"f1_macro"``, ``"csmf_accuracy"``).
        save_path:  File path to save the figure (e.g.
                    ``"runs/hpo/hpo_convergence.png"``).  When ``None``
                    the figure is not saved.  Supports any format accepted
                    by ``matplotlib.savefig``.
        plot:       Call ``plt.show()``.  Set ``False`` when saving
                    headlessly.  Default True.

    Returns:
        The ``matplotlib.figure.Figure`` object.

    Raises:
        ValueError: If ``metric`` column is not found in the trials DataFrame.

    Example::

        from multimodalva.results import hpo_convergence_plot

        fig = hpo_convergence_plot(study, metric="f1_macro",
                                   save_path="runs/hpo/convergence.png",
                                   plot=False)
    """
    trials_df = _load_trials(source)
    if trials_df.empty:
        logger.warning("No completed trials to plot.")
        return plt.figure()

    col = metric or "value"
    if col not in trials_df.columns:
        if metric is not None:
            raise ValueError(
                f"metric={col!r} not found.  "
                f"Available columns: {list(trials_df.columns)}."
            )
        # Auto-fallback for Ray Tune DataFrames, which expose metric columns
        # directly and have no "value" (Optuna objective) column.
        _candidates = ["accuracy", "f1_macro", "f1_weighted", "csmf_accuracy",
                       "balanced_accuracy", "log_loss"]
        col = next((m for m in _candidates if m in trials_df.columns), None)
        if col is None:
            logger.warning(
                "No 'value' column and no standard metric columns found.  "
                "Available columns: %s", list(trials_df.columns),
            )
            return plt.figure()
        logger.info(
            "No 'value' column found (Ray Tune DataFrame?); "
            "using %r as the convergence metric.", col,
        )

    vals = trials_df.sort_values("trial")[col].dropna().values
    best = vals.max()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(vals)), vals, "o-", label=col)
    ax.axhline(best, color="red", linestyle="--",
               label=f"best = {best:.4f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel(col)
    ax.set_title(f"HPO trial values — {col}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Convergence plot saved to %s", save_path)

    if plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# 5. Val metric variance across trials
# ---------------------------------------------------------------------------

def hpo_metric_variance(
    source,
    *,
    metric_cols: list[str] | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    """Show the distribution of every metric stored across Optuna trials.

    High variance across trials for a metric that is NOT the objective
    suggests that some hyperparameter configurations are strongly unstable —
    worth investigating via ``hyperparameter_importance()``.

    Args:
        source:      An ``optuna.Study`` or ``pd.DataFrame`` from
                     ``study.trials_dataframe()``.
        metric_cols: Columns to include.  Defaults to all recognised metric
                     columns: ``value``, ``accuracy``, ``f1_macro``,
                     ``f1_weighted``, ``csmf_accuracy``.  Any subset that
                     exists in the trials DataFrame will be shown.
        plot:        Show a box + jitter plot for each metric.  Default True.

    Returns:
        DataFrame indexed by metric name with columns
        ``mean``, ``std``, ``min``, ``max``, ``cv`` (coefficient of variation).
    """
    trials_df = _load_trials(source)
    if trials_df.empty:
        logger.warning("No completed trials found.")
        return pd.DataFrame()

    default_cols = ["value", "accuracy", "f1_macro", "f1_weighted",
                    "csmf_accuracy", "f1_weighted"]
    candidates   = metric_cols if metric_cols is not None else default_cols
    cols         = [c for c in candidates if c in trials_df.columns]
    if not cols:
        raise ValueError(
            f"None of the requested metric columns {candidates} found in trials. "
            f"Available: {list(trials_df.columns)}."
        )

    rows = []
    for c in cols:
        vals = trials_df[c].dropna()
        rows.append({
            "metric": c,
            "mean":   vals.mean(),
            "std":    vals.std(),
            "min":    vals.min(),
            "max":    vals.max(),
            "cv":     vals.std() / vals.mean() if vals.mean() != 0 else float("nan"),
        })
    summary = pd.DataFrame(rows).set_index("metric")

    if plot:
        data = [trials_df[c].dropna().values for c in cols]
        _, ax = plt.subplots(figsize=(max(6, len(cols) * 1.8), 5))
        bp = ax.boxplot(data, labels=cols, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        # Jitter overlay
        for i, (c, d) in enumerate(zip(cols, data), start=1):
            jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(d))
            ax.scatter(i + jitter, d, s=15, alpha=0.5, color="steelblue", zorder=3)
        ax.set_ylabel("Metric value")
        ax.set_title("HPO metric variance across trials")
        ax.grid(True, axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return summary


# ---------------------------------------------------------------------------
# 6. Hyperparameter importance
# ---------------------------------------------------------------------------

def hyperparameter_importance(
    source,
    *,
    evaluator: str = "auto",
    plot: bool = True,
) -> pd.DataFrame:
    """Rank hyperparameters by their importance for the Optuna objective.

    Two backends are supported:

    * ``"optuna"``  — uses ``optuna.importance.get_param_importances()``
      (FAnova or Mean Decrease Impurity; requires ``optuna >= 3.0``).
    * ``"pearson"`` — Pearson |correlation| between each ``params_*`` column
      and the objective value.  Always available; less statistically rigorous
      but useful when too few trials exist for FAnova.
    * ``"auto"``    — tries ``"optuna"``; falls back to ``"pearson"``.

    Args:
        source:    An ``optuna.Study`` object (required for the ``"optuna"``
                   backend) or a ``pd.DataFrame`` from
                   ``study.trials_dataframe()`` (``"pearson"`` only).
        evaluator: ``"auto"`` | ``"optuna"`` | ``"pearson"``.  Default ``"auto"``.
        plot:      Show a horizontal bar chart.  Default True.

    Returns:
        DataFrame with columns ``param`` and ``importance``, sorted
        descending by importance.

    Raises:
        ValueError: If ``evaluator="optuna"`` but ``source`` is not a Study,
                    or if no ``params_*`` columns are found for Pearson.
    """
    if evaluator not in {"auto", "optuna", "pearson"}:
        raise ValueError(
            f"evaluator must be 'auto', 'optuna', or 'pearson', got {evaluator!r}."
        )

    importance: dict[str, float] = {}
    used_backend = evaluator

    # --- Optuna backend ---
    if evaluator in {"auto", "optuna"}:
        if not hasattr(source, "trials_dataframe"):
            if evaluator == "optuna":
                raise ValueError(
                    "evaluator='optuna' requires an optuna.Study object, "
                    "not a DataFrame.  Pass the Study directly or use evaluator='pearson'."
                )
        else:
            try:
                import optuna  # noqa: PLC0415
                imp = optuna.importance.get_param_importances(source)
                importance = dict(imp)
                used_backend = "optuna"
            except Exception as exc:  # noqa: BLE001
                if evaluator == "optuna":
                    raise
                logger.warning(
                    "optuna importance failed (%s); falling back to Pearson.", exc
                )
                used_backend = "pearson"

    # --- Pearson fallback ---
    if not importance:
        trials_df = _load_trials(source)
        param_cols = [c for c in trials_df.columns if c.startswith("params_")]
        if not param_cols:
            raise ValueError(
                "No 'params_*' columns found in trials DataFrame.  "
                "Cannot compute Pearson importance."
            )
        obj = trials_df["value"].dropna()
        for col in param_cols:
            vals = pd.to_numeric(trials_df[col], errors="coerce")
            both = pd.concat([obj, vals], axis=1).dropna()
            if len(both) < 3:
                importance[col.replace("params_", "")] = float("nan")
                continue
            corr = np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]
            importance[col.replace("params_", "")] = abs(float(corr))
        used_backend = "pearson"

    out = (
        pd.DataFrame(list(importance.items()), columns=["param", "importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if plot:
        _, ax = plt.subplots(figsize=(7, max(3, len(out) * 0.45)))
        ax.barh(out["param"][::-1], out["importance"][::-1], color="steelblue")
        ax.set_xlabel(
            "Importance (FAnova)" if used_backend == "optuna"
            else "Importance (|Pearson r|)"
        )
        ax.set_title("Hyperparameter importance")
        ax.grid(True, axis="x", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return out


# ---------------------------------------------------------------------------
# 6b. Optuna Plotly visualizations
# ---------------------------------------------------------------------------

_PLOT_KINDS = frozenset({
    "importance",
    "parallel_coordinate",
    "contour",
    "slice",
    "optimization_history",
})


def plot_param_importances(
    source,
    *,
    study_name: str = "text_hpo",
    metric: str | None = None,
    kind: str = "importance",
    params: list[str] | None = None,
    save_path=None,
    plot: bool = True,
):
    """Interactive Plotly visualization of HPO results via optuna.visualization.

    Produces a richer, interactive figure than the matplotlib-based
    ``hyperparameter_importance()``.  The figure is returned as a Plotly
    ``Figure`` object so it can be embedded in notebooks, saved as HTML, or
    exported to a static image.

    ``source`` can be any of:

    * An ``optuna.Study`` object — used directly.
    * A path to an SQLite ``.db`` file (``str`` or ``pathlib.Path``) —
      loaded with ``optuna.load_study(study_name, storage=...)``.
    * A ``dict`` with keys ``"study_name"`` and ``"storage"`` (SQLite URL or
      RDB URL) — e.g. ``{"study_name": "text_hpo",
      "storage": "sqlite:///runs/hpo.db"}``.

    Args:
        source:      See above.  Study object, ``.db`` path, or config dict.
        study_name:  Study name used when loading from a ``.db`` file or when
                     ``source`` is a dict without a ``"study_name"`` key.
                     Default ``"text_hpo"`` (matches our HPO default).
        metric:      User-attribute metric to use as the optimization target
                     for the importance / contour / slice plots.  ``None``
                     uses the Optuna objective value (default).  Pass a name
                     such as ``"f1_macro"`` or ``"csmf_accuracy"`` to rank
                     parameters by their effect on a secondary metric.
        kind:        Which Optuna visualization to produce:

                     * ``"importance"``            — bar chart of FAnova / MDI
                       hyperparameter importances (default).
                     * ``"parallel_coordinate"``   — parallel coordinates of
                       all hyperparameters coloured by objective value.
                     * ``"contour"``               — 2-D contour grid over
                       pairs of hyperparameters.
                     * ``"slice"``                 — 1-D slice plots for each
                       hyperparameter vs objective.
                     * ``"optimization_history"``  — trial-by-trial objective
                       with running best value overlaid.

        params:      Restrict the plot to this subset of hyperparameter names.
                     ``None`` = all parameters.  Useful for ``"contour"`` and
                     ``"slice"`` when the search space is large.
        save_path:   File path to save the figure.  Supports two formats:

                     * ``.html``         — interactive HTML (no extra deps).
                     * ``.png`` / ``.pdf`` / ``.svg`` — static image via
                       ``kaleido`` (``pip install kaleido``).

                     ``None`` = do not save.  Default ``None``.
        plot:        Call ``fig.show()`` to display the figure.  In Jupyter
                     the figure renders inline; in a script it opens a browser
                     tab.  Set ``False`` when saving headlessly.  Default True.

    Returns:
        ``plotly.graph_objects.Figure`` — the Plotly figure object.

    Raises:
        ImportError:  ``optuna`` is not installed, or ``kaleido`` is needed
                      for a static image format but is not installed.
        ValueError:   ``kind`` is not one of the supported values.
        FileNotFoundError: ``.db`` file does not exist.

    Example — load from SQLite and save interactive HTML::

        from multimodalva.results import plot_param_importances

        fig = plot_param_importances(
            "runs/hpo/hpo_allenai_longformer-base-4096.db",
            study_name="text_hpo",
            metric="f1_macro",
            kind="importance",
            save_path="runs/hpo/param_importance.html",
            plot=False,
        )

    Example — load study object directly::

        import optuna
        study = optuna.load_study(
            study_name="text_hpo",
            storage="sqlite:///runs/hpo/hpo_bert.db",
        )
        fig = plot_param_importances(study, kind="parallel_coordinate")
    """
    try:
        import optuna
        import optuna.visualization as ov
    except ImportError as exc:
        raise ImportError(
            "optuna is required for plot_param_importances().  "
            "Install with:  pip install 'optuna>=3.4'"
        ) from exc

    if kind not in _PLOT_KINDS:
        raise ValueError(
            f"kind={kind!r} is not supported.  "
            f"Choose from: {sorted(_PLOT_KINDS)}."
        )

    # --- Resolve source to an optuna.Study -----------------------------------
    if hasattr(source, "trials"):
        # Already a Study object
        study = source
    elif isinstance(source, dict):
        _sname   = source.get("study_name", study_name)
        _storage = source["storage"]
        logger.info("Loading Optuna study %r from %s", _sname, _storage)
        study = optuna.load_study(study_name=_sname, storage=_storage)
    else:
        # Treat as a path to a .db file
        from pathlib import Path as _Path
        db_path = _Path(source)
        if not db_path.exists():
            raise FileNotFoundError(
                f"SQLite database not found: {db_path}.  "
                "Pass the correct path or a study object directly."
            )
        storage_url = f"sqlite:///{db_path.resolve()}"
        logger.info("Loading Optuna study %r from %s", study_name, storage_url)
        study = optuna.load_study(study_name=study_name, storage=storage_url)

    # --- Build target callable when metric != objective ---------------------
    target = None
    target_name = metric
    if metric is not None:
        def target(trial: optuna.trial.FrozenTrial) -> float:  # noqa: E306
            return trial.user_attrs.get(metric, float("nan"))

    # --- Generate figure ----------------------------------------------------
    _kw_target = {}
    if target is not None:
        _kw_target = {"target": target, "target_name": target_name}

    _kw_params = {}
    if params is not None:
        _kw_params = {"params": params}

    if kind == "importance":
        fig = ov.plot_param_importances(study, **_kw_target)
    elif kind == "parallel_coordinate":
        fig = ov.plot_parallel_coordinate(study, **_kw_params, **_kw_target)
    elif kind == "contour":
        fig = ov.plot_contour(study, **_kw_params, **_kw_target)
    elif kind == "slice":
        fig = ov.plot_slice(study, **_kw_params, **_kw_target)
    else:  # optimization_history
        fig = ov.plot_optimization_history(study, **_kw_target)

    # --- Save ---------------------------------------------------------------
    if save_path is not None:
        from pathlib import Path as _Path
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        if suffix == ".html":
            fig.write_html(str(save_path))
            logger.info("Saved interactive HPO plot to %s", save_path)
        else:
            try:
                fig.write_image(str(save_path))
                logger.info("Saved static HPO plot to %s", save_path)
            except Exception as exc:
                raise ImportError(
                    f"Saving as {suffix!r} requires kaleido.  "
                    "Install with:  pip install kaleido"
                ) from exc

    # --- Display ------------------------------------------------------------
    if plot:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# 7. Train vs eval loss gap
# ---------------------------------------------------------------------------

def train_eval_gap(
    log_history: list[dict],
    *,
    skip_steps: int = 0,
    plot: bool = True,
) -> pd.DataFrame:
    """Compute the gap between training loss and evaluation loss at matched steps.

    A persistent positive gap (eval > train) is a sign of overfitting; a
    negative gap may indicate data leakage or label noise.  The gap is
    computed only at steps where an evaluation was logged (the HuggingFace
    Trainer logs eval less frequently than train).

    Train loss is interpolated to the nearest logged eval step by selecting
    the last training-log entry at or before each eval step.

    Args:
        log_history: List of dicts logged by the HuggingFace Trainer.
                     Pass ``metadata["log_history"]`` or
                     ``trainer.state.log_history``.
        skip_steps:  Ignore entries at or below this step.  Default 0.
        plot:        Show the gap over training steps.  Default True.

    Returns:
        DataFrame with columns:

            ``step``        — training step of the eval checkpoint
            ``train_loss``  — interpolated training loss at that step
            ``eval_loss``   — evaluation loss at that step
            ``gap``         — ``eval_loss − train_loss``

    Raises:
        ValueError: If no training loss or eval loss is found in log_history.
    """
    df = pd.DataFrame(log_history)

    if "loss" not in df.columns and "eval_loss" not in df.columns:
        raise ValueError("No loss values found in log_history.")

    df_train = df[df["loss"].notna()].copy() if "loss" in df.columns else pd.DataFrame()
    df_eval  = df[df["eval_loss"].notna()].copy() if "eval_loss" in df.columns else pd.DataFrame()

    if df_train.empty:
        raise ValueError("No training loss entries found in log_history.")
    if df_eval.empty:
        raise ValueError("No eval_loss entries found in log_history.")

    if skip_steps > 0:
        df_train = df_train[df_train["step"] > skip_steps]
        df_eval  = df_eval[df_eval["step"] > skip_steps]

    df_train = df_train.sort_values("step")
    df_eval  = df_eval.sort_values("step")

    # Interpolate: for each eval step, find the most recent preceding train step
    rows = []
    for _, eval_row in df_eval.iterrows():
        step      = eval_row["step"]
        preceding = df_train[df_train["step"] <= step]
        if preceding.empty:
            continue
        train_loss = preceding.iloc[-1]["loss"]
        eval_loss  = eval_row["eval_loss"]
        rows.append({
            "step":       step,
            "train_loss": train_loss,
            "eval_loss":  eval_loss,
            "gap":        eval_loss - train_loss,
        })

    if not rows:
        raise ValueError(
            "Could not match any eval steps to train steps.  "
            "Check that log_history contains both 'loss' and 'eval_loss' entries."
        )

    out = pd.DataFrame(rows)

    if plot:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.plot(out["step"], out["train_loss"], label="Train loss")
        ax1.plot(out["step"], out["eval_loss"],  label="Eval loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training vs Evaluation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.4)

        colors = ["red" if g > 0 else "green" for g in out["gap"]]
        ax2.bar(out["step"], out["gap"], color=colors, alpha=0.7, width=out["step"].diff().median() * 0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Gap (eval − train)")
        ax2.set_title("Train–Eval Loss Gap  (red = overfitting signal)")
        ax2.grid(True, axis="y", alpha=0.4)

        plt.tight_layout()
        plt.show()

    return out


# ---------------------------------------------------------------------------
# 8. Loss curve diagnostics (programmatic, no plot)
# ---------------------------------------------------------------------------

def loss_curve_diagnostics(
    log_history: list[dict],
    warmup_steps: int = 0,
) -> dict:
    """Return quantitative diagnostics for a HuggingFace training loss curve.

    Checks performed:

    * **Post-warmup slope** — fraction of loss dropped in the first 20 % of
      post-warmup steps.  > 15 % → ``"steep"``; 0–15 % → ``"modest"``;
      ≤ 0 % → ``"not_decreasing"``.
    * **Stabilisation** — compares loss variance in the last 20 % vs the
      first 20 % of post-warmup steps.  A ratio < 0.5 indicates the curve
      has stabilised.
    * **Train/eval gap** — mean gap (eval − train) at matched steps.
      Positive → overfitting risk; negative → possible leakage.
    * **Monotone decrease** — fraction of consecutive training-loss steps
      where loss did not increase.

    Args:
        log_history:  List of dicts from HuggingFace Trainer.
        warmup_steps: LR warmup steps; used to split the curve.  Default 0.

    Returns:
        dict with keys:

            ``post_warmup_drop_pct``   — % drop in first 20 % of post-warmup steps
            ``post_warmup_verdict``    — ``"steep"`` | ``"modest"`` | ``"not_decreasing"``
            ``stabilised``             — bool; variance ratio last/first 20 % < 0.5
            ``variance_ratio``         — var(last 20 %) / var(first 20 %)
            ``mean_train_eval_gap``    — mean of eval_loss − train_loss (NaN if no eval)
            ``overfitting_flag``       — bool; mean gap > 0.05
            ``monotone_decrease_frac`` — fraction of consecutive steps with non-increase

    Raises:
        ValueError: If no loss values found in log_history.
    """
    df = pd.DataFrame(log_history)
    if "loss" not in df.columns:
        raise ValueError("No 'loss' column found in log_history.")

    df_train = df[df["loss"].notna()].sort_values("step")
    if df_train.empty:
        raise ValueError("No training loss entries found in log_history.")

    losses = df_train["loss"].values

    # Post-warmup slope
    if warmup_steps > 0:
        post = df_train[df_train["step"] > warmup_steps]["loss"].values
    else:
        post = losses

    if len(post) >= 4:
        split       = max(1, len(post) // 5)
        early_mean  = post[:split].mean()
        late_mean   = post[-split:].mean()
        drop_pct    = (early_mean - late_mean) / (early_mean + 1e-12) * 100
        var_first   = post[:split].var()
        var_last    = post[-split:].var()
        var_ratio   = var_last / (var_first + 1e-12)
        stabilised  = bool(var_ratio < 0.5)
    else:
        drop_pct   = float("nan")
        var_ratio  = float("nan")
        stabilised = False

    if np.isnan(drop_pct):
        verdict = "insufficient_data"
    elif drop_pct > 15:
        verdict = "steep"
    elif drop_pct > 0:
        verdict = "modest"
    else:
        verdict = "not_decreasing"

    # Monotone decrease fraction
    if len(losses) > 1:
        diffs               = np.diff(losses)
        monotone_frac       = float((diffs <= 0).mean())
    else:
        monotone_frac = float("nan")

    # Train/eval gap
    mean_gap        = float("nan")
    overfitting     = False
    df_eval = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
    if not df_eval.empty:
        gaps = []
        for _, row in df_eval.iterrows():
            preceding = df_train[df_train["step"] <= row["step"]]
            if preceding.empty:
                continue
            gaps.append(row["eval_loss"] - preceding.iloc[-1]["loss"])
        if gaps:
            mean_gap    = float(np.mean(gaps))
            overfitting = mean_gap > 0.05

    return {
        "post_warmup_drop_pct":    round(float(drop_pct), 2),
        "post_warmup_verdict":     verdict,
        "stabilised":              stabilised,
        "variance_ratio":          round(float(var_ratio), 4) if not np.isnan(var_ratio) else float("nan"),
        "mean_train_eval_gap":     round(mean_gap, 4) if not np.isnan(mean_gap) else float("nan"),
        "overfitting_flag":        overfitting,
        "monotone_decrease_frac":  round(monotone_frac, 4) if not np.isnan(monotone_frac) else float("nan"),
    }
