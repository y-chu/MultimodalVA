"""
Training and HPO result visualization.

Public API
----------
    plot_loss_curves(log_history, skip_steps)
        Training/evaluation loss curves from a HuggingFace Trainer log.

    hpo_leaderboard(source, sort_by, top_n, ascending, param_cols)
        Formatted HPO trial table from an Optuna study or trials DataFrame.
"""

from __future__ import annotations

import logging

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(log_history: list[dict], skip_steps: int = 0) -> None:
    """Plot training and evaluation loss curves from a HuggingFace log history.

    The log_history comes directly from train() metadata:
        metadata["log_history"]
    or from a live trainer:
        trainer.state.log_history

    Args:
        log_history: List of dicts logged by the HuggingFace Trainer.
        skip_steps: Ignore all entries at or below this step number.
                    Useful for skipping noisy early steps. Default 0.
    """
    df = pd.DataFrame(log_history)

    df_loss = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
    df_eval = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()

    if skip_steps > 0:
        df_loss = df_loss[df_loss["step"] > skip_steps]
        df_eval = df_eval[df_eval["step"] > skip_steps]

    if df_loss.empty and df_eval.empty:
        raise ValueError("No loss values found in log_history.")

    plt.figure(figsize=(10, 6))
    if not df_loss.empty:
        plt.plot(df_loss["step"], df_loss["loss"], label="Training Loss")
    if not df_eval.empty:
        plt.plot(df_eval["step"], df_eval["eval_loss"], label="Evaluation Loss")

    title = "Loss Curves" + (f" (after step {skip_steps})" if skip_steps > 0 else "")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
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
        sort_by:   Column to sort by.  Defaults to the Optuna objective
                   value (``"value"`` column).  Can be any metric stored as
                   a user attribute (e.g. ``"accuracy"``, ``"f1_macro"``,
                   ``"f1_weighted"``, ``"csmf_accuracy"``).
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

    # Rename "number" → "trial"
    if "number" in trials_df.columns:
        trials_df = trials_df.rename(columns={"number": "trial"})

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

    # Determine sort column
    if sort_by is None:
        sort_col = "value"
    elif sort_by in trials_df.columns:
        sort_col = sort_by
    else:
        raise ValueError(
            f"sort_by={sort_by!r} not found in trials.  "
            f"Available columns: {list(trials_df.columns)}."
        )

    trials_df = trials_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    if top_n is not None:
        trials_df = trials_df.head(top_n)

    return trials_df
