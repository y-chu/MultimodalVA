"""
Bootstrap confidence intervals for model performance.

Point estimates on a single test set say nothing about how precise they are.
These functions resample the test cases with replacement, recompute every metric
on each resample, and report the percentile interval of the resulting
distribution.

Two functions, for two different questions:

- :func:`bootstrap_ci` — how precise is each model's score?
- :func:`paired_bootstrap_ci` — is model A better than model B?

Use the paired function for comparisons. Two models evaluated on the same cases
make correlated errors, so overlapping individual intervals do **not** mean the
difference is negligible; only resampling the difference itself answers that.

    from multimodalva.results import bootstrap_ci, paired_bootstrap_ci

    ci = bootstrap_ci({"bert": bert_result, "lightgbm": lgbm_result})
    diff = paired_bootstrap_ci({"bert": bert_result, "lightgbm": lgbm_result},
                               reference="lightgbm")

**What the interval covers.** It describes the performance of a *fixed trained
model*, treating the held-out test set as one sample from the population of
interest. Model weights, hyperparameters, training data and the train/test split
are all held fixed, so the interval does **not** capture training randomness
(seed, initialisation), variation between splits, or hyperparameter-search
variance. Re-training with a different seed can move a score outside the
interval. This is the usual convention in published clinical model evaluations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.metrics import REPORT_METRICS

logger = logging.getLogger(__name__)

DEFAULT_METRICS = list(REPORT_METRICS)

_CCCSMF_CHANCE = 0.632


# ---------------------------------------------------------------------------
# Assembling predictions from several models
# ---------------------------------------------------------------------------

def predictions_frame(
    results: dict,
    true_col: str = "true_label",
) -> pd.DataFrame:
    """Collect several models' predictions into one DataFrame.

    Builds the wide table used by :func:`bootstrap_ci`,
    :func:`paired_bootstrap_ci` and
    :func:`~multimodalva.results.performance_leaderboard`: one column of true
    labels, then one column of predicted labels per model.

    Example::

        from multimodalva import run
        from multimodalva.results import predictions_frame

        bert = run(task="text",    data=df, label="cause", text_col="narrative",
                   model="bioclinicalbert", output_dir="runs/bert")
        lgbm = run(task="tabular", data=df, label="cause",
                   model="lightgbm", output_dir="runs/lgbm")

        frame = predictions_frame({"bert": bert, "lightgbm": lgbm})

    Args:
        results:  Mapping of model name → predictions. Each value may be
                  whatever is easiest to hand over: the dict returned by
                  :func:`multimodalva.run`, a ``PredictionResult``, its ``top1``
                  DataFrame, or the path to a run directory (the function then
                  reads ``predictions/predictions_top1.csv``).
        true_col: Name to give the true-label column in the result.

    Returns:
        DataFrame with ``true_col`` plus one column per model, in the order the
        models were given. When the predictions carry an ``id`` column (from
        running a pipeline with ``id_col``), it is kept as the first column and
        used to match models to each other.

    Raises:
        ValueError: If no models are given, if a model's predictions cannot be
                    read, or if the models were not scored on the same test
                    records. Models carrying an ``id`` column are matched on it
                    (order does not matter); without ids they are matched by row
                    position and their true-label sequences must agree. Either
                    way, predictions from a different train/test split are
                    rejected rather than silently mismatched.
    """
    if not results:
        raise ValueError("results is empty — pass at least one model.")

    frame: dict[str, Any] = {}
    reference_truth: np.ndarray | None = None
    reference_ids: np.ndarray | None = None
    reference_name = ""

    for name, value in results.items():
        top1 = _as_top1_frame(name, value)

        # When every model carries row identifiers, match on them instead of on
        # row position — that survives reordering and catches a mismatched split
        # outright. Predictions written by this package include an ``id`` column
        # whenever the pipeline was given ``id_col``.
        if reference_ids is not None or (reference_truth is None and "id" in top1.columns):
            if "id" not in top1.columns:
                raise ValueError(
                    f"Model {name!r} has no 'id' column while other models do. "
                    "Either re-run it with id_col set, or drop the id columns so "
                    "the models are matched by row order."
                )
            ids = top1["id"].to_numpy()
            if reference_ids is None:
                reference_ids = ids
                frame["id"] = ids
            else:
                if set(ids.tolist()) != set(reference_ids.tolist()):
                    raise ValueError(
                        f"Model {name!r} was scored on different records than "
                        f"{reference_name!r}. Re-run both with the same split "
                        "(see the `split` argument of multimodalva.run)."
                    )
                top1 = (top1.set_index("id")
                            .reindex(reference_ids)
                            .reset_index())

        truth = top1["true_label"].astype(str).to_numpy()
        preds = top1["predicted_label"].astype(str).to_numpy()

        if reference_truth is None:
            reference_truth, reference_name = truth, name
            frame[true_col] = truth
        elif len(truth) != len(reference_truth):
            raise ValueError(
                f"Model {name!r} has {len(truth)} test cases but {reference_name!r} "
                f"has {len(reference_truth)}. All models must be evaluated on the "
                "same test set — re-run them with the same split (see the `split` "
                "argument of multimodalva.run)."
            )
        elif not np.array_equal(truth, reference_truth):
            raise ValueError(
                f"Model {name!r} has the same number of test cases as "
                f"{reference_name!r} but their true labels are in a different "
                "order, so the rows do not refer to the same deaths. Re-run both "
                "models with the same split (see the `split` argument of "
                "multimodalva.run)."
            )
        frame[name] = preds

    return pd.DataFrame(frame)


def _as_top1_frame(name: str, value: Any) -> pd.DataFrame:
    """Coerce one model's predictions into a top-1 DataFrame."""
    if isinstance(value, dict) and "predictions" in value:      # run() output
        value = value["predictions"]
    if hasattr(value, "top1"):                                   # PredictionResult
        value = value.top1
    if isinstance(value, (str, Path)):                           # run directory
        base = Path(value)
        for candidate in (base / "predictions" / "predictions_top1.csv",
                          base / "predictions_top1.csv",
                          base):
            if candidate.is_file():
                value = pd.read_csv(candidate)
                break
        else:
            raise ValueError(
                f"Model {name!r}: no predictions_top1.csv found under {base}."
            )
    if not isinstance(value, pd.DataFrame):
        raise ValueError(
            f"Model {name!r}: expected a run() result, a PredictionResult, a "
            f"top1 DataFrame or a run directory, got {type(value).__name__}."
        )

    missing = {"true_label", "predicted_label"} - set(value.columns)
    if missing:
        raise ValueError(
            f"Model {name!r}: prediction table is missing {sorted(missing)}."
        )
    return value


# ---------------------------------------------------------------------------
# Metrics, computed from a confusion matrix so resampling stays fast
# ---------------------------------------------------------------------------

def _confusion_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, metrics: list[str]
) -> dict[str, float]:
    """All label-based metrics for one resample, from a single confusion matrix.

    Building the confusion matrix with ``bincount`` and deriving every metric
    from it costs one pass over the data, instead of one pass per metric. Results
    match scikit-learn's ``zero_division=0`` behaviour.
    """
    n = y_true.size
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)

    hits = np.diagonal(cm).astype(float)
    support_true = cm.sum(axis=1).astype(float)   # rows: true labels
    support_pred = cm.sum(axis=0).astype(float)   # columns: predicted labels

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support_true > 0, hits / support_true, 0.0)
        precision = np.where(support_pred > 0, hits / support_pred, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    # scikit-learn's macro averages run over the labels appearing in either
    # vector; weighted averages use the true-label support as weights.
    present = (support_true > 0) | (support_pred > 0)
    weights = support_true / n

    out: dict[str, float] = {}
    for m in metrics:
        if m == "accuracy":
            out[m] = float(hits.sum() / n)
        elif m == "balanced_accuracy":
            in_truth = support_true > 0
            out[m] = float(recall[in_truth].mean()) if in_truth.any() else 0.0
        elif m == "f1_macro":
            out[m] = float(f1[present].mean()) if present.any() else 0.0
        elif m == "f1_weighted":
            out[m] = float((f1 * weights).sum())
        elif m == "precision_macro":
            out[m] = float(precision[present].mean()) if present.any() else 0.0
        elif m == "precision_weighted":
            out[m] = float((precision * weights).sum())
        elif m == "recall_macro":
            out[m] = float(recall[present].mean()) if present.any() else 0.0
        elif m == "recall_weighted":
            out[m] = float((recall * weights).sum())
        elif m in ("csmf_accuracy", "cccsmf_accuracy"):
            csmf = _csmf_from_counts(support_true, support_pred, n)
            out[m] = csmf if m == "csmf_accuracy" else (
                (csmf - _CCCSMF_CHANCE) / (1.0 - _CCCSMF_CHANCE)
            )
    return out


def _csmf_from_counts(
    support_true: np.ndarray, support_pred: np.ndarray, n: int
) -> float:
    """CSMF accuracy from class counts (same definition as utils.metrics)."""
    present = (support_true > 0) | (support_pred > 0)
    true_frac = support_true[present] / n
    pred_frac = support_pred[present] / n
    min_true = true_frac.min()
    if min_true == 1.0:
        return 1.0 if np.allclose(true_frac, pred_frac) else 0.0
    return float(1.0 - np.abs(pred_frac - true_frac).sum() / (2 * (1 - min_true)))


def _encode(frame: pd.DataFrame, true_col: str, model_cols: list[str]):
    """Map labels to integer codes shared by every model."""
    classes = sorted(
        set(frame[true_col].astype(str))
        | {v for c in model_cols for v in frame[c].astype(str)}
    )
    code = {c: i for i, c in enumerate(classes)}
    y_true = frame[true_col].astype(str).map(code).to_numpy()
    y_pred = {c: frame[c].astype(str).map(code).to_numpy() for c in model_cols}
    return y_true, y_pred, len(classes)


def _resolve_input(data, true_col, model_cols):
    """Accept either a wide DataFrame or a {name: predictions} mapping."""
    if isinstance(data, dict):
        frame = predictions_frame(data, true_col=true_col)
        cols = model_cols or [c for c in frame.columns if c not in {true_col, "id"}]
        return frame, cols
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "data must be a DataFrame of predictions or a "
            "{model name: predictions} mapping, got "
            f"{type(data).__name__}."
        )
    if true_col not in data.columns:
        raise ValueError(f"true_col {true_col!r} not found in the DataFrame.")
    cols = model_cols or [c for c in data.columns if c not in {true_col, "id"}]
    if not cols:
        raise ValueError("No model columns found — pass model_cols explicitly.")
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Model columns not in the DataFrame: {missing}.")
    return data, cols


def _draw_indices(n: int, n_boot: int, random_state: int) -> np.ndarray:
    """One resample-index matrix, reused by every model.

    Every model is scored on the same resampled cases. Drawing separately per
    model would make the intervals incomparable and would shift them whenever
    the set of models changed.
    """
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n, size=(n_boot, n))


def _topk_hits(topk_df: pd.DataFrame, top_k: int) -> np.ndarray:
    """Boolean vector: was the true label within the model's top *k*?"""
    label_cols = [f"top{i}_label" for i in range(1, top_k + 1)
                  if f"top{i}_label" in topk_df.columns]
    if not label_cols:
        raise ValueError(
            f"topk table has no top1_label…top{top_k}_label columns."
        )
    truth = topk_df["true_label"].astype(str).to_numpy()
    ranked = topk_df[label_cols].astype(str).to_numpy()
    return (ranked == truth[:, None]).any(axis=1)


# ---------------------------------------------------------------------------
# 1. Confidence interval per model
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data,
    true_col: str = "true_label",
    model_cols: list[str] | None = None,
    metrics: list[str] | None = None,
    n_boot: int = 1000,
    ci: float = 95.0,
    random_state: int = 42,
    topk_dfs: dict | None = None,
    top_k: int = 3,
    percentage: bool = True,
    long_format: bool = False,
    formatted: bool = False,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for one or more models.

    Resamples the test cases with replacement ``n_boot`` times, recomputes each
    metric on every resample, and reports the percentile interval. All models are
    scored on the same resamples, so the intervals are comparable.

    Example::

        from multimodalva.results import bootstrap_ci

        ci = bootstrap_ci({"bert": bert_result, "lightgbm": lgbm_result})
        print(ci[["accuracy", "f1_macro", "csmf_accuracy"]])

        # Ready-to-paste "62.1 (58.4-65.7)" cells:
        table = bootstrap_ci(frame, formatted=True)

    Args:
        data:        Either a ``{model name: predictions}`` mapping — values may
                     be :func:`multimodalva.run` results, ``PredictionResult``
                     objects, ``top1`` DataFrames or run directories — or a wide
                     DataFrame with a true-label column and one predicted-label
                     column per model.
        true_col:    Name of the true-label column. Default ``"true_label"``.
        model_cols:  Which model columns to evaluate. Default: all of them.
        metrics:     Metrics to compute. Default: the ten label-based metrics
                     also used by
                     :func:`~multimodalva.results.performance_leaderboard`.
        n_boot:      Number of resamples. Default 1000, enough for stable
                     endpoints at one-decimal reporting; raise to 10000 for
                     publication tables if runtime allows.
        ci:          Interval width in percent. Default 95 (2.5th–97.5th
                     percentiles).
        random_state: Seed for the resample draws, so a rerun reproduces the
                     same intervals.
        topk_dfs:    Optional ``{model name: topk DataFrame}`` (``result.topk``).
                     When given, ``top2_accuracy`` … ``top{top_k}_accuracy`` are
                     added.
        top_k:       Highest rank to report when ``topk_dfs`` is given.
        percentage:  Report metrics on a 0–100 scale. Default True.
        long_format: Return one row per model × metric (columns ``model``,
                     ``metric``, ``estimate``, ``ci_lower``, ``ci_upper``,
                     ``bootstrap_se``, ``n``, ``n_boot``) instead of the wide
                     layout. Convenient for plotting and for report tables.
        formatted:   Add (wide) or replace with (long) ``"estimate (lower-upper)"``
                     strings rounded to one decimal.

    Returns:
        Wide: a DataFrame indexed by model, with one column per metric holding
        the point estimate, plus ``ci_lower`` / ``ci_upper`` columns per metric.
        Long: one row per model and metric. See ``long_format``.

    Raises:
        ValueError: If the input cannot be read, the models were evaluated on
                    different test sets, or an unknown metric is requested.
    """
    metrics = list(metrics or DEFAULT_METRICS)
    unknown = set(metrics) - set(DEFAULT_METRICS)
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {sorted(unknown)}. Supported: {DEFAULT_METRICS}."
        )
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if not 0 < ci < 100:
        raise ValueError("ci must be between 0 and 100 (exclusive).")

    frame, cols = _resolve_input(data, true_col, model_cols)
    y_true, y_pred, n_classes = _encode(frame, true_col, cols)
    n = y_true.size
    boot_idx = _draw_indices(n, n_boot, random_state)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    scale = 100.0 if percentage else 1.0

    topk_metric_names: list[str] = []
    if topk_dfs:
        topk_metric_names = [f"top{k}_accuracy" for k in range(2, top_k + 1)]

    rows: list[dict] = []
    for col in cols:
        yp = y_pred[col]
        point = _confusion_metrics(y_true, yp, n_classes, metrics)
        draws = {m: np.empty(n_boot) for m in metrics}
        for b in range(n_boot):
            idx = boot_idx[b]
            values = _confusion_metrics(y_true[idx], yp[idx], n_classes, metrics)
            for m in metrics:
                draws[m][b] = values[m]

        if topk_dfs and col in topk_dfs:
            hits = _topk_hits(topk_dfs[col], top_k)
            if hits.size != n:
                raise ValueError(
                    f"Model {col!r}: topk table has {hits.size} rows but the "
                    f"predictions have {n}."
                )
            for name in topk_metric_names:
                k = int(name.split("top")[1].split("_")[0])
                hits_k = _topk_hits(topk_dfs[col], k)
                point[name] = float(hits_k.mean())
                draws[name] = hits_k[boot_idx].mean(axis=1)

        for m in metrics + [x for x in topk_metric_names if x in draws]:
            values = draws[m] * scale
            lo, hi = np.percentile(values, [lo_q, hi_q])
            rows.append({
                "model": col,
                "metric": m,
                "estimate": point[m] * scale,
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "bootstrap_se": float(values.std(ddof=1)) if n_boot > 1 else np.nan,
                "n": n,
                "n_boot": n_boot,
            })
        logger.info("Bootstrap CI: %s done (n=%d, B=%d)", col, n, n_boot)

    long_df = pd.DataFrame(rows)
    if formatted:
        long_df["formatted"] = long_df.apply(
            lambda r: f"{r['estimate']:.1f} ({r['ci_lower']:.1f}-{r['ci_upper']:.1f})",
            axis=1,
        )
    if long_format:
        return long_df

    if formatted:
        wide = long_df.pivot(index="model", columns="metric", values="formatted")
        return wide.reindex(cols)

    wide = long_df.pivot(index="model", columns="metric",
                         values=["estimate", "ci_lower", "ci_upper"])
    ordered = [m for m in metrics + topk_metric_names
               if m in long_df["metric"].unique()]
    flat = pd.DataFrame(index=cols)
    for m in ordered:
        flat[m] = wide[("estimate", m)]
        flat[f"{m}_ci_lower"] = wide[("ci_lower", m)]
        flat[f"{m}_ci_upper"] = wide[("ci_upper", m)]
    flat.index.name = "model"
    return flat


# ---------------------------------------------------------------------------
# 2. Paired comparison between models
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    data,
    true_col: str = "true_label",
    reference: str | None = None,
    model_cols: list[str] | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
    metrics: list[str] | None = None,
    n_boot: int = 1000,
    ci: float = 95.0,
    random_state: int = 42,
    percentage: bool = True,
    formatted: bool = False,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for the *difference* between models.

    Use this to compare models, not the individual intervals from
    :func:`bootstrap_ci`. Models evaluated on the same cases make correlated
    errors, so their individual intervals can overlap even when one is reliably
    better. Here both models are scored on each resample and the difference is
    recorded, which measures the comparison directly.

    Example::

        from multimodalva.results import paired_bootstrap_ci

        # every model against a baseline
        paired_bootstrap_ci(frame, reference="insilicova")

        # a single head-to-head comparison
        paired_bootstrap_ci(frame, model_a="ensemble", model_b="bert")

    Args:
        data:        Same input as :func:`bootstrap_ci`: a
                     ``{model name: predictions}`` mapping or a wide DataFrame.
        true_col:    Name of the true-label column.
        reference:   Compare every model against this one. The difference is
                     reported as *model minus reference*, so a positive value
                     means the model scores higher.
        model_cols:  Which models to compare against ``reference``. Default: all
                     except the reference itself.
        model_a, model_b: A single comparison, reported as *a minus b*. Use
                     instead of ``reference``.
        metrics:     Metrics to compare. Default: the ten label-based metrics.
        n_boot:      Number of resamples. Default 1000.
        ci:          Interval width in percent. Default 95.
        random_state: Seed for the resample draws.
        percentage:  Report differences on a 0–100 scale. Default True.
        formatted:   Add an ``"estimate (lower-upper)"`` string column.

    Returns:
        One row per comparison and metric, with the observed ``difference``, its
        interval (``ci_lower``, ``ci_upper``), ``bootstrap_se``, and
        ``prob_reversed`` — the share of resamples in which the difference
        changed sign. A small ``prob_reversed`` and an interval excluding zero
        together support a claim that one model is better; an interval spanning
        zero does not.

    Raises:
        ValueError: If neither ``reference`` nor both of ``model_a``/``model_b``
                    are given, if a named model is absent, or if the models were
                    evaluated on different test sets.
    """
    metrics = list(metrics or DEFAULT_METRICS)
    unknown = set(metrics) - set(DEFAULT_METRICS)
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {sorted(unknown)}. Supported: {DEFAULT_METRICS}."
        )

    frame, cols = _resolve_input(data, true_col, model_cols)

    if model_a is not None or model_b is not None:
        if model_a is None or model_b is None:
            raise ValueError("Pass both model_a and model_b, or use reference=.")
        pairs = [(model_a, model_b)]
        needed = {model_a, model_b}
    elif reference is not None:
        others = [c for c in cols if c != reference]
        if not others:
            raise ValueError(
                f"reference={reference!r} leaves no other model to compare."
            )
        pairs = [(c, reference) for c in others]
        needed = set(cols) | {reference}
    else:
        raise ValueError(
            "Pass reference= to compare every model against one baseline, or "
            "model_a= and model_b= for a single comparison."
        )

    absent = [m for m in needed if m not in frame.columns]
    if absent:
        raise ValueError(f"Model(s) not found in the predictions: {absent}.")

    all_models = sorted(needed)
    y_true, y_pred, n_classes = _encode(frame, true_col, all_models)
    n = y_true.size
    boot_idx = _draw_indices(n, n_boot, random_state)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    scale = 100.0 if percentage else 1.0

    # Score every model once per resample, then take differences, so a model
    # appearing in several comparisons is not rescored.
    per_model: dict[str, dict[str, np.ndarray]] = {}
    point: dict[str, dict[str, float]] = {}
    for name in all_models:
        yp = y_pred[name]
        point[name] = _confusion_metrics(y_true, yp, n_classes, metrics)
        draws = {m: np.empty(n_boot) for m in metrics}
        for b in range(n_boot):
            idx = boot_idx[b]
            values = _confusion_metrics(y_true[idx], yp[idx], n_classes, metrics)
            for m in metrics:
                draws[m][b] = values[m]
        per_model[name] = draws
        logger.info("Paired bootstrap: %s scored (n=%d, B=%d)", name, n, n_boot)

    rows: list[dict] = []
    for a, b in pairs:
        for m in metrics:
            diff = (per_model[a][m] - per_model[b][m]) * scale
            observed = (point[a][m] - point[b][m]) * scale
            lo, hi = np.percentile(diff, [lo_q, hi_q])
            reversed_share = (
                float((diff < 0).mean()) if observed > 0
                else float((diff > 0).mean()) if observed < 0
                else float("nan")
            )
            rows.append({
                "model": a,
                "compared_with": b,
                "metric": m,
                f"{a}_estimate": point[a][m] * scale,
                f"{b}_estimate": point[b][m] * scale,
                "difference": observed,
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "bootstrap_se": float(diff.std(ddof=1)) if n_boot > 1 else np.nan,
                "prob_reversed": reversed_share,
                "n": n,
                "n_boot": n_boot,
            })

    out = pd.DataFrame(rows)
    # Per-pair estimate columns only make sense for a single comparison.
    if len(pairs) > 1:
        out = out.drop(columns=[c for c in out.columns if c.endswith("_estimate")])
    if formatted:
        out["formatted"] = out.apply(
            lambda r: f"{r['difference']:+.1f} ({r['ci_lower']:+.1f}, {r['ci_upper']:+.1f})",
            axis=1,
        )
    return out
