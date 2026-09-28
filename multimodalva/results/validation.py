"""Validation scores: the numbers for choosing between candidate models.

Picking a representative model — the best text backbone, the best fusion
strategy — on **test** scores leaks the test set into the choice, and the chosen
model's test score is then optimistic. The choice has to be made on data the
test set never touched. Every pipeline already produces such a score; this
module finds it, whichever pipeline made the run:

=====================  ==========================================================
``source``             where the score comes from
=====================  ==========================================================
``hpo_cv``             the best search trial's cross-validated (or held-out)
                       score, from ``hpo_trials.csv``
``early_stopping``     the best epoch on the early-stopping slice of the training
                       split (text models trained with fixed hyperparameters)
``automm_holdout``     AutoMM's validation curve on the holdout it carves from
                       the training split (feature fusion), from its tfevents
``combiner_cv``        stacking with ``combiner="best"``: the chosen combiner's
                       nested-CV score on the out-of-fold predictions, from
                       ``combiner_comparison/combiner_comparison.json``
``meta_cv``            the chosen stacking meta-learner's cross-validated score
                       on the out-of-fold predictions
``none``               no validation data was used — for example a tabular model
                       fitted with fixed hyperparameters. Reported as missing,
                       never filled in from the test set.
=====================  ==========================================================

These are **selection** scores. Each is the best of several — trials, epochs,
checkpoints, candidates — so it is optimistic, and it is not a performance
estimate to report. Compare candidates with it; report the chosen model's test
score.

Each pipeline writes the result to ``validation.json`` when it finishes; runs
made before that are read from the artifacts above, so a leaderboard can be
built over runs already on disk.

    from multimodalva.results import validation_leaderboard

    board = validation_leaderboard({
        "biomedbert":      "runs/text/biomedbert",
        "clinicalbert":    "runs/text/clinicalbert",
        "fusion_attention": "runs/feature_fusion/attention",
    })
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "summarize_validation",
    "write_validation",
    "read_validation",
    "validation_leaderboard",
]

#: Metrics where lower is better.
_LOWER_IS_BETTER = {"log_loss", "loss"}

#: Names the Hugging Face trainer logs, mapped to this package's metric names.
_TRAINER_METRIC_NAMES = {"macro_f1": "f1_macro", "weighted_f1": "f1_weighted"}

# Where the per-trial metrics live in an Optuna trials table.
_ATTR = "user_attrs_"


def _clean(value):
    """JSON-safe float, or None for missing/NaN."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ---------------------------------------------------------------------------
# One reader per source
# ---------------------------------------------------------------------------

def _from_hpo_trials(csv: Path, metric: str) -> dict | None:
    trials = pd.read_csv(csv)
    if "state" in trials.columns:
        trials = trials[trials["state"] == "COMPLETE"]
    if trials.empty or "value" not in trials.columns:
        return None
    # The objective is ``value``; pick the trial the search itself picked.
    direction_min = metric in _LOWER_IS_BETTER
    best = trials.loc[trials["value"].idxmin() if direction_min else trials["value"].idxmax()]
    scores = {
        c[len(_ATTR):]: _clean(best[c]) for c in trials.columns
        if c.startswith(_ATTR) and not c[len(_ATTR):].startswith(("fold_", "elapsed", "trial_dir"))
    }
    scores = {k: v for k, v in scores.items() if v is not None}
    folds = sorted(c for c in trials.columns if c.startswith(f"{_ATTR}fold_"))
    return {
        "source": "hpo_cv",
        "scores": scores,
        "objective": _clean(best["value"]),
        "n_trials": int(len(trials)),
        "n_folds": len({c.split("_")[3] for c in folds if c.count("_") >= 4}) or None,
        "file": str(csv),
    }


def _from_log_history(meta_json: Path) -> dict | None:
    with open(meta_json) as fh:
        history = json.load(fh).get("log_history") or []
    evals = [e for e in history if any(k.startswith("eval_") for k in e)]
    if not evals:
        return None
    keys = {k for e in evals for k in e if k.startswith("eval_")
            and k not in ("eval_runtime", "eval_samples_per_second", "eval_steps_per_second")}
    scores = {}
    for k in keys:
        name = k[len("eval_"):]
        # The trainer logs its own names (eval_macro_f1); report them under the
        # package's metric names so they line up with every other source.
        name = _TRAINER_METRIC_NAMES.get(name, name)
        vals = [e[k] for e in evals if k in e and e[k] is not None]
        if vals:
            scores[name] = float(min(vals) if name in _LOWER_IS_BETTER else max(vals))
    return {"source": "early_stopping", "scores": scores, "n_epochs": len(evals),
            "file": str(meta_json)} if scores else None


def _from_tfevents(model_dir: Path) -> dict | None:
    files = sorted(model_dir.glob("events.out.tfevents*"))
    if not files:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("tensorboard is not installed; cannot read %s", model_dir)
        return None
    scores = {}
    for f in files:
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if not tag.startswith("val_"):
                continue
            name = tag[len("val_"):]
            vals = [e.value for e in acc.Scalars(tag)]
            if vals:
                best = min(vals) if name in _LOWER_IS_BETTER else max(vals)
                prev = scores.get(name)
                better = prev is None or (best < prev if name in _LOWER_IS_BETTER else best > prev)
                if better:
                    scores[name] = float(best)
    return {"source": "automm_holdout", "scores": scores, "file": str(model_dir)} if scores else None


def _from_meta_learner(meta_json: Path) -> dict | None:
    with open(meta_json) as fh:
        meta = json.load(fh)
    scores = meta.get("meta_scores") or {}
    best = meta.get("best_meta_name")
    metric = meta.get("meta_select_metric")
    if best is None or scores.get(best) is None or metric is None:
        return None
    return {"source": "meta_cv", "scores": {metric: float(scores[best])},
            "chosen": best, "file": str(meta_json)}


def _from_combiner_comparison(comparison_json: Path) -> dict | None:
    """Stacking run with ``combiner="best"``: the chosen combiner's nested-CV score."""
    with open(comparison_json) as fh:
        comparison = json.load(fh)
    best = comparison.get("best_combiner")
    metric = comparison.get("metric")
    result = (comparison.get("results") or {}).get(best) or {}
    if best is None or metric is None or result.get("mean") is None:
        return None
    return {"source": "combiner_cv",
            "scores": {metric: float(result["mean"]),
                       f"cv_std_{metric}": float(result.get("std", np.nan))},
            "chosen": best, "file": str(comparison_json)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_validation(run_dir: "str | Path", metric: str = "f1_macro") -> dict:
    """Work out a run's validation score from what the run left on disk.

    Looks, in order, for: a stacking combiner comparison (``combiner="best"``),
    a stacking meta-learner, a search (``hpo_trials.csv``
    in ``hpo/`` or the run root), an AutoMM validation curve, and a text model's
    early-stopping history. Base-model searches inside an ensemble
    (``hpo/text_0/`` …, ``base_models/``) are never mistaken for the ensemble's
    own score.

    Args:
        run_dir: A pipeline's ``output_dir``.
        metric:  Only used to read a search's objective in the right direction.

    Returns:
        ``{"source": ..., "scores": {metric: value, ...}, ...}``, with
        ``source="none"`` and empty ``scores`` when no validation data was used.
    """
    run = Path(run_dir)
    # Only when combiner="best" made the choice: a comparison run on its own
    # sits beside whichever combiner the run actually used.
    comparison = run / "combiner_comparison" / "combiner_comparison.json"
    run_meta = run / "training_metadata.json"
    if comparison.is_file() and run_meta.is_file():
        with open(run_meta) as fh:
            chosen = json.load(fh).get("combiner_chosen")
        if chosen:
            found = _from_combiner_comparison(comparison)
            if found and found["chosen"] == chosen:
                return found
    meta_learner = run / "meta_learner" / "meta_learner_metadata.json"
    if meta_learner.is_file():
        found = _from_meta_learner(meta_learner)
        if found:
            return found
    if (run / "oof").is_dir() or (run / "base_models").is_dir():
        return {"source": "none", "scores": {},
                "note": "ensemble without a cross-validated second stage"}

    # hpo_trials_ray.csv is what the Ray backend wrote before 2026-09-27; both
    # backends write hpo_trials.csv now, but runs made before then are still on
    # disk and their hpo_cv score has to stay readable.
    for csv in (
        run / "hpo" / "hpo_trials.csv",
        run / "hpo_trials.csv",
        run / "hpo" / "hpo_trials_ray.csv",
        run / "hpo_trials_ray.csv",
    ):
        if csv.is_file():
            found = _from_hpo_trials(csv, metric)
            if found:
                return found

    for model_dir in (run / "automm_model",):
        if model_dir.is_dir():
            found = _from_tfevents(model_dir)
            if found:
                return found

    for meta in (run / "final" / "training_metadata.json", run / "training_metadata.json"):
        if meta.is_file():
            found = _from_log_history(meta)
            if found:
                return found

    return {"source": "none", "scores": {}}


def write_validation(run_dir: "str | Path", metric: str = "f1_macro") -> dict:
    """Summarise a finished run's validation score into ``validation.json``.

    Called by every pipeline at the end of ``run()``, so each run carries its
    selection score in one uniform file.
    """
    summary = summarize_validation(run_dir, metric=metric)
    summary = {"metric": metric, **summary}
    with open(Path(run_dir) / "validation.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if summary["source"] == "none":
        logger.info("No validation score for this run (no validation data was used).")
    else:
        value = summary["scores"].get(metric)
        logger.info(
            "Validation %s = %s (source: %s) — for choosing between candidates, "
            "not for reporting.", metric,
            "n/a" if value is None else f"{value:.4f}", summary["source"],
        )
    return summary


def read_validation(run_dir: "str | Path", metric: str = "f1_macro") -> dict:
    """A run's validation summary: ``validation.json`` if present, else derived."""
    path = Path(run_dir) / "validation.json"
    if path.is_file():
        with open(path) as fh:
            return json.load(fh)
    return {"metric": metric, **summarize_validation(run_dir, metric=metric)}


def validation_leaderboard(
    runs: "dict | list",
    metric: str = "f1_macro",
    metrics: "list[str] | None" = None,
) -> pd.DataFrame:
    """Rank candidate runs on their **validation** scores.

    The counterpart of :func:`performance_leaderboard`, which scores the test
    set. Use this one to choose a representative model; then report that
    model's test score.

    Args:
        runs:    ``{name: run}`` or a list of runs, where a run is an
                 ``output_dir`` or the dict ``run()`` returned.
        metric:  The metric to rank by. Default ``"f1_macro"``.
        metrics: Extra validation metrics to show beside it, where the source
                 recorded them (searches record accuracy, balanced accuracy,
                 CSMF accuracy, weighted F1 and log loss).

    Returns:
        DataFrame indexed by name, sorted best first, with ``val_<metric>``
        columns, ``cv_std`` where a cross-validated search recorded it,
        ``source`` and ``n_trials``. Runs without a validation score sort last
        with ``NaN`` — they are not filled in from the test set.

    Compare runs whose ``source`` is the same. A score from one holdout's best
    epoch (``automm_holdout``, ``early_stopping``) runs higher than a
    cross-validated search score (``hpo_cv``) for the same model; a mixed table
    logs a warning.
    """
    if isinstance(runs, (list, tuple)):
        runs = {Path(r if not isinstance(r, dict) else r["output_dir"]).name: r for r in runs}
    shown = [metric] + [m for m in (metrics or []) if m != metric]

    rows = []
    for name, run in runs.items():
        run_dir = run["output_dir"] if isinstance(run, dict) else run
        summary = read_validation(run_dir, metric=metric)
        scores = summary.get("scores") or {}
        row = {"model": name}
        for m in shown:
            row[f"val_{m}"] = scores.get(m, np.nan)
        row["cv_std"] = scores.get(f"cv_std_{metric}", np.nan)
        row["source"] = summary.get("source", "none")
        row["n_trials"] = summary.get("n_trials", np.nan)
        rows.append(row)

    board = pd.DataFrame(rows).set_index("model")
    sources = sorted(set(board["source"]) - {"none"})
    if len(sources) > 1:
        logger.warning(
            "This leaderboard mixes validation sources (%s). They are not on the "
            "same footing — one holdout's best epoch is more optimistic than a "
            "cross-validated search score — so rank candidates within one "
            "source, e.g. text backbones against each other.", ", ".join(sources),
        )
    ascending = metric in _LOWER_IS_BETTER
    return board.sort_values(f"val_{metric}", ascending=ascending, na_position="last")
