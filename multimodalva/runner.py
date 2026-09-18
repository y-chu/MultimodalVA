"""
One-call, config-driven entry point for every MultimodalVA pipeline.

The classifier classes (:class:`TextClassifier`, :class:`TabularClassifier`,
:class:`EnsembleClassifier`) already chain split → prepare → HPO → train →
predict and save artifacts to ``output_dir/{final,hpo,predictions}``.  This
module adds the *surrounding* glue that every analysis script otherwise
re-implements — loading data, filtering rows, resolving feature columns,
optional fixed train/test split, diagnostics — behind a single function::

    from multimodalva import run

    run(task="text", data="clean.csv", label="cause",
        text_col="narrative", model="bluebert", output_dir="runs/bluebert",
        optimize=True, n_trials=50)

``data`` accepts a CSV/Parquet path **or** an in-memory DataFrame, so a
statistician can preprocess in a notebook and pass the frame directly.

Heavy imports (torch/transformers/autogluon) are performed lazily inside
:func:`run`, so ``from multimodalva import run`` stays lightweight.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["run", "SUPPORTED_TASKS"]

# Canonical task name -> internal handler key. Aliases fold onto canonical names
# so users can write the natural word ("voting") or the internal method name.
_TASK_ALIASES: dict[str, str] = {
    "text": "text",
    "unimodal_text": "text",
    "tabular": "tabular",
    "unimodal_tabular": "tabular",
    "data_fusion": "data_fusion",
    "datafusion": "data_fusion",
    "feature_fusion": "feature_fusion",
    "featurefusion": "feature_fusion",
    "automm": "feature_fusion",
    "voting": "soft_voting",
    "soft_voting": "soft_voting",
    "softvoting": "soft_voting",
    "stacking": "stacking",
    "stack": "stacking",
}

#: Public list of task names accepted by :func:`run` (canonical + friendly).
SUPPORTED_TASKS: tuple[str, ...] = (
    "text", "tabular", "data_fusion", "feature_fusion", "voting", "stacking",
)

# Tasks that consume a free-text narrative column.
_TEXT_TASKS = {"text", "data_fusion", "feature_fusion"}
# Tasks that consume tabular feature columns.
_FEATURE_TASKS = {"tabular", "data_fusion", "feature_fusion"}
# Ensemble tasks dispatched through EnsembleClassifier(method=...).
_ENSEMBLE_METHODS = {"data_fusion", "feature_fusion", "soft_voting", "stacking"}

# Column injected to carry a pre-defined train/test assignment down to split().
_SPLIT_MARKER_COL = "__mmva_split__"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(
    task: str,
    data: "str | Path | pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]",
    label: str,
    output_dir: "str | Path",
    *,
    text_col: str | None = None,
    features: "list[str] | str | None" = None,
    model: str | None = None,
    filters: dict | None = None,
    optimize: bool = False,
    n_trials: int | None = None,
    metric: str | None = None,
    # tabular preprocessing
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    encode_categoricals: str | None = "ordinal",
    scale_numeric: bool = False,
    # split
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    split: "str | dict | Path | tuple | None" = None,
    id_col: str | None = None,
    top_k: int = 3,
    # ensemble base-model specs (voting / stacking)
    text_models: list[dict] | None = None,
    tabular_models: list[dict] | None = None,
    init_kwargs: dict | None = None,
    save_diagnostics: bool = True,
    **run_kwargs: Any,
) -> dict:
    """Run any MultimodalVA pipeline end-to-end from a flat config.

    Args:
        task: One of ``SUPPORTED_TASKS`` (aliases like ``"voting"`` accepted).
        data: CSV/Parquet path, an in-memory DataFrame, or an explicit
            ``(train_df, test_df)`` tuple (which also fixes the split).
        label: Label (cause-of-death) column name.
        output_dir: Root directory for all artifacts.
        text_col: Narrative column (required for text/fusion tasks).
        features: Tabular feature columns. A list is used verbatim; ``"re:PAT"``
            selects columns whose name matches the regex ``PAT`` (e.g.
            ``"re:^i\\d{3}[a-zA-Z]$"``); ``None`` / ``"auto"`` uses every column
            except the label, text, filter, and split columns.
        model: Model name or HuggingFace checkpoint / alias (e.g. ``"bluebert"``,
            ``"lightgbm"``). Falls back to each pipeline's default when ``None``.
        filters: ``{column: value | [values]}`` row filter applied before
            splitting. Omit for no filtering (df treated as ready-to-run).
        optimize: Run HPO before final training.
        n_trials: HPO trial count. ``None`` keeps each pipeline's own default.
        metric: HPO metric. ``None`` uses ``"f1_macro"`` when ``optimize`` is set.
        cat_cols / num_cols / encode_categoricals / scale_numeric: tabular
            preprocessing options (tabular / voting / stacking tabular models).
        test_size / random_state / stratify: random-split controls (ignored when
            an external split is supplied).
        split: Fixed train/test assignment for cross-experiment comparison —
            a column name, a ``{"train_ids": [...], "test_ids": [...]}`` mapping
            (or path to a JSON file of that shape, resolved against ``id_col``),
            or ``None`` for a random stratified split.
        id_col: Row-identifier column. Used to resolve id-based ``split`` specs,
            and carried into the prediction tables as a leading ``id`` column so
            results can be joined back to the source records.
        top_k: Number of top classes to report.
        text_models / tabular_models: base-model specs for voting / stacking.
        init_kwargs: extra keyword args forwarded to the classifier *constructor*
            (e.g. feature_fusion ``preset``/``fusion_strategy``; stacking
            ``n_folds``/``meta_learners``; voting ``weights``).
        save_diagnostics: write ``hpo_leaderboard.csv`` + ``hpo_convergence.png``
            when HPO trial data is found.
        **run_kwargs: forwarded verbatim to the underlying ``.run()`` — advanced
            knobs such as ``use_lora``, ``gradient_checkpointing``, ``time_limit``,
            and Hub publishing (``push_to_hub``, ``hub_repo_id``, ...).

    Returns:
        The underlying results dict: ``predictions`` (a ``PredictionResult``),
        ``label2id``, ``id2label``, ``output_dir``, and pipeline-specific keys.
    """
    canonical = _normalize_task(task)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = dict(init_kwargs or {})

    # --- 1-3. Load, filter, drop-missing, resolve split marker --------------
    df, split_col = _prepare_dataframe(
        data=data,
        canonical=canonical,
        label=label,
        text_col=text_col,
        filters=filters,
        split=split,
        id_col=id_col,
    )

    # --- 4. Resolve feature columns (tabular / fusion / tabular base models) --
    feature_cols = None
    if canonical in _FEATURE_TASKS or (tabular_models and canonical in _ENSEMBLE_METHODS):
        feature_cols = _resolve_features(
            df, features, label, text_col, filters, split_col
        )
        logger.info("Running on %d feature columns.", len(feature_cols))

    if canonical in _TEXT_TASKS and not text_col:
        raise ValueError(f"task={task!r} requires text_col.")

    # --- 5. Dispatch --------------------------------------------------------
    results = _dispatch(
        canonical=canonical,
        df=df,
        label=label,
        text_col=text_col,
        feature_cols=feature_cols,
        model=model,
        output_dir=output_dir,
        optimize=optimize,
        n_trials=n_trials,
        metric=metric,
        cat_cols=cat_cols,
        num_cols=num_cols,
        encode_categoricals=encode_categoricals,
        scale_numeric=scale_numeric,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        split_col=split_col,
        top_k=top_k,
        text_models=text_models,
        tabular_models=tabular_models,
        init_kwargs=init_kwargs,
        id_col=id_col,
        run_kwargs=run_kwargs,
    )

    # --- 6. Diagnostics -----------------------------------------------------
    if save_diagnostics and optimize:
        _save_diagnostics(output_dir, metric or "f1_macro")

    _log_summary(results, label)
    return results


# ---------------------------------------------------------------------------
# Task normalization
# ---------------------------------------------------------------------------
def _normalize_task(task: str) -> str:
    key = str(task).strip().lower()
    if key not in _TASK_ALIASES:
        raise ValueError(
            f"Unknown task {task!r}. Choose from: {list(SUPPORTED_TASKS)}."
        )
    return _TASK_ALIASES[key]


# ---------------------------------------------------------------------------
# Data loading / filtering / split resolution
# ---------------------------------------------------------------------------
def _load_df(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return df
    out = df
    for col, val in filters.items():
        if col not in out.columns:
            raise ValueError(
                f"filter column {col!r} not in DataFrame. "
                f"Available: {out.columns.tolist()}"
            )
        if isinstance(val, (list, tuple, set)):
            out = out[out[col].isin(list(val))]
        else:
            out = out[out[col] == val]
    logger.info("Filters %s -> %d rows.", filters, len(out))
    return out.reset_index(drop=True)


def _prepare_dataframe(
    *, data, canonical, label, text_col, filters, split, id_col
) -> tuple[pd.DataFrame, str | None]:
    """Load, apply an explicit split (if any), filter, drop-missing.

    Returns the prepared DataFrame and the split-marker column name (or None).
    """
    # Explicit (train_df, test_df) tuple — either via ``data`` or ``split``.
    tuple_split = None
    if isinstance(data, tuple):
        tuple_split = data
    elif isinstance(split, tuple):
        tuple_split = split

    if tuple_split is not None:
        train_df, test_df = tuple_split
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[_SPLIT_MARKER_COL] = "train"
        test_df[_SPLIT_MARKER_COL] = "test"
        df = pd.concat([train_df, test_df], ignore_index=True)
        split_col = _SPLIT_MARKER_COL
    else:
        df = _load_df(data)
        split_col = _resolve_split_col(df, split, id_col)

    df = _apply_filters(df, filters)
    df = _drop_missing(df, canonical, label, text_col)
    if split_col is not None:
        _validate_split_nonempty(df, split_col)
    return df, split_col


def _resolve_split_col(df: pd.DataFrame, split, id_col) -> str | None:
    """Normalize a ``split=`` spec into a marker column on ``df`` (mutates df)."""
    if split is None:
        return None

    # A column name that already marks train/test.
    if isinstance(split, str) and not _looks_like_path(split):
        if split not in df.columns:
            raise ValueError(
                f"split column {split!r} not in DataFrame. "
                f"Available: {df.columns.tolist()}"
            )
        return split

    # A JSON file path -> load into a dict of train_ids/test_ids.
    if isinstance(split, (str, Path)) and _looks_like_path(split):
        split = json.loads(Path(split).read_text())

    if isinstance(split, dict):
        return _apply_id_split(df, split, id_col)

    raise ValueError(
        "split must be a column name, a {'train_ids': [...], 'test_ids': [...]} "
        "dict, a path to such a JSON file, or an explicit (train_df, test_df) tuple."
    )


def _apply_id_split(df: pd.DataFrame, spec: dict, id_col: str | None) -> str:
    if id_col is None:
        raise ValueError("id_col is required when split is given as train/test ids.")
    if id_col not in df.columns:
        raise ValueError(f"id_col {id_col!r} not in DataFrame.")
    train_ids = set(spec.get("train_ids", []))
    test_ids = set(spec.get("test_ids", []))
    if not train_ids and not test_ids:
        raise ValueError("split dict must contain non-empty 'train_ids' or 'test_ids'.")

    def _assign(rid):
        if rid in test_ids:
            return "test"
        if rid in train_ids:
            return "train"
        return None  # rows not in either list are dropped below

    df[_SPLIT_MARKER_COL] = df[id_col].map(_assign)
    n_drop = int(df[_SPLIT_MARKER_COL].isna().sum())
    if n_drop:
        logger.info("Dropping %d rows not present in the supplied split.", n_drop)
        df.drop(df.index[df[_SPLIT_MARKER_COL].isna()], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return _SPLIT_MARKER_COL


def _looks_like_path(value) -> bool:
    if isinstance(value, Path):
        return True
    if isinstance(value, str):
        return value.lower().endswith((".json", ".yaml", ".yml"))
    return False


def _drop_missing(df, canonical, label, text_col) -> pd.DataFrame:
    if label not in df.columns:
        raise ValueError(
            f"label column {label!r} not in DataFrame. "
            f"Available: {df.columns.tolist()}"
        )
    before = len(df)
    mask = df[label].notna() & (df[label].astype(str).str.strip() != "")
    if canonical in _TEXT_TASKS and text_col and text_col in df.columns:
        mask &= df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    out = df[mask].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %d rows with missing label/text.", dropped)
    if len(out) == 0:
        raise ValueError("No rows remain after dropping missing label/text.")
    return out


def _validate_split_nonempty(df, split_col) -> None:
    from .utils.split import _is_test_marker

    is_test = df[split_col].map(_is_test_marker)
    n_test = int(is_test.sum())
    n_train = len(df) - n_test
    if n_train == 0 or n_test == 0:
        raise ValueError(
            f"After filtering, the supplied split has {n_train} train / {n_test} "
            "test rows. Both sides must be non-empty."
        )


# ---------------------------------------------------------------------------
# Feature resolution
# ---------------------------------------------------------------------------
def _resolve_features(df, features, label, text_col, filters, split_col) -> list[str]:
    if isinstance(features, list):
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns not in DataFrame: {missing}")
        return features

    reserved = {label}
    if text_col:
        reserved.add(text_col)
    if filters:
        reserved.update(filters.keys())
    if split_col:
        reserved.add(split_col)

    if isinstance(features, str) and features.startswith("re:"):
        pattern = re.compile(features[3:])
        cols = [c for c in df.columns if c not in reserved and pattern.search(str(c))]
        if not cols:
            raise ValueError(f"feature regex {features!r} matched no columns.")
        return sorted(cols)

    if features is None or features == "auto":
        cols = [c for c in df.columns if c not in reserved]
        if not cols:
            raise ValueError("No feature columns left after excluding label/text.")
        return cols

    raise ValueError(
        "features must be a list, 're:PATTERN', 'auto', or None; "
        f"got {features!r}."
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _dispatch(
    *, canonical, df, label, text_col, feature_cols, model, output_dir,
    optimize, n_trials, metric, cat_cols, num_cols, encode_categoricals,
    scale_numeric, test_size, random_state, stratify, split_col, top_k,
    text_models, tabular_models, init_kwargs, run_kwargs, id_col=None,
) -> dict:
    metric_val = metric or ("f1_macro" if optimize else None)

    common_split = dict(
        test_size=test_size, random_state=random_state,
        stratify=stratify, split_col=split_col,
    )

    if canonical == "text":
        from .text.text_classifier import TextClassifier

        clf = TextClassifier(
            model_name=_resolve_model_name(model) or "bert-base-uncased",
            output_dir=output_dir,
        )
        kwargs = dict(
            df=df, text_col=text_col, label_col=label,
            use_optimize=optimize, top_k=top_k, **common_split, **run_kwargs,
        )
        _put(kwargs, "id_col", id_col)
        _put(kwargs, "optimize_metric", metric_val)
        _put(kwargs, "n_trials", n_trials)
        return clf.run(**kwargs)

    if canonical == "tabular":
        from .tabular.tabular_classifier import TabularClassifier

        clf = TabularClassifier(
            model_name=model or "random_forest", output_dir=output_dir
        )
        kwargs = dict(
            df=df, feature_cols=feature_cols, label_col=label,
            use_optimize=optimize, top_k=top_k,
            cat_cols=cat_cols, num_cols=num_cols,
            encode_categoricals=encode_categoricals, scale_numeric=scale_numeric,
            **common_split, **run_kwargs,
        )
        _put(kwargs, "id_col", id_col)
        _put(kwargs, "optimize_metric", metric_val)
        _put(kwargs, "n_trials", n_trials)
        return clf.run(**kwargs)

    # ---- ensemble strategies ----
    from .ensemble.ensemble_classifier import EnsembleClassifier

    ctor = dict(output_dir=output_dir)
    run_args = dict(df=df, text_col=text_col, feature_cols=feature_cols,
                    label_col=label, **common_split, **run_kwargs)

    if canonical == "data_fusion":
        if model:
            ctor["model_name"] = _resolve_model_name(model)
        run_args.update(use_optimize=optimize, top_k=top_k)
        _put(run_args, "optimize_metric", metric_val)
        _put(run_args, "n_trials", n_trials)

    elif canonical == "feature_fusion":
        if model:
            ctor["model_name"] = model
        if metric_val:
            ctor["eval_metric"] = metric_val
        run_args.update(use_hpo=optimize, top_k=top_k)
        _put(run_args, "n_hpo_trials", n_trials)

    elif canonical in {"soft_voting", "stacking"}:
        ctor["text_models"] = text_models or []
        ctor["tabular_models"] = tabular_models or []
        run_args["top_k"] = top_k

    ctor.update(init_kwargs)
    clf = EnsembleClassifier(method=canonical, **ctor)
    return clf.run(**run_args)


def _put(d: dict, key: str, value) -> None:
    """Insert key only when value is not None (lets class defaults win)."""
    if value is not None:
        d[key] = value


def _resolve_model_name(model: str | None) -> str | None:
    """Resolve a text model name with :func:`multimodalva.text.models.resolve_model_name`.

    Package aliases map to Hugging Face Hub IDs, remote keys such as
    ``"roberta-pm"`` are downloaded once to ``~/.cache/multimodalva/``, and
    Hub IDs or local directories are returned unchanged. ``None`` is returned
    unchanged so pipeline defaults apply.
    """
    if not model:
        return model
    from .text.models import resolve_model_name

    return resolve_model_name(model)


# ---------------------------------------------------------------------------
# Diagnostics + summary
# ---------------------------------------------------------------------------
def _save_diagnostics(output_dir: Path, metric: str) -> None:
    try:
        trials_csv = _find_trials_csv(output_dir)
        if trials_csv is None:
            return
        from .results import hpo_leaderboard, hpo_convergence_plot

        trials_df = pd.read_csv(trials_csv)
        dest = trials_csv.parent
        hpo_leaderboard(trials_df).to_csv(dest / "hpo_leaderboard.csv", index=False)
        hpo_convergence_plot(
            trials_df, metric=metric,
            save_path=dest / "hpo_convergence.png", plot=False,
        )
        logger.info("Wrote HPO diagnostics to %s", dest)
    except Exception as exc:  # diagnostics are non-fatal
        logger.warning("HPO diagnostics failed (non-fatal): %s", exc)


def _find_trials_csv(output_dir: Path) -> Path | None:
    for name in ("hpo_trials_ray.csv", "hpo_trials.csv"):
        hits = sorted(output_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def _log_summary(results: dict, label: str) -> None:
    try:
        pred = results.get("predictions")
        if pred is None or getattr(pred, "top1", None) is None:
            return
        top1 = pred.top1
        from .utils.metrics import score_predictions

        parts = []
        for m in ("accuracy", "f1_macro", "csmf_accuracy"):
            try:
                parts.append(f"{m}={score_predictions(top1, m):.4f}")
            except Exception:
                pass
        if parts:
            logger.info("Test metrics — %s", "  ".join(parts))
    except Exception as exc:
        logger.debug("Summary metric computation skipped: %s", exc)
