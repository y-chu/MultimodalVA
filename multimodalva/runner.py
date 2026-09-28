"""
One-call, config-driven entry point for every MultimodalVA pipeline.

The classifier classes (:class:`TextClassifier`, :class:`TabularClassifier`,
:class:`EnsembleClassifier`) already chain split → prepare → HPO → train →
predict and save artifacts to ``output_dir/{final,hpo,predictions}``.  This
module adds the *surrounding* glue that every analysis script otherwise
re-implements — loading data, filtering rows, resolving feature columns,
optional fixed train/test split, diagnostics — behind a single function::

    from multimodalva import run

    run(task="text", data="clean.csv", label_col="cause",
        text_col="narrative", model="bluebert", output_dir="runs/bluebert",
        hyperparams=Optimize(n_trials=50))

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

from .utils.seeds import reject_removed_seed_args
from .utils.optimize_config import Optimize  # noqa: E402

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
    data: "str | Path | pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | None" = None,
    label_col: str | None = None,
    output_dir: "str | Path" = "runs/run",
    *,
    text_col: str | None = None,
    features: "list[str] | str | None" = None,
    model: str | None = None,
    filters: dict | None = None,
    hyperparams: "dict | str | Optimize | None" = None,
    # tabular preprocessing
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    encode_categoricals: str | None = "ordinal",
    scale_numeric: bool = False,
    # split
    test_size: float = 0.2,
    split_seed: int = 42,
    train_seed: int = 42,
    deterministic: bool = False,
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
        label_col: Label (cause-of-death) column name.
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
        hyperparams: Where the model's hyperparameters come from. Pass a dict
            to use exactly those values, ``Optimize(...)`` to search for them,
            or leave it out for the model library's own defaults. For ``voting``
            and ``stacking`` this applies to every base model that does not set
            its own in ``text_models`` / ``tabular_models``.
        cat_cols / num_cols / encode_categoricals / scale_numeric: tabular
            preprocessing options (tabular / voting / stacking tabular models).
        test_size / stratify: random-split controls (ignored when an external
            split is supplied).
        split_seed: Seed for every row-partitioning decision — the train/test
            split, each search's cross-validation folds, the early-stopping
            slice and stacking's out-of-fold partition. Vary it on its own to
            measure sampling uncertainty. Default 42.
        train_seed: Seed for everything a model does with the rows it is given —
            initialisation, dropout, batch order, the Optuna sampler, each
            estimator's ``random_state`` and AutoMM's trainer. Vary it on its
            own, with ``hyperparams`` fixed, to measure model stochasticity.
            Default 42. See :mod:`multimodalva.utils.seeds` for why the folds
            follow ``split_seed`` rather than this.
        deterministic: Demand bit-for-bit repeatable kernels. Slower, and an
            operation with no deterministic implementation raises instead of
            falling back. For the run behind a published number, not everyday
            use. Default False.
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

    Example:
        Fine-tune a clinical BERT on the narrative::

            import multimodalva as mv

            out = mv.run(task="text", data="clean.csv", label_col="cause",
                         text_col="narrative", model="bioclinicalbert",
                         output_dir="runs/text")
            out["predictions"].top1.head()

        The questionnaire indicators instead, with a hyperparameter search::

            out = mv.run(task="tabular", data="clean.csv", label_col="cause",
                         features="re:^i\\d{3}[a-zA-Z]$", model="lightgbm",
                         hyperparams=Optimize(n_trials=30),
                         output_dir="runs/tabular")

        Both together, fused before the model sees them::

            out = mv.run(task="data_fusion", data="clean.csv", label_col="cause",
                         text_col="narrative", features="auto",
                         model="clinicallongformer", output_dir="runs/fusion")

        Reusing one split across models so the numbers are comparable::

            out = mv.run(task="text", data=(train_df, test_df), label_col="cause",
                         text_col="narrative", output_dir="runs/fixed_split")
    """
    # Seed arguments that no longer exist would otherwise slip through
    # **run_kwargs into the classifier and fail there with an unhelpful
    # message. Catch them here and name the replacement.
    reject_removed_seed_args(
        f"run(task={task!r})",
        **{k: run_kwargs.pop(k) for k in ("random_state", "set_seed", "automm_seed")
           if k in run_kwargs},
    )
    if "resume_training" in run_kwargs:
        raise TypeError(
            "resume_training= is now resume=, the same name every task uses."
        )

    canonical = _normalize_task(task)
    # Stacking's stage 2 reads everything — rows, split, seeds, label maps —
    # from the run oof_from points at, so it needs no data of its own. Every
    # other call does; saying so here beats a TypeError from a positional
    # argument the caller deliberately left out.
    reuses_oof = canonical == "stacking" and run_kwargs.get("oof_from") is not None
    if not reuses_oof and (data is None or label_col is None):
        raise TypeError(
            f"run(task={task!r}) needs data= and label_col=. "
            "Only stacking with oof_from= may omit them: that call reuses the "
            "rows, split and seeds of the run it points at."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = dict(init_kwargs or {})

    # --- 1-3. Load, filter, drop-missing, resolve split marker --------------
    if reuses_oof:
        df, split_col = None, None
    else:
        df, split_col = _prepare_dataframe(
            data=data,
            canonical=canonical,
            label_col=label_col,
            text_col=text_col,
            filters=filters,
            split=split,
            id_col=id_col,
        )

    # --- 4. Resolve feature columns (tabular / fusion / tabular base models) --
    feature_cols = None
    if not reuses_oof and (
        canonical in _FEATURE_TASKS or (tabular_models and canonical in _ENSEMBLE_METHODS)
    ):
        feature_cols = _resolve_features(
            df, features, label_col, text_col, filters, split_col, id_col=id_col
        )
        logger.info("Running on %d feature columns.", len(feature_cols))

    if canonical in _TEXT_TASKS and not text_col:
        raise ValueError(f"task={task!r} requires text_col.")

    # --- 5. Dispatch --------------------------------------------------------
    results = _dispatch(
        canonical=canonical,
        df=df,
        label_col=label_col,
        text_col=text_col,
        feature_cols=feature_cols,
        model=model,
        output_dir=output_dir,
        hyperparams=hyperparams,
        cat_cols=cat_cols,
        num_cols=num_cols,
        encode_categoricals=encode_categoricals,
        scale_numeric=scale_numeric,
        test_size=test_size,
        split_seed=split_seed,
        train_seed=train_seed,
        deterministic=deterministic,
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
    # Not gated on hyperparams=: ensembles configure the search per base-model
    # spec, so the only reliable signal that one ran is a trials CSV on disk.
    if save_diagnostics:
        _save_diagnostics(output_dir, _diagnostics_metric(hyperparams))

    _log_summary(results, label_col)
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
    *, data, canonical, label_col, text_col, filters, split, id_col
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
    df = _drop_missing(df, canonical, label_col, text_col)
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


def _drop_missing(df, canonical, label_col, text_col) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(
            f"label column {label_col!r} not in DataFrame. "
            f"Available: {df.columns.tolist()}"
        )
    before = len(df)
    mask = df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")
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
def _resolve_features(df, features, label_col, text_col, filters, split_col, id_col=None) -> list[str]:
    if isinstance(features, list):
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns not in DataFrame: {missing}")
        if id_col is not None and id_col in features:
            raise ValueError(
                f"id_col {id_col!r} is also listed in features. A record identifier "
                "carries no information about the cause of death, and where it "
                "correlates with site or date it leaks. Remove it from features."
            )
        return features

    # An identifier is never a feature: it would be learned as noise, or as a
    # leak wherever ids track site, date or interviewer.
    reserved = {label_col}
    if id_col:
        reserved.add(id_col)
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
    *, canonical, df, label_col, text_col, feature_cols, model, output_dir,
    hyperparams, cat_cols, num_cols, encode_categoricals,
    scale_numeric, test_size, split_seed, train_seed, deterministic,
    stratify, split_col, top_k,
    text_models, tabular_models, init_kwargs, run_kwargs, id_col=None,
) -> dict:


    common_split = dict(
        test_size=test_size, split_seed=split_seed, train_seed=train_seed,
        deterministic=deterministic, stratify=stratify, split_col=split_col,
    )

    if canonical == "text":
        from .text.text_classifier import TextClassifier

        clf = TextClassifier(
            model_name=_resolve_model_name(model) or "bert-base-uncased",
            output_dir=output_dir,
        )
        kwargs = dict(
            df=df, text_col=text_col, label_col=label_col,
            hyperparams=hyperparams, top_k=top_k, **common_split, **run_kwargs,
        )
        _put(kwargs, "id_col", id_col)
        return clf.run(**kwargs)

    if canonical == "tabular":
        from .tabular.tabular_classifier import TabularClassifier

        clf = TabularClassifier(
            model_name=model or "random_forest", output_dir=output_dir
        )
        kwargs = dict(
            df=df, feature_cols=feature_cols, label_col=label_col,
            hyperparams=hyperparams, top_k=top_k,
            cat_cols=cat_cols, num_cols=num_cols,
            encode_categoricals=encode_categoricals, scale_numeric=scale_numeric,
            **common_split, **run_kwargs,
        )
        _put(kwargs, "id_col", id_col)
        return clf.run(**kwargs)

    # ---- ensemble strategies ----
    from .ensemble.ensemble_classifier import EnsembleClassifier

    ctor = dict(output_dir=output_dir)
    run_args = dict(df=df, text_col=text_col, feature_cols=feature_cols,
                    label_col=label_col, **common_split, **run_kwargs)
    # Every ensemble accepts id_col, so predictions from any task can be joined
    # back to the source records and matched against each other by id.
    _put(run_args, "id_col", id_col)

    if canonical == "data_fusion":
        if model:
            ctor["model_name"] = _resolve_model_name(model)
        run_args.update(hyperparams=hyperparams, top_k=top_k)

    elif canonical == "feature_fusion":
        if model:
            ctor["model_name"] = model
        _m = _diagnostics_metric(hyperparams)
        if _m:
            ctor["eval_metric"] = _m
        run_args.update(hyperparams=hyperparams, top_k=top_k)

    elif canonical in {"soft_voting", "stacking"}:
        if canonical == "stacking" and "resume" in run_kwargs:
            # Stacking holds resume on the object, because one object spans
            # several stage calls; every other task takes it on run().
            ctor["resume"] = run_kwargs.pop("resume")
        # Voting and stacking configure hyperparameters per base model, inside
        # each spec dict. A run-level hyperparams= applies to every spec that
        # does not already say otherwise, so the argument means the same thing
        # here as it does for a single-model task.
        # Preprocessing is a named argument of run(), so it is not in
        # run_kwargs; pass it on explicitly or the ensembles never see it.
        run_args.update(encode_categoricals=encode_categoricals,
                        scale_numeric=scale_numeric)
        ctor["text_models"] = _apply_hyperparams_to_specs(text_models, hyperparams)
        ctor["tabular_models"] = _apply_hyperparams_to_specs(tabular_models, hyperparams)
        run_args["top_k"] = top_k

    ctor.update(init_kwargs)
    clf = EnsembleClassifier(method=canonical, **ctor)
    return clf.run(**run_args)



def _apply_hyperparams_to_specs(specs, hyperparams):
    """Give every base model the run-level ``hyperparams`` it did not set itself.

    A spec that already says where its hyperparameters come from keeps saying
    so — the per-model setting is the more specific one. In particular a spec
    carrying explicit values is never overwritten by a run-level
    ``Optimize(...)``, because those values are usually the result of an earlier
    single-model search and searching again would discard them.

    Returns new dicts; the caller's list is not modified.
    """
    if not specs:
        return []
    if hyperparams is None:
        return [dict(s) for s in specs]
    out = []
    for spec in specs:
        spec = dict(spec)
        if "hyperparams" in spec:
            logger.info(
                "Base model %r sets its own hyperparams; the run-level value "
                "does not apply to it.", spec.get("model_name"),
            )
        else:
            spec["hyperparams"] = hyperparams
        out.append(spec)
    return out



def _diagnostics_metric(hyperparams) -> str:
    """The metric to label HPO diagnostics with.

    Whatever the caller asked a search to maximise, or the default objective
    when they did not configure one.
    """
    if isinstance(hyperparams, Optimize):
        return hyperparams.metric
    return "f1_macro"


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
    """Write hpo_leaderboard.csv and hpo_convergence.png beside every trials CSV.

    An ensemble runs one search per base model, so there is one trials CSV per
    ``base_models/*/hpo/`` as well as one for a single-model run. Each gets its
    own diagnostics, next to the trials file it came from.
    """
    trials_csvs = _find_trials_csvs(output_dir)
    if not trials_csvs:
        return
    from .results import hpo_leaderboard, hpo_convergence_plot

    for trials_csv in trials_csvs:
        dest = trials_csv.parent
        try:
            trials_df = pd.read_csv(trials_csv)
            hpo_leaderboard(trials_df).to_csv(dest / "hpo_leaderboard.csv", index=False)
            hpo_convergence_plot(
                trials_df, metric=metric,
                save_path=dest / "hpo_convergence.png", plot=False,
            )
            logger.info("Wrote HPO diagnostics to %s", dest)
        except Exception as exc:  # diagnostics are non-fatal
            logger.warning(
                "HPO diagnostics failed for %s (non-fatal): %s", dest, exc
            )


def _find_trials_csvs(output_dir: Path) -> list[Path]:
    """Every Optuna/Ray trials CSV under ``output_dir``, deduplicated by folder."""
    found: dict[Path, Path] = {}
    for name in ("hpo_trials_ray.csv", "hpo_trials.csv"):
        for hit in sorted(Path(output_dir).rglob(name)):
            found.setdefault(hit.parent, hit)
    return list(found.values())


def _log_summary(results: dict, label_col: str) -> None:
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
