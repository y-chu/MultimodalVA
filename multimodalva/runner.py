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

__all__ = ["run", "preflight", "SUPPORTED_TASKS"]

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
    if "resume" in run_kwargs:
        # Checked here rather than at the guard inside the pipeline, which is
        # reached only after the data has been loaded and split.
        from .utils.provenance import _normalise_resume

        _normalise_resume(run_kwargs["resume"], f"run(task={task!r})")

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
    requires_text, requires_features = _required_modalities(
        canonical, text_models, tabular_models, init_kwargs
    )

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
            requires_text=requires_text,
        )

    # --- 4. Resolve feature columns (tabular / fusion / tabular base models) --
    feature_cols = None
    if not reuses_oof and (
        requires_features
    ):
        feature_cols = _resolve_features(
            df, features, label_col, text_col, filters, split_col, id_col=id_col
        )
        logger.info("Running on %d feature columns.", len(feature_cols))

    if requires_text and not text_col:
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
# Preflight
# ---------------------------------------------------------------------------

#: Destination class per single-model task, as ``(module, class name)``. The four
#: ensemble tasks read theirs from ``SUPPORTED_METHODS``, so that mapping is not
#: copied here and cannot drift from it.
_SINGLE_MODEL_TARGETS: dict[str, tuple[str, str]] = {
    "text": ("multimodalva.text.text_classifier", "TextClassifier"),
    "tabular": ("multimodalva.tabular.tabular_classifier", "TabularClassifier"),
}

#: Keyword names :func:`_dispatch` computes and supplies itself, and that are not
#: also named parameters of :func:`run`. One of these passed to :func:`run`
#: travels in ``**run_kwargs`` and arrives at the pipeline a second time, where
#: it raises a duplicate-keyword ``TypeError``.
_DISPATCH_SUPPLIED = frozenset({"df", "split_col", "model_name"})

#: Flat keywords :func:`_dispatch` deliberately routes to the *constructor* for a
#: given task, rather than forwarding to ``.run()``. Stacking keeps ``resume`` on
#: the object because one object spans several stage calls, so ``resume=`` is the
#: documented spelling there even though it is not a parameter of its ``run()``.
_DISPATCH_ROUTED_TO_CTOR: dict[str, frozenset[str]] = {
    "stacking": frozenset({"resume"}),
}

#: What each pipeline counts as "there is prior work here", mirroring the
#: ``artifacts_exist=`` expression at its own ``enforce_resume_manifest()`` call.
#: Preflight has to agree with those exactly, or it reports a directory as
#: resumable that the run then refuses (or the reverse). Kept in step by
#: ``test_every_task_has_a_resume_artifact_list``.
_RESUME_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "text":           ("hpo", "final", "predictions"),
    "tabular":        ("hpo", "final", "predictions"),
    "data_fusion":    ("hpo", "final", "predictions"),
    "feature_fusion": ("automm_model", "hpo", "predictions"),
    "soft_voting":    ("base_models", "predictions"),
    "stacking":       ("oof", "hpo", "final", "label2id.json"),
}


def _resume_dir(output_dir: "str | Path", canonical: str) -> Path:
    """Where the pipeline for ``canonical`` keeps its own artifacts.

    An ensemble strategy is handed ``output_dir / method`` by
    :class:`EnsembleClassifier`, so its manifest is one level down from the
    ``output_dir`` the caller passed.
    """
    root = Path(str(output_dir))
    return root / canonical if canonical in _ENSEMBLE_METHODS else root


class _LogCapture(logging.Handler):
    """Collects the INFO lines the package emits while a preflight runs.

    The interesting facts — how many rows the filters left, which columns the
    feature regex matched, which base model kept its own hyperparameters — are
    already logged by the functions a preflight calls. Capturing them is exact
    by construction, where a second implementation in the report would drift.
    """

    def __init__(self, sink: list[str]):
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record.getMessage())


def _run_default(name: str) -> Any:
    """The default :func:`run` gives ``name``, read from its own signature."""
    import inspect

    return inspect.signature(run).parameters[name].default


def _signature_names(fn) -> set[str] | None:
    """Parameter names ``fn`` accepts, or ``None`` when it takes ``**kwargs``.

    ``None`` means *not checkable*: a catch-all accepts every spelling, so
    calling one of them unknown would be a guess. Every pipeline class spells
    its arguments out today; this keeps the check honest if one stops.
    """
    import inspect

    sig = inspect.signature(fn)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return {p.name for p in sig.parameters.values() if p.name != "self"}


def _kwarg_targets(canonical: str) -> tuple[set[str] | None, set[str] | None, str]:
    """``(constructor params, run params, class name)`` for a task's pipeline class.

    Raises:
        ImportError: when the task's optional dependency is not installed. That
            is worth reporting rather than hiding: it is the failure a cluster
            job discovers only after it has waited for the queue.
    """
    import importlib

    if canonical in _SINGLE_MODEL_TARGETS:
        module, cls_name = _SINGLE_MODEL_TARGETS[canonical]
        cls = getattr(importlib.import_module(module), cls_name)
    else:
        from .ensemble.ensemble_classifier import SUPPORTED_METHODS

        cls = SUPPORTED_METHODS[canonical]
    return _signature_names(cls.__init__), _signature_names(cls.run), cls.__name__


def _describe_text_model(name: str) -> tuple[str, str | None]:
    """``(description, problem)`` for a text model name, **without downloading it**.

    A remote model is reported as a pending download rather than fetched: a
    preflight that pulls half a gigabyte is no longer a preflight.
    """
    import difflib

    from .text.models import REMOTE_MODELS, TEXT_MODELS, resolve_model_name

    if name in TEXT_MODELS:
        return f"{name} -> {TEXT_MODELS[name]}", None
    if name in REMOTE_MODELS:
        return f"{name} -> downloaded to ~/.cache/multimodalva on first use", None
    if Path(name).expanduser().is_dir():
        return f"{name} (local directory)", None
    try:
        # Everything that downloads has already returned; this only raises for
        # the spellings that could mean either biomedical RoBERTa.
        resolve_model_name(name)
    except ValueError as exc:
        return name, str(exc)
    # A Hub ID cannot be verified without network access, so an unknown name is
    # not an error. It usually is a typo though, and a close package alias says
    # so much more usefully than a download failure an hour later would.
    close = difflib.get_close_matches(name, sorted(TEXT_MODELS), n=1)
    if close and "/" not in name:
        return (f"{name} (no package alias of this name — did you mean "
                f"{close[0]!r}? Otherwise it is passed to Hugging Face as a Hub "
                "ID, which cannot be verified without network access)"), None
    return (f"{name} (no package alias of this name — passed to Hugging Face as a "
            "Hub ID, which cannot be verified without network access)"), None


def _describe_tabular_model(name: str) -> tuple[str, str | None]:
    """``(description, problem)`` for a tabular model name."""
    import difflib

    from .tabular.train import TABULAR_MODELS

    if name in TABULAR_MODELS:
        return f"{name} -> {TABULAR_MODELS[name][1]}", None
    close = difflib.get_close_matches(name, sorted(TABULAR_MODELS), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return name, (f"model={name!r} is not a tabular model. Choose from: "
                  f"{', '.join(sorted(TABULAR_MODELS))}.{hint}")


def preflight(**config: Any) -> dict:
    """Check a :func:`run` configuration without training anything.

    Takes the same keyword arguments as :func:`run` and performs every step that
    happens *before* a model is built — argument policing, loading, filtering,
    the missing-row drop, the train/test split, feature-column resolution — then
    stops. On a cluster that turns a mistyped column name from a job that
    queues, starts, and dies into an answer in seconds on the login node.

    Unlike :func:`run` it does not raise on a bad configuration. Every problem it
    finds is collected, so one call names all of them instead of only the first,
    and ``ok`` says whether :func:`run` would get past this point.

    Args:
        **config: Exactly what :func:`run` takes. Keywords :func:`run` does not
            name are checked against the signature of the class the task
            dispatches to, which is where an unrecognised one would land.

    Returns:
        A report dict:

        ``ok``
            ``False`` if anything in ``problems`` would stop the run.
        ``task``
            The canonical task name, or ``None`` if even that could not be read.
        ``problems``
            Configuration errors, each already phrased for a reader.
        ``notes``
            Things worth knowing that are not errors — a degradation the run
            would accept silently, a setting that overrides another.
        ``facts``
            Ordered label -> value pairs describing the run that would happen.
        ``log``
            The INFO lines the package emitted while checking.

    What a clean preflight does **not** promise. It checks that the
    configuration is consistent with the data, not that the run will succeed:
    nothing here trains, so it cannot tell whether a batch size fits in the GPU,
    whether the trial budget is enough for the search space, or whether an
    ``Optimize(extra=...)`` key means anything to the backend.

    Example:
        ::

            import multimodalva as mv

            report = mv.preflight(task="tabular", data="clean.csv",
                                  label_col="cause", features="re:^i\\d{3}$",
                                  model="lightgbm")
            if not report["ok"]:
                raise SystemExit("\\n".join(report["problems"]))
    """
    problems: list[str] = []
    notes: list[str] = []
    facts: dict[str, str] = {}
    captured: list[str] = []

    pkg_logger = logging.getLogger(__package__)
    handler = _LogCapture(captured)
    previous_level = pkg_logger.level
    pkg_logger.addHandler(handler)
    if not pkg_logger.isEnabledFor(logging.INFO):
        pkg_logger.setLevel(logging.INFO)
    try:
        canonical = _preflight_checks(dict(config), problems, notes, facts)
    finally:
        pkg_logger.removeHandler(handler)
        pkg_logger.setLevel(previous_level)

    return {
        "ok": not problems,
        "task": canonical,
        "problems": problems,
        "notes": notes,
        "facts": facts,
        "log": captured,
    }


def _preflight_checks(config: dict, problems: list[str], notes: list[str],
                      facts: dict[str, str]) -> str | None:
    """Every check :func:`preflight` runs, in dependency order.

    Later stages are skipped when what they need is already known to be broken,
    but independent stages all run, so one report names every problem at once.
    """
    import difflib
    import inspect

    # --- the task name ------------------------------------------------------
    task = config.get("task")
    try:
        canonical = _normalize_task(task)
    except (ValueError, KeyError) as exc:
        problems.append(str(exc))
        return None
    facts["task"] = canonical if canonical == task else f"{canonical} (from {task!r})"

    # --- arguments that no longer exist ------------------------------------
    removed = {k: config[k] for k in ("random_state", "set_seed", "automm_seed")
               if k in config}
    if removed:
        try:
            reject_removed_seed_args(f"run(task={task!r})", **removed)
        except TypeError as exc:
            problems.append(str(exc))
    if "resume_training" in config:
        problems.append(
            "resume_training= is now resume=, the same name every task uses."
        )

    # --- keywords the pipeline would not recognise ---------------------------
    try:
        ctor_params, run_params, cls_name = _kwarg_targets(canonical)
    except ImportError as exc:
        ctor_params = run_params = None
        cls_name = canonical
        problems.append(
            f"task={canonical!r} cannot be imported in this environment: {exc}. "
            "Install the extra it needs before submitting — on a cluster this is "
            "the failure that waits for the queue first."
        )
    facts["pipeline"] = cls_name
    init_kwargs = config.get("init_kwargs") or {}
    requires_text, requires_features = _required_modalities(
        canonical, config.get("text_models"), config.get("tabular_models"),
        init_kwargs if isinstance(init_kwargs, dict) else {},
    )

    named = set(inspect.signature(run).parameters) - {"run_kwargs"}
    already_reported = set(removed) | {"resume_training"}
    extras = sorted(k for k in config
                    if k not in named and k not in already_reported)
    if extras and ctor_params is None and run_params is None:
        notes.append(
            f"{cls_name} accepts **kwargs, so these could not be checked: "
            f"{', '.join(extras)}."
        )
    else:
        accepted = (ctor_params or set()) | (run_params or set())
        for key in extras:
            if key in _DISPATCH_SUPPLIED:
                problems.append(
                    f"{key}= is computed by run() itself. Passing it as well "
                    f"makes {cls_name} receive it twice, which raises there."
                )
                continue
            if key in (run_params or set()):
                continue
            if key in _DISPATCH_ROUTED_TO_CTOR.get(canonical, frozenset()):
                # run() moves this one to the constructor itself, so a flat
                # keyword is the documented spelling rather than a mistake.
                continue
            if key in (ctor_params or set()):
                problems.append(
                    f"{key}= is a constructor setting of {cls_name}, but flat "
                    "extra keywords are forwarded to .run(). Put it inside "
                    f"init_kwargs={{'{key}': ...}}."
                )
                continue
            close = difflib.get_close_matches(key, sorted(accepted | named), n=1)
            hint = f" Did you mean {close[0]}=?" if close else ""
            problems.append(f"{key}= is not accepted by {cls_name}.{hint}")

    if isinstance(init_kwargs, dict) and ctor_params:
        for key in sorted(init_kwargs):
            if key in ctor_params:
                if key in {"text_models", "tabular_models"} and config.get(key):
                    notes.append(
                        f"init_kwargs[{key!r}] and {key}= are both set. "
                        f"init_kwargs wins, so {key}= would have no effect."
                    )
                continue
            close = difflib.get_close_matches(key, sorted(ctor_params), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            problems.append(
                f"init_kwargs[{key!r}] is not accepted by {cls_name}(...).{hint}"
            )

    # --- the narrative column ----------------------------------------------
    text_col = config.get("text_col")
    if requires_text and not text_col:
        problems.append(f"task={task!r} requires text_col.")
    if canonical == "feature_fusion" and not requires_text and config.get("model"):
        notes.append(
            "fusion_strategy='tabular_only' does not use model=; no text "
            "checkpoint will be resolved or downloaded."
        )

    # --- the model ----------------------------------------------------------
    model = config.get("model")
    if model:
        if canonical in {"text", "data_fusion"}:
            described, problem = _describe_text_model(str(model))
        elif canonical == "tabular":
            described, problem = _describe_tabular_model(str(model))
        else:
            described, problem = str(model), None
        facts["model"] = described
        if problem:
            problems.append(problem)
    elif canonical in _ENSEMBLE_METHODS:
        facts["model"] = "per base-model spec"
    else:
        facts["model"] = "the pipeline default"

    effective_specs = {
        "text_models": init_kwargs.get("text_models", config.get("text_models"))
        if isinstance(init_kwargs, dict) else config.get("text_models"),
        "tabular_models": init_kwargs.get("tabular_models", config.get("tabular_models"))
        if isinstance(init_kwargs, dict) else config.get("tabular_models"),
    }
    if canonical == "soft_voting" and sum(
        len(effective_specs[k] or []) for k in effective_specs
    ) < 2:
        problems.append("voting requires at least 2 base models in total.")
    if (canonical == "stacking" and config.get("oof_from") is None
            and not any(effective_specs.values())):
        problems.append(
            "stacking requires at least one base model unless oof_from= reuses "
            "a completed stage 1."
        )

    if canonical in {"soft_voting", "stacking"}:
        from .utils.optimize_config import validate_base_model_specs

        if canonical == "soft_voting":
            from .ensemble.voting import (
                VOTING_TABULAR_SPEC_KEYS, VOTING_TEXT_SPEC_KEYS,
            )
            allowed_text, allowed_tabular = (
                VOTING_TEXT_SPEC_KEYS, VOTING_TABULAR_SPEC_KEYS,
            )
        else:
            from .ensemble.stacking import (
                STACKING_TABULAR_SPEC_KEYS, STACKING_TEXT_SPEC_KEYS,
            )
            allowed_text, allowed_tabular = (
                STACKING_TEXT_SPEC_KEYS, STACKING_TABULAR_SPEC_KEYS,
            )
        for key, allowed in (("text_models", allowed_text),
                             ("tabular_models", allowed_tabular)):
            try:
                validate_base_model_specs(
                    effective_specs[key], allowed, setting=key
                )
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))

    for key, describe in (("text_models", _describe_text_model),
                          ("tabular_models", _describe_tabular_model)):
        specs = effective_specs[key]
        if not specs:
            continue
        facts[key] = f"{len(specs)} spec(s)"
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = (spec or {}).get("model_name")
            if not name:
                problems.append(f"a {key} spec has no model_name.")
                continue
            described, problem = describe(str(name))
            if problem:
                problems.append(f"{key}: {problem}")
            elif "did you mean" in described:
                notes.append(f"{key}: {described}")

    # --- the data -----------------------------------------------------------
    df: "pd.DataFrame | None" = None
    train_preview: "pd.DataFrame | None" = None
    split_col: str | None = None
    label_col = config.get("label_col")
    reuses_oof = canonical == "stacking" and config.get("oof_from") is not None
    if reuses_oof:
        oof = Path(str(config["oof_from"]))
        facts["stage 2 only"] = f"rows, split and seeds come from {oof}"
        if not oof.exists():
            problems.append(f"oof_from={str(oof)!r} does not exist.")
    elif config.get("data") is None or label_col is None:
        problems.append(
            f"run(task={task!r}) needs data= and label_col=. Only stacking with "
            "oof_from= may omit them: that call reuses the rows, split and seeds "
            "of the run it points at."
        )
    else:
        try:
            df, split_col = _prepare_dataframe(
                data=config["data"], canonical=canonical, label_col=label_col,
                text_col=text_col, filters=config.get("filters"),
                split=config.get("split"), id_col=config.get("id_col"),
                requires_text=requires_text,
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            problems.append(str(exc))
            # "no rows remain" names the step that noticed, not the filter value
            # that emptied the frame. Say which value matched nothing, since a
            # misspelt filter is the usual reason a run ends up with no rows.
            notes.extend(_unmatched_filter_values(config))

    if df is not None:
        facts["rows"] = f"{len(df)} after filtering and the missing-row drop"
        counts = df[label_col].value_counts()
        facts["classes"] = str(len(counts))
        smallest, rarest = int(counts.iloc[-1]), counts.index[-1]
        facts["smallest class"] = f"{smallest} row(s) ({rarest!r})"
        # Perform the same split the real pipeline will perform. Besides making
        # the reported counts exact, this ensures HPO feasibility is checked on
        # the training side rather than on rows that will be held out for test.
        try:
            from .utils.split import split as split_dataframe

            train_preview, test_preview = split_dataframe(
                df,
                label_col=label_col,
                text_col=text_col if requires_text else None,
                test_size=config.get("test_size", _run_default("test_size")),
                random_state=config.get("split_seed", _run_default("split_seed")),
                stratify=config.get("stratify", _run_default("stratify")),
                split_col=split_col,
            )
            source = "supplied" if split_col is not None else "random"
            if split_col is None:
                facts["split"] = (
                    f"{source}: {len(train_preview)} train / {len(test_preview)} "
                    f"test (test_size={config.get('test_size', _run_default('test_size'))}, "
                    f"stratify={config.get('stratify', _run_default('stratify'))})"
                )
            else:
                facts["split"] = (
                    f"{source}: {len(train_preview)} train / {len(test_preview)} test"
                )
        except (ValueError, TypeError) as exc:
            problems.append(str(exc))

    # --- the feature columns ------------------------------------------------
    wants_features = requires_features
    if df is not None and wants_features:
        try:
            cols = _resolve_features(
                df, config.get("features"), label_col, text_col,
                config.get("filters"), split_col, id_col=config.get("id_col"),
            )
            facts["feature columns"] = (
                f"{len(cols)} resolved" + (f" (first: {cols[0]})" if cols else "")
            )
        except ValueError as exc:
            problems.append(str(exc))

    # --- resume -------------------------------------------------------------
    _preflight_resume(config, canonical, problems, notes, facts)

    # --- hyperparameters ----------------------------------------------------
    _preflight_hyperparams(
        config, canonical, train_preview, problems, notes, facts
    )
    return canonical


def _preflight_resume(config: dict, canonical: str, problems: list[str],
                      notes: list[str], facts: dict[str, str]) -> None:
    """Report what ``resume=`` would do to whatever is already in output_dir.

    The expensive mistake is discovering after the queue that a directory cannot
    be resumed, so this looks at the directory now.
    """
    from .utils.provenance import _normalise_resume

    try:
        resume, adopt = _normalise_resume(config.get("resume", True),
                                          f"run(task={canonical!r})")
    except ValueError as exc:
        problems.append(str(exc))
        return

    output_dir = _resume_dir(config.get("output_dir", _run_default("output_dir")),
                             canonical)
    manifest = output_dir / "resume_manifest.json"
    artifacts = [name for name in _RESUME_ARTIFACTS[canonical]
                 if (output_dir / name).exists()]

    if not resume:
        facts["resume"] = "off — artifacts in output_dir are rebuilt"
        return
    if not artifacts:
        facts["resume"] = "on — no reusable artifacts in output_dir yet"
        return
    present = ", ".join(artifacts)
    if manifest.is_file():
        facts["resume"] = (
            f"on — {present} present with a signature, reused only if it "
            "matches this call"
        )
    elif adopt:
        facts["resume"] = f"adopt — {present} present, no signature"
        notes.append(
            f'resume="adopt" will reuse the existing {present} in {output_dir} '
            "without verifying them: nothing recorded what data, split or "
            "settings produced them. The signature of this call is written, so "
            "later runs are checked normally. Use resume=False to rebuild them."
        )
    else:
        facts["resume"] = f"BLOCKED — {present} present, no signature"
        problems.append(
            f"{output_dir} already holds {present} but no resume_manifest.json, "
            "so resume=True cannot verify they came from this data and "
            'configuration. Use a new output_dir, resume="adopt" to accept them '
            "deliberately (runs made before resume manifests existed), or "
            "move/remove them and rerun."
        )


def _unmatched_filter_values(config: dict) -> list[str]:
    """Filter values that match no row, for when the data stage left nothing.

    Reloads the data — only on the failure path, where a second read costs far
    less than the wrong diagnosis does.
    """
    filters = config.get("filters")
    if not filters:
        return []
    try:
        raw = _load_df(config.get("data"))
    except Exception:  # the load itself is what failed; nothing to add
        return []
    out = []
    for col, value in filters.items():
        if col not in raw.columns:
            continue
        wanted = list(value) if isinstance(value, (list, tuple, set)) else [value]
        present = set(raw[col].dropna().unique())
        missing = [v for v in wanted if v not in present]
        if missing:
            sample = sorted(str(v) for v in present)[:8]
            out.append(
                f"filters[{col!r}]: {missing} match no row. The column holds "
                f"{len(present)} distinct value(s), e.g. {sample}."
            )
    return out


def _preflight_hyperparams(config: dict, canonical: str, train_df,
                           problems: list[str],
                           notes: list[str], facts: dict[str, str]) -> None:
    """Report where hyperparameters come from, and what the class sizes allow.

    The class-size checks are the ones worth having: a search splits the
    training rows again, and a class too small to appear in every fold is the
    difference between a silent degradation and a run that stops.
    """
    hyperparams = config.get("hyperparams")
    if hyperparams is None:
        facts["hyperparameters"] = "the model library's own defaults — no search"
        return
    if not isinstance(hyperparams, Optimize):
        n = len(hyperparams) if isinstance(hyperparams, dict) else "?"
        facts["hyperparameters"] = f"fixed: {n} value(s) as given — no search"
        return

    import os

    from .utils.optimize_config import ray_is_available, resolve_backend

    facts["hyperparameters"] = (
        f"search: metric={hyperparams.metric}, n_trials="
        f"{hyperparams.n_trials if hyperparams.n_trials else 'pipeline default'}"
    )

    if canonical == "feature_fusion":
        from .ensemble.feature_fusion import FEATURE_FUSION_HONOURED_SETTINGS
        from .utils.optimize_config import warn_unused_settings

        dropped = warn_unused_settings(
            hyperparams,
            FEATURE_FUSION_HONOURED_SETTINGS,
            pipeline="feature_fusion",
            note=(
                "AutoMM uses hpo_scheduler=/hpo_searcher=, its own holdout, "
                "and the classifier's eval_metric=."
            ),
            log=logger,
        )
        if dropped:
            notes.append(
                "feature_fusion ignores these explicitly changed Optimize "
                f"settings: {', '.join(dropped)}. Configure AutoMM with "
                "init_kwargs={'eval_metric': ...} and the pipeline's "
                "hpo_scheduler=/hpo_searcher= arguments instead."
            )
        facts["hpo backend"] = "AutoMM built-in search"
        facts["search split"] = "AutoMM's own training holdout"
        if hyperparams.space:
            facts["search space"] = (
                f"{len(hyperparams.space)} AutoMM key(s) given; package "
                "defaults supply the remaining search dimensions"
            )
        else:
            facts["search space"] = "the package's default AutoMM space"
        return

    requested = hyperparams.backend
    resolved = resolve_backend(requested, warn=False)
    facts["hpo backend"] = (
        resolved if resolved == requested else f"{resolved} (requested {requested!r})"
    )
    # An explicit backend='ray' is never downgraded: the search raises an
    # ImportError naming the extra. That is the right behaviour and the wrong
    # moment — after the job has queued — so say it here instead.
    if requested == "ray" and not ray_is_available():
        problems.append(
            "backend='ray' was asked for explicitly but ray is not installed in "
            "this environment. An explicit request is never downgraded, so the "
            "search raises instead of falling back: pip install "
            "'multimodalva[ray]', or pass backend='auto'."
        )
    elif (requested == "auto" and resolved == "optuna"
            and os.environ.get("SLURM_JOB_ID") and not ray_is_available()):
        notes.append(
            "backend='auto' resolves to Optuna here because ray is not installed, "
            "on a machine where it would have run trials in parallel. The results "
            "will be correct, only slower: pip install 'multimodalva[ray]'."
        )
    if hyperparams.space:
        facts["search space"] = (
            f"{len(hyperparams.space)} key(s) given and used exactly as passed "
            f"({', '.join(sorted(hyperparams.space))}); every other key of the "
            "adaptive space for this data is searched as well"
        )
    else:
        facts["search space"] = "the package default, adapted to this data"

    if train_df is None:
        return
    label_col = config.get("label_col")
    train_labels = train_df[label_col].to_numpy()
    smallest = int(train_df[label_col].value_counts().iloc[-1])
    if hyperparams.cv:
        # Ask the resolver the searches use, so preflight cannot disagree with
        # them about what this data supports.
        from .utils.optimize_config import (
            CV_FOLDS_AUTO, CV_FOLDS_DEFAULT, resolve_cv_folds,
        )

        quiet = logging.getLogger(f"{__name__}._preflight_quiet")
        quiet.propagate = False
        try:
            folds = resolve_cv_folds(hyperparams.cv_folds, train_labels,
                                     where="the search", log=quiet)
        except ValueError as exc:
            facts["search split"] = (
                "cross-validation is not possible on this data"
                if hyperparams.cv_folds == CV_FOLDS_AUTO
                else f"{hyperparams.cv_folds}-fold cross-validation (not possible)"
            )
            problems.append(str(exc))
        else:
            if hyperparams.cv_folds == CV_FOLDS_AUTO and folds < CV_FOLDS_DEFAULT:
                facts["search split"] = (
                    f'{folds}-fold cross-validation (cv_folds="auto" reduced it '
                    f"from {CV_FOLDS_DEFAULT}; the rarest training class has "
                    f"{smallest} row(s))"
                )
                notes.append(
                    f'cv_folds="auto" will use {folds} folds instead of '
                    f"{CV_FOLDS_DEFAULT}, because the rarest class in the "
                    f"training split has {smallest} row(s). Trial scores average "
                    f"over {folds} folds, so they are noisier than a "
                    f"{CV_FOLDS_DEFAULT}-fold search and not comparable with one."
                )
            else:
                facts["search split"] = f"{folds}-fold cross-validation"
    else:
        facts["search split"] = "a single stratified holdout (cv=False)"
        if smallest < 2:
            notes.append(
                "the training split contains a singleton class, so the HPO "
                "holdout cannot be stratified and will use a seeded random "
                "split. That class may be absent from one side."
            )


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


def _required_modalities(
    canonical: str,
    text_models: list[dict] | None,
    tabular_models: list[dict] | None,
    init_kwargs: dict | None,
) -> tuple[bool, bool]:
    """Return whether a call actually consumes text and tabular columns.

    Feature-fusion ablations and voting/stacking model rosters are modality
    choices, not cosmetic settings. Centralising this prevents the loader,
    preflight and dispatcher from disagreeing about columns that are required,
    dropped for missing values, or resolved as features.
    """
    init_kwargs = init_kwargs or {}
    if canonical == "text":
        return True, False
    if canonical == "tabular":
        return False, True
    if canonical == "data_fusion":
        return True, True
    if canonical == "feature_fusion":
        strategy = init_kwargs.get("fusion_strategy", "default")
        return strategy != "tabular_only", strategy != "text_only"
    if canonical in {"soft_voting", "stacking"}:
        effective_text = init_kwargs.get("text_models", text_models)
        effective_tabular = init_kwargs.get("tabular_models", tabular_models)
        return bool(effective_text), bool(effective_tabular)
    return False, False


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
    *, data, canonical, label_col, text_col, filters, split, id_col,
    requires_text: bool | None = None,
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
    if requires_text is None:
        requires_text = canonical in _TEXT_TASKS
    df = _drop_missing(df, label_col, text_col, requires_text=requires_text)
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
    overlap = train_ids & test_ids
    if overlap:
        shown = sorted(map(str, overlap))[:10]
        raise ValueError(
            "The id-based split assigns the same record(s) to both train and "
            f"test: {shown}. The two sets must be disjoint."
        )
    if df[id_col].isna().any():
        raise ValueError(
            f"id_col {id_col!r} contains missing values; an id-based split "
            "requires one non-missing identifier per row."
        )
    duplicated = df.loc[df[id_col].duplicated(keep=False), id_col]
    if not duplicated.empty:
        shown = sorted(map(str, duplicated.unique()))[:10]
        raise ValueError(
            f"id_col {id_col!r} is not unique; duplicated identifier(s): "
            f"{shown}. A row identifier must identify exactly one record."
        )
    present = set(df[id_col])
    absent = (train_ids | test_ids) - present
    if absent:
        shown = sorted(map(str, absent))[:10]
        raise ValueError(
            f"The supplied split names {len(absent)} id(s) that are not in "
            f"{id_col!r}, e.g. {shown}. Refusing a partial/mistyped split."
        )

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


def _drop_missing(df, label_col, text_col, *, requires_text: bool) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(
            f"label column {label_col!r} not in DataFrame. "
            f"Available: {df.columns.tolist()}"
        )
    if requires_text:
        if not text_col:
            raise ValueError("text_col is required for this pipeline.")
        if text_col not in df.columns:
            raise ValueError(
                f"text column {text_col!r} is not in the data. "
                f"Available: {df.columns.tolist()}"
            )
    before = len(df)
    mask = df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")
    if requires_text and text_col and text_col in df.columns:
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

        ctor = dict(
            model_name=_resolve_model_name(model) or "bert-base-uncased",
            output_dir=output_dir,
        )
        ctor.update(init_kwargs)
        clf = TextClassifier(**ctor)
        kwargs = dict(
            df=df, text_col=text_col, label_col=label_col,
            hyperparams=hyperparams, top_k=top_k, **common_split, **run_kwargs,
        )
        _put(kwargs, "id_col", id_col)
        return clf.run(**kwargs)

    if canonical == "tabular":
        from .tabular.tabular_classifier import TabularClassifier

        ctor = dict(
            model_name=model or "random_forest", output_dir=output_dir
        )
        ctor.update(init_kwargs)
        clf = TabularClassifier(**ctor)
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
    # Stacking holds resume on the object, because one object spans several stage
    # calls; every other pipeline takes it on run(). This has to happen *before*
    # run_args is built: popping it from run_kwargs afterwards leaves the copy
    # that **run_kwargs already made, and StackingClassifier.run() has no such
    # parameter, so `run(task="stacking", resume=...)` and the CLI's
    # --no-resume died with "unexpected keyword argument 'resume'".
    if canonical == "stacking" and "resume" in run_kwargs:
        ctor["resume"] = run_kwargs.pop("resume")
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
        run_args.update(hyperparams=hyperparams, top_k=top_k)

    elif canonical in {"soft_voting", "stacking"}:
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
