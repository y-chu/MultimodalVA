"""``predict_from_pretrained()``: score new data with a model that is already trained."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..utils.predictions import assemble_predictions, resolve_test_ids, save_predictions
from ..utils.types import PredictionResult
from .backends import PRETRAINED_MISSING_FEATURE_METHODS, load_pretrained_backend
from .checks import PretrainedChecks, show_label_disclaimer

logger = logging.getLogger(__name__)

_HUB_URL_PREFIX = "https://huggingface.co/"


def predict_from_pretrained(
    source: str | Path,
    data: pd.DataFrame | str | Path,
    *,
    text_col: str | None = None,
    label_col: str | None = None,
    id_col: str | None = None,
    feature_cols: list[str] | None = None,
    missing_feature_method: str = "error",
    id2label: dict | None = None,
    task: str | None = None,
    force: bool = False,
    batch_size: int = 32,
    top_k: int = 3,
    save_dir: str | Path | None = None,
    backend_kwargs: dict | None = None,
) -> PredictionResult:
    """Load a trained model and predict on new data, labelled or not.

    ``source`` is a local directory (a run's ``output_dir`` or its ``final/``),
    a Hugging Face Hub repo id (``"org/name"``) or a Hub URL. Models trained
    outside MultimodalVA work too, as long as they say what their classes are
    and, for tabular models, what their feature columns are called.

    Args:
        source:         Where the model is.
        data:           New records: a DataFrame, or a CSV / Parquet path.
        text_col:       Text column (text and data-fusion models). A data-fusion
                        model expects the already-converted long text.
        label_col:      Ground-truth column, if the data is labelled. Then
                        ``top1`` / ``full`` / ``topk`` carry ``true_label`` and
                        every ``results`` function can score them. Without it
                        the ``true_label`` column is left out entirely, never
                        filled with NaN.
        id_col:         Column copied onto every table as a leading ``id``.
        feature_cols:   Tabular only, and only when the model does not record
                        its own column names. Columns are always matched by
                        name, never by position.
        missing_feature_method: ``"error"`` (default) when a trained-on column is
                        absent from ``data``, or ``"fill_na"`` to fill it with NA.
                        NA *values* in a present column are normal and never warn.
        id2label:       Class names, used only when the artifact has none.
        task:           Override task detection (``"text"``, ``"data_fusion"``,
                        ``"tabular"``, ``"feature_fusion"``). Detection still
                        runs: when it disagrees with a task the *run itself*
                        recorded, the call **raises** rather than spending an
                        expensive inference pass on the wrong pipeline. A guess
                        from the files present is overridden with a warning.
        force:          Proceed with ``task`` even against the run's own
                        recorded pipeline. Only reach for this when you know the
                        recorded metadata is wrong.
        batch_size:     Text inference batch size. Unrelated to the training
                        batch size: each batch is padded to its own longest
                        sequence, so the value shifts probabilities slightly —
                        measured at ~3e-8 on bert-tiny and ~1e-6 on a
                        bert-base run over 1000 records, with identical top-1
                        labels in both. Reproducing a saved
                        ``predictions_full.csv`` byte-for-byte needs the same value.
        top_k:          Classes listed in ``topk``.
        save_dir:       If given, the three tables are written there with
                        ``save_predictions()``.
        backend_kwargs: Passed to the backend's loader, e.g. ``max_length``,
                        ``use_fast`` (text), ``revision`` / ``token`` for the Hub.

    Returns:
        ``PredictionResult(top1, full, topk, id2label, checks)``. ``checks`` is a
        :class:`PretrainedChecks` recording what was detected and every warning.

    Label meaning is your responsibility: predictions use the artifact's label
    vocabulary, and whether another site's labels mean the same thing is a
    judgement the package cannot make.
    """
    if missing_feature_method not in PRETRAINED_MISSING_FEATURE_METHODS:
        raise ValueError(
            f"missing_feature_method must be one of "
            f"{PRETRAINED_MISSING_FEATURE_METHODS}; got {missing_feature_method!r}."
        )
    backend_kwargs = dict(backend_kwargs or {})
    hub_kwargs = {k: backend_kwargs.pop(k) for k in ("revision", "token")
                  if k in backend_kwargs}

    root = descend_to_pretrained_run(_resolve_pretrained_source(source, hub_kwargs))
    artifact_dir = _find_artifact_dir(root)
    meta = _read_training_metadata(root, artifact_dir)
    task, task_source, conflict = resolve_pretrained_task(
        meta, artifact_dir, task, force=force)

    checks = PretrainedChecks(
        source=str(source), artifact_dir=str(artifact_dir), task=task,
        task_source=task_source,
        multimodalva_version=meta.get("multimodalva_version"),
    )
    if conflict:
        checks.warn(conflict)
    logger.info("predict_from_pretrained(): task %r (from %s), model at %s",
                task, task_source, artifact_dir)
    if checks.multimodalva_version is None:
        checks.warn("The artifact does not record which MultimodalVA version "
                    "trained it (a foreign or older model); continuing.")

    backend = build_pretrained_backend(
        artifact_dir, task, checks, text_col=text_col, feature_cols=feature_cols,
        missing_feature_method=missing_feature_method, id2label=id2label,
        batch_size=batch_size, backend_kwargs=backend_kwargs,
    )

    df = _load_pretrained_input(data)
    true_labels = None
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"label_col {label_col!r} not found in the input.")
        true_labels = df[label_col].astype(str).where(df[label_col].notna())
        checks.labelled = True
        unknown = sorted(set(true_labels.dropna()) - set(backend.id2label.values()))
        if unknown:
            checks.warn(f"{len(unknown)} label(s) in {label_col!r} are not classes "
                        f"of this model and can never be predicted: {unknown[:10]}")
    ids = resolve_test_ids(df, id_col)
    inputs = backend.prepare_inputs(df.drop(columns=[c for c in (label_col, id_col)
                                                     if c is not None]))
    proba = backend.predict_proba(inputs)
    checks.n_rows = len(df)

    result = assemble_predictions(
        proba, backend.id2label,
        true_labels=None if true_labels is None else true_labels.tolist(),
        ids=ids, top_k=top_k,
    )
    if save_dir is not None:
        save_predictions(result, save_dir, "predictions")
    show_label_disclaimer()
    return result._replace(checks=checks)


def build_pretrained_backend(artifact_dir: Path, task: str, checks: PretrainedChecks,
                             *, text_col=None, feature_cols=None,
                             missing_feature_method="error", id2label=None,
                             batch_size=32, backend_kwargs=None):
    """Load the backend for *task* and let it report on the artifact.

    One place that knows which arguments each backend takes, so the single-model
    and ensemble entry points load a model the same way.
    """
    backend_cls = load_pretrained_backend(task)
    if task in ("tabular", "feature_fusion"):
        task_kwargs = dict(feature_cols=feature_cols,
                           missing_feature_method=missing_feature_method)
    else:
        task_kwargs = dict(text_col=text_col, batch_size=batch_size)
        logger.info("Inference batch_size=%d (adjustable with batch_size=).", batch_size)
    backend = backend_cls.from_pretrained(
        artifact_dir, checks, id2label=id2label, **task_kwargs,
        **(backend_kwargs or {})
    )
    backend.check_artifact()
    return backend


def _resolve_pretrained_source(source, hub_kwargs: dict) -> Path:
    """A local directory as is; a Hub repo id or URL downloaded to the HF cache."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    repo_id = str(source)
    if repo_id.startswith(_HUB_URL_PREFIX):
        repo_id = repo_id[len(_HUB_URL_PREFIX):].strip("/")
        repo_id = "/".join(repo_id.split("/")[:2])  # drop /tree/<rev>, /blob/...
    if repo_id.count("/") != 1:
        raise FileNotFoundError(
            f"{source!r} is neither a local directory nor a Hugging Face Hub repo "
            "id of the form 'org/name'."
        )
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(repo_id, **hub_kwargs))


# EnsembleClassifier writes each method into its own subdirectory of the
# output_dir it was given, so that root holds no run of its own.
_ENSEMBLE_SUBDIRS = ("soft_voting", "stacking", "data_fusion", "feature_fusion")


def descend_to_pretrained_run(root: Path) -> Path:
    """Step into ``output_dir/<method>/`` when that is where the run actually is.

    ``run(task="voting", output_dir=D)`` writes to ``D/soft_voting/``, so a
    caller who passes ``D`` means the one run inside it. Ambiguity is an error,
    never a guess.
    """
    if (root / "training_metadata.json").is_file():
        return root
    found = [root / name for name in _ENSEMBLE_SUBDIRS
             if (root / name / "training_metadata.json").is_file()]
    if len(found) == 1:
        logger.info("Using the run in %s.", found[0])
        return found[0]
    if len(found) > 1:
        raise ValueError(
            f"{root} holds several runs ({[d.name for d in found]}). Name the "
            "one you mean, e.g. source=<output_dir>/stacking."
        )
    return root


def _find_artifact_dir(root: Path) -> Path:
    """A run's ``output_dir`` keeps its weights in ``final/``; accept either."""
    final = root / "final"
    for d in (final, root):
        if ((d / "config.json").is_file() or (d / "model.joblib").is_file()
                or (d / "automm_model").is_dir()):
            return d
    return root


def _read_training_metadata(root: Path, artifact_dir: Path) -> dict:
    """Merge the run root's and ``final/``'s ``training_metadata.json``."""
    meta: dict = {}
    for d in (artifact_dir, root):  # the root's pipeline name wins
        p = d / "training_metadata.json"
        if p.is_file():
            try:
                meta.update(json.loads(p.read_text()))
            except ValueError:
                pass
    return meta


def _detect_pretrained_task(meta: dict, artifact_dir: Path) -> tuple[str, str]:
    """Task from recorded metadata first, then from the files present."""
    pipeline = meta.get("pipeline")
    if pipeline:
        return pipeline, "training_metadata"
    if (artifact_dir / "config.json").is_file():
        return "text", "structure"
    if (artifact_dir / "model.joblib").is_file():
        return "tabular", "structure"
    if (artifact_dir / "automm_model").is_dir():
        return "feature_fusion", "structure"
    raise ValueError(
        f"Cannot tell what kind of model is in {artifact_dir}: no "
        "training_metadata.json 'pipeline', no config.json (Hugging Face), no "
        "model.joblib, no automm_model/. Pass task=... if you know it."
    )


def resolve_pretrained_task(meta: dict, artifact_dir: Path, requested: str | None,
                            *, force: bool = False) -> tuple[str, str, str | None]:
    """Settle which task to use: the caller's, the artifact's, or an error.

    Detection always runs, even when the caller names a task, because the two
    disagreeing is worth knowing before a long inference pass starts — pointing
    at the wrong directory otherwise costs a whole GPU job and yields
    predictions that look fine.

    Returns:
        ``(task, task_source, conflict_note)``. ``conflict_note`` is a warning
        to record when the caller overrode a *guess*; it is ``None`` otherwise.

    Raises:
        ValueError: The caller's task contradicts the pipeline the run itself
            recorded and ``force`` is not set; or nothing could be determined
            and the caller named no task.
    """
    detected = detected_source = None
    try:
        detected, detected_source = _detect_pretrained_task(meta, artifact_dir)
    except ValueError:
        if requested is None:
            raise

    if requested is None:
        return detected, detected_source, None
    if detected is None:
        return requested, "caller", None
    if detected == requested:
        return requested, "caller", None

    if detected_source == "training_metadata" and not force:
        raise ValueError(
            f"task={requested!r} contradicts this run: its own "
            f"training_metadata.json says {detected!r}. Predicting anyway would "
            f"spend a full inference pass producing {requested!r} predictions "
            f"from a {detected!r} run. Check that {artifact_dir} is the "
            "directory you meant; pass force=True (CLI: --force) to override a "
            "recorded pipeline you know to be wrong."
        )
    how = ("the run's own metadata" if detected_source == "training_metadata"
           else "the files present")
    return requested, "caller", (
        f"Using task={requested!r} although {how} suggests {detected!r}"
        + (" (forced)." if force else ".")
    )


def _load_pretrained_input(data) -> pd.DataFrame:
    from ..runner import _load_df  # noqa: PLC0415 — the one data loader

    return _load_df(data).reset_index(drop=True)
